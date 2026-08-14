"""Database connection settings, read from the environment and never logged.

Three Supabase-specific traps are handled here rather than left for a caller to trip over:

**SUPABASE_URL is not a database URL.** It is the PostgREST endpoint
(``https://<ref>.supabase.co``) and no Postgres driver can connect to it. The direct
connection string lives under Settings > Database, and is a separate value.

**`sslmode` is a libpq parameter and asyncpg rejects it.** Supabase's copy-paste connection
strings carry ``?sslmode=require``, so pasting one in unmodified fails at connect with an
obscure ``unexpected keyword argument``. It is translated here into asyncpg's own ``ssl``,
preserving each mode's real meaning — ``require`` encrypts without verifying, and is *not*
``verify-full``. See `_ssl_from_sslmode`; getting that distinction wrong makes Supabase's own
connection string impossible to use, because its pooler serves a self-signed certificate on
the Postgres port even though the same hostname serves a properly signed one on 443.

**The transaction pooler forbids prepared statements.** Port 6543 is pgbouncer in transaction
mode, where SQLAlchemy's default statement caching breaks with ``prepared statement "__asyncpg_
stmt_1__" already exists``. Detected by port and disabled automatically.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

__all__ = ["DatabaseConfig", "DatabaseConfigError"]

#: pgbouncer in transaction mode. Prepared statements do not survive between statements here.
TRANSACTION_POOLER_PORT = 6543

#: Query parameters that are libpq's, not asyncpg's. Carried in every Supabase copy-paste
#: string; passing them through to asyncpg raises TypeError at connect time.
_LIBPQ_ONLY_PARAMS = frozenset({"sslmode", "target_session_attrs", "options", "connect_timeout"})

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "postgres", "db"})


class DatabaseConfigError(RuntimeError):
    """The database configuration is missing or unusable. Never contains the password."""


def _redact(url: str) -> str:
    """Render a DSN with the password replaced. Safe to log, and the only form that is."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable dsn>"
    if parts.hostname is None:
        return "<unparseable dsn>"
    user = parts.username or ""
    auth = f"{user}:***@" if parts.password else (f"{user}@" if user else "")
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{auth}{parts.hostname}{port}{parts.path}"


@dataclass(frozen=True)
class DatabaseConfig:
    """Everything needed to open a pool, with the password held but never rendered."""

    url: str
    pool_size: int = 5
    max_overflow: int = 5
    pool_timeout_seconds: float = 30.0
    pool_recycle_seconds: int = 1800
    command_timeout_seconds: float = 30.0
    connect_args: dict[str, Any] = field(default_factory=dict)
    uses_transaction_pooler: bool = False

    def __repr__(self) -> str:
        """Redacted on purpose. This object reaches logs, tracebacks and pytest output."""
        return (
            f"DatabaseConfig(url={_redact(self.url)!r}, pool_size={self.pool_size}, "
            f"max_overflow={self.max_overflow}, "
            f"uses_transaction_pooler={self.uses_transaction_pooler})"
        )

    __str__ = __repr__

    @property
    def safe_url(self) -> str:
        """The DSN with its password masked — the only form permitted in a log line."""
        return _redact(self.url)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **overrides: Any) -> DatabaseConfig:
        """Build from ``SUPABASE_DB_URL``, falling back to ``DATABASE_URL`` for local runs."""
        source = os.environ if env is None else env

        raw = (source.get("SUPABASE_DB_URL") or source.get("DATABASE_URL") or "").strip()
        if not raw:
            if (source.get("SUPABASE_URL") or "").strip():
                raise DatabaseConfigError(
                    "SUPABASE_URL is set but SUPABASE_DB_URL is not. SUPABASE_URL is the "
                    "PostgREST endpoint and cannot be used to open a Postgres connection. "
                    "Copy the connection string from Supabase > Settings > Database and set it "
                    "as SUPABASE_DB_URL — prefer the Session pooler (IPv4) unless the host has "
                    "IPv6."
                )
            raise DatabaseConfigError(
                "no database URL configured; set SUPABASE_DB_URL (or DATABASE_URL for a local "
                "Postgres). See .env.example under STORE."
            )

        return cls.from_url(raw, **overrides)

    @classmethod
    def from_url(cls, raw: str, **overrides: Any) -> DatabaseConfig:
        """Normalise a DSN into something the asyncpg dialect actually accepts."""
        url, connect_args, port = _normalise(raw)

        uses_pooler = port == TRANSACTION_POOLER_PORT
        if uses_pooler:
            # pgbouncer hands out a different backend per statement, so a cached prepared
            # statement name collides with one from another session.
            connect_args["statement_cache_size"] = 0

        defaults: dict[str, Any] = {
            "url": url,
            "connect_args": connect_args,
            "uses_transaction_pooler": uses_pooler,
        }
        defaults.update(overrides)
        config = cls(**defaults)
        config.connect_args.setdefault("command_timeout", config.command_timeout_seconds)
        return config


def _normalise(raw: str) -> tuple[str, dict[str, Any], int | None]:
    """Rewrite the scheme for asyncpg and translate libpq-only query parameters."""
    parts = urlsplit(raw)
    if not parts.scheme:
        raise DatabaseConfigError(
            f"database URL has no scheme: {_redact(raw)}. Expected postgresql://..."
        )
    if not parts.hostname:
        raise DatabaseConfigError(f"database URL has no host: {_redact(raw)}")

    scheme = parts.scheme
    if scheme in ("postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"):
        scheme = "postgresql+asyncpg"
    elif scheme != "postgresql+asyncpg":
        raise DatabaseConfigError(
            f"unsupported database scheme {parts.scheme!r}; expected a postgresql:// URL"
        )

    connect_args: dict[str, Any] = {}
    kept: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key not in _LIBPQ_ONLY_PARAMS:
            kept.append((key, value))
            continue
        if key == "sslmode":
            ssl = _ssl_from_sslmode(value)
            if ssl is not None:
                connect_args["ssl"] = ssl
        # Remaining libpq-only params are dropped rather than forwarded; asyncpg would raise.

    host = parts.hostname
    if "ssl" not in connect_args and host not in _LOCAL_HOSTS:
        # A remote host with no explicit sslmode gets full verification. Deliberately the
        # strict end: an omitted parameter should not silently buy the weaker posture, and
        # a URL that must not verify can say so with `?sslmode=require`. Against Supabase
        # this fails with "self-signed certificate in certificate chain" until it does.
        connect_args["ssl"] = True

    normalised = urlunsplit((scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))
    return normalised, connect_args, parts.port


def _ssl_from_sslmode(value: str) -> bool | ssl.SSLContext | None:
    """Map libpq's sslmode onto what asyncpg accepts, preserving libpq's actual semantics.

    **`require` is not `verify-full`.** In libpq, `require` means "the connection must be
    encrypted" and deliberately says nothing about trusting the certificate; only the two
    `verify-*` modes check it. Collapsing them onto `ssl=True` — which makes asyncpg build a
    fully verifying context — silently asks for something *stricter* than the caller wrote,
    and that is not a safe direction to be wrong in either: it fails closed against Supabase,
    whose pooler presents a self-signed certificate on the Postgres port, so the connection
    string Supabase itself hands you cannot connect.

    So the modes are mapped to what they actually mean:

    * `disable`/`allow` — no TLS.
    * `prefer`/`require` — encrypt, do not verify. Safe against passive eavesdropping, not
      against an active man-in-the-middle. This is what Supabase's own string asks for.
    * `verify-ca` — verify the chain, allow a hostname mismatch.
    * `verify-full` — verify chain and hostname, against the system trust store. Supabase
      needs its CA certificate installed for this to succeed.
    """
    mode = value.strip().lower()
    if mode in ("disable", "allow"):
        return False
    if mode in ("prefer", "require"):
        return _unverified_context()
    if mode == "verify-ca":
        context = ssl.create_default_context()
        context.check_hostname = False
        return context
    if mode == "verify-full":
        return True
    raise DatabaseConfigError(f"unrecognised sslmode {value!r} in database URL")


def _unverified_context() -> ssl.SSLContext:
    """Encrypt without verifying, which is what libpq's `require` promises and no more."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context
