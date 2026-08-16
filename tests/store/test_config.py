"""DSN normalisation and secret redaction. No database required."""

from __future__ import annotations

import ssl

import pytest

from fxagent.store.config import DatabaseConfig, DatabaseConfigError

SUPABASE_POOLER = (
    "postgresql://postgres.abcdefghijklm:s3cr3t-p4ss"
    "@aws-0-eu-west-2.pooler.supabase.com:5432/postgres?sslmode=require"
)
LOCAL = "postgresql://fxagent:fxagent@localhost:15432/fxagent_test"


def test_scheme_is_rewritten_for_asyncpg() -> None:
    config = DatabaseConfig.from_url(LOCAL)
    assert config.url.startswith("postgresql+asyncpg://")


@pytest.mark.parametrize("scheme", ["postgres", "postgresql", "postgresql+psycopg2"])
def test_common_schemes_are_accepted(scheme: str) -> None:
    config = DatabaseConfig.from_url(f"{scheme}://u:p@example.com:5432/db")
    assert config.url.startswith("postgresql+asyncpg://")


def test_sslmode_is_translated_and_removed_from_the_query() -> None:
    """asyncpg rejects `sslmode`; Supabase's copy-paste string always carries it."""
    config = DatabaseConfig.from_url(SUPABASE_POOLER)

    assert "sslmode" not in config.url
    assert isinstance(config.connect_args["ssl"], ssl.SSLContext)


@pytest.mark.parametrize("mode", ["disable", "allow"])
def test_sslmode_can_turn_tls_off(mode: str) -> None:
    config = DatabaseConfig.from_url(f"postgresql://u:p@example.com:5432/db?sslmode={mode}")
    assert config.connect_args["ssl"] is False


@pytest.mark.parametrize("mode", ["prefer", "require"])
def test_require_encrypts_without_verifying(mode: str) -> None:
    """libpq's `require` promises encryption and says nothing about trust.

    Mapping it to `ssl=True` would make asyncpg verify the chain and the hostname, which is
    `verify-full` — stricter than asked, and the reason Supabase's own connection string
    could not connect: its pooler serves a self-signed certificate on the Postgres port.
    """
    context = DatabaseConfig.from_url(
        f"postgresql://u:p@example.com:5432/db?sslmode={mode}"
    ).connect_args["ssl"]

    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode is ssl.CERT_NONE
    assert context.check_hostname is False


def test_verify_ca_checks_the_chain_but_not_the_hostname() -> None:
    context = DatabaseConfig.from_url(
        "postgresql://u:p@example.com:5432/db?sslmode=verify-ca"
    ).connect_args["ssl"]

    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is False


def test_verify_full_is_the_only_mode_that_checks_the_hostname() -> None:
    config = DatabaseConfig.from_url("postgresql://u:p@example.com:5432/db?sslmode=verify-full")
    assert config.connect_args["ssl"] is True


def test_the_modes_are_ordered_from_weakest_to_strongest() -> None:
    """A regression guard: the four TLS modes must not collapse onto one another again."""

    def posture(mode: str) -> object:
        value = DatabaseConfig.from_url(
            f"postgresql://u:p@example.com:5432/db?sslmode={mode}"
        ).connect_args["ssl"]
        if isinstance(value, ssl.SSLContext):
            return (value.verify_mode, value.check_hostname)
        return value

    assert posture("disable") is False
    assert posture("require") != posture("verify-ca") != posture("verify-full")
    assert posture("require") != posture("verify-full")


def test_unknown_sslmode_is_rejected_rather_than_guessed() -> None:
    with pytest.raises(DatabaseConfigError, match="sslmode"):
        DatabaseConfig.from_url("postgresql://u:p@example.com/db?sslmode=maybe")


def test_remote_host_gets_full_verification_when_no_sslmode_is_given() -> None:
    """Strict by default. An omitted parameter must not silently buy the weaker posture."""
    config = DatabaseConfig.from_url("postgresql://u:p@db.abc.supabase.co:5432/postgres")
    assert config.connect_args["ssl"] is True


def test_local_host_does_not_get_tls() -> None:
    """The test container speaks plaintext; forcing TLS would fail every db test."""
    config = DatabaseConfig.from_url(LOCAL)
    assert "ssl" not in config.connect_args


def test_transaction_pooler_port_disables_the_statement_cache() -> None:
    """Port 6543 is pgbouncer in transaction mode, where prepared statements collide."""
    config = DatabaseConfig.from_url(SUPABASE_POOLER.replace(":5432/", ":6543/"))

    assert config.uses_transaction_pooler is True
    assert config.connect_args["statement_cache_size"] == 0


def test_session_pooler_port_keeps_the_statement_cache() -> None:
    config = DatabaseConfig.from_url(SUPABASE_POOLER)

    assert config.uses_transaction_pooler is False
    assert "statement_cache_size" not in config.connect_args


def test_repr_never_contains_the_password() -> None:
    """This object reaches logs, tracebacks and pytest output."""
    config = DatabaseConfig.from_url(SUPABASE_POOLER)

    for rendered in (repr(config), str(config), config.safe_url, f"{config}"):
        assert "s3cr3t-p4ss" not in rendered
    assert "***" in repr(config)
    assert "aws-0-eu-west-2.pooler.supabase.com" in repr(config), "host stays, for diagnosis"


def test_supabase_url_alone_gives_an_actionable_error() -> None:
    """SUPABASE_URL is PostgREST's endpoint and cannot open a Postgres connection."""
    with pytest.raises(DatabaseConfigError, match="PostgREST endpoint"):
        DatabaseConfig.from_env({"SUPABASE_URL": "https://abcdefg.supabase.co"})


def test_missing_configuration_is_rejected() -> None:
    with pytest.raises(DatabaseConfigError, match="no database URL configured"):
        DatabaseConfig.from_env({})


def test_database_url_is_the_fallback() -> None:
    config = DatabaseConfig.from_env({"DATABASE_URL": LOCAL})
    assert config.url.startswith("postgresql+asyncpg://")


def test_supabase_db_url_wins_over_database_url() -> None:
    config = DatabaseConfig.from_env({"SUPABASE_DB_URL": SUPABASE_POOLER, "DATABASE_URL": LOCAL})
    assert "pooler.supabase.com" in config.url


def test_malformed_url_is_rejected() -> None:
    with pytest.raises(DatabaseConfigError):
        DatabaseConfig.from_url("not-a-url")


def test_non_postgres_scheme_is_rejected() -> None:
    with pytest.raises(DatabaseConfigError, match="unsupported database scheme"):
        DatabaseConfig.from_url("mysql://u:p@example.com/db")


# --- connection hints ---------------------------------------------------------------


def test_a_self_signed_certificate_failure_names_the_fix() -> None:
    """A deployment with the right host, the right pooler port and the right credentials
    failed every request on this, sixty lines down a serverless traceback. The DSN was missing
    `?sslmode=require`, which .env.example warns about and nobody reads while pasting a URL
    into a hosting dashboard."""
    from fxagent.store.config import connection_hint

    error = Exception(
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
        "self-signed certificate in certificate chain (_ssl.c:1010)"
    )
    hint = connection_hint(error)

    assert hint is not None
    assert "sslmode=require" in hint


def test_an_unrecognised_failure_gets_no_invented_hint() -> None:
    """Guessing at a cause is worse than saying nothing: it sends the reader somewhere."""
    from fxagent.store.config import connection_hint

    assert connection_hint(OSError("connection refused")) is None


def test_a_dsn_with_no_sslmode_still_verifies_in_full() -> None:
    """The strict default, kept on purpose even though it is what refuses Supabase's pooler.

    `require` means encrypt-without-verifying. Inferring it from the hostname — "this looks
    like Supabase, so stop checking the certificate" — would downgrade someone else's TLS on
    their behalf, silently, to save them typing eight characters. So the default stays strict,
    the connection fails, and `connection_hint` says which eight characters to type.
    """
    from fxagent.store.config import DatabaseConfig

    config = DatabaseConfig.from_url(
        "postgresql://user:pw@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"
    )

    assert "sslmode" not in config.url
    assert config.connect_args["ssl"] is True

    relaxed = DatabaseConfig.from_url(
        "postgresql://user:pw@aws-1-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require"
    )

    assert relaxed.connect_args["ssl"] is not True  # an unverified context, not a bare bool
