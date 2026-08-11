"""DSN normalisation and secret redaction. No database required."""

from __future__ import annotations

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
    assert config.connect_args["ssl"] is True


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("disable", False),
        ("allow", False),
        ("prefer", True),
        ("require", True),
        ("verify-full", True),
    ],
)
def test_sslmode_values_map_to_asyncpg_ssl(mode: str, expected: bool) -> None:
    config = DatabaseConfig.from_url(f"postgresql://u:p@example.com:5432/db?sslmode={mode}")
    assert config.connect_args["ssl"] is expected


def test_unknown_sslmode_is_rejected_rather_than_guessed() -> None:
    with pytest.raises(DatabaseConfigError, match="sslmode"):
        DatabaseConfig.from_url("postgresql://u:p@example.com/db?sslmode=maybe")


def test_remote_host_gets_tls_even_without_sslmode() -> None:
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
