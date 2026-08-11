"""Startup health check, and a guard against schema.py drifting from the migrations."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from fxagent.store.engine import Database
from fxagent.store.health import REQUIRED_TABLES, check_health
from fxagent.store.schema import metadata

from .conftest import make_database, requires_postgres

pytestmark = [pytest.mark.db, requires_postgres]


# -- health -------------------------------------------------------------------


async def test_health_check_passes_on_a_migrated_database(database: Database) -> None:
    report = await check_health(database)

    assert report.healthy is True
    assert bool(report) is True
    assert report.pgvector_installed is True
    assert report.missing_tables == ()
    assert report.pending_migrations == ()
    assert report.server_version.startswith("16")
    assert "healthy" in report.summary()


async def test_health_check_reports_unreachable_rather_than_raising() -> None:
    """A service must be able to log the reason and exit, not crash in a stack trace."""
    unreachable = make_database()
    broken = Database(
        type(unreachable.config)(
            url=unreachable.config.url.replace("15432", "15499"),
            connect_args=dict(unreachable.config.connect_args),
        ),
        retry=type(unreachable.retry_policy)(attempts=1, base_delay_seconds=0.01),
    )
    await unreachable.dispose()

    try:
        report = await check_health(broken)
    finally:
        await broken.dispose()

    assert report.healthy is False
    assert bool(report) is False
    assert report.error
    assert "unhealthy" in report.summary()


async def test_health_report_never_contains_the_password(database: Database) -> None:
    report = await check_health(database)
    assert "fxagent:fxagent" not in report.summary()


async def test_health_check_detects_a_missing_table(database: Database) -> None:
    """Drop a required table and the check must refuse the startup."""
    async with database.session() as session:
        await session.execute(text("drop table if exists windows cascade"))
        await session.commit()

    try:
        report = await check_health(database, require_migrations=False)
        assert report.healthy is False
        assert "windows" in report.missing_tables
        assert "windows" in report.summary()
    finally:
        # Restore for the next test: the migration record still says 0006 ran.
        async with database.connect() as connection:
            raw = await connection.get_raw_connection()
            await raw.driver_connection.execute(
                "delete from schema_migrations where version = '0006'"
            )
            await connection.commit()
        import tests.store.conftest as conftest_module

        conftest_module._migrated = False


# -- schema drift -------------------------------------------------------------


async def test_every_declared_table_exists_in_the_database(database: Database) -> None:
    async with database.engine.connect() as connection:
        names = await connection.run_sync(lambda sync: inspect(sync).get_table_names())

    for table in metadata.tables:
        assert table in names, f"schema.py declares {table!r} but no migration creates it"


async def test_required_tables_match_the_declared_schema() -> None:
    """health.py's list and schema.py's tables must not drift apart."""
    assert set(REQUIRED_TABLES) == set(metadata.tables)


async def test_declared_columns_match_the_migrated_columns(database: Database) -> None:
    """schema.py mirrors the SQL by hand, so drift is possible and must fail loudly."""
    async with database.engine.connect() as connection:
        for name, table in metadata.tables.items():
            actual = await connection.run_sync(
                lambda sync, n=name: {c["name"] for c in inspect(sync).get_columns(n)}
            )
            declared = {column.name for column in table.c}

            assert declared == actual, (
                f"{name}: schema.py and the migrations disagree. "
                f"only in schema.py: {declared - actual}; only in the database: "
                f"{actual - declared}"
            )
