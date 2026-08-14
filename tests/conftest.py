"""Postgres fixtures shared by every suite that needs a database.

Lives at the root rather than under `tests/store/` because the collector needs the same
database the store tests use, and a fixture defined in a sibling package is not visible.

These run against the local container in `docker-compose.test.yml`, never against Supabase.
When `TEST_DATABASE_URL` is unset every dependent test skips, so the suite stays runnable on a
machine without Docker — the trade-off being that a green run does not by itself prove the
store works. `uv run pytest -m db` is the run that does.

**No connection outlives its event loop.** pytest-asyncio gives each test a fresh loop, and an
asyncpg connection is bound to the loop that opened it — reusing a session-scoped engine across
tests fails with "attached to a different loop", masking whatever the test was really checking.
So migrations run once behind a module-level flag using an engine that is disposed before the
fixture returns, and every test builds its own `Database`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from fxagent.store.config import DatabaseConfig
from fxagent.store.engine import Database
from fxagent.store.migrate import apply_migrations

#: Emptied between tests. `cascade` handles trades -> evaluations without ordering games.
#: Emptied between tests. A table missing from here leaks rows into the next test, where it
#: shows up as a count that is mysteriously too high rather than as an isolation failure.
_TABLES = (
    "trades",
    "evaluations",
    "events",
    "bars",
    "windows",
    "service_heartbeats",
    "statistical_observations",
    "cot_reports",
)

_migrated = False


def database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL") or None


requires_postgres = pytest.mark.skipif(
    database_url() is None,
    reason=(
        "TEST_DATABASE_URL is not set. Start the container with "
        "`docker compose -f docker-compose.test.yml up -d` and set "
        "TEST_DATABASE_URL=postgresql://fxagent:fxagent@localhost:15432/fxagent_test"
    ),
)


def reset_migration_state() -> None:
    """Force the next `database` fixture to re-run migrations.

    For the one test that deliberately drops a table to prove the health check notices. Without
    this the table stays dropped for the rest of the session and every later truncate fails on
    a table that no longer exists — a failure that looks nothing like its cause.
    """
    global _migrated
    _migrated = False


def make_database(**overrides: object) -> Database:
    """A `Database` on the test container. Callers dispose it."""
    url = database_url()
    if url is None:
        pytest.skip("TEST_DATABASE_URL is not set")
    return Database(DatabaseConfig.from_url(url, pool_size=2, max_overflow=2, **overrides))


@pytest_asyncio.fixture
async def database() -> AsyncIterator[Database]:
    """A migrated, empty database with an engine belonging to this test's loop."""
    global _migrated

    instance = make_database()
    try:
        if not _migrated:
            async with instance.connect() as connection:
                await apply_migrations(connection)
            _migrated = True

        async with instance.session() as cleaner:
            await cleaner.execute(text(f"truncate {', '.join(_TABLES)} restart identity cascade"))
            await cleaner.commit()

        yield instance
    finally:
        await instance.dispose()


@pytest_asyncio.fixture
async def session(database: Database) -> AsyncIterator[AsyncSession]:
    """A committing session for a single test."""
    async with database.session() as active:
        yield active
        await active.commit()
