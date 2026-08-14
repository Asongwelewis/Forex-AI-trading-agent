"""Fixtures for the Postgres-backed store tests.

These run against the local container in `docker-compose.test.yml`, never against Supabase.
When `TEST_DATABASE_URL` is unset every test here skips, so the suite stays runnable on a
machine without Docker — the trade-off being that a green run does not by itself prove the
store works. `uv run pytest -m db` is the run that does.

**No connection outlives its event loop.** pytest-asyncio gives each test a fresh loop, and an
asyncpg connection is bound to the loop that opened it — reusing a session-scoped engine across
tests fails with "attached to a different loop", which masks whatever the test was really
checking. So migrations run once behind a session-scoped flag using a throwaway engine that is
disposed before the fixture returns, and every test builds its own `Database`.

Each test truncates rather than re-migrating: milliseconds instead of seconds, same schema.
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
from fxagent.store.repositories import (
    BarRepository,
    EvaluationRepository,
    EventRepository,
    TradeRepository,
    WindowRepository,
)

#: Emptied between tests. `cascade` handles trades -> evaluations without ordering games.
_TABLES = ("trades", "evaluations", "events", "bars", "windows")

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


@pytest_asyncio.fixture
async def events_repo(session: AsyncSession) -> EventRepository:
    return EventRepository(session)


@pytest_asyncio.fixture
async def bars_repo(session: AsyncSession) -> BarRepository:
    return BarRepository(session)


@pytest_asyncio.fixture
async def evaluations_repo(session: AsyncSession) -> EvaluationRepository:
    return EvaluationRepository(session)


@pytest_asyncio.fixture
async def trades_repo(session: AsyncSession) -> TradeRepository:
    return TradeRepository(session)


@pytest_asyncio.fixture
async def windows_repo(session: AsyncSession) -> WindowRepository:
    return WindowRepository(session)
