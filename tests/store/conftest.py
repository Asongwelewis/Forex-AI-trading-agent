"""Repository fixtures for the store suite.

The database, session and skip-marker fixtures live in `tests/conftest.py` so the collector
suite can share them. They are re-exported here because the store tests import them by name
from this module.
"""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from fxagent.store.repositories import (
    BarRepository,
    EvaluationRepository,
    EventRepository,
    HeartbeatRepository,
    TradeRepository,
    WindowRepository,
)
from tests.conftest import database_url, make_database, requires_postgres

__all__ = ["database_url", "make_database", "requires_postgres"]


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


@pytest_asyncio.fixture
async def heartbeats_repo(session: AsyncSession) -> HeartbeatRepository:
    return HeartbeatRepository(session)
