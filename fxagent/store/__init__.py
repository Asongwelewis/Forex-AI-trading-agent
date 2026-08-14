"""Supabase Postgres store: engine, migrations, health check and repositories.

SQLAlchemy Core over asyncpg, talking the Postgres wire protocol directly rather than going
through PostgREST. That choice is what makes the schema testable against a bare `postgres`
container, transactions available, and the point-in-time filters expressible in SQL — see
`docs/ADR-001-store.md`.

Typical startup, shared by the collector and the analyst:

    database = Database.from_env()
    async with database.connect() as connection:
        await apply_migrations(connection)
    report = await check_health(database)
    if not report:
        raise SystemExit(report.summary())
"""

from __future__ import annotations

from fxagent.store.config import DatabaseConfig, DatabaseConfigError
from fxagent.store.engine import Database
from fxagent.store.health import HealthReport, check_health
from fxagent.store.migrate import Migration, MigrationError, apply_migrations, load_migrations
from fxagent.store.repositories import (
    BarRepository,
    EvaluationRepository,
    EventRepository,
    HeartbeatRepository,
    TradeRepository,
    WindowRepository,
)
from fxagent.store.retry import RetryPolicy, is_transient, with_retry
from fxagent.store.schema import EMBEDDING_DIMENSIONS, metadata

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "BarRepository",
    "Database",
    "DatabaseConfig",
    "DatabaseConfigError",
    "EvaluationRepository",
    "EventRepository",
    "HealthReport",
    "HeartbeatRepository",
    "Migration",
    "MigrationError",
    "RetryPolicy",
    "TradeRepository",
    "WindowRepository",
    "apply_migrations",
    "check_health",
    "is_transient",
    "load_migrations",
    "metadata",
    "with_retry",
]
