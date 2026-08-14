"""Startup health check, called by the collector and the analyst before they do any work.

Checks four things, in the order that fails most cheaply first. A service that starts, runs a
cycle, and only then discovers pgvector is missing has already written partial data; the point
is to refuse to start instead.

The result is a value, not an exception, so a caller can log the detail and decide. Nothing
here interpolates the connection password — only `config.safe_url` is ever rendered.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text

from fxagent.store.engine import Database
from fxagent.store.migrate import load_migrations

__all__ = ["HealthReport", "check_health"]

logger = logging.getLogger(__name__)

#: Tables the schema must expose before any service is allowed to start.
REQUIRED_TABLES = (
    "bars",
    "events",
    "evaluations",
    "trades",
    "windows",
    "service_heartbeats",
)


@dataclass(frozen=True)
class HealthReport:
    """Outcome of a startup check. Falsy when the service must not start."""

    healthy: bool
    latency_ms: float
    server_version: str = ""
    pgvector_installed: bool = False
    missing_tables: tuple[str, ...] = ()
    pending_migrations: tuple[str, ...] = ()
    error: str = ""
    details: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.healthy

    def summary(self) -> str:
        """One line for a startup log."""
        if self.healthy:
            return (
                f"database healthy in {self.latency_ms:.0f}ms "
                f"(postgres {self.server_version}, pgvector present)"
            )
        if self.error:
            return f"database unhealthy: {self.error}"
        problems: list[str] = []
        if not self.pgvector_installed:
            problems.append("pgvector extension missing")
        if self.missing_tables:
            problems.append(f"missing tables: {', '.join(self.missing_tables)}")
        if self.pending_migrations:
            problems.append(f"unapplied migrations: {', '.join(self.pending_migrations)}")
        return f"database unhealthy: {'; '.join(problems) or 'unknown reason'}"


async def check_health(database: Database, *, require_migrations: bool = True) -> HealthReport:
    """Verify the database is reachable, extended, migrated and queryable."""
    started = time.perf_counter()

    async def probe() -> HealthReport:
        async with database.session() as session:
            version = (await session.execute(text("select version()"))).scalar_one()

            has_vector = (
                await session.execute(
                    text("select exists (select 1 from pg_extension where extname = 'vector')")
                )
            ).scalar_one()

            rows = await session.execute(
                text(
                    "select table_name from information_schema.tables where table_schema = 'public'"
                )
            )
            present = {row.table_name for row in rows}
            missing = tuple(t for t in REQUIRED_TABLES if t not in present)

            pending: tuple[str, ...] = ()
            if require_migrations and "schema_migrations" in present:
                applied_rows = await session.execute(text("select version from schema_migrations"))
                applied = {row.version for row in applied_rows}
                pending = tuple(
                    f"{m.version}_{m.name}" for m in load_migrations() if m.version not in applied
                )
            elif require_migrations:
                pending = tuple(f"{m.version}_{m.name}" for m in load_migrations())

            elapsed = (time.perf_counter() - started) * 1000
            return HealthReport(
                healthy=bool(has_vector) and not missing and not pending,
                latency_ms=elapsed,
                server_version=_short_version(str(version)),
                pgvector_installed=bool(has_vector),
                missing_tables=missing,
                pending_migrations=pending,
            )

    try:
        report = await database.run(probe, description="health check")
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        # type(exc).__name__ and str(exc) — never the config, which holds the password.
        report = HealthReport(
            healthy=False,
            latency_ms=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )

    log = logger.info if report.healthy else logger.error
    log("%s [%s]", report.summary(), database.config.safe_url)
    return report


def _short_version(banner: str) -> str:
    """`select version()` returns a paragraph; keep the number."""
    parts = banner.split()
    return parts[1] if len(parts) > 1 and parts[0] == "PostgreSQL" else banner[:40]
