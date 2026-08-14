"""Service liveness records."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Row, select
from sqlalchemy.dialects.postgresql import insert

from fxagent.store.repositories.base import Repository, require_utc
from fxagent.store.schema import service_heartbeats

__all__ = ["HeartbeatRecord", "HeartbeatRepository"]

logger = logging.getLogger(__name__)

_COLUMNS = tuple(column.name for column in service_heartbeats.c)


@dataclass(frozen=True)
class HeartbeatRecord:
    """One service's liveness row."""

    service: str
    started_at_utc: datetime
    last_beat_utc: datetime
    beats: int
    detail: dict[str, Any] | None

    @property
    def uptime(self) -> timedelta:
        """How long this run has been going — not the sum of all runs."""
        return self.last_beat_utc - self.started_at_utc

    def is_stale(self, *, as_of: datetime, tolerance: timedelta) -> bool:
        return (as_of - self.last_beat_utc) > tolerance

    @classmethod
    def from_row(cls, row: Row[Any]) -> HeartbeatRecord:
        mapping = row._mapping  # noqa: SLF001
        return cls(**{name: mapping[name] for name in _COLUMNS})


class HeartbeatRepository(Repository):
    """Reads and writes service heartbeats."""

    async def beat(
        self,
        service: str,
        *,
        now: datetime,
        started_at: datetime,
        detail: dict[str, Any] | None = None,
    ) -> int:
        """Record a heartbeat and return the running beat count for this service.

        `started_at` is written only on the first beat of a run and then left alone, so a
        crash-looping service shows a start time that keeps moving while a healthy one shows a
        start time that does not. `beats` increments in SQL rather than being read and written
        back, so two collectors racing cannot lose a count between them.
        """
        moment = require_utc(now, "now")
        started = require_utc(started_at, "started_at")
        if moment < started:
            raise ValueError(f"now {moment} is before started_at {started}")

        statement = insert(service_heartbeats).values(
            service=service,
            started_at_utc=started,
            last_beat_utc=moment,
            beats=1,
            detail=detail,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[service_heartbeats.c.service],
            set_={
                "last_beat_utc": statement.excluded.last_beat_utc,
                "started_at_utc": statement.excluded.started_at_utc,
                "beats": service_heartbeats.c.beats + 1,
                "detail": statement.excluded.detail,
            },
        ).returning(service_heartbeats.c.beats)

        result = await self._session.execute(statement)
        return int(result.scalar_one())

    async def get(self, service: str) -> HeartbeatRecord | None:
        result = await self._session.execute(
            select(service_heartbeats).where(service_heartbeats.c.service == service)
        )
        row = result.first()
        return HeartbeatRecord.from_row(row) if row is not None else None

    async def all_services(self) -> list[HeartbeatRecord]:
        result = await self._session.execute(
            select(service_heartbeats).order_by(service_heartbeats.c.service)
        )
        return [HeartbeatRecord.from_row(row) for row in result]

    async def stale_services(
        self, *, tolerance: timedelta, as_of: datetime | None = None
    ) -> list[HeartbeatRecord]:
        """Services whose last beat is older than `tolerance` — the uptime alarm."""
        moment = require_utc(as_of, "as_of") if as_of is not None else datetime.now(UTC)
        cutoff = moment - tolerance
        result = await self._session.execute(
            select(service_heartbeats).where(service_heartbeats.c.last_beat_utc < cutoff)
        )
        return [HeartbeatRecord.from_row(row) for row in result]
