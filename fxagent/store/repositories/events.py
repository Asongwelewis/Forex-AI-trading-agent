"""Fundamental events, readable only as of a point in time.

Every public read takes `as_of` as its first positional argument and there is no method that
omits it. That is deliberate and is the whole design of this class: an optional `as_of` with a
sensible default is an `as_of` that gets left off in the one code path that mattered.

All reads compose from `_visible()`, which selects from the `events_visible_at()` SQL function
rather than from the `events` table. The filter therefore lives in the database (hard rule 6),
so it holds for an analyst's ad-hoc query too, not only for traffic through this class.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Row, func, select
from sqlalchemy.dialects.postgresql import insert

from fxagent.store.repositories.base import Repository, require_utc
from fxagent.store.schema import events

__all__ = ["EventRecord", "EventRepository", "HIGH_IMPACT"]

logger = logging.getLogger(__name__)

#: Importance values that gate execution. 'Holiday' is thin liquidity, not a release.
HIGH_IMPACT = ("High",)

_COLUMNS = tuple(column.name for column in events.c)


@dataclass(frozen=True)
class EventRecord:
    """One event as stored. Mirrors the table; constructed only from a database row."""

    id: int
    event_time_utc: datetime
    publication_time_utc: datetime
    source: str
    currency: str
    category: str | None
    importance: str
    title: str
    body: str | None
    forecast: float | None
    actual: float | None
    previous: float | None
    surprise_score: float | None
    ingested_at: datetime

    @classmethod
    def from_row(cls, row: Row[Any]) -> EventRecord:
        mapping = row._mapping  # noqa: SLF001 - the sanctioned Row -> dict accessor
        return cls(**{name: mapping[name] for name in _COLUMNS})


class EventRepository(Repository):
    """Reads and writes fundamental events. Reads are point-in-time by construction."""

    # -- writes ----------------------------------------------------------------

    async def upsert_many(self, records: Iterable[dict[str, Any]]) -> int:
        """Insert events, updating the mutable fields of ones already seen.

        Calendar rows are republished as `actual` lands, so a plain insert would either
        collide or keep the stale forecast-only version. The identity columns and
        `publication_time_utc` are never updated — rewriting publication time would let a
        later revision retroactively become visible earlier.
        """
        rows = [self._prepare(record) for record in records]
        if not rows:
            return 0

        statement = insert(events).values(rows)
        statement = statement.on_conflict_do_update(
            constraint="events_unique",
            set_={
                "category": statement.excluded.category,
                "importance": statement.excluded.importance,
                "body": statement.excluded.body,
                "forecast": statement.excluded.forecast,
                "actual": statement.excluded.actual,
                "previous": statement.excluded.previous,
                "surprise_score": statement.excluded.surprise_score,
            },
        )
        result = await self._session.execute(statement)
        return result.rowcount or 0

    @staticmethod
    def _prepare(record: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(record)
        for field in ("event_time_utc", "publication_time_utc"):
            if field not in prepared:
                raise ValueError(f"event is missing required field {field!r}")
            prepared[field] = require_utc(prepared[field], field)
        prepared.setdefault("ingested_at", datetime.now(UTC))
        return prepared

    # -- reads (all point-in-time) ---------------------------------------------

    def _visible(self, as_of: datetime) -> Any:
        """The only sanctioned source for a read. Selects through the SQL gate function."""
        checked = require_utc(as_of, "as_of")
        return func.events_visible_at(checked).table_valued(*_COLUMNS)

    async def visible_at(
        self,
        as_of: datetime,
        *,
        currencies: Sequence[str] | None = None,
        importance: Sequence[str] | None = None,
        limit: int = 500,
    ) -> list[EventRecord]:
        """Every event published at or before `as_of`, newest publication first."""
        source = self._visible(as_of)
        statement = select(source).order_by(source.c.publication_time_utc.desc()).limit(limit)
        if currencies:
            statement = statement.where(source.c.currency.in_(tuple(currencies)))
        if importance:
            statement = statement.where(source.c.importance.in_(tuple(importance)))
        result = await self._session.execute(statement)
        return [EventRecord.from_row(row) for row in result]

    async def in_event_window(
        self,
        as_of: datetime,
        *,
        window_start: datetime,
        window_end: datetime,
        currencies: Sequence[str] | None = None,
        importance: Sequence[str] | None = None,
    ) -> list[EventRecord]:
        """Events *occurring* in a window, among those already published at `as_of`.

        Two different times, both applied: `event_time_utc` bounds the window, and
        `publication_time_utc` bounds what we were allowed to know.
        """
        start = require_utc(window_start, "window_start")
        end = require_utc(window_end, "window_end")
        if end < start:
            raise ValueError(f"window_end {end} is before window_start {start}")

        source = self._visible(as_of)
        statement = (
            select(source)
            .where(source.c.event_time_utc >= start, source.c.event_time_utc <= end)
            .order_by(source.c.event_time_utc.asc())
        )
        if currencies:
            statement = statement.where(source.c.currency.in_(tuple(currencies)))
        if importance:
            statement = statement.where(source.c.importance.in_(tuple(importance)))
        result = await self._session.execute(statement)
        return [EventRecord.from_row(row) for row in result]

    async def high_impact_near(
        self,
        as_of: datetime,
        *,
        currencies: Sequence[str],
        lookahead: timedelta = timedelta(minutes=15),
        lookbehind: timedelta = timedelta(minutes=5),
    ) -> list[EventRecord]:
        """Backs the "high-impact event within 15 minutes" auto-revoke trigger.

        `lookbehind` is not symmetry for its own sake — the spread damage from a release
        outlasts the release, so a grant should stay revoked for a few minutes afterwards.
        """
        checked = require_utc(as_of, "as_of")
        return await self.in_event_window(
            checked,
            window_start=checked - lookbehind,
            window_end=checked + lookahead,
            currencies=currencies,
            importance=HIGH_IMPACT,
        )

    async def count_visible(self, as_of: datetime) -> int:
        """How many events were published at or before `as_of`."""
        source = self._visible(as_of)
        result = await self._session.execute(select(func.count()).select_from(source))
        return int(result.scalar_one())

    async def latest_publication_time(self, as_of: datetime) -> datetime | None:
        """Most recent publication time visible at `as_of`, for staleness checks.

        The calendar feed covers the current week only. If this falls too far behind, the
        lookahead window has run out and the permission layer must fail closed rather than
        read the absence of events as safety.
        """
        source = self._visible(as_of)
        result = await self._session.execute(
            select(func.max(source.c.publication_time_utc)).select_from(source)
        )
        return result.scalar_one_or_none()
