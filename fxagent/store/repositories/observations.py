"""Raw statistical prints. Write-mostly, and deliberately outside the point-in-time gate.

This repository has no `as_of` parameter, and that is not an oversight — it is why nothing in
the analysis pipeline may read it. These rows carry no publication timestamp, because no
primary source publishes one (docs/ADR-003-fundamentals.md), so there is no honest way to
answer "what was knowable at time T" from them. A repository that offered an `as_of` here would
be answering that question with a guess.

They are collected anyway because `surprise_score` needs eight-plus monthly samples before it
means anything, and that clock only runs while the data is being stored. The join into calendar
events is the expensive half and waits.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Row, case, func, select
from sqlalchemy.dialects.postgresql import insert

from fxagent.store.repositories.base import Repository
from fxagent.store.schema import statistical_observations

__all__ = ["ObservationRecord", "ObservationRepository"]

logger = logging.getLogger(__name__)

_COLUMNS = tuple(column.name for column in statistical_observations.c)


@dataclass(frozen=True)
class ObservationRecord:
    """One statistical print as stored."""

    id: int
    source: str
    series_id: str
    reference_period: str
    period_start: date
    value: float
    first_value: float
    first_seen_at: datetime
    revised_at: datetime | None
    revisions: int

    @classmethod
    def from_row(cls, row: Row[Any]) -> ObservationRecord:
        mapping = row._mapping  # noqa: SLF001 - the sanctioned Row -> dict accessor
        return cls(**{name: mapping[name] for name in _COLUMNS})

    @property
    def was_revised(self) -> bool:
        return self.revisions > 0


class ObservationRepository(Repository):
    """Stores raw prints and reads them back per series. No point-in-time semantics."""

    async def upsert_many(self, records: Iterable[dict[str, Any]]) -> int:
        """Insert prints, tracking revisions without ever losing the original.

        On conflict, `value` moves to the newly-reported number and `revisions` increments, but
        `first_value` and `first_seen_at` are left alone. BLS revises the prior two months at
        every release, and the market traded the *first* print — a plain overwrite would
        quietly destroy the only number a surprise history can legitimately be built from, and
        would do it months before anyone thought to look.

        `revised_at` only moves when the value actually changed. Re-fetching an unchanged series
        is the normal case and must not look like a revision.
        """
        rows = [self._prepare(record) for record in records]
        if not rows:
            return 0

        statement = insert(statistical_observations).values(rows)
        excluded = statement.excluded
        changed = excluded.value.is_distinct_from(statistical_observations.c.value)
        statement = statement.on_conflict_do_update(
            constraint="statobs_unique",
            set_={
                "value": excluded.value,
                "revised_at": case(
                    (changed, func.now()), else_=statistical_observations.c.revised_at
                ),
                "revisions": case(
                    (changed, statistical_observations.c.revisions + 1),
                    else_=statistical_observations.c.revisions,
                ),
            },
        )
        result = await self._session.execute(statement)
        return result.rowcount or 0

    @staticmethod
    def _prepare(record: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(record)
        for required in ("source", "series_id", "reference_period", "period_start", "value"):
            if prepared.get(required) is None:
                raise ValueError(f"observation is missing required field {required!r}")
        # first_value mirrors value on insert and is then frozen by the upsert above.
        prepared.setdefault("first_value", prepared["value"])
        prepared.setdefault("first_seen_at", datetime.now(UTC))
        prepared.setdefault("revisions", 0)
        prepared.setdefault("revised_at", None)
        return prepared

    async def series(
        self, source: str, series_id: str, *, limit: int = 240
    ) -> list[ObservationRecord]:
        """One series, oldest first — the shape a history is built from."""
        statement = (
            select(statistical_observations)
            .where(
                statistical_observations.c.source == source,
                statistical_observations.c.series_id == series_id,
            )
            .order_by(statistical_observations.c.period_start.asc())
            .limit(limit)
        )
        result = await self._session.execute(statement)
        return [ObservationRecord.from_row(row) for row in result]

    async def known_series(self) -> Sequence[tuple[str, str]]:
        """Every (source, series_id) with at least one stored print."""
        result = await self._session.execute(
            select(statistical_observations.c.source, statistical_observations.c.series_id)
            .distinct()
            .order_by(statistical_observations.c.source, statistical_observations.c.series_id)
        )
        return [(row.source, row.series_id) for row in result]

    async def count(self) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(statistical_observations)
        )
        return int(result.scalar_one())
