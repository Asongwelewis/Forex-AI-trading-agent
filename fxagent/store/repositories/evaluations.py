"""Consensus evaluations — every cycle, including the ones that fired nothing."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Row, func, select
from sqlalchemy.dialects.postgresql import insert

from fxagent.store.repositories.base import Repository, require_utc
from fxagent.store.schema import evaluations

__all__ = ["EvaluationRecord", "EvaluationRepository"]

logger = logging.getLogger(__name__)

_COLUMNS = tuple(column.name for column in evaluations.c)


@dataclass(frozen=True)
class EvaluationRecord:
    """One consensus evaluation as stored."""

    id: int
    cycle_id: UUID
    ts_utc: datetime
    symbol: str
    regime: dict[str, Any]
    votes: dict[str, Any]
    consensus_score: float
    fired: bool
    reason: str
    created_at: datetime

    @classmethod
    def from_row(cls, row: Row[Any]) -> EvaluationRecord:
        mapping = row._mapping  # noqa: SLF001
        return cls(**{name: mapping[name] for name in _COLUMNS})


class EvaluationRepository(Repository):
    """Writes and reads consensus evaluations."""

    async def record(
        self,
        *,
        cycle_id: UUID,
        ts_utc: datetime,
        symbol: str,
        regime: dict[str, Any],
        votes: dict[str, Any],
        consensus_score: float,
        fired: bool,
        reason: str,
    ) -> int:
        """Store one evaluation and return its id.

        Idempotent on (cycle_id, symbol): a cycle retried after a dropped connection updates
        its row rather than writing a second one, which would double-count the disagreement
        statistics the router is meant to learn from.
        """
        if not reason:
            raise ValueError(
                "reason is required, including when nothing fired — a silent evaluation with "
                "no recorded reason is not training data"
            )

        statement = insert(evaluations).values(
            cycle_id=cycle_id,
            ts_utc=require_utc(ts_utc, "ts_utc"),
            symbol=symbol,
            regime=regime,
            votes=votes,
            consensus_score=consensus_score,
            fired=fired,
            reason=reason,
            created_at=datetime.now(UTC),
        )
        statement = statement.on_conflict_do_update(
            constraint="evaluations_unique",
            set_={
                "regime": statement.excluded.regime,
                "votes": statement.excluded.votes,
                "consensus_score": statement.excluded.consensus_score,
                "fired": statement.excluded.fired,
                "reason": statement.excluded.reason,
            },
        ).returning(evaluations.c.id)

        result = await self._session.execute(statement)
        return int(result.scalar_one())

    async def get(self, evaluation_id: int) -> EvaluationRecord | None:
        result = await self._session.execute(
            select(evaluations).where(evaluations.c.id == evaluation_id)
        )
        row = result.first()
        return EvaluationRecord.from_row(row) if row is not None else None

    async def for_cycle(self, cycle_id: UUID) -> list[EvaluationRecord]:
        result = await self._session.execute(
            select(evaluations)
            .where(evaluations.c.cycle_id == cycle_id)
            .order_by(evaluations.c.symbol)
        )
        return [EvaluationRecord.from_row(row) for row in result]

    async def recent(
        self,
        *,
        symbol: str | None = None,
        fired_only: bool = False,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[EvaluationRecord]:
        statement = select(evaluations).order_by(evaluations.c.ts_utc.desc()).limit(limit)
        if symbol is not None:
            statement = statement.where(evaluations.c.symbol == symbol)
        if fired_only:
            statement = statement.where(evaluations.c.fired.is_(True))
        if since is not None:
            statement = statement.where(evaluations.c.ts_utc >= require_utc(since, "since"))
        result = await self._session.execute(statement)
        return [EvaluationRecord.from_row(row) for row in result]

    async def disagreement_counts(
        self, *, since: datetime | None = None
    ) -> Sequence[tuple[str, bool, int]]:
        """(symbol, fired, count) — the shape the router's training set is built from."""
        statement = (
            select(evaluations.c.symbol, evaluations.c.fired, func.count())
            .group_by(evaluations.c.symbol, evaluations.c.fired)
            .order_by(evaluations.c.symbol)
        )
        if since is not None:
            statement = statement.where(evaluations.c.ts_utc >= require_utc(since, "since"))
        result = await self._session.execute(statement)
        return [(row[0], row[1], int(row[2])) for row in result]
