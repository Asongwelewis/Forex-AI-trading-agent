"""Encoded price windows and their nearest-neighbour retrieval, for the historian agent.

Search is point-in-time by the same reasoning as the events repository. `forward_outcome`
describes what happened after the window, so a neighbour whose outcome had not resolved by the
bar being analysed is the future. `search` therefore requires `as_of` and filters on
`outcome_resolved_at`, and there is no unfiltered search method to reach for instead.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Row, select
from sqlalchemy.dialects.postgresql import insert

from fxagent.store.repositories.base import Repository, require_utc
from fxagent.store.schema import EMBEDDING_DIMENSIONS, windows

__all__ = ["Neighbour", "WindowRecord", "WindowRepository"]

logger = logging.getLogger(__name__)

_COLUMNS = tuple(column.name for column in windows.c)


@dataclass(frozen=True)
class WindowRecord:
    """One stored window."""

    id: int
    symbol: str
    ts_utc: datetime
    timeframe: str
    embedding: Sequence[float]
    normalised_ohlc: dict[str, Any]
    forward_outcome: dict[str, Any] | None
    outcome_resolved_at: datetime | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: Row[Any]) -> WindowRecord:
        mapping = row._mapping  # noqa: SLF001
        return cls(**{name: mapping[name] for name in _COLUMNS})


@dataclass(frozen=True)
class Neighbour:
    """A retrieved analogue and its cosine distance. Lower is more similar."""

    window: WindowRecord
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity in [-1, 1], for reporting."""
        return 1.0 - self.distance


class WindowRepository(Repository):
    """Stores encoded windows and retrieves resolved analogues."""

    async def upsert(
        self,
        *,
        symbol: str,
        ts_utc: datetime,
        timeframe: str,
        embedding: Sequence[float],
        normalised_ohlc: dict[str, Any],
        forward_outcome: dict[str, Any] | None = None,
        outcome_resolved_at: datetime | None = None,
    ) -> int:
        """Store a window. Outcome and its resolution time must arrive together."""
        self._check_dimensions(embedding)
        if (forward_outcome is None) != (outcome_resolved_at is None):
            raise ValueError(
                "forward_outcome and outcome_resolved_at must be set together: an outcome with "
                "no resolution time can never become visible to retrieval"
            )

        statement = insert(windows).values(
            symbol=symbol,
            ts_utc=require_utc(ts_utc, "ts_utc"),
            timeframe=timeframe,
            embedding=list(embedding),
            normalised_ohlc=normalised_ohlc,
            forward_outcome=forward_outcome,
            outcome_resolved_at=(
                require_utc(outcome_resolved_at, "outcome_resolved_at")
                if outcome_resolved_at is not None
                else None
            ),
            created_at=datetime.now(UTC),
        )
        statement = statement.on_conflict_do_update(
            constraint="windows_unique",
            set_={
                "embedding": statement.excluded.embedding,
                "normalised_ohlc": statement.excluded.normalised_ohlc,
                "forward_outcome": statement.excluded.forward_outcome,
                "outcome_resolved_at": statement.excluded.outcome_resolved_at,
            },
        ).returning(windows.c.id)

        result = await self._session.execute(statement)
        return int(result.scalar_one())

    async def resolve_outcome(
        self,
        window_id: int,
        *,
        forward_outcome: dict[str, Any],
        outcome_resolved_at: datetime,
    ) -> bool:
        """Attach an outcome once it has actually resolved, making the window retrievable."""
        from sqlalchemy import update

        statement = (
            update(windows)
            .where(windows.c.id == window_id)
            .values(
                forward_outcome=forward_outcome,
                outcome_resolved_at=require_utc(outcome_resolved_at, "outcome_resolved_at"),
            )
        )
        result = await self._session.execute(statement)
        return (result.rowcount or 0) > 0

    async def search(
        self,
        as_of: datetime,
        *,
        embedding: Sequence[float],
        limit: int = 10,
        symbol: str | None = None,
        timeframe: str | None = None,
        exclude_window_id: int | None = None,
    ) -> list[Neighbour]:
        """Nearest resolved analogues to `embedding`, as of `as_of`.

        Two filters, both required for correctness: the outcome must exist, and it must have
        resolved at or before `as_of`. A window encoded from bars up to yesterday whose outcome
        lands tomorrow is invisible today, which is the whole point.
        """
        self._check_dimensions(embedding)
        checked = require_utc(as_of, "as_of")
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        distance = windows.c.embedding.cosine_distance(list(embedding)).label("distance")
        statement = (
            select(windows, distance)
            .where(
                windows.c.outcome_resolved_at.is_not(None),
                windows.c.outcome_resolved_at <= checked,
            )
            .order_by(distance)
            .limit(limit)
        )
        if symbol is not None:
            statement = statement.where(windows.c.symbol == symbol)
        if timeframe is not None:
            statement = statement.where(windows.c.timeframe == timeframe)
        if exclude_window_id is not None:
            statement = statement.where(windows.c.id != exclude_window_id)

        result = await self._session.execute(statement)
        return [
            Neighbour(window=WindowRecord.from_row(row), distance=float(row._mapping["distance"]))  # noqa: SLF001
            for row in result
        ]

    async def get(self, window_id: int) -> WindowRecord | None:
        result = await self._session.execute(select(windows).where(windows.c.id == window_id))
        row = result.first()
        return WindowRecord.from_row(row) if row is not None else None

    async def count_retrievable(self, as_of: datetime) -> int:
        """How many analogues retrieval could actually return at `as_of`."""
        from sqlalchemy import func

        checked = require_utc(as_of, "as_of")
        result = await self._session.execute(
            select(func.count())
            .select_from(windows)
            .where(
                windows.c.outcome_resolved_at.is_not(None),
                windows.c.outcome_resolved_at <= checked,
            )
        )
        return int(result.scalar_one())

    @staticmethod
    def _check_dimensions(embedding: Sequence[float]) -> None:
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding has {len(embedding)} dimensions, but the windows table is "
                f"vector({EMBEDDING_DIMENSIONS}). Changing the dimension needs a migration and "
                "a re-encode of every stored row."
            )
