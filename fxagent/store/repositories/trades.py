"""Trades, each linked to the evaluation that produced it."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Row, and_, or_, select, update
from sqlalchemy.dialects.postgresql import insert

from fxagent.store.repositories.base import Repository, require_utc
from fxagent.store.schema import trades

__all__ = ["BARRIERS", "MODES", "TradeRecord", "TradeRepository"]

logger = logging.getLogger(__name__)

BARRIERS = ("TARGET", "STOP", "TIME")

#: Every mode the column accepts. 'LIVE' is in the type but blocked by a database constraint
#: (0007) — the enum stays whole so enabling it in v2 is a one-line migration, not a type change.
MODES = ("ADVISORY", "DEMO_AUTO", "LIVE")

#: Modes this system is permitted to write. The database enforces the same thing independently;
#: this check exists only to fail with a readable message instead of an IntegrityError.
PERMITTED_MODES = ("ADVISORY", "DEMO_AUTO")

_COLUMNS = tuple(column.name for column in trades.c)


@dataclass(frozen=True)
class TradeRecord:
    """One trade as stored."""

    id: int
    evaluation_id: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    entry_time_utc: datetime
    exit_price: float | None
    exit_time_utc: datetime | None
    stop_price: float
    target_price: float
    barrier_touched: str | None
    label_span_start: datetime
    label_span_end: datetime
    pnl: float | None
    r_multiple: float | None
    mode: str
    created_at: datetime

    @property
    def is_open(self) -> bool:
        return self.exit_time_utc is None

    @classmethod
    def from_row(cls, row: Row[Any]) -> TradeRecord:
        mapping = row._mapping  # noqa: SLF001
        return cls(**{name: mapping[name] for name in _COLUMNS})


class TradeRepository(Repository):
    """Writes and reads trades. Opening and closing are separate operations."""

    async def open_trade(
        self,
        *,
        evaluation_id: int,
        symbol: str,
        direction: str,
        volume: float,
        entry_price: float,
        entry_time_utc: datetime,
        stop_price: float,
        target_price: float,
        label_span_start: datetime,
        label_span_end: datetime,
        mode: str,
    ) -> int:
        """Record a newly opened trade and return its id.

        Stop and target are required, mirroring `OrderRequest` — hard rule 3 says an order
        without protection cannot exist, so a row describing one cannot either.
        """
        if direction not in ("LONG", "SHORT"):
            raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode not in PERMITTED_MODES:
            # The database rejects this too (constraint trades_no_live_mode). Raising here only
            # buys a readable error; deleting this check does NOT enable live trading.
            raise ValueError(
                f"mode {mode!r} is not permitted: this system trades demo accounts only "
                "(CLAUDE.md hard rule 1). The database enforces the same rule via constraint "
                "trades_no_live_mode, which a v2 must drop deliberately in a migration."
            )

        statement = (
            insert(trades)
            .values(
                evaluation_id=evaluation_id,
                symbol=symbol,
                direction=direction,
                volume=volume,
                entry_price=entry_price,
                entry_time_utc=require_utc(entry_time_utc, "entry_time_utc"),
                stop_price=stop_price,
                target_price=target_price,
                label_span_start=require_utc(label_span_start, "label_span_start"),
                label_span_end=require_utc(label_span_end, "label_span_end"),
                mode=mode,
                created_at=datetime.now(UTC),
            )
            .returning(trades.c.id)
        )
        result = await self._session.execute(statement)
        return int(result.scalar_one())

    async def close_trade(
        self,
        trade_id: int,
        *,
        exit_price: float,
        exit_time_utc: datetime,
        barrier_touched: str,
        pnl: float | None = None,
        r_multiple: float | None = None,
    ) -> bool:
        """Close an open trade. The exit fields land together or the row check rejects them."""
        if barrier_touched not in BARRIERS:
            raise ValueError(f"barrier_touched must be one of {BARRIERS}, got {barrier_touched!r}")

        statement = (
            update(trades)
            .where(trades.c.id == trade_id, trades.c.exit_time_utc.is_(None))
            .values(
                exit_price=exit_price,
                exit_time_utc=require_utc(exit_time_utc, "exit_time_utc"),
                barrier_touched=barrier_touched,
                pnl=pnl,
                r_multiple=r_multiple,
            )
        )
        result = await self._session.execute(statement)
        return (result.rowcount or 0) > 0

    async def get(self, trade_id: int) -> TradeRecord | None:
        result = await self._session.execute(select(trades).where(trades.c.id == trade_id))
        row = result.first()
        return TradeRecord.from_row(row) if row is not None else None

    async def open_trades(self, *, symbol: str | None = None) -> list[TradeRecord]:
        statement = (
            select(trades)
            .where(trades.c.exit_time_utc.is_(None))
            .order_by(trades.c.entry_time_utc.asc())
        )
        if symbol is not None:
            statement = statement.where(trades.c.symbol == symbol)
        result = await self._session.execute(statement)
        return [TradeRecord.from_row(row) for row in result]

    async def for_evaluation(self, evaluation_id: int) -> list[TradeRecord]:
        result = await self._session.execute(
            select(trades)
            .where(trades.c.evaluation_id == evaluation_id)
            .order_by(trades.c.entry_time_utc.asc())
        )
        return [TradeRecord.from_row(row) for row in result]

    async def labels_overlapping(
        self, *, start: datetime, end: datetime, symbol: str | None = None
    ) -> list[TradeRecord]:
        """Trades whose label span overlaps [start, end].

        This is the purge step of purged walk-forward cross-validation (hard rule 7). A
        training trade whose outcome resolved inside the test window has seen the test period,
        and leaving it in the training fold leaks the answer. Overlap, not containment: a label
        that merely straddles the boundary leaks just as much as one wholly inside it.
        """
        first = require_utc(start, "start")
        last = require_utc(end, "end")
        if last < first:
            raise ValueError(f"end {last} is before start {first}")

        statement = (
            select(trades)
            .where(
                and_(
                    trades.c.label_span_start <= last,
                    trades.c.label_span_end >= first,
                )
            )
            .order_by(trades.c.label_span_start.asc())
        )
        if symbol is not None:
            statement = statement.where(trades.c.symbol == symbol)
        result = await self._session.execute(statement)
        return [TradeRecord.from_row(row) for row in result]

    async def closed_trades(
        self, *, symbol: str | None = None, mode: str | None = None, limit: int = 1000
    ) -> list[TradeRecord]:
        """Closed trades, for expectancy and profit-factor reporting."""
        statement = (
            select(trades)
            .where(trades.c.exit_time_utc.is_not(None))
            .order_by(trades.c.exit_time_utc.desc())
            .limit(limit)
        )
        if symbol is not None:
            statement = statement.where(trades.c.symbol == symbol)
        if mode is not None:
            statement = statement.where(trades.c.mode == mode)
        result = await self._session.execute(statement)
        return [TradeRecord.from_row(row) for row in result]

    async def unlabelled(self, *, as_of: datetime) -> list[TradeRecord]:
        """Closed trades still missing an r_multiple, or open ones whose span has expired."""
        checked = require_utc(as_of, "as_of")
        statement = select(trades).where(
            or_(
                and_(trades.c.exit_time_utc.is_not(None), trades.c.r_multiple.is_(None)),
                and_(trades.c.exit_time_utc.is_(None), trades.c.label_span_end <= checked),
            )
        )
        result = await self._session.execute(statement)
        return [TradeRecord.from_row(row) for row in result]
