"""Row builders for the dashboard suite. No database, no network, no clock.

The dashboard reads `EvaluationRecord` and `TradeRecord`, which are store dataclasses rather
than Pydantic models, so they are constructed directly here. That is the point of the seam:
every assertion in this suite is about what the panel does with a row, not about whether
Postgres can produce one — the store suite already answers that, and answering it twice would
mean the panel could only be tested on a machine with Docker running.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from fxagent.adapters.base import Bar, BarSeries
from fxagent.store.repositories.evaluations import EvaluationRecord
from fxagent.store.repositories.trades import TradeRecord

__all__ = ["bar_series", "evaluation", "trade", "vote"]


def bar_series(
    *,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    start: datetime | None = None,
    count: int = 48,
    price: float = 1.1000,
    step: float = 0.0004,
) -> BarSeries:
    """A gently rising hourly series starting at midnight UTC, so whole Asian sessions exist."""
    first = start or datetime(2026, 1, 12, 0, 0, tzinfo=UTC)
    bars = []
    for index in range(count):
        open_ = price + index * step
        close = open_ + step / 2
        bars.append(
            Bar(
                timestamp=first + timedelta(hours=index),
                open=open_,
                high=max(open_, close) + 0.0003,
                low=min(open_, close) - 0.0003,
                close=close,
                volume=100 + index,
            )
        )
    return BarSeries(symbol=symbol, timeframe=timeframe, bars=tuple(bars))


def vote(
    strategy: str,
    *,
    weight: float = 1.0,
    direction: str | None = None,
    confidence: float | None = None,
    participated: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    """One row of the diagnostics vote list, shaped exactly as `Consensus._tally` writes it."""
    return {
        "strategy": strategy,
        "weight": weight,
        "direction": direction,
        "confidence": confidence,
        "participated": participated,
        "reason": reason,
    }


def evaluation(
    *,
    identifier: int = 1,
    ts_utc: datetime | None = None,
    symbol: str = "EURUSD",
    votes: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    regime: dict[str, Any] | None = None,
    fired: bool = False,
    reason: str = "no strategy offered a tradeable direction",
    consensus_score: float = 0.0,
    cycle_id: UUID | None = None,
) -> EvaluationRecord:
    document: dict[str, Any] = {"votes": votes if votes is not None else []}
    document.update(extra or {})

    moment = ts_utc or datetime(2026, 1, 12, 9, 0, tzinfo=UTC)
    return EvaluationRecord(
        id=identifier,
        cycle_id=cycle_id or uuid4(),
        ts_utc=moment,
        symbol=symbol,
        regime=regime
        if regime is not None
        else {
            "sessions": ["LONDON"],
            "market_open": True,
            "trend_strength": 27.5,
            "volatility_percentile": 62.0,
            "is_trending": True,
            "is_ranging": False,
            "minutes_until_weekly_close": 3000,
        },
        votes=document,
        consensus_score=consensus_score,
        fired=fired,
        reason=reason,
        created_at=moment,
    )


def trade(
    *,
    identifier: int = 1,
    evaluation_id: int = 1,
    symbol: str = "EURUSD",
    direction: str = "LONG",
    entry_time: datetime | None = None,
    exit_time: datetime | None = None,
    entry_price: float = 1.1050,
    stop_price: float = 1.1020,
    target_price: float = 1.1110,
    r_multiple: float | None = None,
    barrier: str | None = None,
    mode: str = "DEMO_AUTO",
) -> TradeRecord:
    entered = entry_time or datetime(2026, 1, 12, 9, 0, tzinfo=UTC)
    return TradeRecord(
        id=identifier,
        evaluation_id=evaluation_id,
        symbol=symbol,
        direction=direction,
        volume=0.02,
        entry_price=entry_price,
        entry_time_utc=entered,
        exit_price=target_price if exit_time else None,
        exit_time_utc=exit_time,
        stop_price=stop_price,
        target_price=target_price,
        barrier_touched=barrier,
        label_span_start=entered,
        label_span_end=entered + timedelta(days=2),
        pnl=None,
        r_multiple=r_multiple,
        mode=mode,
        created_at=entered,
    )
