"""Walk the open paper trades and close the ones the bars have settled."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from fxagent.adapters.base import TIMEFRAMES, BarSeries, OrderSide
from fxagent.backtest.barriers import Barrier, resolve_barriers
from fxagent.costs import CostConfig, Quote, fill, swap_cost
from fxagent.risk.symbols import SymbolSpec
from fxagent.stats.returns import r_multiple
from fxagent.store.engine import Database
from fxagent.store.repositories import BarRepository, TradeRepository
from fxagent.store.repositories.trades import TradeRecord

__all__ = ["ResolveConfig", "ResolveStats", "resolve_open_trades"]

logger = logging.getLogger(__name__)

#: Matches `ReplayConfig.max_bars_held`. 24 on H1 is a day, past which an intraday setup is not
#: what it was, whatever the price is doing. Restated here rather than imported so that the
#: resolver does not pull the backtest package in — but the number must not drift, and
#: `tests/resolve/test_resolver_matches_the_replay.py` asserts the two are equal.
DEFAULT_MAX_BARS_HELD: Final = 24


@dataclass(frozen=True)
class ResolveConfig:
    """What the resolver needs that is not the trades themselves."""

    source: str
    timeframe: str = "H1"
    max_bars_held: int = DEFAULT_MAX_BARS_HELD
    costs: CostConfig = field(default_factory=CostConfig)
    #: Per-symbol contract specs. A symbol without one is skipped and counted, never resolved
    #: against a guessed contract size — the R multiple would be wrong in a way nothing catches.
    specs: dict[str, SymbolSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source is required; a bar's identity includes where it came from")
        if self.timeframe not in TIMEFRAMES:
            raise ValueError(f"unknown timeframe {self.timeframe!r}")
        if self.max_bars_held < 1:
            raise ValueError(f"max_bars_held must be at least 1, got {self.max_bars_held}")


@dataclass
class ResolveStats:
    """What one resolver run did. Every category is counted, including the no-ops."""

    examined: int = 0
    closed: int = 0
    #: Still open because the bars to settle them have not arrived. The normal case.
    still_running: int = 0
    #: Skipped because something was missing — no spec, no bars, entry not in the series.
    skipped: int = 0
    errors: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)

    def as_detail(self) -> dict[str, object]:
        return {
            "examined": self.examined,
            "closed": self.closed,
            "still_running": self.still_running,
            "skipped": self.skipped,
            "errors": self.errors,
            "outcomes": dict(self.outcomes),
        }


def _side(direction: str) -> OrderSide:
    return OrderSide.BUY if direction == "LONG" else OrderSide.SELL


def _entry_index(bars: BarSeries, entry_time: datetime) -> int | None:
    for index, bar in enumerate(bars.bars):
        if bar.timestamp == entry_time:
            return index
    return None


async def resolve_open_trades(
    database: Database,
    config: ResolveConfig,
    *,
    now: datetime | None = None,
) -> ResolveStats:
    """Close every open trade the stored bars have settled. Returns what happened.

    Idempotent: `close_trade` only updates a row whose `exit_time_utc` is still NULL, so a run
    repeated after a dropped connection closes nothing twice. That matters more than it sounds —
    a double-closed trade would appear twice in the expectancy denominator.
    """
    moment = now or datetime.now(UTC)
    stats = ResolveStats()

    async with database.session() as session:
        open_trades = await TradeRepository(session).open_trades()

    for trade in open_trades:
        stats.examined += 1
        try:
            closed = await _resolve_one(database, config, trade, stats=stats, now=moment)
        except Exception as exc:  # noqa: BLE001 - one trade must not end the run
            stats.errors += 1
            logger.exception("could not resolve trade %s (%s): %s", trade.id, trade.symbol, exc)
            continue
        if closed:
            stats.closed += 1

    logger.info(
        "resolver: examined %d, closed %d, still running %d, skipped %d, errors %d",
        stats.examined,
        stats.closed,
        stats.still_running,
        stats.skipped,
        stats.errors,
    )
    return stats


async def _resolve_one(
    database: Database,
    config: ResolveConfig,
    trade: TradeRecord,
    *,
    stats: ResolveStats,
    now: datetime,
) -> bool:
    spec = config.specs.get(trade.symbol)
    if spec is None:
        stats.skipped += 1
        logger.warning(
            "no SymbolSpec for %s; trade %s left open rather than resolved against a guess",
            trade.symbol,
            trade.id,
        )
        return False

    step = TIMEFRAMES[config.timeframe]
    # One bar of slack past the time barrier, so the horizon bar itself is inside the window.
    window_end = trade.entry_time_utc + step * (config.max_bars_held + 1)

    async with database.session() as session:
        repository = BarRepository(session)
        bars = await repository.bars_between(
            trade.symbol,
            config.timeframe,
            start=trade.entry_time_utc,
            end=min(window_end, now),
            source=config.source,
        )
        quotes = await repository.quotes_between(
            trade.symbol,
            config.timeframe,
            start=trade.entry_time_utc,
            end=min(window_end, now),
            source=config.source,
        )

    entry_index = _entry_index(bars, trade.entry_time_utc)
    if entry_index is None:
        # The entry bar is not in the series under this source. Skipped rather than resolved
        # from a neighbouring bar: an outcome measured from the wrong entry is worse than an
        # outcome not yet measured, and it would be indistinguishable afterwards.
        stats.skipped += 1
        logger.warning(
            "trade %s: no %s bar at %s from %s; leaving it open",
            trade.id,
            config.timeframe,
            trade.entry_time_utc.isoformat(),
            config.source,
        )
        return False

    if len(bars) - 1 <= entry_index:
        # No bar after entry yet. The normal case for a signal from the current hour.
        stats.still_running += 1
        return False

    side = _side(trade.direction)
    outcome = resolve_barriers(
        bars,
        entry_index,
        side=side,
        stop_price=trade.stop_price,
        target_price=trade.target_price,
        max_bars=config.max_bars_held,
    )

    settled = outcome.touched is not Barrier.TIME or (
        outcome.exit_index >= entry_index + config.max_bars_held
    )
    if not settled:
        # `resolve_barriers` returns TIME when it runs out of bars as well as when the horizon
        # is genuinely reached, and those are different facts: the first means "ask again
        # tomorrow" and the second means "this trade is over". Closing the first as a TIME exit
        # would stamp a horizon outcome on a position that had not reached one.
        stats.still_running += 1
        return False

    exit_bid, exit_ask = quotes.get(outcome.exit_time, (None, None))
    # Closing reverses the side: a long is closed by selling, and sells hit the bid.
    exit_side = OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY
    exit_fill = fill(
        outcome.exit_price, exit_side, spec, config.costs, Quote(bid=exit_bid, ask=exit_ask)
    )

    swap = swap_cost(side, trade.volume, trade.entry_time_utc, outcome.exit_time, config.costs)

    direction = 1.0 if side is OrderSide.BUY else -1.0
    pnl = spec.money_per_lot((exit_fill.price - trade.entry_price) * direction) * trade.volume
    pnl += swap

    # `r_multiple` takes prices, not money, and carries the direction in the sign of
    # (entry - stop) so it cannot be passed in wrongly. It is computed on the *filled* exit,
    # which is what makes this comparable with the replay: both charge the spread before the
    # result is expressed in R, rather than measuring a clean price and deducting costs after.
    realised_r = r_multiple(trade.entry_price, exit_fill.price, trade.stop_price)

    async with database.begin() as session:
        did_close = await TradeRepository(session).close_trade(
            trade.id,
            exit_price=exit_fill.price,
            exit_time_utc=outcome.exit_time,
            barrier_touched=str(outcome.touched),
            pnl=pnl,
            r_multiple=realised_r,
        )

    if not did_close:
        # Another run closed it between the read and the write. Not an error; the row is right.
        logger.debug("trade %s was already closed by a concurrent run", trade.id)
        return False

    key = str(outcome.touched)
    stats.outcomes[key] = stats.outcomes.get(key, 0) + 1
    logger.info(
        "closed trade %s (%s %s): %s at %s, %.3fR%s",
        trade.id,
        trade.symbol,
        trade.direction,
        key,
        outcome.exit_time.isoformat(),
        realised_r if realised_r is not None else float("nan"),
        " [ambiguous bar, resolved to STOP]" if outcome.ambiguous_resolution else "",
    )
    return True


def horizon_for(entry: datetime, timeframe: str, max_bars_held: int) -> datetime:
    """When a trade opened at `entry` reaches its time barrier. Used by the label span."""
    return entry + TIMEFRAMES[timeframe] * max_bars_held
