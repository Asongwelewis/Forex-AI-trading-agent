"""Spread, slippage and swap. **One implementation, and it does not live in `backtest/`.**

Card 16's backtest and card 22's paper resolver must both charge the same costs, or the two
stop being comparable — and comparing them is the entire reason for running both. A backtest
that is optimistic by half a pip per trade and a paper run that is not will disagree by exactly
that, and the disagreement will be read as alpha decay, or as a broken strategy, or as anything
except the arithmetic difference it is.

So this module sits at the top level rather than inside `fxagent/backtest/`. That placement is
the safeguard: a live resolver that had to write `from fxagent.backtest.costs import ...` is a
resolver whose author will eventually decide that importing the backtest package into the
execution path is wrong, and write a small local version instead. That is how the two fork.
Here the module belongs to neither caller and both import it as a peer, and
`tests/test_costs_are_shared.py` asserts it reaches back into neither.

**Every cost is charged against the trader.** Fills happen on the wrong side of the spread,
slippage is always adverse, and swap is a debit unless a configured rate says otherwise. There
is no path here that improves a fill, because a cost model that can flatter a backtest is a
cost model that eventually will.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

from fxagent.adapters.base import OrderSide
from fxagent.risk.symbols import SymbolSpec

__all__ = [
    "ROLLOVER_HOUR_UTC",
    "TRIPLE_SWAP_WEEKDAY",
    "CostConfig",
    "Fill",
    "Quote",
    "SpreadSource",
    "fill",
    "rollover_nights",
    "swap_cost",
]

#: Exness rolls at 21:00 server time and its server clock is UTC — measured, see CLAUDE.md. A
#: broker on GMT+2 rolls at 19:00 UTC and this constant would be wrong for it, which is why it
#: is named and documented rather than written as a bare 21 inside a comparison.
ROLLOVER_HOUR_UTC: Final = 21

#: Wednesday. Spot FX settles T+2, so the position rolled on Wednesday night carries its value
#: date over the weekend and is charged three nights. Monday is 0, so Wednesday is 2.
TRIPLE_SWAP_WEEKDAY: Final = 2


class SpreadSource(StrEnum):
    """Where the spread on a fill came from. **Reported, never assumed.**

    A run on stored quotes and a run on a configured constant are different experiments, and the
    second is only as good as the guess in its config. Every `Fill` carries which one it used, so
    a run that fell back to the fixed spread because the feed had no quotes cannot be read as if
    it had not.
    """

    STORED = "STORED"
    FIXED = "FIXED"


@dataclass(frozen=True)
class Quote:
    """Bid and ask at a bar close, as the feed stored them.

    Both are optional and both must be present to be usable: a row with a bid and no ask has no
    spread, and inventing the missing half would produce a fill that looks sourced and is not.
    """

    bid: float | None = None
    ask: float | None = None

    @property
    def usable(self) -> bool:
        return self.bid is not None and self.ask is not None and self.ask >= self.bid


@dataclass(frozen=True)
class CostConfig:
    """What the broker charges. The defaults are deliberately not free.

    `swap_long_per_lot` and `swap_short_per_lot` are account currency per lot per night, signed
    the way a broker quotes them: negative is a debit. They default to zero because a wrong swap
    number is worse than an absent one — a carry strategy sized against an invented rate is
    being backtested against a fiction — and `swap_is_configured` lets a report say so out loud.
    """

    fixed_spread_pips: float = 1.0
    slippage_pips: float = 0.5
    swap_long_per_lot: float = 0.0
    swap_short_per_lot: float = 0.0

    def __post_init__(self) -> None:
        if self.fixed_spread_pips < 0:
            raise ValueError(
                f"fixed_spread_pips must not be negative, got {self.fixed_spread_pips}"
            )
        if self.slippage_pips < 0:
            raise ValueError(
                f"slippage_pips must not be negative, got {self.slippage_pips}. Negative "
                "slippage is a fill better than the price asked for, and a cost model that can "
                "improve a fill will eventually be tuned until it does."
            )

    @property
    def swap_is_configured(self) -> bool:
        """Whether either rate was actually supplied. Reported beside any multi-day result."""
        return self.swap_long_per_lot != 0.0 or self.swap_short_per_lot != 0.0


@dataclass(frozen=True)
class Fill:
    """A price to trade at, and every adjustment that produced it.

    The components travel separately from the total so a report can attribute a result to the
    spread rather than to the strategy. "Expectancy fell 0.3R when slippage went from 0.5 pips
    to 1.0" is a finding; "expectancy is 0.1R", with the costs folded invisibly into it, is not.
    """

    price: float
    reference_price: float
    side: OrderSide
    spread: float
    spread_source: SpreadSource
    slippage: float

    @property
    def total_cost(self) -> float:
        """How far this fill sits from the reference price. Always against the trader."""
        return abs(self.price - self.reference_price)


def fill(
    reference_price: float,
    side: OrderSide,
    spec: SymbolSpec,
    config: CostConfig,
    quote: Quote | None = None,
) -> Fill:
    """Where this order actually fills: wrong side of the spread, then slippage on top.

    A BUY lifts the ask and a SELL hits the bid — never the mid, and never the bar close, which
    is a mid-ish print no counterparty ever offered. Where the feed stored real quotes they are
    used and the fill is marked `STORED`; otherwise the configured spread is applied
    symmetrically around the reference and marked `FIXED`, because a symmetric guess is the only
    honest thing to do with a number nobody measured.

    Slippage is adverse on both sides. Exness is a market-maker CFD broker whose spreads widen
    on news, so the default half-pip is a floor rather than an estimate — this system takes
    session-open breakouts, which is exactly when it is worst.
    """
    if reference_price <= 0:
        raise ValueError(f"reference_price must be positive, got {reference_price}")

    slippage = config.slippage_pips * spec.pip

    if quote is not None and quote.usable:
        # `usable` has already established both are present; the locals keep the type narrow.
        bid, ask = quote.bid, quote.ask
        assert bid is not None and ask is not None
        spread = ask - bid
        base = ask if side is OrderSide.BUY else bid
        source = SpreadSource.STORED
    else:
        spread = config.fixed_spread_pips * spec.pip
        half = spread / 2.0
        base = reference_price + half if side is OrderSide.BUY else reference_price - half
        source = SpreadSource.FIXED

    price = base + slippage if side is OrderSide.BUY else base - slippage
    return Fill(
        price=price,
        reference_price=reference_price,
        side=side,
        spread=spread,
        spread_source=source,
        slippage=slippage,
    )


def rollover_nights(entry: datetime, exit_time: datetime) -> int:
    """Swap-charging nights between two instants, counting Wednesday three times.

    A night is charged when the position is open across a rollover instant, so the boundary is
    `entry < rollover <= exit`. A trade opened at 21:30 and closed at 22:00 the same evening
    crosses nothing and pays nothing; one opened at 20:30 and closed at 21:30 crosses once, even
    though it lived for an hour. Intraday strategies pay no swap and multi-day ones pay for
    every night they were actually open, which is the distinction that matters to
    `carry_divergence`.

    Wednesday counts triple because spot FX settles T+2 and Wednesday's roll carries the value
    date across the weekend. Getting this wrong understates the cost of every held position by
    two nights a week, which for a multi-day carry strategy is most of the carry.
    """
    if entry.tzinfo is None or exit_time.tzinfo is None:
        raise ValueError("entry and exit must be timezone-aware; a rollover is a wall-clock event")
    if exit_time < entry:
        raise ValueError(f"exit {exit_time} is before entry {entry}")

    nights = 0
    # Start from the rollover instant on the entry's own date. Beginning a day early would cost
    # one harmless iteration; beginning a day late is the off-by-one when entry is after 21:00.
    moment = entry.replace(hour=ROLLOVER_HOUR_UTC, minute=0, second=0, microsecond=0)
    while moment <= exit_time:
        if moment > entry:
            nights += 3 if moment.weekday() == TRIPLE_SWAP_WEEKDAY else 1
        moment += timedelta(days=1)
    return nights


def swap_cost(
    side: OrderSide, volume: float, entry: datetime, exit_time: datetime, config: CostConfig
) -> float:
    """Total swap over the life of a position, signed as a P&L adjustment.

    Negative is a debit, which is the usual case on both sides of most pairs once the broker's
    markup is applied. Returns exactly 0.0 when no rate is configured — see `CostConfig`, where
    the default is zero precisely so an unconfigured swap is visibly absent rather than quietly
    invented.
    """
    if volume <= 0:
        raise ValueError(f"volume must be positive, got {volume}")
    rate = config.swap_long_per_lot if side is OrderSide.BUY else config.swap_short_per_lot
    return rollover_nights(entry, exit_time) * volume * rate
