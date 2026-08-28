"""Fade a statistical extreme, but only in a market that is actually ranging.

The z-score finds the stretch; the regime decides whether fading it is sane. That gate is
the whole strategy. A price two standard deviations above its mean is a reversion setup in a
range and a *trend confirmation* in a trend — the same number, opposite meaning — so
mean-reversion without a trend filter is a machine for selling breakouts.

**"Ranging" is not defined here.** It is `regime.is_ranging`, computed once by
`RegimeClassifier` from `ClassifierConfig`. This file used to own an `ADX_TREND_CEILING` and
an `ADX_PERIOD` of its own, which meant the threshold the router gates on and the threshold
this strategy gates on were two numbers that happened to be equal. They would have stayed
equal right up until somebody tuned one of them, at which point the router would permit a
strategy that then refused — the same silent drift the breakout window had, with ADX in
place of hours. Reading the classifier's boolean makes the two agree by construction rather
than by coincidence, and `generate` refuses loudly if handed a regime describing some other
bar.

The stop therefore sits beyond the current bar's extreme rather than beyond the entry: if
this really is the turn, the bar that made the extreme should not be exceeded. The target
is the mean the price is stretched away from, which is the definition of the trade.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fxagent.adapters.base import Bar, BarSeries
from fxagent.indicators import atr, rolling_zscore
from fxagent.strategies.base import (
    MarketContext,
    Signal,
    SignalDirection,
    Strategy,
    bars_to_frame,
)

if TYPE_CHECKING:
    from fxagent.regime.classifier import Regime

__all__ = ["RangeReversion"]

ZSCORE_PERIOD = 20
#: How stretched the price must be before it is worth fading.
ZSCORE_TRIGGER = 2.0

ATR_PERIOD = 14
#: Stop distance beyond the current bar's extreme.
#: How much of the bar's range must sit beyond its close, on the side the stretch came from.
#: 0.25 means a quarter of the bar is rejection wick. Below that the bar is still going.
MIN_REJECTION = 0.25

ATR_STOP_MULTIPLE = 1.5

#: Denominator that maps |z| onto confidence: the trigger scores 0.5, twice it scores 1.
CONFIDENCE_SCALE = 2.0 * ZSCORE_TRIGGER


def _rejection_fraction(bar: Bar, direction: SignalDirection) -> float:
    """Share of the bar's range left as a wick against the stretch, 0 to 1.

    Shorting an overstretched bar wants an upper wick: price went higher and came back. A bar
    closing on its high has rejected nothing and the move is still running. A zero-range bar
    returns 0.0 — no evidence, which is not the same as evidence of continuation.
    """
    span = bar.high - bar.low
    if span <= 0.0:
        return 0.0
    wick = bar.high - bar.close if direction is SignalDirection.SHORT else bar.close - bar.low
    return wick / span


class RangeReversion(Strategy):
    """Fade a 20-period z-score extreme back to its mean, only while the regime is ranging."""

    @property
    def name(self) -> str:
        return "range_reversion"

    @property
    def required_bars(self) -> int:
        """What this file actually computes. The range gate's own warm-up is the classifier's.

        Smaller than it was, because the ADX it no longer measures no longer dominates. That
        is not a loosening: a classifier still warming up reports `is_ranging` as False, so
        the gate below stays shut until the regime is genuinely known.
        """
        return max(ZSCORE_PERIOD, ATR_PERIOD + 1)

    def generate(
        self, bars: BarSeries, context: MarketContext, regime: Regime | None = None
    ) -> Signal | None:
        if regime is None:
            raise ValueError(
                f"{self.name} gates on the regime and was given none. Pass the Regime the "
                "classifier produced for this bar — silently declining to trade would hide a "
                "wiring error as a missing setup."
            )
        self._check_regime_describes(bars, regime)

        if not self._has_enough_history(bars):
            return None

        # The one definition of "ranging", shared with the router. Not re-derived here.
        if not regime.is_ranging:
            return None

        frame = bars_to_frame(bars)
        high, low, close = frame["high"], frame["low"], frame["close"]

        stretch = rolling_zscore(close, ZSCORE_PERIOD).iloc[-1]
        if math.isnan(stretch) or abs(stretch) <= ZSCORE_TRIGGER:
            return None

        average_range = atr(high, low, close, ATR_PERIOD).iloc[-1]
        if math.isnan(average_range) or average_range <= 0.0:
            return None

        mean = close.rolling(ZSCORE_PERIOD).mean().iloc[-1]
        last = bars.bars[-1]
        entry = last.close
        buffer = ATR_STOP_MULTIPLE * float(average_range)

        if stretch > 0:
            direction = SignalDirection.SHORT
            stop = last.high + buffer
        else:
            direction = SignalDirection.LONG
            stop = last.low - buffer

        # The mean must still be a real target on the correct side of entry. A window whose
        # mean has been dragged past the current price offers no trade, only a wrong one.
        if direction is SignalDirection.SHORT and mean >= entry:
            return None
        if direction is SignalDirection.LONG and mean <= entry:
            return None
        if abs(entry - stop) <= 0.0:
            return None

        # Second evidence family, and the sleeve's own confirmation. The z-score says price is
        # stretched; the rejection wick says the stretch is being sold rather than continuing.
        # Stretch alone is what makes a mean-reversion sleeve short a trend all the way up — it
        # is a measure of how far, never of whether it has stopped. Requiring the bar to close
        # back off its own extreme is the cheapest independent read of that, and it is
        # independent of the z-score in a way another strategy's opinion never was.
        rejection = _rejection_fraction(last, direction)
        if rejection < MIN_REJECTION:
            return None

        return Signal(
            symbol=bars.symbol,
            direction=direction,
            confidence=min(1.0, abs(stretch) / CONFIDENCE_SCALE),
            entry_price=entry,
            stop_loss=stop,
            take_profit=float(mean),
            strategy_name=self.name,
            timestamp=last.timestamp,
            reasoning={
                "zscore": float(stretch),
                "rejection_fraction": rejection,
                "confirmation": "zscore + rejection wick",
                "rolling_mean": float(mean),
                # Recorded from the regime, not measured here — so the journal shows the
                # number the gate was actually taken on.
                "adx": regime.trend_strength,
                "atr": float(average_range),
                "stop_buffer": buffer,
                "extreme_faded": last.high if direction is SignalDirection.SHORT else last.low,
                "risk_distance": abs(entry - stop),
            },
        )

    def _check_regime_describes(self, bars: BarSeries, regime: Regime) -> None:
        """Refuse a regime measured from some other symbol or some other bar.

        Sharing the classifier's boolean only guarantees agreement if the boolean describes
        the bar being traded. A regime from the previous hour, or from another pair, is a
        wiring error that would otherwise read as a perfectly ordinary range.
        """
        last = bars.bars[-1]
        if regime.symbol != bars.symbol:
            raise ValueError(
                f"{self.name} was given a regime for {regime.symbol!r} while trading "
                f"{bars.symbol!r}"
            )
        if regime.timestamp != last.timestamp:
            raise ValueError(
                f"{self.name} was given a regime measured at {regime.timestamp.isoformat()} "
                f"but the last bar opens at {last.timestamp.isoformat()}; the range gate would "
                "be taken from a different bar than the trade"
            )
