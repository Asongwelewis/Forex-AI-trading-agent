"""Hold the currency that pays you, but only while the daily trend is not fighting you.

Carry is the edge; price is the veto. A positive rate differential says holding the pair
long earns interest, but carry trades die in drawdowns, not in flat markets — so the
50-period daily EMA must be sloping the same way the carry points, and the injected macro
bias must not be arguing the opposite. Any of the three disagreeing means no trade.

`rate_differential` and `macro_bias` arrive on `MarketContext` and are never fetched here.
That is what keeps the strategy replayable: a backtest supplies the differential that was
true on the bar being replayed, not the one that is true now.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fxagent.adapters.base import BarSeries
from fxagent.indicators import atr, ema
from fxagent.strategies.base import (
    MarketContext,
    Signal,
    SignalDirection,
    Strategy,
    bars_to_frame,
)

if TYPE_CHECKING:
    from fxagent.regime.classifier import Regime

__all__ = ["CarryDivergence"]

EMA_PERIOD = 50
#: Bars between the two EMA readings whose difference defines "slope".
SLOPE_LOOKBACK = 1

ATR_PERIOD = 14
#: Multi-day holds need room; this is wider than the intraday strategies on purpose.
ATR_STOP_MULTIPLE = 2.0
REWARD_RISK = 2.0

TIMEFRAME = "D1"


class CarryDivergence(Strategy):
    """Daily carry trade, gated on EMA slope agreement and a non-opposing macro bias."""

    @property
    def name(self) -> str:
        return "carry_divergence"

    @property
    def required_bars(self) -> int:
        """The EMA seed plus one more bar to measure a slope across."""
        return max(EMA_PERIOD + SLOPE_LOOKBACK, ATR_PERIOD + 1)

    def generate(
        self, bars: BarSeries, context: MarketContext, regime: Regime | None = None
    ) -> Signal | None:
        """`regime` is accepted and ignored: this strategy gates on the rate differential
        and its own EMA slope, neither of which the classifier measures."""
        if bars.timeframe != TIMEFRAME:
            raise ValueError(
                f"{self.name} reads {TIMEFRAME} bars; got {bars.timeframe!r}. Carry is a "
                "multi-day hold and must not be judged on an intraday series."
            )
        if not self._has_enough_history(bars):
            return None

        direction = self._carry_direction(context.rate_differential)
        if direction is None:
            return None
        if self._macro_opposes(direction, context.macro_bias):
            return None

        frame = bars_to_frame(bars)
        close = frame["close"]

        trend = ema(close, EMA_PERIOD)
        slope = trend.iloc[-1] - trend.iloc[-1 - SLOPE_LOOKBACK]
        if math.isnan(slope) or not self._slope_agrees(direction, slope):
            return None

        average_range = atr(frame["high"], frame["low"], close, ATR_PERIOD).iloc[-1]
        if math.isnan(average_range) or average_range <= 0.0:
            return None

        last = bars.bars[-1]
        entry = last.close
        risk = ATR_STOP_MULTIPLE * float(average_range)
        if risk <= 0.0 or risk >= entry:
            # A stop at or below zero is not a stop. Only reachable on absurd inputs, but
            # `Signal` would reject it downstream and a None here says why.
            return None

        if direction is SignalDirection.LONG:
            stop, target = entry - risk, entry + REWARD_RISK * risk
        else:
            stop, target = entry + risk, entry - REWARD_RISK * risk

        return Signal(
            symbol=bars.symbol,
            direction=direction,
            confidence=min(1.0, 0.5 + 0.5 * abs(context.macro_bias)),
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            strategy_name=self.name,
            timestamp=last.timestamp,
            reasoning={
                "rate_differential": context.rate_differential,
                "macro_bias": context.macro_bias,
                "ema_slope": float(slope),
                "ema": float(trend.iloc[-1]),
                "atr": float(average_range),
                "risk_distance": risk,
            },
        )

    def _carry_direction(self, rate_differential: float) -> SignalDirection | None:
        """Positive carry is long the pair; negative is short. Exactly zero has no side."""
        if rate_differential > 0.0:
            return SignalDirection.LONG
        if rate_differential < 0.0:
            return SignalDirection.SHORT
        return None

    def _macro_opposes(self, direction: SignalDirection, macro_bias: float) -> bool:
        """Block only an actively contrary view.

        A neutral bias permits the trade rather than blocking it. Phase 5 falls back to
        neutral whenever the macro brief fails validation, and a strategy that goes silent
        every time an LLM call fails would make deterministic carry logic hostage to it.
        """
        if direction is SignalDirection.LONG:
            return macro_bias < 0.0
        return macro_bias > 0.0

    def _slope_agrees(self, direction: SignalDirection, slope: float) -> bool:
        """A flat EMA agrees with nothing; the trend must actively point the same way."""
        if direction is SignalDirection.LONG:
            return slope > 0.0
        return slope < 0.0
