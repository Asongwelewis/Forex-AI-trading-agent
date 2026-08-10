"""Fade a statistical extreme, but only in a market that is actually ranging.

The z-score finds the stretch; ADX decides whether fading it is sane. That gate is the
whole strategy. A price two standard deviations above its mean is a reversion setup in a
range and a *trend confirmation* in a trend — the same number, opposite meaning — so
mean-reversion without a trend filter is a machine for selling breakouts.

The stop therefore sits beyond the current bar's extreme rather than beyond the entry: if
this really is the turn, the bar that made the extreme should not be exceeded. The target
is the mean the price is stretched away from, which is the definition of the trade.
"""

from __future__ import annotations

import math

from fxagent.adapters.base import BarSeries
from fxagent.indicators import adx, atr, rolling_zscore
from fxagent.strategies.base import (
    MarketContext,
    Signal,
    SignalDirection,
    Strategy,
    bars_to_frame,
)

__all__ = ["RangeReversion"]

ZSCORE_PERIOD = 20
#: How stretched the price must be before it is worth fading.
ZSCORE_TRIGGER = 2.0

ADX_PERIOD = 14
#: At or above this, the market is trending and a stretch is confirmation, not excess.
ADX_TREND_CEILING = 20.0

ATR_PERIOD = 14
#: Stop distance beyond the current bar's extreme.
ATR_STOP_MULTIPLE = 1.5

#: Denominator that maps |z| onto confidence: the trigger scores 0.5, twice it scores 1.
CONFIDENCE_SCALE = 2.0 * ZSCORE_TRIGGER


class RangeReversion(Strategy):
    """Fade a 20-period z-score extreme back to its mean, only while ADX(14) < 20."""

    @property
    def name(self) -> str:
        return "range_reversion"

    @property
    def required_bars(self) -> int:
        """ADX dominates: it needs `2 * period` bars before its first value exists."""
        return max(ZSCORE_PERIOD, 2 * ADX_PERIOD, ATR_PERIOD + 1)

    def generate(self, bars: BarSeries, context: MarketContext) -> Signal | None:
        if not self._has_enough_history(bars):
            return None

        frame = bars_to_frame(bars)
        high, low, close = frame["high"], frame["low"], frame["close"]

        trend_strength = adx(high, low, close, ADX_PERIOD).iloc[-1]
        if math.isnan(trend_strength) or trend_strength >= ADX_TREND_CEILING:
            return None

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
                "rolling_mean": float(mean),
                "adx": float(trend_strength),
                "atr": float(average_range),
                "stop_buffer": buffer,
                "extreme_faded": last.high if direction is SignalDirection.SHORT else last.low,
                "risk_distance": abs(entry - stop),
            },
        )
