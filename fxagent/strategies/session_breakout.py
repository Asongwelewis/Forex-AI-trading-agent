"""London-open breakout of the Asian session range.

Tokyo grinds out a range while London sleeps; when London arrives, the first hourly close
that leaves that range is the setup. Everything here is decided from the bars themselves,
including which UTC day is "today" — the strategy never reads a clock.

**The breakout window is not ours to define.** It comes from `regime.sessions.LONDON_MORNING`
— the same object the router gates on — because a strategy that decides for itself when
London opens will disagree with the router that decides whether it may speak. It did: fixed
UTC hours here meant the router permitted 07:00-10:59 UTC every summer while this file only
spoke from 08:00, losing the first permitted hour and offering a signal in an hour the router
refused. Deriving the window rather than restating it makes that class of drift unavailable.

**The two windows are asymmetric, and deliberately so.** London's hours move against UTC
twice a year, so the breakout window must be derived. Tokyo's do not — Japan has observed no
daylight saving since 1951 — so the pre-London range genuinely is fixed in UTC and is left
alone. Shifting both would have been symmetry for its own sake, and would have moved a window
that was never wrong.

Three refusals matter as much as the entry:

* Outside the London morning there is no setup, so the strategy is silent. Later breakouts
  are a different animal driven by New York, not by London's open.
* If an earlier bar in the window already closed outside the range, the move is not new
  and we do not chase it. Only the *first* close beyond the range trades.
* A day whose Asian session is not fully present cannot have its range measured, so no
  signal is offered for it at all rather than one drawn from a partial range.
"""

from __future__ import annotations

import math
from datetime import date

from fxagent.adapters.base import Bar, BarSeries
from fxagent.indicators import atr

# Imported from the leaf module rather than the `fxagent.regime` package on purpose:
# `regime.sessions` imports nothing from fxagent, so this stays safe in both directions even
# though `regime.consensus` imports `strategies.base`.
from fxagent.regime.sessions import LONDON_MORNING, SessionOpening
from fxagent.strategies.base import (
    MarketContext,
    Signal,
    SignalDirection,
    Strategy,
    bars_to_frame,
)

__all__ = ["SessionBreakout"]

#: Asian range is measured from bars OPENING in [00:00, 07:00) UTC, i.e. hours 0-6.
#:
#: Legitimately fixed in UTC, unlike the breakout window. Asia/Tokyo is UTC+9 the whole year
#: round, so these bars are the same local hours in January and July. The window also ends
#: exactly as London opens in summer, and an hour before it in winter — it never overlaps the
#: session it is supposed to precede.
ASIAN_HOURS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

ATR_PERIOD = 14
#: Stop distance ceiling. The range's far side is used instead whenever it is nearer.
ATR_STOP_MULTIPLE = 1.5
#: Target distance as a multiple of the risk actually taken.
REWARD_RISK = 2.0

#: Two days of H1. The ATR warm-up needs 15 bars; the session logic needs the whole day.
REQUIRED_BARS = 48

TIMEFRAME = "H1"


def _bars_on(bars: BarSeries, day: date) -> list[Bar]:
    return [bar for bar in bars.bars if bar.timestamp.date() == day]


class SessionBreakout(Strategy):
    """First H1 close outside the Asian range, taken in the direction of the break."""

    def __init__(self, window: SessionOpening = LONDON_MORNING) -> None:
        #: Pass the router's own `RouterConfig.breakout_window` to keep them aligned when
        #: either is tuned. The default is the shared object, so they start aligned.
        self._window = window

    @property
    def window(self) -> SessionOpening:
        """The session window this strategy will speak in. The router gates on the same one."""
        return self._window

    @property
    def name(self) -> str:
        return "session_breakout"

    @property
    def required_bars(self) -> int:
        return REQUIRED_BARS

    def generate(self, bars: BarSeries, context: MarketContext) -> Signal | None:
        if bars.timeframe != TIMEFRAME:
            raise ValueError(
                f"{self.name} reads {TIMEFRAME} bars; got {bars.timeframe!r}. The caller "
                "chose the wrong series — this is wiring, not a missing setup."
            )
        if not self._has_enough_history(bars):
            return None

        last = bars.bars[-1]
        if not self._window.permits(last.timestamp):
            return None

        session_range = self._asian_range(bars, last.timestamp.date())
        if session_range is None:
            return None
        range_high, range_low = session_range

        direction = self._break_direction(last.close, range_high, range_low)
        if direction is None:
            return None
        if self._already_broke_out(bars, last, range_high, range_low):
            return None

        average_range = atr(
            *(bars_to_frame(bars)[column] for column in ("high", "low", "close")),
            ATR_PERIOD,
        ).iloc[-1]
        if math.isnan(average_range) or average_range <= 0.0:
            return None

        entry = last.close
        stop, stop_source = self._stop_for(direction, entry, range_high, range_low, average_range)
        risk = abs(entry - stop)
        if risk <= 0.0:
            return None

        if direction is SignalDirection.LONG:
            target = entry + REWARD_RISK * risk
            beyond = entry - range_high
        else:
            target = entry - REWARD_RISK * risk
            beyond = range_low - entry

        return Signal(
            symbol=bars.symbol,
            direction=direction,
            confidence=min(1.0, beyond / average_range),
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            strategy_name=self.name,
            timestamp=last.timestamp,
            reasoning={
                "asian_high": range_high,
                "asian_low": range_low,
                "range_width": range_high - range_low,
                "break_distance": beyond,
                "atr": float(average_range),
                "stop_source": stop_source,
                "risk_distance": risk,
                "breakout_hour_utc": last.timestamp.hour,
                # The local hour is the one the rule is actually written in; logging only
                # the UTC hour makes a summer signal look an hour early forever after.
                "breakout_hour_local": self._window.local_hour(last.timestamp),
                "breakout_session": str(self._window.session),
            },
        )

    def _asian_range(self, bars: BarSeries, day: date) -> tuple[float, float] | None:
        """High and low of the day's 00:00-07:00 UTC bars, or None if any hour is missing."""
        session = [bar for bar in _bars_on(bars, day) if bar.timestamp.hour in ASIAN_HOURS]
        if {bar.timestamp.hour for bar in session} != set(ASIAN_HOURS):
            return None
        return max(bar.high for bar in session), min(bar.low for bar in session)

    def _break_direction(
        self, close: float, range_high: float, range_low: float
    ) -> SignalDirection | None:
        if close > range_high:
            return SignalDirection.LONG
        if close < range_low:
            return SignalDirection.SHORT
        return None

    def _already_broke_out(
        self, bars: BarSeries, last: Bar, range_high: float, range_low: float
    ) -> bool:
        """True if a bar earlier in today's window already closed outside the range.

        "Earlier in the window" is asked of the same `permits` the entry gate uses, so the
        first-break rule covers exactly the hours a signal could have been offered in — not
        an hour more in winter or one fewer in summer.
        """
        earlier = [
            bar
            for bar in _bars_on(bars, last.timestamp.date())
            if bar.timestamp < last.timestamp and self._window.permits(bar.timestamp)
        ]
        return any(bar.close > range_high or bar.close < range_low for bar in earlier)

    def _stop_for(
        self,
        direction: SignalDirection,
        entry: float,
        range_high: float,
        range_low: float,
        average_range: float,
    ) -> tuple[float, str]:
        """The tighter of the range's far side and an ATR-scaled stop, and which one won.

        Tighter means nearer to entry, which is the higher of the two for a long and the
        lower for a short — not simply the smaller number.
        """
        atr_stop = ATR_STOP_MULTIPLE * average_range
        if direction is SignalDirection.LONG:
            far_side, scaled = range_low, entry - atr_stop
            return (far_side, "range") if far_side > scaled else (scaled, "atr")
        far_side, scaled = range_high, entry + atr_stop
        return (far_side, "range") if far_side < scaled else (scaled, "atr")
