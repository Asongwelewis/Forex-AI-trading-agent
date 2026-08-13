"""Turn a bar series into a `Regime`: where we are in the day, and what the market is doing.

The classifier measures; it does not decide. Nothing here knows a strategy exists. It reads
the indicator layer, reads the session clock, and produces a value object that the router
then applies rules to — which keeps "what is true" and "what is therefore allowed" in
separate files that can be tested and tuned independently.

**Warm-up is reported, not faked.** ADX and the volatility percentile are `None` until enough
history exists to compute them honestly. A classifier that returned 0.0 during warm-up would
be indistinguishable from a genuinely flat market, and `is_ranging` would fire on the first
hundred bars of every backtest. `None` propagates as "unknown", and unknown means neither
ranging nor trending — so the gated strategies stay shut rather than guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, computed_field

from fxagent.adapters.base import BarSeries
from fxagent.indicators import adx, atr, rolling_percentile
from fxagent.regime.sessions import (
    Session,
    active_sessions,
    dominant_session,
    is_market_open,
    minutes_until_close,
)
from fxagent.strategies.base import UtcDatetime, bars_to_frame

__all__ = ["ClassifierConfig", "Regime", "RegimeClassifier"]


@dataclass(frozen=True)
class ClassifierConfig:
    """Measurement periods and the two thresholds that split trending from ranging.

    Thresholds live here rather than in the `Regime` model so a tuning change never touches
    the value object that gets logged, and so the same stored regime can be re-read under a
    different rule set later.
    """

    adx_period: int = 14
    atr_period: int = 14
    #: Trailing window the current ATR is ranked against, in bars.
    volatility_lookback: int = 100
    #: ADX below this is a range.
    ranging_below: float = 20.0
    #: ADX above this is a trend. The gap between the two is deliberate: it is neither.
    trending_above: float = 25.0

    def __post_init__(self) -> None:
        if self.ranging_below >= self.trending_above:
            raise ValueError(
                f"ranging_below ({self.ranging_below}) must sit under trending_above "
                f"({self.trending_above}); overlapping thresholds would let one bar be both"
            )

    @property
    def required_bars(self) -> int:
        """History needed before every field can be measured.

        ADX needs about two periods to settle. The percentile needs a full lookback window of
        ATR values, and ATR needs its own warm-up first, so those compose rather than max.
        """
        return max(2 * self.adx_period, self.atr_period + self.volatility_lookback) + 1


class Regime(BaseModel):
    """What the market looked like at one bar. A measurement, not a decision.

    `sessions` is the authoritative field and `session` is a derived label. The set is what
    the router tests against, because the rules ask overlapping questions — "is London
    trading" and "are we in the overlap" are both true at 14:00 UTC in January, and a single
    field could only answer one of them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    timestamp: UtcDatetime
    sessions: tuple[Session, ...] = Field(
        description="Every session trading at this bar, sorted for a stable journal line."
    )
    market_open: bool
    minutes_until_weekly_close: int = Field(ge=0)
    trend_strength: float | None = Field(
        default=None, description="ADX. None until the warm-up has passed."
    )
    volatility_percentile: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Current ATR's rank in its trailing window."
    )
    is_ranging: bool = False
    is_trending: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def session(self) -> Session | None:
        """The single best label for this bar, or None when nothing is trading."""
        return dominant_session(frozenset(self.sessions))

    def in_session(self, session: Session) -> bool:
        return session in self.sessions


class RegimeClassifier:
    """Measures a `Regime` from bars. Pure: the timestamp comes from the last bar, not a clock."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self._config = config or ClassifierConfig()

    @property
    def config(self) -> ClassifierConfig:
        return self._config

    @property
    def required_bars(self) -> int:
        return self._config.required_bars

    def classify(self, bars: BarSeries) -> Regime:
        moment = bars.bars[-1].timestamp
        sessions = active_sessions(moment)

        trend_strength = self._trend_strength(bars)
        return Regime(
            symbol=bars.symbol,
            timestamp=moment,
            sessions=tuple(sorted(sessions)),
            market_open=is_market_open(moment),
            minutes_until_weekly_close=minutes_until_close(moment),
            trend_strength=trend_strength,
            volatility_percentile=self._volatility_percentile(bars),
            # Unknown strength is neither state. See the module docstring.
            is_ranging=trend_strength is not None and trend_strength < self._config.ranging_below,
            is_trending=(
                trend_strength is not None and trend_strength > self._config.trending_above
            ),
        )

    def _trend_strength(self, bars: BarSeries) -> float | None:
        frame = bars_to_frame(bars)
        series = adx(frame["high"], frame["low"], frame["close"], self._config.adx_period)
        return _last_finite(series)

    def _volatility_percentile(self, bars: BarSeries) -> float | None:
        frame = bars_to_frame(bars)
        ranges = atr(frame["high"], frame["low"], frame["close"], self._config.atr_period)
        return _last_finite(rolling_percentile(ranges, self._config.volatility_lookback))


def _last_finite(series: pd.Series) -> float | None:
    """The final value of an indicator series, or None if it is still warming up."""
    value = float(series.iloc[-1])
    return None if math.isnan(value) else value
