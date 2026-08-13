"""Fixtures for the regime tests.

`regime_at` derives its session fields from the real session clock rather than accepting
them as arguments, so a router test written against "09:00 UTC in January" is testing the
same conversion the live path uses. Only the measured indicator values are injected.
"""

from __future__ import annotations

from datetime import datetime

from fxagent.regime.classifier import ClassifierConfig, Regime
from fxagent.regime.sessions import active_sessions, is_market_open, minutes_until_close
from fxagent.strategies.base import Signal, SignalDirection

SYMBOL = "EURUSD"
ENTRY = 1.1000


def regime_at(
    moment: datetime,
    *,
    trend_strength: float | None = 30.0,
    volatility_percentile: float | None = 50.0,
    symbol: str = SYMBOL,
    config: ClassifierConfig | None = None,
) -> Regime:
    """A `Regime` at a real instant, with the ranging/trending flags derived as the classifier
    derives them so a test cannot accidentally assert an impossible combination."""
    thresholds = config or ClassifierConfig()
    return Regime(
        symbol=symbol,
        timestamp=moment,
        sessions=tuple(sorted(active_sessions(moment))),
        market_open=is_market_open(moment),
        minutes_until_weekly_close=minutes_until_close(moment),
        trend_strength=trend_strength,
        volatility_percentile=volatility_percentile,
        is_ranging=trend_strength is not None and trend_strength < thresholds.ranging_below,
        is_trending=trend_strength is not None and trend_strength > thresholds.trending_above,
    )


def signal(
    strategy_name: str,
    direction: SignalDirection,
    *,
    timestamp: datetime,
    confidence: float = 0.6,
    symbol: str = SYMBOL,
    entry: float = ENTRY,
) -> Signal:
    """A structurally valid signal, with the stop and target on the correct sides of entry."""
    if direction is SignalDirection.FLAT:
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry,
            strategy_name=strategy_name,
            timestamp=timestamp,
        )

    offset = 0.0020
    if direction is SignalDirection.LONG:
        stop, target = entry - offset, entry + 2 * offset
    else:
        stop, target = entry + offset, entry - 2 * offset

    return Signal(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        strategy_name=strategy_name,
        timestamp=timestamp,
    )
