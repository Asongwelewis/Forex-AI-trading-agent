"""Synthetic bars and a scripted strategy, for testing the loop rather than the strategies.

The replay engine's job is the clock, the costs and the bookkeeping. Driving it with the real
`SessionBreakout` would mean constructing bars that satisfy its gates before any of that could
be asserted, and a test that fails because a gate moved has told you nothing about the loop. So
the engine takes its strategies as an argument and these fixtures supply a scripted one.

`tests/backtest/test_replay.py` separately asserts that `default_strategies()` returns the real
three, so the injection point cannot become a way for the harness to drift from the pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fxagent.adapters.base import Bar, BarSeries
from fxagent.regime.classifier import Regime
from fxagent.regime.router import RegimeRouter
from fxagent.strategies.base import MarketContext, Signal, SignalDirection, Strategy

START = datetime(2026, 3, 2, 0, 0, tzinfo=UTC)  # a Monday


def bar(
    index: int,
    *,
    open_: float = 1.1000,
    high: float | None = None,
    low: float | None = None,
    close: float | None = None,
    start: datetime = START,
) -> Bar:
    """One H1 bar `index` hours after `start`, defaulting to a flat doji."""
    close = open_ if close is None else close
    high = max(open_, close) if high is None else high
    low = min(open_, close) if low is None else low
    return Bar(
        timestamp=start + timedelta(hours=index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100,
    )


def series(bars: list[Bar], symbol: str = "EURUSD") -> BarSeries:
    return BarSeries(symbol=symbol, timeframe="H1", bars=tuple(bars))


def flat_series(count: int, *, price: float = 1.1000, symbol: str = "EURUSD") -> BarSeries:
    """A run of identical bars — enough history to clear every warm-up, and no setup in it."""
    return series([bar(index, open_=price) for index in range(count)], symbol=symbol)


class ScriptedStrategy(Strategy):
    """Fires a fixed signal at chosen bar indices and is silent everywhere else.

    Deliberately not a mock: it returns real `Signal` objects, so consensus, sizing and the
    barrier resolver all see exactly what they would see in production. Only the *reason* for
    the signal is scripted.
    """

    def __init__(
        self,
        name: str,
        fire_at: set[int],
        *,
        direction: SignalDirection = SignalDirection.LONG,
        stop_distance: float = 0.0025,
        reward: float = 2.0,
        confidence: float = 0.8,
    ) -> None:
        self._name = name
        self._fire_at = fire_at
        self._direction = direction
        self._stop_distance = stop_distance
        self._reward = reward
        self._confidence = confidence
        self.seen_lengths: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_bars(self) -> int:
        return 2

    def generate(
        self, bars: BarSeries, context: MarketContext, regime: Regime | None = None
    ) -> Signal | None:
        self.seen_lengths.append(len(bars))
        last = bars.bars[-1]
        # Index within the whole run is recovered from the timestamp, so the script stays valid
        # however much history the engine chose to hand over.
        index = int((last.timestamp - START).total_seconds() // 3600)
        if index not in self._fire_at:
            return None

        entry = last.close
        if self._direction is SignalDirection.LONG:
            stop = entry - self._stop_distance
            target = entry + self._stop_distance * self._reward
        else:
            stop = entry + self._stop_distance
            target = entry - self._stop_distance * self._reward

        return Signal(
            symbol=bars.symbol,
            direction=self._direction,
            confidence=self._confidence,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            strategy_name=self._name,
            timestamp=last.timestamp,
        )


def agreeing_pair(fire_at: set[int], **kwargs: object) -> dict[str, Strategy]:
    """Two scripted strategies that agree — the minimum consensus will act on."""
    first = ScriptedStrategy("session_breakout", fire_at, **kwargs)  # type: ignore[arg-type]
    second = ScriptedStrategy("carry_divergence", fire_at, **kwargs)  # type: ignore[arg-type]
    return {first.name: first, second.name: second}


class FullWeightRouter(RegimeRouter):
    """A real router that weights every strategy fully, whatever the regime.

    Subclassing rather than stubbing, so the engine still receives a `RegimeRouter` and the
    consensus path is exercised for real. What is removed is only the regime gating, which is
    tested exhaustively in `tests/regime/` and which would otherwise force every fixture here to
    manufacture a trending London morning before the replay loop could be exercised at all.
    """

    def weights(self, regime: Regime) -> dict[str, float]:  # noqa: ARG002 - deliberately ignored
        return {
            "session_breakout": 1.0,
            "range_reversion": 1.0,
            "carry_divergence": 1.0,
        }
