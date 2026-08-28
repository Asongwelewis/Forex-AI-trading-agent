"""The daily directional bias: two families must align, and it never raises off D1.

`test_an_intraday_series_returns_no_view_rather_than_raising` is the one that matters. The
intraday path asks for a bias on every bar, and the old `carry_divergence` raised on anything
but D1 — so a filter that kept that behaviour would have to be wrapped in a try, and a filter
wrapped in a try is one that ends up silently absent.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.adapters.base import Bar, BarSeries
from fxagent.regime.bias import (
    MIN_DIFFERENTIAL,
    BiasMode,
    BiasPolicy,
    DirectionalBias,
    apply_bias,
    carry_bias,
)
from fxagent.strategies.base import MarketContext, Signal, SignalDirection

LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT
START = datetime(2026, 1, 5, tzinfo=UTC)


def daily(count: int = 60, *, step: float = 0.001, timeframe: str = "D1") -> BarSeries:
    """A steadily rising daily series — the EMA slopes up."""
    return BarSeries(
        symbol="EURUSD",
        timeframe=timeframe,
        bars=tuple(
            Bar(
                timestamp=START + timedelta(days=index),
                open=1.10 + index * step,
                high=1.10 + index * step + 0.002,
                low=1.10 + index * step - 0.002,
                close=1.10 + index * step,
                volume=1000,
            )
            for index in range(count)
        ),
    )


def falling(count: int = 60) -> BarSeries:
    return daily(count, step=-0.001)


def context(differential: float = 1.0, macro: float = 0.0) -> MarketContext:
    return MarketContext(rate_differential=differential, macro_bias=macro)


def intraday_signal(direction: SignalDirection, confidence: float = 0.8) -> Signal:
    entry = 1.1000
    offset = 0.0025 if direction is LONG else -0.0025
    return Signal(
        symbol="EURUSD",
        direction=direction,
        confidence=confidence,
        entry_price=entry,
        stop_loss=entry - offset,
        take_profit=entry + 2 * offset,
        strategy_name="session_breakout",
        timestamp=START,
    )


class TestTheDailyView:
    def test_positive_carry_with_a_rising_daily_trend_is_a_long_view(self) -> None:
        bias = carry_bias(daily(), context(differential=1.0))
        assert bias.direction is LONG
        assert bias.has_view

    def test_negative_carry_with_a_falling_daily_trend_is_a_short_view(self) -> None:
        bias = carry_bias(falling(), context(differential=-1.0))
        assert bias.direction is SHORT

    def test_carry_and_trend_disagreeing_is_no_view_not_a_weak_one(self) -> None:
        """Carry trades die in drawdowns. A differential fighting the daily trend is not an edge."""
        bias = carry_bias(falling(), context(differential=1.0))
        assert not bias.has_view
        assert "two families disagreeing" in bias.reason

    def test_a_differential_inside_the_noise_band_is_no_view(self) -> None:
        bias = carry_bias(daily(), context(differential=MIN_DIFFERENTIAL / 2))
        assert not bias.has_view
        assert "noise band" in bias.reason

    def test_an_intraday_series_returns_no_view_rather_than_raising(self) -> None:
        """The intraday path asks every bar. A filter that throws here becomes a filter in a try."""
        bias = carry_bias(daily(timeframe="H1"), context(differential=1.0))
        assert not bias.has_view
        assert "H1" in bias.reason

    def test_a_short_series_returns_no_view(self) -> None:
        bias = carry_bias(daily(count=10), context(differential=1.0))
        assert not bias.has_view

    def test_a_stronger_differential_carries_more_strength(self) -> None:
        weak = carry_bias(daily(), context(differential=0.5))
        strong = carry_bias(daily(), context(differential=2.0))
        assert strong.strength > weak.strength

    def test_the_reason_is_recorded_even_when_there_is_no_view(self) -> None:
        """The ledger is the product on a bar where nothing happened."""
        assert carry_bias(daily(timeframe="H1"), context()).as_dict()["reason"]


class TestOpposition:
    def test_no_view_opposes_nothing(self) -> None:
        """Absence of a view must never read as a view."""
        none = DirectionalBias.none("warming up")
        assert not none.opposes(LONG)
        assert not none.opposes(SHORT)

    def test_a_view_opposes_only_the_other_side(self) -> None:
        bias = DirectionalBias(direction=LONG, strength=1.0, reason="test")
        assert bias.opposes(SHORT)
        assert not bias.opposes(LONG)

    def test_flat_is_never_opposed(self) -> None:
        bias = DirectionalBias(direction=LONG, strength=1.0, reason="test")
        assert not bias.opposes(SignalDirection.FLAT)


class TestApplication:
    def test_an_agreeing_signal_passes_untouched(self) -> None:
        bias = DirectionalBias(direction=LONG, strength=1.0, reason="test")
        result, note = apply_bias(intraday_signal(LONG, 0.8), bias)
        assert result is not None
        assert result.confidence == pytest.approx(0.8)
        assert not note["opposed"]

    def test_an_opposing_signal_is_downsized_by_default(self) -> None:
        bias = DirectionalBias(direction=SHORT, strength=1.0, reason="test")
        result, note = apply_bias(intraday_signal(LONG, 0.8), bias)
        assert result is not None
        assert result.confidence == pytest.approx(0.4)
        assert note["opposed"]
        assert "downsized" in note["action"]

    def test_suppress_refuses_the_trade_outright(self) -> None:
        bias = DirectionalBias(direction=SHORT, strength=1.0, reason="test")
        result, note = apply_bias(intraday_signal(LONG), bias, BiasPolicy(mode=BiasMode.SUPPRESS))
        assert result is None
        assert "suppressed" in note["action"]

    def test_downsizing_never_moves_the_levels(self) -> None:
        """The bias grades a signal. It does not get to move a stop."""
        original = intraday_signal(LONG, 0.8)
        bias = DirectionalBias(direction=SHORT, strength=1.0, reason="test")
        result, _ = apply_bias(original, bias)
        assert result is not None
        assert result.stop_loss == original.stop_loss
        assert result.take_profit == original.take_profit
        assert result.entry_price == original.entry_price

    def test_a_zero_factor_is_refused_as_a_disguised_suppress(self) -> None:
        with pytest.raises(ValueError, match="silent SUPPRESS"):
            BiasPolicy(opposed_factor=0.0)

    def test_the_note_records_the_untouched_path_too(self) -> None:
        """A bar where the filter did nothing must still record that it looked."""
        _, note = apply_bias(intraday_signal(LONG), DirectionalBias.none("no data"))
        assert note["action"] == "not opposed"
        assert note["bias"]["direction"] is None
