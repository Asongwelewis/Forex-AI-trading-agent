"""Carry divergence: three things must agree, and any one of them can veto.

Carry sets the direction, the daily EMA slope has to point the same way, and the injected
macro bias must not argue the opposite. Each negative test below holds two of the three
steady and flips only the one under test, asserting the other two were satisfied — so a
refusal is attributable rather than merely observed.

The daily ramp fixture has a constant true range of 0.0015, so ATR(14) is exactly that and
every stop and target below is an exact number rather than a tolerance.
"""

from __future__ import annotations

import pytest

from fxagent.strategies import MarketContext, SignalDirection, bars_to_frame
from fxagent.strategies.carry_divergence import CarryDivergence
from tests.strategies.builders import (
    daily_series,
    falling_daily,
    h1_series,
    rising_daily,
)

BARS = 60
EXPECTED_ATR = 0.0015
EXPECTED_RISK = 2 * EXPECTED_ATR

RISING_LAST_CLOSE = 1.1000 + 59 * 0.0010
FALLING_LAST_CLOSE = 1.1000 - 59 * 0.0010

POSITIVE_CARRY = MarketContext(rate_differential=1.5, macro_bias=0.4)
NEGATIVE_CARRY = MarketContext(rate_differential=-1.5, macro_bias=-0.4)


def _ema_slope(bars) -> float:
    from fxagent.indicators import ema

    trend = ema(bars_to_frame(bars)["close"], 50)
    return float(trend.iloc[-1] - trend.iloc[-2])


# --- it fires ------------------------------------------------------------------


def test_positive_carry_with_a_rising_daily_trend_signals_long() -> None:
    bars = daily_series(rising_daily(BARS))
    signal = CarryDivergence().generate(bars, POSITIVE_CARRY)

    assert signal is not None
    assert signal.direction is SignalDirection.LONG
    assert signal.entry_price == pytest.approx(RISING_LAST_CLOSE)
    assert signal.stop_loss == pytest.approx(RISING_LAST_CLOSE - EXPECTED_RISK)
    assert signal.take_profit == pytest.approx(RISING_LAST_CLOSE + 2 * EXPECTED_RISK)
    assert signal.reasoning["atr"] == pytest.approx(EXPECTED_ATR)
    assert signal.reward_risk == pytest.approx(2.0)
    assert signal.strategy_name == "carry_divergence"


def test_negative_carry_with_a_falling_daily_trend_signals_short() -> None:
    bars = daily_series(falling_daily(BARS))
    signal = CarryDivergence().generate(bars, NEGATIVE_CARRY)

    assert signal is not None
    assert signal.direction is SignalDirection.SHORT
    assert signal.entry_price == pytest.approx(FALLING_LAST_CLOSE)
    assert signal.stop_loss == pytest.approx(FALLING_LAST_CLOSE + EXPECTED_RISK)
    assert signal.take_profit == pytest.approx(FALLING_LAST_CLOSE - 2 * EXPECTED_RISK)


@pytest.mark.parametrize(
    ("bars_factory", "context", "direction"),
    [
        (rising_daily, POSITIVE_CARRY, SignalDirection.LONG),
        (falling_daily, NEGATIVE_CARRY, SignalDirection.SHORT),
    ],
    ids=["long", "short"],
)
def test_stop_is_always_on_the_losing_side_of_entry(
    bars_factory, context: MarketContext, direction: SignalDirection
) -> None:
    signal = CarryDivergence().generate(daily_series(bars_factory(BARS)), context)

    assert signal is not None
    if direction is SignalDirection.LONG:
        assert signal.stop_loss < signal.entry_price < signal.take_profit
    else:
        assert signal.take_profit < signal.entry_price < signal.stop_loss


def test_the_stop_is_two_atr_wide_not_one_and_a_half() -> None:
    """A multi-day hold gets more room than the intraday strategies. Pin the multiple."""
    signal = CarryDivergence().generate(daily_series(rising_daily(BARS)), POSITIVE_CARRY)

    assert signal is not None
    assert signal.risk_distance == pytest.approx(2 * EXPECTED_ATR)


def test_context_is_read_rather_than_fetched() -> None:
    """Same bars, different injected carry, opposite conclusions — the input is the context."""
    bars = daily_series(rising_daily(BARS))

    assert CarryDivergence().generate(bars, POSITIVE_CARRY) is not None
    assert CarryDivergence().generate(bars, MarketContext.neutral()) is None


# --- it refuses ----------------------------------------------------------------


def test_positive_carry_against_a_falling_trend_is_refused() -> None:
    """Carry says long, the daily EMA says down. The price veto wins."""
    bars = daily_series(falling_daily(BARS))

    assert POSITIVE_CARRY.rate_differential > 0  # carry direction is LONG
    assert _ema_slope(bars) < 0  # and the slope disagrees
    assert POSITIVE_CARRY.macro_bias > 0  # macro is not the blocker

    assert CarryDivergence().generate(bars, POSITIVE_CARRY) is None


def test_negative_carry_against_a_rising_trend_is_refused() -> None:
    bars = daily_series(rising_daily(BARS))

    assert NEGATIVE_CARRY.rate_differential < 0
    assert _ema_slope(bars) > 0

    assert CarryDivergence().generate(bars, NEGATIVE_CARRY) is None


def test_no_rate_differential_means_no_carry_and_no_trade() -> None:
    """A flat differential has no side. The trend agreeing with nothing is still nothing."""
    bars = daily_series(rising_daily(BARS))
    no_carry = MarketContext(rate_differential=0.0, macro_bias=0.9)

    assert _ema_slope(bars) > 0  # trend would have agreed with a long
    assert CarryDivergence().generate(bars, no_carry) is None


def test_an_opposing_macro_bias_vetoes_an_otherwise_valid_setup() -> None:
    bars = daily_series(rising_daily(BARS))
    opposed = MarketContext(rate_differential=1.5, macro_bias=-0.5)

    assert opposed.rate_differential > 0  # carry says long
    assert _ema_slope(bars) > 0  # slope says long
    assert CarryDivergence().generate(bars, opposed) is None  # macro says no


def test_a_neutral_macro_bias_permits_the_trade() -> None:
    """Deliberate: Phase 5 falls back to neutral, and that must not silence the strategy."""
    bars = daily_series(rising_daily(BARS))
    unknown_macro = MarketContext(rate_differential=1.5, macro_bias=0.0)

    signal = CarryDivergence().generate(bars, unknown_macro)
    assert signal is not None
    assert signal.confidence == pytest.approx(0.5)


def test_confidence_rises_with_macro_conviction() -> None:
    bars = daily_series(rising_daily(BARS))

    weak = CarryDivergence().generate(bars, MarketContext(rate_differential=1.0, macro_bias=0.0))
    strong = CarryDivergence().generate(bars, MarketContext(rate_differential=1.0, macro_bias=1.0))

    assert weak is not None and strong is not None
    assert strong.confidence > weak.confidence
    assert strong.confidence == pytest.approx(1.0)


def test_too_little_history_returns_none() -> None:
    strategy = CarryDivergence()
    short = daily_series(rising_daily(strategy.required_bars - 1))
    assert strategy.generate(short, POSITIVE_CARRY) is None


def test_exactly_enough_history_is_enough() -> None:
    """The boundary is a real boundary: one more bar than the refusal above must work."""
    strategy = CarryDivergence()
    bars = daily_series(rising_daily(strategy.required_bars))
    assert strategy.generate(bars, POSITIVE_CARRY) is not None


def test_an_intraday_series_is_a_wiring_error() -> None:
    hourly = h1_series(list(rising_daily(BARS)), timeframe="H1")
    with pytest.raises(ValueError, match="reads D1 bars"):
        CarryDivergence().generate(hourly, POSITIVE_CARRY)


def test_required_bars_covers_the_ema_seed_plus_a_slope() -> None:
    strategy = CarryDivergence()
    assert strategy.required_bars == 51
    assert strategy.name == "carry_divergence"
