"""Range reversion: fade the stretch, but only when ADX says the market is ranging.

The negative tests carry the weight. Each one proves *which* gate refused: the ADX test
asserts the z-score was past its trigger anyway, and the z-score test asserts ADX was
comfortably inside its ceiling. Without that, a fixture that simply never set up would
pass as a working gate.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from fxagent.indicators import adx, rolling_zscore
from fxagent.strategies import MarketContext, SignalDirection, bars_to_frame
from fxagent.strategies.range_reversion import RangeReversion
from tests.strategies.builders import bar, flat_run, h1_series

CONTEXT = MarketContext.neutral()
FLAT_END = datetime(2026, 1, 5, 12, tzinfo=UTC)
BAND = 0.0010
BASELINE_TR = 2 * BAND
SPIKE_TR = 0.0061

#: Wilder's step from a settled ATR of 0.0020 onto the single 0.0061 spike bar.
EXPECTED_ATR = (BASELINE_TR * 13 + SPIKE_TR) / 14
#: The flat run produces no directional movement at all, so ADX sits at 0 until the spike,
#: whose DX of 100 enters the average once: (0 * 13 + 100) / 14.
EXPECTED_ADX = 100.0 / 14.0
#: 19 identical closes plus one outlier always gives this, whatever the outlier's size.
EXPECTED_ABS_Z = 0.95 / math.sqrt(0.05)

SPIKE_UP = {"open_": 1.1000, "high": 1.1060, "low": 1.0999, "close": 1.1050}
SPIKE_DOWN = {"open_": 1.1000, "high": 1.1001, "low": 1.0940, "close": 1.0950}


def _spiked(shape: dict[str, float], *, count: int = 40):
    """A long flat run — zero directional movement — with one violent bar on the end."""
    bars = flat_run(end=FLAT_END, count=count, band=BAND)
    return h1_series([*bars, bar(FLAT_END + timedelta(hours=1), **shape)])


def _h1_ramp(count: int, *, step: float = 0.0010, final_jump: float = 0.0):
    """A staircase trend: every bar a higher high and a higher low, so ADX saturates."""
    start = FLAT_END - timedelta(hours=count - 1)
    bars = []
    price = 1.1000
    for index in range(count):
        price = 1.1000 + step * index
        if index == count - 1:
            price += final_jump
        bars.append(
            bar(
                start + timedelta(hours=index),
                open_=price,
                high=price + 0.0005,
                low=price - 0.0005,
                close=price,
            )
        )
    return h1_series(bars)


def _oscillating(count: int = 40):
    """Closes alternating by a hair: ADX stays low but no z-score ever reaches the trigger."""
    start = FLAT_END - timedelta(hours=count - 1)
    return h1_series(
        [
            bar(
                start + timedelta(hours=index),
                open_=1.1000 + 0.0002 * (index % 2),
                high=1.1000 + 0.0002 * (index % 2) + 0.0010,
                low=1.1000 + 0.0002 * (index % 2) - 0.0010,
                close=1.1000 + 0.0002 * (index % 2),
            )
            for index in range(count)
        ]
    )


def _indicator(bars, name: str) -> float:
    frame = bars_to_frame(bars)
    if name == "adx":
        return float(adx(frame["high"], frame["low"], frame["close"], 14).iloc[-1])
    return float(rolling_zscore(frame["close"], 20).iloc[-1])


# --- it fires ------------------------------------------------------------------


def test_a_stretch_above_the_mean_is_faded_short() -> None:
    signal = RangeReversion().generate(_spiked(SPIKE_UP), CONTEXT)

    assert signal is not None
    assert signal.direction is SignalDirection.SHORT
    assert signal.entry_price == pytest.approx(1.1050)
    assert signal.reasoning["zscore"] == pytest.approx(EXPECTED_ABS_Z)
    assert signal.reasoning["adx"] == pytest.approx(EXPECTED_ADX)
    assert signal.reasoning["atr"] == pytest.approx(EXPECTED_ATR)


def test_a_stretch_below_the_mean_is_faded_long() -> None:
    signal = RangeReversion().generate(_spiked(SPIKE_DOWN), CONTEXT)

    assert signal is not None
    assert signal.direction is SignalDirection.LONG
    assert signal.entry_price == pytest.approx(1.0950)
    assert signal.reasoning["zscore"] == pytest.approx(-EXPECTED_ABS_Z)


def test_the_short_stop_sits_an_atr_multiple_beyond_the_bars_high() -> None:
    """Beyond the extreme, not beyond the entry — if this is the turn, the high holds."""
    signal = RangeReversion().generate(_spiked(SPIKE_UP), CONTEXT)

    assert signal is not None
    assert signal.stop_loss == pytest.approx(1.1060 + 1.5 * EXPECTED_ATR)
    assert signal.stop_loss > 1.1060  # strictly beyond the high the bar actually printed


def test_the_long_stop_sits_an_atr_multiple_beyond_the_bars_low() -> None:
    signal = RangeReversion().generate(_spiked(SPIKE_DOWN), CONTEXT)

    assert signal is not None
    assert signal.stop_loss == pytest.approx(1.0940 - 1.5 * EXPECTED_ATR)
    assert signal.stop_loss < 1.0940


@pytest.mark.parametrize(
    ("shape", "direction"),
    [(SPIKE_UP, SignalDirection.SHORT), (SPIKE_DOWN, SignalDirection.LONG)],
    ids=["short", "long"],
)
def test_stop_is_always_on_the_losing_side_of_entry(
    shape: dict[str, float], direction: SignalDirection
) -> None:
    signal = RangeReversion().generate(_spiked(shape), CONTEXT)

    assert signal is not None
    if direction is SignalDirection.SHORT:
        assert signal.take_profit < signal.entry_price < signal.stop_loss
    else:
        assert signal.stop_loss < signal.entry_price < signal.take_profit


def test_the_target_is_the_rolling_mean_it_is_reverting_to() -> None:
    """19 closes at 1.1000 and one at 1.1050 average to exactly 1.10025."""
    signal = RangeReversion().generate(_spiked(SPIKE_UP), CONTEXT)

    assert signal is not None
    assert signal.take_profit == pytest.approx(1.10025)
    assert signal.reasoning["rolling_mean"] == pytest.approx(1.10025)


# --- it refuses ----------------------------------------------------------------


def test_a_trending_market_blocks_the_fade_even_when_the_stretch_qualifies() -> None:
    """The gate that defines the strategy: same stretch, but ADX says this is a trend."""
    trending = _h1_ramp(40, final_jump=0.0100)

    # Prove the setup is otherwise present, so the ADX gate is demonstrably what refused.
    assert _indicator(trending, "zscore") > 2.0
    assert _indicator(trending, "adx") >= 20.0

    assert RangeReversion().generate(trending, CONTEXT) is None


def test_adx_exactly_at_the_ceiling_is_treated_as_trending() -> None:
    """The rule is ADX < 20, so 20 itself blocks. A boundary this cheap should be pinned."""
    from fxagent.strategies.range_reversion import ADX_TREND_CEILING

    assert ADX_TREND_CEILING == 20.0
    trending = _h1_ramp(40, final_jump=0.0100)
    assert _indicator(trending, "adx") >= ADX_TREND_CEILING
    assert RangeReversion().generate(trending, CONTEXT) is None


def test_a_calm_market_without_a_stretch_produces_nothing() -> None:
    """ADX is well inside its ceiling here, so the z-score trigger is what refused."""
    calm = _oscillating()

    assert _indicator(calm, "adx") < 20.0
    assert abs(_indicator(calm, "zscore")) <= 2.0

    assert RangeReversion().generate(calm, CONTEXT) is None


def test_a_perfectly_flat_market_has_no_z_score_and_no_signal() -> None:
    """Zero variance makes the z-score undefined; NaN must not slip through as a trigger."""
    flat = h1_series(flat_run(end=FLAT_END, count=40, band=BAND))

    assert math.isnan(_indicator(flat, "zscore"))
    assert RangeReversion().generate(flat, CONTEXT) is None


def test_too_little_history_returns_none() -> None:
    strategy = RangeReversion()
    short = _spiked(SPIKE_UP, count=strategy.required_bars - 2)
    assert strategy.generate(short, CONTEXT) is None


def test_required_bars_is_driven_by_the_adx_warm_up() -> None:
    """ADX(14) has no value at all until bar 2*14-1, which outlasts every other input."""
    strategy = RangeReversion()
    assert strategy.required_bars == 28
    assert strategy.name == "range_reversion"
