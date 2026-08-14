"""Range reversion: fade the stretch, but only when the *regime* says the market is ranging.

The negative tests carry the weight. Each one proves *which* gate refused: the regime test
asserts the z-score was past its trigger anyway, and the z-score test hands in a regime that
is explicitly ranging. Without that, a fixture that simply never set up would pass as a
working gate.

The range gate itself is no longer measured here — it arrives on the `Regime`. These tests
therefore inject the regime directly rather than classifying the fixtures, because they are
about this strategy's arithmetic. That the injected boolean is the same one the router reads
is proved in `tests/regime/test_gate_coupling.py`, which classifies for real.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from fxagent.adapters.base import BarSeries
from fxagent.indicators import rolling_zscore
from fxagent.regime.classifier import Regime
from fxagent.strategies import MarketContext, SignalDirection, bars_to_frame
from fxagent.strategies.range_reversion import RangeReversion
from tests.regime.builders import regime_at
from tests.strategies.builders import bar, flat_run, h1_series

CONTEXT = MarketContext.neutral()
FLAT_END = datetime(2026, 1, 5, 12, tzinfo=UTC)
BAND = 0.0010
BASELINE_TR = 2 * BAND
SPIKE_TR = 0.0061

#: Wilder's step from a settled ATR of 0.0020 onto the single 0.0061 spike bar.
EXPECTED_ATR = (BASELINE_TR * 13 + SPIKE_TR) / 14
#: 19 identical closes plus one outlier always gives this, whatever the outlier's size.
EXPECTED_ABS_Z = 0.95 / math.sqrt(0.05)

#: ADX values either side of the classifier's default 20.0 boundary.
RANGING = 10.0
TRENDING = 30.0

SPIKE_UP = {"open_": 1.1000, "high": 1.1060, "low": 1.0999, "close": 1.1050}
SPIKE_DOWN = {"open_": 1.1000, "high": 1.1001, "low": 1.0940, "close": 1.0950}


def _spiked(shape: dict[str, float], *, count: int = 40) -> BarSeries:
    """A long flat run — zero directional movement — with one violent bar on the end."""
    bars = flat_run(end=FLAT_END, count=count, band=BAND)
    return h1_series([*bars, bar(FLAT_END + timedelta(hours=1), **shape)])


def _regime(bars: BarSeries, *, trend_strength: float = RANGING) -> Regime:
    """A regime describing exactly the last bar of `bars`, as the strategy demands."""
    return regime_at(bars.bars[-1].timestamp, trend_strength=trend_strength, symbol=bars.symbol)


def _oscillating(count: int = 40) -> BarSeries:
    """Closes alternating by a hair, so no z-score ever reaches the trigger."""
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


def _zscore(bars: BarSeries) -> float:
    return float(rolling_zscore(bars_to_frame(bars)["close"], 20).iloc[-1])


# --- it fires ------------------------------------------------------------------


def test_a_stretch_above_the_mean_is_faded_short() -> None:
    bars = _spiked(SPIKE_UP)
    signal = RangeReversion().generate(bars, CONTEXT, _regime(bars))

    assert signal is not None
    assert signal.direction is SignalDirection.SHORT
    assert signal.entry_price == pytest.approx(1.1050)
    assert signal.reasoning["zscore"] == pytest.approx(EXPECTED_ABS_Z)
    assert signal.reasoning["atr"] == pytest.approx(EXPECTED_ATR)


def test_the_logged_adx_is_the_regimes_not_a_locally_recomputed_one() -> None:
    """The journal must show the number the gate was actually taken on."""
    bars = _spiked(SPIKE_UP)
    signal = RangeReversion().generate(bars, CONTEXT, _regime(bars, trend_strength=12.5))

    assert signal is not None
    assert signal.reasoning["adx"] == pytest.approx(12.5)


def test_a_stretch_below_the_mean_is_faded_long() -> None:
    bars = _spiked(SPIKE_DOWN)
    signal = RangeReversion().generate(bars, CONTEXT, _regime(bars))

    assert signal is not None
    assert signal.direction is SignalDirection.LONG
    assert signal.entry_price == pytest.approx(1.0950)
    assert signal.reasoning["zscore"] == pytest.approx(-EXPECTED_ABS_Z)


def test_the_short_stop_sits_an_atr_multiple_beyond_the_bars_high() -> None:
    """Beyond the extreme, not beyond the entry — if this is the turn, the high holds."""
    bars = _spiked(SPIKE_UP)
    signal = RangeReversion().generate(bars, CONTEXT, _regime(bars))

    assert signal is not None
    assert signal.stop_loss == pytest.approx(1.1060 + 1.5 * EXPECTED_ATR)
    assert signal.stop_loss > 1.1060  # strictly beyond the high the bar actually printed


def test_the_long_stop_sits_an_atr_multiple_beyond_the_bars_low() -> None:
    bars = _spiked(SPIKE_DOWN)
    signal = RangeReversion().generate(bars, CONTEXT, _regime(bars))

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
    bars = _spiked(shape)
    signal = RangeReversion().generate(bars, CONTEXT, _regime(bars))

    assert signal is not None
    if direction is SignalDirection.SHORT:
        assert signal.take_profit < signal.entry_price < signal.stop_loss
    else:
        assert signal.stop_loss < signal.entry_price < signal.take_profit


def test_the_target_is_the_rolling_mean_it_is_reverting_to() -> None:
    """19 closes at 1.1000 and one at 1.1050 average to exactly 1.10025."""
    bars = _spiked(SPIKE_UP)
    signal = RangeReversion().generate(bars, CONTEXT, _regime(bars))

    assert signal is not None
    assert signal.take_profit == pytest.approx(1.10025)
    assert signal.reasoning["rolling_mean"] == pytest.approx(1.10025)


# --- it refuses ----------------------------------------------------------------


def test_a_trending_regime_blocks_the_fade_even_when_the_stretch_qualifies() -> None:
    """The gate that defines the strategy — now asked of the regime, not of a local ADX."""
    bars = _spiked(SPIKE_UP)

    # Prove the setup is otherwise present, so the regime gate is demonstrably what refused.
    assert _zscore(bars) > 2.0
    assert RangeReversion().generate(bars, CONTEXT, _regime(bars)) is not None

    trending = _regime(bars, trend_strength=TRENDING)
    assert trending.is_ranging is False
    assert RangeReversion().generate(bars, CONTEXT, trending) is None


def test_a_regime_that_is_neither_ranging_nor_trending_also_blocks() -> None:
    """ADX in the gap between the thresholds is not a range, so there is nothing to fade."""
    bars = _spiked(SPIKE_UP)
    neither = _regime(bars, trend_strength=22.0)

    assert neither.is_ranging is False
    assert neither.is_trending is False
    assert RangeReversion().generate(bars, CONTEXT, neither) is None


def test_an_unknown_regime_blocks_rather_than_guessing() -> None:
    """During classifier warm-up `is_ranging` is False, so the gate stays shut."""
    bars = _spiked(SPIKE_UP)
    warming_up = _regime(bars, trend_strength=None)

    assert warming_up.trend_strength is None
    assert warming_up.is_ranging is False
    assert RangeReversion().generate(bars, CONTEXT, warming_up) is None


def test_a_calm_market_without_a_stretch_produces_nothing() -> None:
    """The regime is explicitly ranging here, so the z-score trigger is what refused."""
    calm = _oscillating()
    ranging = _regime(calm)

    assert ranging.is_ranging is True
    assert abs(_zscore(calm)) <= 2.0
    assert RangeReversion().generate(calm, CONTEXT, ranging) is None


def test_a_perfectly_flat_market_has_no_z_score_and_no_signal() -> None:
    """Zero variance makes the z-score undefined; NaN must not slip through as a trigger."""
    flat = h1_series(flat_run(end=FLAT_END, count=40, band=BAND))

    assert math.isnan(_zscore(flat))
    assert RangeReversion().generate(flat, CONTEXT, _regime(flat)) is None


def test_too_little_history_returns_none() -> None:
    strategy = RangeReversion()
    short = _spiked(SPIKE_UP, count=strategy.required_bars - 2)
    assert strategy.generate(short, CONTEXT, _regime(short)) is None


# --- the regime is required, and must describe this bar ------------------------


def test_a_missing_regime_is_a_loud_wiring_error_not_a_quiet_none() -> None:
    """Returning None here would hide a broken caller as an absent setup."""
    bars = _spiked(SPIKE_UP)
    with pytest.raises(ValueError, match="gates on the regime and was given none"):
        RangeReversion().generate(bars, CONTEXT)


def test_a_regime_for_another_symbol_is_refused() -> None:
    bars = _spiked(SPIKE_UP)
    wrong = regime_at(bars.bars[-1].timestamp, trend_strength=RANGING, symbol="GBPUSD")
    with pytest.raises(ValueError, match="regime for 'GBPUSD'"):
        RangeReversion().generate(bars, CONTEXT, wrong)


def test_a_regime_from_a_different_bar_is_refused() -> None:
    """A stale regime would gate this bar's trade on the previous bar's market."""
    bars = _spiked(SPIKE_UP)
    stale = regime_at(
        bars.bars[-1].timestamp - timedelta(hours=1), trend_strength=RANGING, symbol=bars.symbol
    )
    with pytest.raises(ValueError, match="different bar than the trade"):
        RangeReversion().generate(bars, CONTEXT, stale)


# --- it owns no definition of "ranging" ----------------------------------------


def test_the_strategy_defines_no_adx_constant_of_its_own() -> None:
    """The regression guard: `ClassifierConfig` is the only place "ranging" is defined."""
    import fxagent.strategies.range_reversion as module

    offenders = [name for name in vars(module) if "ADX" in name.upper()]
    assert not offenders, f"range_reversion re-defines {offenders}; the classifier owns these"


def test_the_strategy_does_not_import_the_adx_indicator() -> None:
    """Importing it would be the first step back to a second definition of the gate."""
    import fxagent.strategies.range_reversion as module

    assert "adx" not in vars(module)


def test_required_bars_covers_only_what_this_file_measures() -> None:
    """No longer driven by an ADX warm-up, because there is no longer an ADX here."""
    strategy = RangeReversion()
    assert strategy.required_bars == 20
    assert strategy.name == "range_reversion"
