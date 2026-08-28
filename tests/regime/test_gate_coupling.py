"""`ClassifierConfig` is the single definition of "ranging", for the router and the strategy.

`range_reversion` used to own an `ADX_TREND_CEILING` of 20.0 and an `ADX_PERIOD` of 14, while
`ClassifierConfig` independently defined the same two numbers. They agreed — until somebody
tuned one. Then the router would permit a strategy that immediately refused, or refuse one
that would have traded, and nothing anywhere would have said so.

These tests classify for real rather than injecting a boolean: one `Regime` goes to the
router and to the strategy, and retuning `ranging_below` has to move both. Testing the
coupling, not the value — the assertions never mention 20.0, so they survive any retuning.

The window equivalent of this file is `test_window_agreement.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.adapters.base import BarSeries
from fxagent.regime.classifier import ClassifierConfig, RegimeClassifier
from fxagent.regime.router import RANGE_REVERSION, RegimeRouter
from fxagent.regime.sessions import Session
from fxagent.strategies.base import MarketContext
from fxagent.strategies.range_reversion import RangeReversion
from tests.strategies.builders import bar, flat_run, h1_series

CONTEXT = MarketContext.neutral()
BAND = 0.0010

#: 09:00 UTC on a January Monday: London only, so the overlap gate cannot interfere with
#: what these tests are actually measuring.
LAST_BAR = datetime(2026, 1, 5, 9, tzinfo=UTC)

#: A flat run scores ADX 0; the single spike lifts it to 100/14, and stretches the z-score
#: well past its trigger at the same time. One fixture, both conditions.
#:
#: The close sits well below the high on purpose. `range_reversion` now requires its own
#: rejection wick as a second evidence family, so a bar closing on its high is declined by the
#: strategy however the router is configured — which would break the coupling this file
#: measures without saying anything about the ADX threshold it is actually testing.
SPIKE = {"open_": 1.1000, "high": 1.1060, "low": 1.0999, "close": 1.1040}
SPIKE_ADX = 100.0 / 14.0


def _bars() -> BarSeries:
    """Long enough for the classifier's own warm-up, not just the strategy's."""
    warm = flat_run(end=LAST_BAR - timedelta(hours=1), count=130, band=BAND)
    return h1_series([*warm, bar(LAST_BAR, **SPIKE)])


def _both(ranging_below: float) -> tuple[float, bool]:
    """Router weight and strategy willingness, from one regime under one config."""
    config = ClassifierConfig(ranging_below=ranging_below, trending_above=ranging_below + 5.0)
    bars = _bars()
    regime = RegimeClassifier(config).classify(bars)

    weight = RegimeRouter().weights(regime)[RANGE_REVERSION]
    spoke = RangeReversion().generate(bars, CONTEXT, regime) is not None
    return weight, spoke


def test_the_fixture_reads_the_adx_the_sweep_assumes() -> None:
    """Guards the sweep below: thresholds are placed relative to this reading."""
    regime = RegimeClassifier().classify(_bars())
    assert regime.trend_strength == pytest.approx(SPIKE_ADX)
    assert regime.sessions == (Session.LONDON,), "the overlap gate must not be involved"


def test_a_threshold_above_the_reading_permits_and_the_strategy_speaks() -> None:
    weight, spoke = _both(ranging_below=SPIKE_ADX + 5.0)
    assert weight > 0.0
    assert spoke is True


def test_a_threshold_below_the_reading_blocks_and_the_strategy_is_silent() -> None:
    weight, spoke = _both(ranging_below=SPIKE_ADX - 5.0)
    assert weight == 0.0
    assert spoke is False


@pytest.mark.parametrize(
    "ranging_below",
    [SPIKE_ADX - 5.0, SPIKE_ADX - 0.5, SPIKE_ADX + 0.5, SPIKE_ADX + 5.0, 20.0, 2.0],
)
def test_router_permission_and_strategy_behaviour_move_together(ranging_below: float) -> None:
    """The property: whatever the threshold, the two sides never disagree.

    Swept either side of the reading, including the old hardcoded 20.0, so a regression that
    reintroduced a local constant would show up as a disagreement at some threshold rather
    than at one specific one.
    """
    weight, spoke = _both(ranging_below)
    assert (weight > 0.0) is spoke, (
        f"with ranging_below={ranging_below:.2f} the router "
        f"{'permits' if weight else 'blocks'} range_reversion but the strategy "
        f"{'speaks' if spoke else 'is silent'}"
    )


def test_the_sweep_exercises_both_outcomes() -> None:
    """Both sides agreeing on "no" everywhere would satisfy the sweep trivially."""
    outcomes = {_both(threshold)[1] for threshold in (SPIKE_ADX - 5.0, SPIKE_ADX + 5.0)}
    assert outcomes == {True, False}


def test_the_adx_period_also_lives_in_one_place() -> None:
    """Retuning the period changes the reading, and both sides follow it in step."""
    bars = _bars()
    slow_regime = RegimeClassifier(ClassifierConfig(adx_period=28)).classify(bars)
    default_regime = RegimeClassifier().classify(bars)

    assert slow_regime.trend_strength != default_regime.trend_strength, (
        "the period must actually affect the reading, or this proves nothing"
    )

    weight = RegimeRouter().weights(slow_regime)[RANGE_REVERSION]
    spoke = RangeReversion().generate(bars, CONTEXT, slow_regime) is not None
    assert (weight > 0.0) is spoke
