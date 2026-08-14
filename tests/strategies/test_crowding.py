"""Crowding grades a decision, never makes one, and by default does not even grade.

`crowding_confidence_factor` is read in exactly two places — `carry_divergence` and
`regime.consensus` — and both are covered here rather than split across two files, because the
property being defended is the same one in both and it is a property about what crowding
*cannot* do. A test that only checked the confidence arithmetic would pass just as happily on an
implementation that had quietly acquired a veto.

**Every test that exercises the arithmetic passes `ON` explicitly.** The shipped default is
off, so a test relying on the default would be asserting the arithmetic of a code path nothing
currently runs — and would keep passing if the flag were later wired backwards. The
`TestDisabledByDefault` block below is the one that pins the shipped behaviour, and it asserts
byte-for-byte equality with the pre-crowding confidence rather than merely "smaller effect".
"""

from __future__ import annotations

import pytest

from fxagent.strategies import MarketContext, SignalDirection
from fxagent.strategies.base import (
    MAX_CROWDING_PENALTY,
    POSITIONING_OFF,
    PositioningConfig,
    crowding_confidence_factor,
)
from fxagent.strategies.carry_divergence import CarryDivergence
from tests.strategies.builders import daily_series, rising_daily

BARS = 60
LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT
FLAT = SignalDirection.FLAT

#: Opt in explicitly. Never the default, so the arithmetic tests cannot pass by accident.
ON = PositioningConfig(enabled=True)


# -- the rule ------------------------------------------------------------------


def test_a_neutral_reading_leaves_confidence_untouched() -> None:
    assert crowding_confidence_factor(LONG, 0.0, config=ON) == pytest.approx(1.0)
    assert crowding_confidence_factor(SHORT, 0.0, config=ON) == pytest.approx(1.0)


def test_buying_a_crowded_long_costs_confidence() -> None:
    assert crowding_confidence_factor(LONG, 1.0, config=ON) == pytest.approx(
        1.0 - MAX_CROWDING_PENALTY
    )
    assert crowding_confidence_factor(LONG, 0.5, config=ON) == pytest.approx(
        1.0 - MAX_CROWDING_PENALTY / 2
    )


def test_selling_a_crowded_short_costs_confidence() -> None:
    """The mirror image. A sign error here would reward exactly the trades it should discount."""
    assert crowding_confidence_factor(SHORT, -1.0, config=ON) == pytest.approx(
        1.0 - MAX_CROWDING_PENALTY
    )


def test_trading_against_the_crowd_is_not_a_bonus() -> None:
    """Scales down, never up — the same asymmetry hard rule 9 imposes on sizing.

    An uncrowded trade is the normal case, not an edge, and a multiplier that can exceed 1.0 is
    one that eventually gets pointed at position size.
    """
    assert crowding_confidence_factor(LONG, -1.0, config=ON) == pytest.approx(1.0)
    assert crowding_confidence_factor(SHORT, 1.0, config=ON) == pytest.approx(1.0)


def test_the_factor_can_never_reach_zero() -> None:
    """The arithmetic guarantee behind "confirmation, not a gate".

    If the penalty could reach 1.0, crowding would zero a signal's confidence and suppress the
    trade downstream — a veto smuggled in through multiplication.
    """
    for direction in (LONG, SHORT):
        for score in (-1.0, -0.5, 0.0, 0.5, 1.0):
            assert (
                crowding_confidence_factor(direction, score, config=ON)
                >= 1.0 - MAX_CROWDING_PENALTY
            )
            assert crowding_confidence_factor(direction, score, config=ON) > 0.0


def test_flat_has_no_exposure_to_be_crowded_into() -> None:
    assert crowding_confidence_factor(FLAT, 1.0, config=ON) == pytest.approx(1.0)


def test_a_score_outside_the_unit_interval_is_refused() -> None:
    """Out of range means an upstream bug, and clamping it silently would hide that bug."""
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        crowding_confidence_factor(LONG, 1.5, config=ON)


def test_market_context_rejects_a_score_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        MarketContext(positioning_score=-2.0)


def test_neutral_context_carries_neutral_positioning() -> None:
    assert MarketContext.neutral().positioning_score == 0.0


# -- carry_divergence: confirmation only, once switched on ---------------------


def _long_context(positioning: float) -> MarketContext:
    """Positive carry and a supportive macro bias, so the setup fires and only crowding varies."""
    return MarketContext(rate_differential=1.5, macro_bias=0.4, positioning_score=positioning)


def test_crowding_lowers_carry_confidence_without_changing_the_trade() -> None:
    bars = daily_series(rising_daily(BARS))

    uncrowded = CarryDivergence(positioning=ON).generate(bars, _long_context(0.0))
    crowded = CarryDivergence(positioning=ON).generate(bars, _long_context(1.0))

    assert uncrowded is not None and crowded is not None
    assert crowded.confidence < uncrowded.confidence
    # Everything that constitutes the trade is untouched.
    assert crowded.direction is uncrowded.direction
    assert crowded.entry_price == pytest.approx(uncrowded.entry_price)
    assert crowded.stop_loss == pytest.approx(uncrowded.stop_loss)
    assert crowded.take_profit == pytest.approx(uncrowded.take_profit)


def test_maximum_crowding_cannot_suppress_a_carry_signal() -> None:
    """The load-bearing test. Positioning is confirmation, so it must not silence the strategy."""
    bars = daily_series(rising_daily(BARS))

    signal = CarryDivergence(positioning=ON).generate(bars, _long_context(1.0))

    assert signal is not None
    assert signal.direction is SignalDirection.LONG
    assert signal.confidence > 0.0


def test_crowding_cannot_create_a_carry_signal_where_carry_says_nothing() -> None:
    """No differential means no trade, at any positioning extreme. Confirmation is not a trigger."""
    bars = daily_series(rising_daily(BARS))

    for score in (-1.0, 0.0, 1.0):
        flat_carry = MarketContext(rate_differential=0.0, macro_bias=0.4, positioning_score=score)
        assert CarryDivergence(positioning=ON).generate(bars, flat_carry) is None


def test_crowding_cannot_flip_the_side_carry_chose() -> None:
    """An extreme crowded long is not a reason to sell. That would be a contrarian trigger."""
    bars = daily_series(rising_daily(BARS))

    signal = CarryDivergence(positioning=ON).generate(bars, _long_context(1.0))

    assert signal is not None
    assert signal.direction is SignalDirection.LONG


def test_the_carry_signal_records_what_crowding_did_to_it() -> None:
    """Every disagreement is training data, so the discount has to be auditable after the fact."""
    bars = daily_series(rising_daily(BARS))

    signal = CarryDivergence(positioning=ON).generate(bars, _long_context(0.8))

    assert signal is not None
    assert signal.reasoning["positioning_score"] == pytest.approx(0.8)
    assert signal.reasoning["crowding_factor"] == pytest.approx(1.0 - MAX_CROWDING_PENALTY * 0.8)
    assert signal.confidence == pytest.approx(
        signal.reasoning["confidence_before_crowding"] * signal.reasoning["crowding_factor"]
    )


# -- off by default ------------------------------------------------------------


class TestDisabledByDefault:
    """The shipped behaviour: crowding is computed, recorded, and applied to nothing.

    Phase 8 has not produced a baseline, and a modifier that was already live during the first
    backtest cannot be measured afterwards — there is nothing clean to difference it against.
    So these tests assert *equality* with the pre-crowding confidence, not merely a smaller
    effect. An implementation that applied a tiny penalty would satisfy "roughly unchanged" and
    would still contaminate the baseline.
    """

    def test_the_shipped_default_is_off(self) -> None:
        assert PositioningConfig().enabled is False
        assert POSITIONING_OFF.enabled is False

    def test_the_factor_is_exactly_one_at_every_extreme_when_disabled(self) -> None:
        for direction in (LONG, SHORT, FLAT):
            for score in (-1.0, -0.5, 0.0, 0.5, 1.0):
                assert crowding_confidence_factor(direction, score) == 1.0

    def test_carry_confidence_is_identical_at_every_positioning_extreme(self) -> None:
        """The baseline guarantee. Crowding must not move this number by so much as a float ulp."""
        bars = daily_series(rising_daily(BARS))

        baseline = CarryDivergence().generate(bars, _long_context(0.0))
        assert baseline is not None

        for score in (-1.0, -0.5, 0.5, 1.0):
            signal = CarryDivergence().generate(bars, _long_context(score))
            assert signal is not None
            assert signal.confidence == baseline.confidence, f"positioning {score} moved it"

    def test_the_disabled_factor_matches_the_pre_crowding_formula(self) -> None:
        """Pinned against the literal expression this strategy used before crowding existed."""
        bars = daily_series(rising_daily(BARS))
        macro_bias = 0.4

        signal = CarryDivergence().generate(bars, _long_context(1.0))

        assert signal is not None
        assert signal.confidence == pytest.approx(min(1.0, 0.5 + 0.5 * abs(macro_bias)))

    def test_the_score_is_still_recorded_while_the_effect_is_off(self) -> None:
        """Disabled means inert, not absent — the journal has to accumulate the input.

        This is what makes turning it on later a measured delta computed from stored rows
        rather than a second backtest run against a moved target.
        """
        bars = daily_series(rising_daily(BARS))

        signal = CarryDivergence().generate(bars, _long_context(0.8))

        assert signal is not None
        assert signal.reasoning["positioning_score"] == pytest.approx(0.8)
        assert signal.reasoning["positioning_enabled"] is False
        assert signal.reasoning["crowding_factor"] == 1.0

    def test_an_out_of_range_score_still_raises_when_disabled(self) -> None:
        """A flag that also suppressed validation would hide the bug until it was flipped."""
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            crowding_confidence_factor(LONG, 1.5)

    def test_a_penalty_that_could_zero_a_confidence_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            PositioningConfig(enabled=True, max_penalty=1.0)

    def test_from_env_defaults_to_off_and_needs_an_explicit_truthy_value(self) -> None:
        assert PositioningConfig.from_env({}).enabled is False
        assert PositioningConfig.from_env({"FX_POSITIONING_ENABLED": "1"}).enabled is True
        assert PositioningConfig.from_env({"FX_POSITIONING_ENABLED": "true"}).enabled is True
        # A typo in a deployment variable must not switch a modifier into the decision path.
        assert PositioningConfig.from_env({"FX_POSITIONING_ENABLED": "ture"}).enabled is False
        assert PositioningConfig.from_env({"FX_POSITIONING_ENABLED": "0"}).enabled is False
