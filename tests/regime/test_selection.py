"""Selection rules, and the ledger that must survive a rejection.

This file replaces `test_consensus.py`. The old suite tested cross-strategy agreement, which the
2024–25 replay showed to be unsatisfiable: the router weighted two sleeves on zero of 12,341
bars, so a "≥2 agree" rule could never be met and the system took no trades at all. The tests
below assert the property that failure taught — **one weighted sleeve is enough** — and, just as
importantly, that a bar which trades nothing still records why.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxagent.regime.bias import BiasMode, BiasPolicy, DirectionalBias
from fxagent.regime.router import CARRY_DIVERGENCE, RANGE_REVERSION, SESSION_BREAKOUT
from fxagent.regime.selection import SelectionConfig, SleeveSelector
from fxagent.strategies.base import PositioningConfig, SignalDirection
from tests.regime.builders import regime_at, signal

MOMENT = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT
FLAT = SignalDirection.FLAT

POSITIONING_ON = PositioningConfig(enabled=True)

#: A normal London morning: the breakout sleeve is valid, the reversion sleeve is not.
BREAKOUT_SLATE = {SESSION_BREAKOUT: 1.0, RANGE_REVERSION: 0.0, CARRY_DIVERGENCE: 0.0}
#: The mirror image, and the point: these two never open together.
REVERSION_SLATE = {SESSION_BREAKOUT: 0.0, RANGE_REVERSION: 1.0, CARRY_DIVERGENCE: 0.0}


def _regime():  # noqa: ANN202 - test helper
    return regime_at(MOMENT, trend_strength=30.0)


class TestOneSleeveIsEnough:
    def test_a_single_weighted_sleeve_trades(self) -> None:
        """The whole fix. Under the old rule this bar produced nothing, forever."""
        result = SleeveSelector().select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT)},
            BREAKOUT_SLATE,
        )
        assert result.fired
        assert result.signal is not None
        assert result.signal.direction is LONG
        assert result.signal.strategy_names == (SESSION_BREAKOUT,)

    def test_the_other_sleeve_trades_on_its_own_bar(self) -> None:
        result = SleeveSelector().select(
            _regime(),
            {RANGE_REVERSION: signal(RANGE_REVERSION, SHORT, timestamp=MOMENT)},
            REVERSION_SLATE,
        )
        assert result.fired
        assert result.signal is not None
        assert result.signal.strategy_names == (RANGE_REVERSION,)

    def test_mutually_exclusive_gates_no_longer_deadlock(self) -> None:
        """The two intraday sleeves are never weighted together, and that is now fine."""
        selector = SleeveSelector()
        breakout = selector.select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT)},
            BREAKOUT_SLATE,
        )
        reversion = selector.select(
            _regime(),
            {RANGE_REVERSION: signal(RANGE_REVERSION, SHORT, timestamp=MOMENT)},
            REVERSION_SLATE,
        )
        assert breakout.fired and reversion.fired

    def test_the_levels_are_taken_whole_from_the_selected_sleeve(self) -> None:
        """Nothing is blended. A blended stop is a level no strategy argued for."""
        own = signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT)
        result = SleeveSelector().select(_regime(), {SESSION_BREAKOUT: own}, BREAKOUT_SLATE)
        assert result.signal is not None
        assert result.signal.primary.stop_loss == own.stop_loss
        assert result.signal.primary.take_profit == own.take_profit


class TestSelectionRules:
    def test_a_gated_sleeve_cannot_trade(self) -> None:
        """The router still decides. Removing agreement did not remove the gate."""
        result = SleeveSelector().select(
            _regime(),
            {RANGE_REVERSION: signal(RANGE_REVERSION, LONG, timestamp=MOMENT)},
            BREAKOUT_SLATE,
        )
        assert not result.fired
        assert "none was selectable" in result.diagnostics["reason"]

    def test_an_underweight_sleeve_cannot_trade(self) -> None:
        result = SleeveSelector(SelectionConfig(min_weight=0.8)).select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT)},
            {SESSION_BREAKOUT: 0.5, RANGE_REVERSION: 0.0, CARRY_DIVERGENCE: 0.0},
        )
        assert not result.fired
        assert "underweight" in result.diagnostics["reason"]

    def test_a_flat_signal_is_recorded_but_never_traded(self) -> None:
        result = SleeveSelector().select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, FLAT, timestamp=MOMENT)},
            BREAKOUT_SLATE,
        )
        assert not result.fired
        vote = next(v for v in result.diagnostics["votes"] if v["strategy"] == SESSION_BREAKOUT)
        assert vote["participated"]
        assert "flat" in vote["reason"]

    def test_the_heaviest_sleeve_wins_when_two_are_somehow_open(self) -> None:
        """Not reachable with today's gates, but the rule must be defined rather than emergent."""
        result = SleeveSelector().select(
            _regime(),
            {
                SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT),
                RANGE_REVERSION: signal(RANGE_REVERSION, SHORT, timestamp=MOMENT),
            },
            {SESSION_BREAKOUT: 1.0, RANGE_REVERSION: 0.6, CARRY_DIVERGENCE: 0.0},
        )
        assert result.fired
        assert result.signal is not None
        assert result.signal.strategy_names == (SESSION_BREAKOUT,)
        assert result.diagnostics["competing_sleeves"] == [RANGE_REVERSION]
        assert result.diagnostics["competing_directions"] == ["LONG", "SHORT"]

    def test_a_signal_for_the_wrong_symbol_is_a_wiring_error(self) -> None:
        stray = signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT).model_copy(
            update={"symbol": "GBPUSD"}
        )
        with pytest.raises(ValueError, match="wiring error"):
            SleeveSelector().select(_regime(), {SESSION_BREAKOUT: stray}, BREAKOUT_SLATE)


class TestTheDailyBiasFilter:
    def test_an_agreeing_bias_leaves_the_signal_untouched(self) -> None:
        bias = DirectionalBias(direction=LONG, strength=0.8, reason="test")
        result = SleeveSelector().select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT, confidence=0.8)},
            BREAKOUT_SLATE,
            bias=bias,
        )
        assert result.fired
        assert result.signal is not None
        assert result.signal.confidence == pytest.approx(0.8)
        assert not result.diagnostics["bias_filter"]["opposed"]

    def test_an_opposing_bias_downsizes_by_default(self) -> None:
        bias = DirectionalBias(direction=SHORT, strength=0.8, reason="test")
        result = SleeveSelector().select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT, confidence=0.8)},
            BREAKOUT_SLATE,
            bias=bias,
        )
        assert result.fired
        assert result.signal is not None
        assert result.signal.confidence == pytest.approx(0.4)
        assert result.diagnostics["bias_filter"]["opposed"]

    def test_an_opposing_bias_can_suppress_instead(self) -> None:
        bias = DirectionalBias(direction=SHORT, strength=0.8, reason="test")
        result = SleeveSelector(bias_policy=BiasPolicy(mode=BiasMode.SUPPRESS)).select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT)},
            BREAKOUT_SLATE,
            bias=bias,
        )
        assert not result.fired
        assert "suppressed" in result.diagnostics["reason"]

    def test_no_bias_supplied_changes_nothing(self) -> None:
        """Absence of a view must never read as a view."""
        with_none = SleeveSelector().select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT, confidence=0.7)},
            BREAKOUT_SLATE,
        )
        assert with_none.fired
        assert with_none.signal is not None
        assert with_none.signal.confidence == pytest.approx(0.7)

    def test_the_bias_never_originates_a_trade(self) -> None:
        """It filters what a sleeve produced and can never produce one itself."""
        bias = DirectionalBias(direction=LONG, strength=1.0, reason="test")
        result = SleeveSelector().select(_regime(), {}, BREAKOUT_SLATE, bias=bias)
        assert not result.fired


class TestTheLedgerSurvivesRejection:
    def test_every_slate_member_gets_a_line_even_when_nothing_fires(self) -> None:
        """This ledger is what produced the counterfactual. Losing it loses the next diagnosis."""
        result = SleeveSelector().select(_regime(), {}, BREAKOUT_SLATE)
        assert not result.fired
        named = {v["strategy"] for v in result.diagnostics["votes"]}
        assert named == {SESSION_BREAKOUT, RANGE_REVERSION, CARRY_DIVERGENCE}

    def test_a_silent_sleeve_is_distinguished_from_a_gated_one(self) -> None:
        """Silence is about the setup and gating is about the router, so a sleeve with a setup
        the router refused reads as gated while one with nothing to say reads as silent."""
        result = SleeveSelector().select(
            _regime(),
            {RANGE_REVERSION: signal(RANGE_REVERSION, SHORT, timestamp=MOMENT)},
            BREAKOUT_SLATE,
        )
        votes = {v["strategy"]: v["reason"] for v in result.diagnostics["votes"]}
        assert "silent" in votes[SESSION_BREAKOUT]
        assert "gated" in votes[RANGE_REVERSION]

    def test_the_regime_is_recorded_on_the_rejection_path(self) -> None:
        diagnostics = SleeveSelector().select(_regime(), {}, BREAKOUT_SLATE).diagnostics
        assert diagnostics["symbol"] == "EURUSD"
        assert diagnostics["trend_strength"] == pytest.approx(30.0)
        assert diagnostics["is_trending"] in (True, False)
        assert diagnostics["fired"] is False

    def test_the_bias_is_recorded_even_when_it_did_nothing(self) -> None:
        result = SleeveSelector().select(_regime(), {}, BREAKOUT_SLATE)
        assert "bias" in result.diagnostics
        assert result.diagnostics["bias"]["direction"] is None

    def test_the_firing_path_names_the_sleeve_that_was_selected(self) -> None:
        result = SleeveSelector().select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT)},
            BREAKOUT_SLATE,
        )
        assert result.diagnostics["selected_sleeve"] == SESSION_BREAKOUT
        assert "router selected" in result.diagnostics["reason"]


class TestCrowdingStillOnlyGrades:
    def test_crowding_lowers_confidence_but_cannot_veto(self) -> None:
        selector = SleeveSelector(positioning=POSITIONING_ON)
        crowded = selector.select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT, confidence=0.8)},
            BREAKOUT_SLATE,
            positioning_score=1.0,
        )
        assert crowded.fired
        assert crowded.signal is not None
        assert crowded.signal.confidence < 0.8

    def test_crowding_is_off_by_default(self) -> None:
        result = SleeveSelector().select(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT, confidence=0.8)},
            BREAKOUT_SLATE,
            positioning_score=1.0,
        )
        assert result.signal is not None
        assert result.signal.confidence == pytest.approx(0.8)
