"""Agreement rules, and the diagnostics that must survive a rejection.

Half of these tests are about the `None` path. That is deliberate: the rejections are what
answer "how often did consensus reject a signal", so a rejection that returns no explanation
is a defect even though nothing crashed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxagent.regime.consensus import Consensus, ConsensusConfig
from fxagent.regime.router import CARRY_DIVERGENCE, RANGE_REVERSION, SESSION_BREAKOUT
from fxagent.strategies.base import SignalDirection
from tests.regime.builders import regime_at, signal

MOMENT = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT
FLAT = SignalDirection.FLAT

#: A slate where the breakout is fully weighted and carry is half — the normal London morning.
OPEN_SLATE = {SESSION_BREAKOUT: 1.0, RANGE_REVERSION: 0.0, CARRY_DIVERGENCE: 0.5}


def _regime():  # noqa: ANN202 - test helper
    return regime_at(MOMENT, trend_strength=30.0)


class TestFiring:
    def test_two_agreeing_strategies_clearing_the_weight_threshold_fire(self) -> None:
        result = Consensus().evaluate(
            _regime(),
            {
                SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT),
                CARRY_DIVERGENCE: signal(CARRY_DIVERGENCE, LONG, timestamp=MOMENT),
            },
            OPEN_SLATE,
        )
        assert result.fired
        assert result.signal is not None
        assert result.signal.direction is LONG
        assert result.signal.total_weight == pytest.approx(1.5)
        assert set(result.signal.strategy_names) == {SESSION_BREAKOUT, CARRY_DIVERGENCE}

    def test_confidence_is_the_router_weighted_mean(self) -> None:
        result = Consensus().evaluate(
            _regime(),
            {
                SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT, confidence=0.8),
                CARRY_DIVERGENCE: signal(CARRY_DIVERGENCE, LONG, timestamp=MOMENT, confidence=0.4),
            },
            OPEN_SLATE,
        )
        assert result.signal is not None
        assert result.signal.confidence == pytest.approx((1.0 * 0.8 + 0.5 * 0.4) / 1.5)

    def test_primary_is_the_heaviest_contribution(self) -> None:
        """Levels are taken whole from one signal, never averaged across several."""
        result = Consensus().evaluate(
            _regime(),
            {
                SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT),
                CARRY_DIVERGENCE: signal(CARRY_DIVERGENCE, LONG, timestamp=MOMENT),
            },
            OPEN_SLATE,
        )
        assert result.signal is not None
        assert result.signal.primary.strategy_name == SESSION_BREAKOUT


class TestRejection:
    def test_a_one_against_one_disagreement_is_rejected(self) -> None:
        result = Consensus().evaluate(
            _regime(),
            {
                SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT),
                RANGE_REVERSION: signal(RANGE_REVERSION, SHORT, timestamp=MOMENT),
            },
            {SESSION_BREAKOUT: 1.0, RANGE_REVERSION: 1.0, CARRY_DIVERGENCE: 0.5},
        )
        assert not result.fired
        assert result.signal is None
        assert result.diagnostics["long_votes"] == 1
        assert result.diagnostics["short_votes"] == 1

    def test_one_strategy_cannot_trade_alone_however_heavy(self) -> None:
        result = Consensus().evaluate(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT)},
            OPEN_SLATE,
        )
        assert not result.fired
        assert result.diagnostics["long_weight"] == pytest.approx(1.0)
        assert result.diagnostics["long_votes"] == 1

    def test_two_agreeing_strategies_below_the_weight_threshold_do_not_fire(self) -> None:
        light = {SESSION_BREAKOUT: 0.4, RANGE_REVERSION: 0.0, CARRY_DIVERGENCE: 0.5}
        result = Consensus().evaluate(
            _regime(),
            {
                SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT),
                CARRY_DIVERGENCE: signal(CARRY_DIVERGENCE, LONG, timestamp=MOMENT),
            },
            light,
        )
        assert not result.fired
        assert result.diagnostics["long_votes"] == 2
        assert result.diagnostics["long_weight"] == pytest.approx(0.9)

    def test_a_contradicting_slate_is_refused_rather_than_broken_arbitrarily(self) -> None:
        """Four strategies, two clearing each side. Neither wins."""
        slate = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        signals = {
            "a": signal("a", LONG, timestamp=MOMENT),
            "b": signal("b", LONG, timestamp=MOMENT),
            "c": signal("c", SHORT, timestamp=MOMENT),
            "d": signal("d", SHORT, timestamp=MOMENT),
        }
        result = Consensus().evaluate(_regime(), signals, slate)
        assert not result.fired
        assert "self-contradicting" in result.diagnostics["reason"]


class TestVoteClassification:
    def test_silent_gated_and_flat_are_distinguished(self) -> None:
        result = Consensus().evaluate(
            _regime(),
            {
                SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT),
                RANGE_REVERSION: signal(RANGE_REVERSION, LONG, timestamp=MOMENT),
                CARRY_DIVERGENCE: signal(CARRY_DIVERGENCE, FLAT, timestamp=MOMENT),
            },
            {SESSION_BREAKOUT: 1.0, RANGE_REVERSION: 0.0, CARRY_DIVERGENCE: 0.5},
        )
        votes = {vote["strategy"]: vote for vote in result.diagnostics["votes"]}

        assert votes[RANGE_REVERSION]["participated"] is False
        assert "gated" in votes[RANGE_REVERSION]["reason"]
        assert votes[CARRY_DIVERGENCE]["participated"] is True
        assert "flat" in votes[CARRY_DIVERGENCE]["reason"]
        assert votes[SESSION_BREAKOUT]["participated"] is True

    def test_a_missing_strategy_is_recorded_as_silent_not_dropped(self) -> None:
        result = Consensus().evaluate(_regime(), {}, OPEN_SLATE)
        votes = {vote["strategy"]: vote for vote in result.diagnostics["votes"]}
        assert set(votes) == set(OPEN_SLATE)
        assert all("silent" in vote["reason"] for vote in votes.values())

    def test_a_gated_strategy_cannot_contribute_weight(self) -> None:
        result = Consensus().evaluate(
            _regime(),
            {
                SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT),
                RANGE_REVERSION: signal(RANGE_REVERSION, LONG, timestamp=MOMENT),
            },
            {SESSION_BREAKOUT: 1.0, RANGE_REVERSION: 0.0, CARRY_DIVERGENCE: 0.5},
        )
        assert not result.fired, "the gated vote must not complete the pair"
        assert result.diagnostics["long_votes"] == 1

    def test_flat_votes_never_win_a_direction(self) -> None:
        result = Consensus().evaluate(
            _regime(),
            {
                SESSION_BREAKOUT: signal(SESSION_BREAKOUT, FLAT, timestamp=MOMENT),
                CARRY_DIVERGENCE: signal(CARRY_DIVERGENCE, FLAT, timestamp=MOMENT),
            },
            OPEN_SLATE,
        )
        assert not result.fired
        assert result.diagnostics["long_weight"] == 0.0
        assert result.diagnostics["short_weight"] == 0.0
        assert "no strategy offered a tradeable direction" in result.diagnostics["reason"]


class TestDiagnosticsAlwaysPopulated:
    @pytest.mark.parametrize(
        ("signals", "label"),
        [
            ({}, "everything silent"),
            (
                {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT)},
                "a lone vote",
            ),
        ],
    )
    def test_the_none_path_still_explains_itself(self, signals: dict, label: str) -> None:
        result = Consensus().evaluate(_regime(), signals, OPEN_SLATE)
        diagnostics = result.diagnostics

        assert result.signal is None, label
        assert diagnostics["fired"] is False
        assert diagnostics["reason"], "a rejection with no reason is the defect this guards"
        assert diagnostics["winning_direction"] is None
        assert len(diagnostics["votes"]) == len(OPEN_SLATE)
        for key in ("symbol", "timestamp", "session", "long_weight", "short_weight"):
            assert key in diagnostics

    def test_thresholds_are_recorded_so_the_line_explains_itself_later(self) -> None:
        result = Consensus().evaluate(_regime(), {}, OPEN_SLATE)
        assert result.diagnostics["min_agreeing"] == 2
        assert result.diagnostics["min_total_weight"] == 1.0

    def test_diagnostics_are_json_safe(self) -> None:
        """They are written to the journal, so no enum or datetime may leak through."""
        import json

        result = Consensus().evaluate(
            _regime(),
            {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT)},
            OPEN_SLATE,
        )
        json.dumps(result.diagnostics)

    def test_the_regime_context_travels_with_the_rejection(self) -> None:
        result = Consensus().evaluate(_regime(), {}, OPEN_SLATE)
        assert result.diagnostics["session"] == "LONDON"
        assert result.diagnostics["is_trending"] is True


class TestPositioning:
    """Crowding may grade a decision and may not make one.

    Every test here holds the slate fixed and varies only `positioning_score`, so any change in
    whether a trade happens is attributable to positioning alone — which is exactly the change
    that must never appear.
    """

    #: Two agreeing strategies clearing both thresholds. Fires with no positioning input at all.
    AGREED = {
        SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT, confidence=0.8),
        CARRY_DIVERGENCE: signal(CARRY_DIVERGENCE, LONG, timestamp=MOMENT, confidence=0.8),
    }
    #: One strategy short of the count threshold. Rejected with no positioning input at all.
    LONE = {SESSION_BREAKOUT: signal(SESSION_BREAKOUT, LONG, timestamp=MOMENT, confidence=0.8)}

    def test_omitting_positioning_behaves_exactly_as_before(self) -> None:
        """The default must be inert, or every existing caller silently changes behaviour."""
        without = Consensus().evaluate(_regime(), self.AGREED, OPEN_SLATE)
        neutral = Consensus().evaluate(_regime(), self.AGREED, OPEN_SLATE, positioning_score=0.0)

        assert without.signal is not None and neutral.signal is not None
        assert without.signal.confidence == pytest.approx(neutral.signal.confidence)

    def test_crowding_discounts_the_agreed_confidence(self) -> None:
        plain = Consensus().evaluate(_regime(), self.AGREED, OPEN_SLATE)
        crowded = Consensus().evaluate(_regime(), self.AGREED, OPEN_SLATE, positioning_score=1.0)

        assert plain.signal is not None and crowded.signal is not None
        assert crowded.signal.confidence < plain.signal.confidence

    def test_positioning_never_turns_a_rejection_into_a_trade(self) -> None:
        """The failure that would matter. A single strategy must stay a single strategy."""
        for score in (-1.0, -0.5, 0.0, 0.5, 1.0):
            result = Consensus().evaluate(_regime(), self.LONE, OPEN_SLATE, positioning_score=score)
            assert result.fired is False, f"positioning {score} manufactured a trade"

    def test_positioning_never_turns_a_trade_into_a_rejection(self) -> None:
        """The mirror. A veto keyed on crowding would be a trigger with the sign flipped."""
        for score in (-1.0, -0.5, 0.0, 0.5, 1.0):
            result = Consensus().evaluate(
                _regime(), self.AGREED, OPEN_SLATE, positioning_score=score
            )
            assert result.fired is True, f"positioning {score} suppressed a trade"

    def test_positioning_does_not_touch_weight_or_the_vote_count(self) -> None:
        plain = Consensus().evaluate(_regime(), self.AGREED, OPEN_SLATE)
        crowded = Consensus().evaluate(_regime(), self.AGREED, OPEN_SLATE, positioning_score=1.0)

        assert plain.signal is not None and crowded.signal is not None
        assert crowded.signal.total_weight == pytest.approx(plain.signal.total_weight)
        assert crowded.diagnostics["long_weight"] == plain.diagnostics["long_weight"]
        assert crowded.diagnostics["long_votes"] == plain.diagnostics["long_votes"]

    def test_the_discount_is_recorded_in_the_diagnostics(self) -> None:
        result = Consensus().evaluate(_regime(), self.AGREED, OPEN_SLATE, positioning_score=0.8)

        assert result.signal is not None
        assert result.diagnostics["positioning_score"] == pytest.approx(0.8)
        assert result.signal.confidence == pytest.approx(
            result.diagnostics["confidence_before_crowding"] * result.diagnostics["crowding_factor"]
        )

    def test_positioning_is_recorded_on_the_rejection_path_too(self) -> None:
        """The rejections are the product; a rejection that drops an input cannot be replayed."""
        result = Consensus().evaluate(_regime(), self.LONE, OPEN_SLATE, positioning_score=-0.6)

        assert result.fired is False
        assert result.diagnostics["positioning_score"] == pytest.approx(-0.6)


class TestWiringErrors:
    def test_a_signal_for_the_wrong_symbol_is_an_error_not_a_disagreement(self) -> None:
        with pytest.raises(ValueError, match="wiring error"):
            Consensus().evaluate(
                _regime(),
                {
                    SESSION_BREAKOUT: signal(
                        SESSION_BREAKOUT, LONG, timestamp=MOMENT, symbol="GBPUSD"
                    )
                },
                OPEN_SLATE,
            )

    def test_a_threshold_of_one_agreeing_strategy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            ConsensusConfig(min_agreeing=1)
