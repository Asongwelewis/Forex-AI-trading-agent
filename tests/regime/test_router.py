"""Routing rules, and proof that each gated strategy stays shut when its gate fails.

`TestDaylightSavingChangesRouting` is the verification that matters: the same UTC clock hour
routes differently in January and July, because the rule is written in London local time.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxagent.regime.router import (
    CARRY_DIVERGENCE,
    RANGE_REVERSION,
    SESSION_BREAKOUT,
    RegimeRouter,
    RouterConfig,
)
from fxagent.regime.sessions import Session
from tests.regime.builders import regime_at

TRENDING = 30.0  # above the 25 threshold
RANGING = 10.0  # below the 20 threshold
NEITHER = 22.0  # in the deliberate gap between them


class TestDaylightSavingChangesRouting:
    def test_same_utc_hour_routes_differently_in_january_and_july(self) -> None:
        """07:00 UTC is before London opens in winter and just after it opens in summer.

        If these two come out equal, the router is reading fixed UTC hours and the breakout
        window is an hour wrong for half the year.
        """
        router = RegimeRouter()
        january = router.weights(
            regime_at(datetime(2026, 1, 15, 7, 0, tzinfo=UTC), trend_strength=TRENDING)
        )
        july = router.weights(
            regime_at(datetime(2026, 7, 15, 7, 0, tzinfo=UTC), trend_strength=TRENDING)
        )

        assert january[SESSION_BREAKOUT] == 0.0, "London has not opened yet in winter"
        assert july[SESSION_BREAKOUT] == 1.0, "London is open at 07:00 UTC in summer"
        assert january != july


class TestSessionBreakoutGate:
    def test_fires_inside_the_london_morning_when_trending(self) -> None:
        weights = RegimeRouter().weights(
            regime_at(datetime(2026, 1, 15, 9, 0, tzinfo=UTC), trend_strength=TRENDING)
        )
        assert weights[SESSION_BREAKOUT] == 1.0

    def test_does_not_fire_when_the_trend_gate_fails(self) -> None:
        """Right session, wrong regime. Required by CLAUDE.md for every gated strategy."""
        for strength in (RANGING, NEITHER, None):
            weights = RegimeRouter().weights(
                regime_at(datetime(2026, 1, 15, 9, 0, tzinfo=UTC), trend_strength=strength)
            )
            assert weights[SESSION_BREAKOUT] == 0.0, f"ADX {strength} is not a trend"

    def test_does_not_fire_outside_the_london_session(self) -> None:
        weights = RegimeRouter().weights(
            regime_at(datetime(2026, 1, 15, 3, 0, tzinfo=UTC), trend_strength=TRENDING)
        )
        assert weights[SESSION_BREAKOUT] == 0.0

    @pytest.mark.parametrize(
        ("utc_hour", "expected"),
        [(8, 1.0), (11, 1.0), (12, 0.0), (13, 0.0)],
    )
    def test_the_local_noon_cutoff_is_exclusive(self, utc_hour: int, expected: float) -> None:
        """In January, London local time equals UTC, so the cutoff is directly observable."""
        weights = RegimeRouter().weights(
            regime_at(datetime(2026, 1, 15, utc_hour, tzinfo=UTC), trend_strength=TRENDING)
        )
        assert weights[SESSION_BREAKOUT] == expected

    def test_the_cutoff_follows_london_local_time_not_utc(self) -> None:
        """11:00 UTC in July is 12:00 in London, so the window has already shut."""
        weights = RegimeRouter().weights(
            regime_at(datetime(2026, 7, 15, 11, 0, tzinfo=UTC), trend_strength=TRENDING)
        )
        assert weights[SESSION_BREAKOUT] == 0.0


class TestRangeReversionGate:
    def test_fires_when_ranging_outside_the_overlap(self) -> None:
        weights = RegimeRouter().weights(
            regime_at(datetime(2026, 1, 15, 9, 0, tzinfo=UTC), trend_strength=RANGING)
        )
        assert weights[RANGE_REVERSION] == 1.0

    def test_does_not_fire_during_the_overlap(self) -> None:
        moment = datetime(2026, 1, 15, 14, 0, tzinfo=UTC)
        regime = regime_at(moment, trend_strength=RANGING)
        assert Session.OVERLAP in regime.sessions, "fixture must actually be in the overlap"
        assert RegimeRouter().weights(regime)[RANGE_REVERSION] == 0.0

    def test_does_not_fire_when_the_range_gate_fails(self) -> None:
        for strength in (TRENDING, NEITHER, None):
            weights = RegimeRouter().weights(
                regime_at(datetime(2026, 1, 15, 9, 0, tzinfo=UTC), trend_strength=strength)
            )
            assert weights[RANGE_REVERSION] == 0.0, f"ADX {strength} is not a range"


class TestCarryDivergence:
    def test_is_always_partially_on(self) -> None:
        router = RegimeRouter()
        for moment in (
            datetime(2026, 1, 15, 3, 0, tzinfo=UTC),
            datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
            datetime(2026, 7, 15, 14, 0, tzinfo=UTC),
        ):
            assert router.weights(regime_at(moment))[CARRY_DIVERGENCE] == 0.5

    def test_is_never_fully_on(self) -> None:
        """A slow strategy must not be able to reach the consensus threshold by itself."""
        assert RouterConfig().carry_weight < 1.0


class TestMarketClosed:
    def test_nothing_is_permitted_at_the_weekend(self) -> None:
        regime = regime_at(datetime(2026, 1, 17, 12, 0, tzinfo=UTC), trend_strength=RANGING)
        assert regime.market_open is False
        assert set(RegimeRouter().weights(regime).values()) == {0.0}

    def test_the_shut_market_rule_can_be_disabled_for_inspection(self) -> None:
        regime = regime_at(datetime(2026, 1, 17, 12, 0, tzinfo=UTC), trend_strength=RANGING)
        router = RegimeRouter(RouterConfig(require_market_open=False))
        assert router.weights(regime)[CARRY_DIVERGENCE] == 0.5


class TestSlateIsTotal:
    def test_every_strategy_is_keyed_even_when_gated(self) -> None:
        """The rejection count in the journal depends on the zeros being present, not absent."""
        weights = RegimeRouter().weights(
            regime_at(datetime(2026, 1, 15, 3, 0, tzinfo=UTC), trend_strength=NEITHER)
        )
        assert set(weights) == {SESSION_BREAKOUT, RANGE_REVERSION, CARRY_DIVERGENCE}
        assert weights[SESSION_BREAKOUT] == 0.0
        assert weights[RANGE_REVERSION] == 0.0


class TestConfigValidation:
    def test_overlap_cannot_anchor_a_local_time_rule(self) -> None:
        with pytest.raises(ValueError, match="no business hours of its own"):
            RouterConfig(breakout_session=Session.OVERLAP)

    def test_weights_must_be_fractions(self) -> None:
        with pytest.raises(ValueError, match=r"must sit in \[0, 1\]"):
            RouterConfig(carry_weight=1.5)
