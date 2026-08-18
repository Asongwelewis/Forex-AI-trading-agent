"""Tail risk against hand-computed quantiles, drawdown against a curve drawn by hand.

The VaR reference values use 101 observations at `alpha=0.05` deliberately: `0.05 * (n - 1)` is
then exactly 5, so the linear-interpolation quantile lands on an order statistic and the
expected numbers can be read off the sample rather than reproduced from the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from fxagent.stats.risk_metrics import (
    TailRisk,
    expected_shortfall,
    max_drawdown,
    probability_of_ruin,
    value_at_risk,
)

#: -50 .. 50 inclusive. The 5% quantile is exactly -45 and the tail below it is -50 .. -45.
LADDER = np.arange(-50.0, 51.0)


class TestTailRisk:
    def test_var_against_a_hand_read_quantile(self) -> None:
        tail = value_at_risk(LADDER, alpha=0.05)
        assert tail.value_at_risk == pytest.approx(45.0)

    def test_expected_shortfall_is_the_mean_of_the_tail(self) -> None:
        """-50 through -45 is six observations averaging -47.5."""
        assert expected_shortfall(LADDER, alpha=0.05) == pytest.approx(47.5)

    def test_var_cannot_be_obtained_without_es(self) -> None:
        """The whole point: there is no call signature that yields VaR on its own."""
        result = value_at_risk(LADDER, alpha=0.05)
        assert isinstance(result, TailRisk)
        assert result.expected_shortfall == pytest.approx(47.5)
        assert "ES" in result.describe()

    def test_es_is_never_less_than_var(self) -> None:
        """A mean of the tail cannot be shallower than the threshold that defines the tail."""
        rng = np.random.default_rng(11)
        for _ in range(50):
            sample = rng.standard_normal(400)
            tail = value_at_risk(sample, alpha=0.05)
            assert tail.expected_shortfall >= tail.value_at_risk - 1e-12

    def test_var_is_signed_so_a_profitable_tail_reads_negative(self) -> None:
        """20 trades, 19 of them winners: the 5% quantile is a gain, and that is worth seeing."""
        sample = [-5.0] + [1.0] * 19
        tail = value_at_risk(sample, alpha=0.05)
        assert tail.value_at_risk < 0
        assert tail.expected_shortfall == pytest.approx(5.0)

    def test_a_tail_with_no_observations_in_it_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 20 observations"):
            value_at_risk(np.arange(10.0), alpha=0.05)

    def test_an_alpha_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha must be in"):
            expected_shortfall(LADDER, alpha=1.5)


class TestMaxDrawdown:
    #: Peak 120 at index 1, trough 60 at index 2, back above 120 at index 4.
    CURVE = [100.0, 120.0, 60.0, 80.0, 130.0]

    def test_depth_peak_and_trough_against_a_hand_drawn_curve(self) -> None:
        drawdown = max_drawdown(self.CURVE)
        assert drawdown.depth == pytest.approx(0.5)
        assert drawdown.peak_index == 1
        assert drawdown.trough_index == 2

    def test_recovery_is_the_first_bar_back_at_the_prior_peak(self) -> None:
        drawdown = max_drawdown(self.CURVE)
        assert drawdown.recovery_index == 4
        assert drawdown.trades_to_recovery == 2
        assert drawdown.length == 1

    def test_a_curve_that_never_recovers_says_so_rather_than_reporting_a_duration(self) -> None:
        drawdown = max_drawdown([100.0, 120.0, 60.0, 80.0])
        assert not drawdown.recovered
        assert drawdown.recovery_index is None
        assert drawdown.trades_to_recovery is None

    def test_a_monotonic_curve_has_no_drawdown(self) -> None:
        drawdown = max_drawdown([100.0, 110.0, 120.0])
        assert drawdown.depth == 0.0
        assert drawdown.recovered

    def test_depth_is_fractional_so_it_compares_across_account_sizes(self) -> None:
        small = max_drawdown([1_000.0, 800.0])
        large = max_drawdown([100_000.0, 80_000.0])
        assert small.depth == large.depth == pytest.approx(0.2)

    def test_the_deepest_fall_wins_not_the_longest(self) -> None:
        drawdown = max_drawdown([100.0, 95.0, 94.0, 93.0, 100.0, 50.0, 100.0])
        assert drawdown.depth == pytest.approx(0.5)
        assert drawdown.trough_index == 5


class TestProbabilityOfRuin:
    def test_no_edge_is_certain_ruin(self) -> None:
        """A coin flip paying 1:1 empties the account eventually, whatever the sizing."""
        assert probability_of_ruin(win_rate=0.5, payoff_ratio=1.0, risk_fraction=0.005) == 1.0

    def test_negative_expectancy_is_certain_ruin(self) -> None:
        assert probability_of_ruin(win_rate=0.3, payoff_ratio=2.0, risk_fraction=0.005) == 1.0

    def test_a_real_edge_at_half_a_percent_is_remote(self) -> None:
        """40% win rate at 2R — `session_breakout`'s design point — risking 0.5%."""
        risk = probability_of_ruin(win_rate=0.40, payoff_ratio=2.0, risk_fraction=0.005)
        assert 0.0 < risk < 1e-6

    def test_the_same_edge_sized_recklessly_is_not(self) -> None:
        """Hard rule 8 exists because the edge is not what saves the account, the sizing is."""
        modest = probability_of_ruin(win_rate=0.40, payoff_ratio=2.0, risk_fraction=0.005)
        reckless = probability_of_ruin(win_rate=0.40, payoff_ratio=2.0, risk_fraction=0.25)
        assert reckless > modest
        assert reckless > 0.01

    def test_a_shallower_ruin_threshold_is_reached_more_often(self) -> None:
        deep = probability_of_ruin(0.40, 2.0, 0.05, ruin_fraction=0.9)
        shallow = probability_of_ruin(0.40, 2.0, 0.05, ruin_fraction=0.2)
        assert shallow > deep

    def test_a_certain_winner_never_goes_broke(self) -> None:
        assert probability_of_ruin(win_rate=1.0, payoff_ratio=1.0, risk_fraction=0.5) == 0.0

    def test_the_docstring_states_its_assumptions_plainly(self) -> None:
        """These caveats are the point of the function; deleting them makes it dangerous."""
        doc = probability_of_ruin.__doc__ or ""
        assert "sanity check, not a forecast" in doc
        assert "fixed" in doc
        assert "independent" in doc

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"win_rate": 1.5, "payoff_ratio": 2.0, "risk_fraction": 0.005}, "win_rate"),
            ({"win_rate": 0.5, "payoff_ratio": 0.0, "risk_fraction": 0.005}, "payoff_ratio"),
            ({"win_rate": 0.5, "payoff_ratio": 2.0, "risk_fraction": 0.0}, "risk_fraction"),
        ],
    )
    def test_nonsense_inputs_are_rejected(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            probability_of_ruin(**kwargs)
