"""Resampling a trade sequence, and the one result that justifies the block bootstrap.

`test_block_resampling_finds_the_losing_streak_that_iid_breaks_up` is the finding the whole
module exists for. A sequence containing one ten-trade losing run is fed to both resamplers; the
iid draw scatters those losses and reports streaks that never exceed a handful, while the block
draw keeps the run whole and reproduces it. The iid drawdown figure is not conservative — it is
wrong in the direction that empties an account.
"""

from __future__ import annotations

import numpy as np
import pytest

from fxagent.stats.resample import (
    Resampler,
    block_bootstrap,
    iid_bootstrap,
    monte_carlo,
    require_single_mode,
    suggested_block_size,
)
from fxagent.stats.returns import EquityMode

#: 90 wins with one contiguous ten-trade losing run buried in the middle.
CLUSTERED = np.array([1.0] * 45 + [-1.0] * 10 + [1.0] * 45)

#: An ordinary-looking record: 40% winners at 2R.
REALISTIC = np.array([2.0] * 40 + [-1.0] * 60)


class TestDrawing:
    def test_iid_returns_the_requested_shape(self) -> None:
        drawn = iid_bootstrap(REALISTIC, paths=50, seed=1)
        assert drawn.shape == (50, REALISTIC.size)

    def test_block_returns_the_requested_shape(self) -> None:
        drawn = block_bootstrap(REALISTIC, block_size=7, paths=50, seed=1)
        assert drawn.shape == (50, REALISTIC.size)

    def test_every_drawn_value_came_from_the_sample(self) -> None:
        """A bootstrap invents nothing; it only reorders and repeats what happened."""
        drawn = block_bootstrap(REALISTIC, block_size=7, paths=20, seed=1)
        assert set(np.unique(drawn)).issubset(set(np.unique(REALISTIC)))

    def test_a_full_length_block_is_a_rotation_of_the_original(self) -> None:
        """Blocks wrap, so at block_size == n each path holds exactly the original multiset."""
        drawn = block_bootstrap(REALISTIC, block_size=REALISTIC.size, paths=25, seed=3)
        for path in drawn:
            assert np.array_equal(np.sort(path), np.sort(REALISTIC))

    def test_the_same_seed_gives_the_same_draw(self) -> None:
        """A report whose interval moves between runs cannot be compared to last week's."""
        first = iid_bootstrap(REALISTIC, paths=20, seed=42)
        second = iid_bootstrap(REALISTIC, paths=20, seed=42)
        assert np.array_equal(first, second)

    def test_a_different_seed_gives_a_different_draw(self) -> None:
        first = iid_bootstrap(REALISTIC, paths=20, seed=42)
        second = iid_bootstrap(REALISTIC, paths=20, seed=43)
        assert not np.array_equal(first, second)

    def test_a_block_longer_than_the_sample_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds the 100 trades available"):
            block_bootstrap(REALISTIC, block_size=200, paths=5, seed=1)

    def test_the_suggested_block_size_is_the_cube_root(self) -> None:
        assert suggested_block_size(1000) == 10
        assert suggested_block_size(8) == 2
        assert suggested_block_size(1) == 2


class TestTheBlockBootstrapEarnsItsPlace:
    def test_block_resampling_finds_the_losing_streak_that_iid_breaks_up(self) -> None:
        """The reason CLAUDE.md's trades are not independent and the iid figure is optimistic."""
        common = {"risk_fraction": 0.005, "paths": 2_000, "seed": 7}
        independent = monte_carlo(CLUSTERED, resampler=Resampler.IID, **common)
        blocked = monte_carlo(CLUSTERED, resampler=Resampler.BLOCK, block_size=10, **common)

        assert blocked.longest_losing_streak.percentile(95) >= 10
        assert independent.longest_losing_streak.percentile(95) < 10

    def test_the_iid_drawdown_is_the_shallower_one(self) -> None:
        common = {"risk_fraction": 0.01, "paths": 2_000, "seed": 7}
        independent = monte_carlo(CLUSTERED, resampler=Resampler.IID, **common)
        blocked = monte_carlo(CLUSTERED, resampler=Resampler.BLOCK, block_size=10, **common)
        assert blocked.max_drawdown.percentile(95) > independent.max_drawdown.percentile(95)

    def test_the_conservative_resampler_is_the_default(self) -> None:
        result = monte_carlo(REALISTIC, risk_fraction=0.005, paths=100, seed=1)
        assert result.resampler is Resampler.BLOCK
        assert result.block_size == suggested_block_size(REALISTIC.size)


class TestTheDistributions:
    def test_percentiles_are_reported_not_only_the_mean(self) -> None:
        result = monte_carlo(REALISTIC, risk_fraction=0.005, paths=500, seed=5)
        percentiles = result.max_drawdown.percentiles
        assert set(percentiles) == {5, 25, 50, 75, 95}
        assert percentiles[5] <= percentiles[50] <= percentiles[95]

    def test_a_deeper_tail_than_the_average_path(self) -> None:
        """The 95th percentile drawdown is the number that decides whether the sizing survives."""
        result = monte_carlo(REALISTIC, risk_fraction=0.005, paths=1_000, seed=5)
        assert result.max_drawdown.percentile(95) > result.max_drawdown.mean

    def test_every_requested_metric_is_present(self) -> None:
        result = monte_carlo(REALISTIC, risk_fraction=0.005, paths=200, seed=5)
        assert result.final_equity.samples.size == 200
        assert result.max_drawdown.samples.size == 200
        assert result.longest_losing_streak.samples.size == 200
        assert 0.0 <= result.never_recovered <= 1.0

    def test_a_losing_record_never_recovers(self) -> None:
        """Every path ends below its peak, so `trades_to_recovery` must not report a number."""
        losing = np.array([-1.0] * 40 + [0.5] * 20)
        result = monte_carlo(losing, risk_fraction=0.01, paths=200, seed=2)
        assert result.never_recovered == pytest.approx(1.0)

    def test_the_settings_travel_with_the_numbers(self) -> None:
        """A drawdown percentile means a different thing under each resampler."""
        result = monte_carlo(REALISTIC, risk_fraction=0.005, paths=100, block_size=5, seed=1)
        assert "BLOCK (block 5)" in result.describe()
        assert result.trades == 100

    def test_an_arbitrary_percentile_can_be_asked_for(self) -> None:
        result = monte_carlo(REALISTIC, risk_fraction=0.005, paths=500, seed=5)
        assert result.max_drawdown.percentile(99) >= result.max_drawdown.percentile(95)

    def test_a_single_trade_cannot_be_resampled(self) -> None:
        with pytest.raises(ValueError, match="at least 2 trades"):
            monte_carlo([1.0], risk_fraction=0.005, paths=10)


class TestTheModeIsAlwaysLabelled:
    """A drawdown figure without its mode cannot be checked against a budget."""

    def test_every_money_metric_carries_the_mode_in_its_name(self) -> None:
        result = monte_carlo(
            REALISTIC, risk_fraction=0.005, paths=100, seed=1, mode=EquityMode.ADDITIVE
        )
        for distribution in (
            result.final_equity,
            result.total_return,
            result.max_drawdown,
            result.trades_to_recovery,
        ):
            assert "additive-R" in distribution.name

    def test_a_metric_the_mode_cannot_change_is_not_labelled(self) -> None:
        """A losing streak belongs to the R sequence. Labelling it would imply a difference."""
        compounded = monte_carlo(REALISTIC, risk_fraction=0.005, paths=100, seed=1)
        additive = monte_carlo(
            REALISTIC, risk_fraction=0.005, paths=100, seed=1, mode=EquityMode.ADDITIVE
        )
        assert compounded.longest_losing_streak.name == "longest losing streak"
        assert compounded.longest_losing_streak.samples.tolist() == (
            additive.longest_losing_streak.samples.tolist()
        )

    def test_the_headline_names_the_mode(self) -> None:
        assert (
            "[compounded]"
            in monte_carlo(REALISTIC, risk_fraction=0.005, paths=50, seed=1).describe()
        )
        assert (
            "[additive-R]"
            in monte_carlo(
                REALISTIC, risk_fraction=0.005, paths=50, seed=1, mode=EquityMode.ADDITIVE
            ).describe()
        )

    def test_compounded_is_the_default(self) -> None:
        assert monte_carlo(REALISTIC, risk_fraction=0.005, paths=50, seed=1).mode is (
            EquityMode.COMPOUNDED
        )


class TestNeverBothInOneChart:
    def test_a_mixed_set_is_refused(self) -> None:
        common = {"risk_fraction": 0.005, "paths": 50, "seed": 1}
        compounded = monte_carlo(REALISTIC, **common)
        additive = monte_carlo(REALISTIC, mode=EquityMode.ADDITIVE, **common)
        with pytest.raises(ValueError, match="mix equity modes"):
            require_single_mode(compounded, additive)

    def test_a_consistent_set_returns_the_mode_it_shares(self) -> None:
        common = {"risk_fraction": 0.005, "paths": 50, "mode": EquityMode.ADDITIVE}
        first = monte_carlo(REALISTIC, seed=1, **common)
        second = monte_carlo(CLUSTERED, seed=2, **common)
        assert require_single_mode(first, second) is EquityMode.ADDITIVE

    def test_an_empty_chart_has_no_mode_to_report(self) -> None:
        with pytest.raises(ValueError, match="no results to check"):
            require_single_mode()
