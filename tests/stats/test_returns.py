"""Log returns against hand-computed values, and the two properties percent returns lack.

`test_percent_returns_have_a_drift_that_logs_do_not` is the reason this module exists. It is
not a style preference: a round trip back to the starting price has a positive mean simple
return, so any statistic that averages percent returns reports a profit on an account that has
made nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fxagent.stats.returns import (
    EquityMode,
    cumulative_log_returns,
    equity_curve,
    log_returns,
    r_multiple,
    r_multiples_from_pnl,
    to_log,
    to_simple,
    total_return,
)
from fxagent.stats.risk_metrics import max_drawdown


class TestLogReturns:
    def test_against_hand_computed_values(self) -> None:
        result = log_returns(pd.Series([100.0, 110.0, 121.0]))
        assert math.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(math.log(1.1))
        assert result.iloc[2] == pytest.approx(math.log(1.1))

    def test_the_first_observation_is_nan_not_zero(self) -> None:
        """A bar that did not move and a bar with no predecessor are different facts."""
        result = log_returns(pd.Series([100.0, 100.0]))
        assert math.isnan(result.iloc[0])
        assert result.iloc[1] == 0.0

    def test_they_are_additive_across_periods(self) -> None:
        """log(p2/p0) == log(p1/p0) + log(p2/p1), exactly. Percent returns are not."""
        prices = pd.Series([100.0, 137.0, 92.0, 181.0])
        returns = log_returns(prices).dropna()
        assert returns.sum() == pytest.approx(math.log(181.0 / 100.0))

    def test_percent_returns_have_a_drift_that_logs_do_not(self) -> None:
        """Up 10% then back to the start: the account is flat and percent says +0.45% a period."""
        prices = pd.Series([100.0, 110.0, 100.0])
        logs = log_returns(prices).dropna()
        simple = prices.pct_change().dropna()

        assert logs.sum() == pytest.approx(0.0, abs=1e-12)
        assert logs.mean() == pytest.approx(0.0, abs=1e-12)
        assert simple.mean() == pytest.approx(0.004545, abs=1e-5)

    def test_a_non_positive_price_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="prices must be positive"):
            log_returns(pd.Series([100.0, 0.0]))

    def test_a_bare_array_is_rejected_rather_than_silently_reindexed(self) -> None:
        with pytest.raises(TypeError, match="pandas Series"):
            log_returns([100.0, 110.0])  # type: ignore[arg-type]


class TestCumulative:
    def test_cumulative_is_a_plain_sum(self) -> None:
        assert cumulative_log_returns([0.1, 0.2, 0.3]) == pytest.approx([0.1, 0.3, 0.6])

    def test_total_return_converts_once_at_the_end(self) -> None:
        """Two 10% gains compound to 21%, not 20%."""
        assert total_return([math.log(1.1), math.log(1.1)]) == pytest.approx(0.21)

    def test_the_round_trip_through_simple_and_back(self) -> None:
        assert to_log(to_simple(0.37)) == pytest.approx(0.37)

    def test_a_total_loss_has_no_log_equivalent(self) -> None:
        with pytest.raises(ValueError, match="no finite log equivalent"):
            to_log(-1.0)

    def test_a_warm_up_nan_raises_rather_than_being_skipped(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            cumulative_log_returns([np.nan, 0.1])


class TestRMultiples:
    def test_a_long_at_twice_its_stop_distance_is_two_r(self) -> None:
        assert r_multiple(entry_price=1.1000, exit_price=1.1050, stop_loss=1.0975) == pytest.approx(
            2.0
        )

    def test_a_short_at_twice_its_stop_distance_is_also_two_r(self) -> None:
        """No direction argument, and the sign still comes out right."""
        assert r_multiple(entry_price=1.1000, exit_price=1.0950, stop_loss=1.1025) == pytest.approx(
            2.0
        )

    def test_being_stopped_out_is_exactly_minus_one(self) -> None:
        assert r_multiple(1.1000, 1.0975, 1.0975) == pytest.approx(-1.0)
        assert r_multiple(1.1000, 1.1025, 1.1025) == pytest.approx(-1.0)

    def test_a_stop_at_the_entry_has_no_r(self) -> None:
        with pytest.raises(ValueError, match="R is undefined"):
            r_multiple(1.1000, 1.1050, 1.1000)

    def test_pnl_is_divided_by_the_risk_actually_taken_per_trade(self) -> None:
        """Sizing rounds down to the lot step, so the money risked differs trade to trade."""
        result = r_multiples_from_pnl(pnl=[10.0, -4.0], risk_amount=[5.0, 4.0])
        assert result == pytest.approx([2.0, -1.0])

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="risk_amount has"):
            r_multiples_from_pnl(pnl=[1.0, 2.0], risk_amount=[1.0])


class TestEquityCurve:
    def test_against_a_hand_computed_curve(self) -> None:
        """+2R then -1R at 0.5%: 1000 -> 1010 -> 1004.95."""
        curve = equity_curve([2.0, -1.0], risk_fraction=0.005, starting_equity=1000.0)
        assert curve == pytest.approx([1000.0, 1010.0, 1004.95])

    def test_it_starts_before_the_first_trade(self) -> None:
        """A drawdown measured from the first result would miss a first losing trade."""
        curve = equity_curve([-1.0], risk_fraction=0.005, starting_equity=1000.0)
        assert len(curve) == 2
        assert curve[0] == 1000.0

    def test_it_compounds_rather_than_adding(self) -> None:
        flat = equity_curve([1.0, 1.0], risk_fraction=0.5, starting_equity=100.0)
        assert flat[-1] == pytest.approx(225.0)  # 100 * 1.5 * 1.5, not 100 + 50 + 50

    def test_equity_is_floored_at_zero_and_stays_there(self) -> None:
        """An account cannot go negative — the broker closes it."""
        curve = equity_curve([-3.0, 5.0], risk_fraction=0.5, starting_equity=100.0)
        assert curve[1] == 0.0
        assert curve[2] == 0.0

    def test_the_fast_path_and_the_floored_path_agree(self) -> None:
        """The cumprod shortcut must be the same arithmetic as the loop, not merely close."""
        multipliers = [0.4, -0.7, 1.3, -0.2]
        fast = equity_curve(multipliers, risk_fraction=0.01, starting_equity=1000.0)
        slow = equity_curve([*multipliers, -200.0], risk_fraction=0.01, starting_equity=1000.0)
        assert fast == pytest.approx(slow[: len(fast)])

    def test_an_absurd_risk_fraction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="risk_fraction must be in"):
            equity_curve([1.0], risk_fraction=1.5)


class TestEquityModes:
    """Compounded and additive-R are different quantities, and the difference has a sign."""

    def test_the_additive_curve_against_a_hand_computed_one(self) -> None:
        """+2R then -1R at a fixed $5 unit: 1000 -> 1010 -> 1005, not 1004.95."""
        curve = equity_curve(
            [2.0, -1.0], risk_fraction=0.005, starting_equity=1000.0, mode=EquityMode.ADDITIVE
        )
        assert curve == pytest.approx([1000.0, 1010.0, 1005.0])

    def test_the_unit_does_not_shrink_after_a_loss(self) -> None:
        """Every trade risks the same money, so a run of losses is a straight line down."""
        curve = equity_curve(
            [-1.0, -1.0, -1.0], risk_fraction=0.01, starting_equity=1000.0, mode=EquityMode.ADDITIVE
        )
        assert np.diff(curve) == pytest.approx([-10.0, -10.0, -10.0])

    def test_the_compounded_unit_does_shrink(self) -> None:
        losses = np.diff(
            equity_curve([-1.0, -1.0, -1.0], risk_fraction=0.01, starting_equity=1000.0)
        )
        assert losses[1] > losses[0]  # less negative: the second loss cost less

    def test_compounded_is_the_default(self) -> None:
        assert (
            equity_curve([-1.0], risk_fraction=0.01, starting_equity=1000.0)[-1]
            == equity_curve(
                [-1.0], risk_fraction=0.01, starting_equity=1000.0, mode=EquityMode.COMPOUNDED
            )[-1]
        )

    def test_a_sixteen_loss_run_passes_a_fifteen_percent_budget_compounded_and_fails_additive(
        self,
    ) -> None:
        """The Card 22 case, exactly. `MAX_ACCEPTABLE_DRAWDOWN = 0.15` was reasoned before this
        distinction existed, so checking it against the compounded curve waves through a system
        that lost sixteen straight trades at 1%."""
        budget = 0.15
        run = [-1.0] * 16
        compounded = max_drawdown(equity_curve(run, risk_fraction=0.01, starting_equity=10_000.0))
        additive = max_drawdown(
            equity_curve(
                run, risk_fraction=0.01, starting_equity=10_000.0, mode=EquityMode.ADDITIVE
            )
        )
        assert additive.depth == pytest.approx(0.16)
        assert compounded.depth < budget < additive.depth

    def test_a_choppy_drawdown_is_usually_deeper_compounded(self) -> None:
        """The other direction, and the one that is easy to get wrong.

        Compounding suffers volatility drag, so a drawdown made of wins and losses interleaved
        costs more compounded than the same net R does at a fixed unit. This record never gets
        above its starting equity and still shows compounded as the deeper measure — so "size
        shrinks after a loss, therefore compounding is gentler" is not a rule.
        """
        record = [
            -1.0,
            -1.0,
            -1.0,
            2.0,
            -1.0,
            -1.0,
            -1.0,
            2.0,
            -1.0,
            -1.0,
            -1.0,
            2.0,
            -1.0,
            -1.0,
            -1.0,
            2.0,
            -1.0,
            2.0,
            2.0,
            -1.0,
            -1.0,
            -1.0,
            2.0,
            -1.0,
            2.0,
            -1.0,
            -1.0,
            -1.0,
            2.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            2.0,
            2.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            2.0,
        ]
        common = {"risk_fraction": 0.01, "starting_equity": 10_000.0}
        curve = equity_curve(record, **common)
        assert curve.max() <= curve[0], "this record must never get above its start"

        compounded = max_drawdown(curve).depth
        additive = max_drawdown(equity_curve(record, mode=EquityMode.ADDITIVE, **common)).depth
        assert compounded == pytest.approx(0.0616, abs=1e-4)
        assert additive == pytest.approx(0.0600, abs=1e-4)
        assert compounded > additive

    def test_neither_mode_dominates_the_other(self) -> None:
        """Both orderings occur often enough that a report must name which one it used."""
        rng = np.random.default_rng(19)
        common = {"risk_fraction": 0.01, "starting_equity": 10_000.0}
        deeper_compounded = deeper_additive = 0
        for _ in range(200):
            record = rng.choice([2.0, -1.0], size=40, p=[0.35, 0.65])
            compounded = max_drawdown(equity_curve(record, **common)).depth
            additive = max_drawdown(equity_curve(record, mode=EquityMode.ADDITIVE, **common)).depth
            deeper_compounded += compounded > additive + 1e-9
            deeper_additive += additive > compounded + 1e-9
        assert deeper_compounded > 20
        assert deeper_additive > 20

    def test_ruin_is_absorbing_in_the_additive_mode_too(self) -> None:
        """A closed account does not go on to win the next trade back."""
        curve = equity_curve(
            [-1.0, -1.0, 5.0], risk_fraction=0.6, starting_equity=100.0, mode=EquityMode.ADDITIVE
        )
        assert curve[2] == 0.0
        assert curve[3] == 0.0
