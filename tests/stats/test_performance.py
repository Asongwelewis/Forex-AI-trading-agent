"""Performance statistics against hand-computed values, and the intervals around them.

`test_sharpe_on_thirty_trades_cannot_tell_an_edge_from_noise` is the one to read first. Thirty
trades drawn from a distribution with a genuinely positive mean produce a Sharpe interval that
comfortably contains zero — which is the whole argument for never returning the point estimate
on its own.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fxagent.stats.performance import (
    INSUFFICIENT_DATA,
    MIN_TRADES_FOR_JUDGEMENT,
    Estimate,
    expectancy_r,
    judgement,
    profit_factor,
    sharpe,
    sortino,
    win_rate,
)

#: Mean 0.5, sample stdev sqrt(3), downside deviation sqrt(0.5). All four statistics below are
#: read off this by hand.
HAND = [2.0, 2.0, -1.0, -1.0]

#: Fewer bootstrap draws than the default so the suite stays quick; the seed pins the interval.
FAST = {"samples": 2_000, "seed": 3}


class TestTheShape:
    def test_an_estimate_unpacks_as_three_values(self) -> None:
        estimate, lower, upper = expectancy_r(HAND, **FAST)
        assert lower <= estimate <= upper

    def test_nothing_here_returns_a_bare_float(self) -> None:
        """The interval is the finding. A scalar return type would let it be dropped silently."""
        for statistic in (sharpe, sortino, profit_factor, expectancy_r, win_rate):
            assert isinstance(statistic(HAND, **FAST), Estimate)

    def test_the_same_seed_reproduces_the_same_interval(self) -> None:
        assert sharpe(HAND, samples=500, seed=9) == sharpe(HAND, samples=500, seed=9)

    def test_a_single_observation_reports_no_interval_rather_than_a_certain_one(self) -> None:
        estimate, lower, upper = expectancy_r([1.5], **FAST)
        assert estimate == pytest.approx(1.5)
        assert math.isnan(lower) and math.isnan(upper)


class TestHandComputedValues:
    def test_sharpe(self) -> None:
        """mean 0.5 over sample stdev sqrt(3) = 0.28868."""
        assert sharpe(HAND, **FAST).estimate == pytest.approx(0.5 / math.sqrt(3.0))

    def test_sharpe_annualises_by_the_square_root_of_the_periods_given(self) -> None:
        per_trade = sharpe(HAND, **FAST).estimate
        annual = sharpe(HAND, periods_per_year=52, **FAST).estimate
        assert annual == pytest.approx(per_trade * math.sqrt(52))

    def test_sharpe_is_not_annualised_by_default(self) -> None:
        """A trade sequence has no time axis, so inventing one is not a default."""
        assert sharpe(HAND, **FAST).estimate == pytest.approx(0.28868, abs=1e-5)

    def test_sortino(self) -> None:
        """mean 0.5 over downside deviation sqrt((0+0+1+1)/4) = 0.70711."""
        assert sortino(HAND, **FAST).estimate == pytest.approx(0.5 / math.sqrt(0.5))

    def test_sortino_scores_above_sharpe_when_the_upside_is_the_volatile_half(self) -> None:
        assert sortino(HAND, **FAST).estimate > sharpe(HAND, **FAST).estimate

    def test_profit_factor(self) -> None:
        """4.0 of gross wins over 2.0 of gross losses."""
        assert profit_factor(HAND, **FAST).estimate == pytest.approx(2.0)

    def test_profit_factor_is_infinite_without_a_single_loss(self) -> None:
        assert math.isinf(profit_factor([1.0, 2.0, 3.0], **FAST).estimate)

    def test_expectancy_r(self) -> None:
        assert expectancy_r(HAND, **FAST).estimate == pytest.approx(0.5)

    def test_win_rate(self) -> None:
        assert win_rate(HAND, **FAST).estimate == pytest.approx(0.5)

    def test_a_scratched_trade_is_not_a_win(self) -> None:
        assert win_rate([1.0, 0.0, 0.0, 0.0], **FAST).estimate == pytest.approx(0.25)


class TestTheIntervalIsThePoint:
    def test_sharpe_on_thirty_trades_cannot_tell_an_edge_from_noise(self) -> None:
        """Thirty draws from a distribution whose mean is genuinely +0.2, and the measured
        Sharpe comes out *negative*. The interval spans zero and is wider than the whole
        plausible range of the statistic — which is the entire case for never reporting the
        point estimate alone."""
        rng = np.random.default_rng(17)
        thirty = rng.normal(loc=0.2, scale=1.0, size=30)
        estimate = sharpe(thirty, samples=4_000, seed=3)
        assert estimate.estimate < 0
        assert estimate.spans_zero
        assert estimate.width > 0.5

    def test_the_same_edge_over_a_thousand_trades_establishes_its_sign(self) -> None:
        rng = np.random.default_rng(17)
        thousand = rng.normal(loc=0.2, scale=1.0, size=1_000)
        estimate = sharpe(thousand, samples=4_000, seed=3)
        assert not estimate.spans_zero
        assert estimate.lower_ci > 0

    def test_more_trades_narrow_the_interval(self) -> None:
        rng = np.random.default_rng(23)
        few = expectancy_r(rng.normal(0.2, 1.0, 50), samples=2_000, seed=3)
        many = expectancy_r(rng.normal(0.2, 1.0, 2_000), samples=2_000, seed=3)
        assert many.width < few.width

    def test_the_sortino_interval_is_the_wider_one(self) -> None:
        """Its denominator is estimated from fewer observations, so it is noisier. Say so."""
        rng = np.random.default_rng(5)
        sample = rng.normal(0.2, 1.0, 200)
        assert sortino(sample, **FAST).width > sharpe(sample, **FAST).width


class TestInsufficientData:
    def test_the_floor_is_a_hundred_trades(self) -> None:
        assert MIN_TRADES_FOR_JUDGEMENT == 100
        assert judgement(99) == INSUFFICIENT_DATA
        assert judgement(100) is None

    def test_the_statistics_still_compute_below_the_floor(self) -> None:
        """A 40-trade interval is informative about how little is known — but it is not a
        number to rank strategies on, and `judgement` is what says so."""
        estimate = expectancy_r([1.0, -1.0] * 20, **FAST)
        assert not math.isnan(estimate.estimate)
        assert judgement(40) == INSUFFICIENT_DATA
