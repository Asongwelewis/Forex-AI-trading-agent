"""Diagnostics against reference series and textbook critical values.

The chi-square tail is checked against the values printed in every statistics table — 3.841 at
one degree of freedom is 5%, 5.991 at two is 5%, 23.209 at ten is 1% — rather than against
another implementation, so the check is independent of the code it is checking.

The Hurst tests assert an *ordering* rather than an absolute value. Classic R/S is biased upward
on finite samples and returns something above 0.5 on genuinely independent data; the honest
reference is the same estimator run on a shuffled copy of the same series, which is exactly what
the docstring tells a caller to do.
"""

from __future__ import annotations

import numpy as np
import pytest

from fxagent.stats.diagnostics import _chi2_survival as chi2_survival
from fxagent.stats.diagnostics import _gamma_continued_fraction as gamma_fraction
from fxagent.stats.diagnostics import _gamma_series as gamma_series
from fxagent.stats.diagnostics import (
    arch_test,
    autocorrelation,
    hurst_exponent,
    realised_volatility,
)


class TestTheChiSquareTail:
    @pytest.mark.parametrize(
        ("statistic", "degrees", "expected"),
        [
            (3.8415, 1, 0.05),
            (6.6349, 1, 0.01),
            (5.9915, 2, 0.05),
            (9.2103, 2, 0.01),
            (11.0705, 5, 0.05),
            (15.0863, 5, 0.01),
            (18.3070, 10, 0.05),
            (23.2093, 10, 0.01),
        ],
    )
    def test_against_printed_critical_values(
        self, statistic: float, degrees: int, expected: float
    ) -> None:
        assert chi2_survival(statistic, degrees) == pytest.approx(expected, abs=1e-4)

    def test_zero_is_certain_to_be_exceeded(self) -> None:
        assert chi2_survival(0.0, 3) == 1.0

    @pytest.mark.parametrize("degrees", [1, 2, 3, 6, 10])
    def test_the_series_and_continued_fraction_branches_meet(self, degrees: int) -> None:
        """Both formulas evaluated at the split point itself, where each is at its worst.

        Compared at `x = a + 1` exactly rather than either side of it: stepping across the
        boundary also steps the statistic, so a slope would masquerade as a discontinuity and
        a real discontinuity would hide inside the tolerance that allowed for it.
        """
        a = degrees / 2.0
        x = a + 1.0
        assert 1.0 - gamma_series(a, x) == pytest.approx(gamma_fraction(a, x), abs=1e-12)


class TestAutocorrelation:
    def test_a_perfect_alternation_against_hand_computed_values(self) -> None:
        """+1, -1, +1 ... correlates -(n-1)/n at lag 1 and +(n-2)/n at lag 2."""
        series = np.array([1.0, -1.0] * 50)
        result = autocorrelation(series, max_lag=2)
        assert result.values[0] == pytest.approx(-0.99)
        assert result.values[1] == pytest.approx(0.98)

    def test_the_lags_are_one_through_max_lag(self) -> None:
        result = autocorrelation(np.arange(100.0), max_lag=5)
        assert list(result.lags) == [1, 2, 3, 4, 5]

    def test_white_noise_stays_inside_the_band(self) -> None:
        rng = np.random.default_rng(4)
        result = autocorrelation(rng.standard_normal(4_000), max_lag=10)
        assert result.confidence_band == pytest.approx(1.96 / np.sqrt(4_000))
        assert len(result.significant_lags) <= 1

    def test_a_strongly_dependent_series_breaks_out_of_it(self) -> None:
        rng = np.random.default_rng(4)
        shocks = rng.standard_normal(2_000)
        series = np.empty(2_000)
        series[0] = shocks[0]
        for i in range(1, 2_000):
            series[i] = 0.8 * series[i - 1] + shocks[i]
        result = autocorrelation(series, max_lag=5)
        assert result.values[0] == pytest.approx(0.8, abs=0.05)
        assert 1 in result.significant_lags

    def test_a_lag_as_long_as_the_sample_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no pairs to correlate"):
            autocorrelation(np.arange(10.0), max_lag=10)

    def test_a_constant_series_has_no_autocorrelation_to_report(self) -> None:
        with pytest.raises(ValueError, match="zero variance"):
            autocorrelation(np.ones(50), max_lag=3)


class TestHurst:
    def test_independent_increments_sit_near_a_half(self) -> None:
        rng = np.random.default_rng(12)
        result = hurst_exponent(rng.standard_normal(8_192))
        assert 0.40 < result.exponent < 0.65

    def test_a_persistent_series_scores_above_an_independent_one(self) -> None:
        """The comparison the docstring asks for: same estimator, ordering destroyed."""
        rng = np.random.default_rng(12)
        shocks = rng.standard_normal(8_192)
        persistent = np.empty(8_192)
        persistent[0] = shocks[0]
        for i in range(1, 8_192):
            persistent[i] = 0.9 * persistent[i - 1] + shocks[i]

        assert hurst_exponent(persistent).exponent > hurst_exponent(shocks).exponent + 0.1

    def test_a_mean_reverting_series_scores_below_one(self) -> None:
        rng = np.random.default_rng(12)
        shocks = rng.standard_normal(8_192)
        reverting = shocks[1:] - shocks[:-1]
        assert hurst_exponent(reverting).exponent < hurst_exponent(shocks).exponent

    def test_the_interpretation_reads_the_exponent(self) -> None:
        rng = np.random.default_rng(12)
        assert "random walk" in hurst_exponent(rng.standard_normal(8_192)).interpretation

    def test_too_short_a_series_is_refused_rather_than_fitted(self) -> None:
        with pytest.raises(ValueError, match="at least 64 observations"):
            hurst_exponent(np.arange(40.0))


class TestRealisedVolatility:
    def test_against_a_hand_computed_value(self) -> None:
        """A hundred moves of 1% each: sqrt(100 * 0.0001) = 0.1."""
        assert realised_volatility([0.01] * 100) == pytest.approx(0.1)

    def test_annualising_scales_by_the_periods_given(self) -> None:
        """sqrt(252 * mean(r^2)) with mean(r^2) = 1e-4 is 0.15875."""
        assert realised_volatility([0.01] * 100, periods_per_year=252) == pytest.approx(
            np.sqrt(252 * 1e-4)
        )

    def test_it_does_not_subtract_the_mean(self) -> None:
        """A constant drift is real variation; a sample stdev would report zero here."""
        assert realised_volatility([0.01] * 100) > 0
        assert np.std([0.01] * 100) == pytest.approx(0.0, abs=1e-15)

    def test_a_non_positive_annualisation_factor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            realised_volatility([0.01, 0.02], periods_per_year=0)


class TestArchEffects:
    def test_independent_returns_show_no_clustering(self) -> None:
        rng = np.random.default_rng(31)
        result = arch_test(rng.standard_normal(2_000), lags=5)
        assert not result.clustering_detected
        assert result.p_value > 0.05

    def test_a_volatility_clustered_series_is_detected(self) -> None:
        """Today's shock scaled by yesterday's size — the textbook ARCH generator."""
        rng = np.random.default_rng(31)
        shocks = rng.standard_normal(2_000)
        series = np.empty(2_000)
        series[0] = shocks[0]
        for i in range(1, 2_000):
            variance = 0.2 + 0.8 * series[i - 1] ** 2
            series[i] = np.sqrt(variance) * shocks[i]

        result = arch_test(series, lags=5)
        assert result.clustering_detected
        assert result.p_value < 0.001
        assert "clustering" in result.describe()

    def test_the_statistic_and_the_p_value_are_reported_together(self) -> None:
        rng = np.random.default_rng(31)
        result = arch_test(rng.standard_normal(500), lags=3)
        assert result.lags == 3
        assert result.observations == 500
        assert result.statistic >= 0

    def test_too_few_observations_for_the_lags_asked_for(self) -> None:
        with pytest.raises(ValueError, match="needs more than"):
            arch_test(np.arange(10.0), lags=5)
