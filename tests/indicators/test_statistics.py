"""Rolling z-score and percentile rank against values worked out by hand.

Expected values are written as the formula applied to the literal window, not as decimal
constants copied from a run — so a reader can check the arithmetic without trusting the
implementation that produced it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fxagent.indicators import rolling_percentile, rolling_zscore


def _sample_std(*values: float) -> float:
    """Sample standard deviation, ddof=1 — what pandas' rolling std uses."""
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


# --- rolling z-score ----------------------------------------------------------


def test_rolling_zscore_matches_the_formula_on_each_window() -> None:
    series = pd.Series([10.0, 12.0, 11.0, 20.0, 9.0])
    result = rolling_zscore(series, 3)

    assert result.iloc[:2].isna().all()
    # window [10, 12, 11]: mean 11, current value 11 -> exactly on the mean.
    assert result.iloc[2] == pytest.approx((11.0 - 11.0) / _sample_std(10.0, 12.0, 11.0))
    # window [12, 11, 20]: mean 43/3.
    assert result.iloc[3] == pytest.approx((20.0 - 43.0 / 3.0) / _sample_std(12.0, 11.0, 20.0))
    # window [11, 20, 9]: mean 40/3.
    assert result.iloc[4] == pytest.approx((9.0 - 40.0 / 3.0) / _sample_std(11.0, 20.0, 9.0))


def test_rolling_zscore_of_a_linear_ramp_is_constant() -> None:
    """Every window of a straight line is the same window shifted, so z is too."""
    result = rolling_zscore(pd.Series(np.arange(10, dtype="float64")), 4)

    values = result.dropna().to_numpy()
    assert values == pytest.approx([values[0]] * len(values))
    # Last point of [0,1,2,3]: (3 - 1.5) / stdev = 1.5 / sqrt(5/3).
    assert values[0] == pytest.approx(1.5 / _sample_std(0.0, 1.0, 2.0, 3.0))


def test_rolling_zscore_of_a_flat_window_is_nan_not_infinity() -> None:
    """Zero variance means no scale; "how many deviations out" has no answer."""
    result = rolling_zscore(pd.Series([5.0, 5.0, 5.0, 5.0]), 3)

    assert result.iloc[2:].isna().all()
    assert not np.isinf(result.to_numpy()).any()


def test_rolling_zscore_uses_a_trailing_window_not_a_centred_one() -> None:
    """A spike must move the z-score on its own bar and the two after it, never before."""
    series = pd.Series([1.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0, 1.0])
    result = rolling_zscore(series, 3)

    assert np.isnan(result.iloc[2])  # window [1,1,1] is flat -> NaN, unaffected by the spike
    assert result.iloc[3] > 1.0  # the spike's own bar
    assert not np.isnan(result.iloc[5])  # spike still inside the trailing window
    assert np.isnan(result.iloc[6])  # spike has left; window is flat again


def test_rolling_zscore_requires_at_least_two_observations() -> None:
    with pytest.raises(ValueError, match="period must be >= 2"):
        rolling_zscore(pd.Series([1.0, 2.0, 3.0]), 1)


# --- rolling percentile -------------------------------------------------------


def test_rolling_percentile_matches_hand_counted_windows() -> None:
    series = pd.Series([10.0, 20.0, 30.0, 5.0, 25.0])
    result = rolling_percentile(series, 3)

    assert result.iloc[:2].isna().all()
    # [10, 20, 30], current 30: 3 of 3 values are <= 30.
    assert result.iloc[2] == pytest.approx(100.0)
    # [20, 30, 5], current 5: only itself.
    assert result.iloc[3] == pytest.approx(100.0 / 3.0)
    # [30, 5, 25], current 25: 5 and 25.
    assert result.iloc[4] == pytest.approx(200.0 / 3.0)


def test_rolling_percentile_scores_the_window_maximum_at_100() -> None:
    result = rolling_percentile(pd.Series([1.0, 2.0, 3.0, 4.0]), 3)
    assert result.dropna().to_numpy() == pytest.approx([100.0, 100.0])


def test_rolling_percentile_floor_is_one_over_period_never_zero() -> None:
    """A value is always <= itself, so nothing can score 0."""
    result = rolling_percentile(pd.Series([4.0, 3.0, 2.0, 1.0]), 4)
    assert result.iloc[3] == pytest.approx(25.0)


def test_rolling_percentile_gives_ties_the_higher_rank() -> None:
    """All three equal: every value is <= the current one, so the rank is 100."""
    result = rolling_percentile(pd.Series([7.0, 7.0, 7.0]), 3)
    assert result.iloc[2] == pytest.approx(100.0)


def test_rolling_percentile_stays_within_bounds_on_noisy_input() -> None:
    rng = np.random.default_rng(20260810)
    series = pd.Series(rng.normal(0, 1, 300))

    values = rolling_percentile(series, 20).dropna()
    assert (values > 0).all()
    assert (values <= 100.0).all()
