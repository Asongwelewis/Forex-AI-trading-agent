"""True range and ATR against values worked out by hand.

Reference bars used throughout, period = 3:

    i   high   low   close    high-low   |high-c_prev|   |low-c_prev|    TR
    0   10.0    9.0    9.5        1.0          -               -         -
    1   11.0    9.5   10.5        1.5         1.5             0.0       1.5
    2   12.0   10.0   11.5        2.0         1.5             0.5       2.0
    3   11.5   10.5   11.0        1.0         0.0             1.0       1.0
    4   12.5   10.8   12.2        1.7         1.5             0.2       1.7
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxagent.indicators import atr, true_range

HIGH = pd.Series([10.0, 11.0, 12.0, 11.5, 12.5])
LOW = pd.Series([9.0, 9.5, 10.0, 10.5, 10.8])
CLOSE = pd.Series([9.5, 10.5, 11.5, 11.0, 12.2])


# --- true range ---------------------------------------------------------------


def test_true_range_matches_the_hand_computed_table() -> None:
    result = true_range(HIGH, LOW, CLOSE)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1:].to_numpy() == pytest.approx([1.5, 2.0, 1.0, 1.7])


def test_true_range_is_undefined_on_the_first_bar() -> None:
    """Bar 0 has no previous close. Degrading it to high-low would bias the ATR seed."""
    assert np.isnan(true_range(HIGH, LOW, CLOSE).iloc[0])


def test_true_range_spans_a_gap_the_bar_itself_does_not_show() -> None:
    """A bar that opens far above the previous close is wider than its own high-low."""
    high = pd.Series([10.0, 20.5])
    low = pd.Series([9.0, 20.0])
    close = pd.Series([9.5, 20.2])

    # high-low is only 0.5, but the gap from 9.5 up to 20.5 is 11.0.
    assert true_range(high, low, close).iloc[1] == pytest.approx(11.0)


def test_true_range_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        true_range(HIGH, LOW.iloc[:-1], CLOSE)


def test_true_range_rejects_a_differently_labelled_index() -> None:
    shifted = LOW.copy()
    shifted.index = range(10, 15)
    with pytest.raises(ValueError, match="index"):
        true_range(HIGH, shifted, CLOSE)


# --- ATR ----------------------------------------------------------------------


def test_atr_matches_a_hand_computed_wilder_recursion() -> None:
    """Seed at index 3 = mean(TR[1..3]) = (1.5 + 2.0 + 1.0) / 3 = 1.5.

    i=4: (1.5 * 2 + 1.7) / 3 = 4.7 / 3
    """
    result = atr(HIGH, LOW, CLOSE, 3)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(1.5)
    assert result.iloc[4] == pytest.approx((1.5 * 2 + 1.7) / 3)


def test_atr_first_value_lands_at_index_period_not_period_minus_one() -> None:
    """It averages the first `period` *defined* true ranges, and TR[0] is not one."""
    result = atr(HIGH, LOW, CLOSE, 3)

    assert result.iloc[:3].isna().all()
    assert not np.isnan(result.iloc[3])


def test_atr_is_not_an_ewm_with_span_equal_to_period() -> None:
    """Wilder's alpha is 1/period, not 2/(period+1). Mixing them shifts every value."""
    result = atr(HIGH, LOW, CLOSE, 3)
    span_based = true_range(HIGH, LOW, CLOSE).ewm(span=3, adjust=False).mean()

    assert result.iloc[4] != pytest.approx(span_based.iloc[4])


def test_atr_of_constant_range_bars_equals_that_range() -> None:
    """Every TR is 1.0, so the seed is 1.0 and the recursion cannot move off it."""
    high = pd.Series([11.0] * 8)
    low = pd.Series([10.0] * 8)
    close = pd.Series([10.5] * 8)

    result = atr(high, low, close, 4)
    assert result.dropna().to_numpy() == pytest.approx([1.0] * 4)


def test_atr_never_goes_negative_on_real_shaped_bars() -> None:
    rng = np.random.default_rng(20260810)
    close = pd.Series(1.1 + np.cumsum(rng.normal(0, 0.002, 200)))
    half_range = pd.Series(np.abs(rng.normal(0, 0.001, 200)) + 0.0001)

    result = atr(close + half_range, close - half_range, close, 14)
    assert (result.dropna() > 0).all()
