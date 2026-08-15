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

from fxagent.indicators import atr, bollinger_bands, true_range

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


# --- Bollinger bands ----------------------------------------------------------
#
# Reference closes, period = 4:
#
#     i   close   window            mean    pop. stdev
#     0    10.0   -                  -        -
#     1    12.0   -                  -        -
#     2    14.0   -                  -        -
#     3    16.0   10,12,14,16       13.0     sqrt(5)  = 2.2360679...
#     4    18.0   12,14,16,18       15.0     sqrt(5)
#     5    10.0   14,16,18,10       14.5     sqrt(35)/2 = 2.9580398...


BB_CLOSE = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 10.0])


def test_bollinger_middle_is_the_simple_moving_average() -> None:
    bands = bollinger_bands(BB_CLOSE, 4)

    assert bands.middle.iloc[3:].to_numpy() == pytest.approx([13.0, 15.0, 14.5])


def test_bollinger_envelope_matches_the_hand_computed_table() -> None:
    bands = bollinger_bands(BB_CLOSE, 4, 2.0)
    spread = 2.0 * np.sqrt(5.0)

    assert bands.upper.iloc[3] == pytest.approx(13.0 + spread)
    assert bands.lower.iloc[3] == pytest.approx(13.0 - spread)
    assert bands.upper.iloc[5] == pytest.approx(14.5 + 2.0 * np.sqrt(35.0) / 2.0)


def test_bollinger_uses_the_population_deviation_not_the_sample_one() -> None:
    """Bollinger's own convention, and the one place this package differs from zscore.

    The two are within 3% at period 20, which is exactly why it is asserted rather than
    assumed: a discrepancy that size makes an overlay disagree with a strategy forever
    without ever looking wrong.
    """
    bands = bollinger_bands(BB_CLOSE, 4, 1.0)
    sample_based = BB_CLOSE.rolling(4).mean() + BB_CLOSE.rolling(4).std(ddof=1)

    assert bands.upper.iloc[3] == pytest.approx(13.0 + np.sqrt(5.0))
    assert bands.upper.iloc[3] != pytest.approx(sample_based.iloc[3])


def test_bollinger_warm_up_ends_on_the_same_bar_for_all_three_bands() -> None:
    """A band that started a bar earlier than its middle would draw an envelope around
    a mean that does not exist yet."""
    bands = bollinger_bands(BB_CLOSE, 4)

    for band in (bands.upper, bands.middle, bands.lower):
        assert band.iloc[:3].isna().all()
        assert not np.isnan(band.iloc[3])


def test_bollinger_bands_collapse_onto_the_mean_when_price_does_not_move() -> None:
    flat = pd.Series([1.2345] * 10)
    bands = bollinger_bands(flat, 4)

    assert bands.upper.dropna().to_numpy() == pytest.approx([1.2345] * 7)
    assert bands.lower.dropna().to_numpy() == pytest.approx([1.2345] * 7)


def test_bollinger_bands_are_ordered_wherever_they_are_defined() -> None:
    rng = np.random.default_rng(20260815)
    close = pd.Series(1.1 + np.cumsum(rng.normal(0, 0.002, 300)))

    bands = bollinger_bands(close, 20)
    defined = bands.middle.notna()

    assert (bands.upper[defined] >= bands.middle[defined]).all()
    assert (bands.middle[defined] >= bands.lower[defined]).all()


def test_a_period_of_one_is_refused() -> None:
    """A single-bar window has no deviation, so the envelope would sit on the price."""
    with pytest.raises(ValueError, match="period must be >= 2"):
        bollinger_bands(BB_CLOSE, 1)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_deviation_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="deviations must be positive"):
        bollinger_bands(BB_CLOSE, 4, bad)
