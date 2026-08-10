"""EMA and ADX against values worked out by hand from the formulas.

The ADX sequence below is derived arithmetically in the docstrings, not captured from a
run. Its anchors are chosen so the early values are exactly checkable without a
calculator: bars 1-5 trend up with no down-move at all, so -DM stays zero, DI- stays zero,
and DX is pinned at 100 by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxagent.indicators import adx, ema

# --- EMA ----------------------------------------------------------------------


def test_ema_matches_a_hand_computed_recursion() -> None:
    """period=3 -> alpha = 2/4 = 0.5, seeded on mean(1, 2, 3) = 2.

    i=3: 0.5 * 4 + 0.5 * 2 = 3
    i=4: 0.5 * 5 + 0.5 * 3 = 4
    """
    result = ema(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_of_a_constant_series_is_that_constant() -> None:
    result = ema(pd.Series([7.0] * 6), 4)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3:].to_numpy() == pytest.approx([7.0] * 3)


def test_ema_period_one_is_the_series_itself() -> None:
    """alpha = 2/2 = 1, so every value is its own average — no smoothing, no warm-up."""
    prices = pd.Series([1.0, 4.0, 2.0])
    assert ema(prices, 1).to_numpy() == pytest.approx(prices.to_numpy())


def test_ema_seed_is_a_simple_average_not_the_first_price() -> None:
    """`ewm(adjust=False)` would report 100.0 at index 0; that is a price, not an average."""
    result = ema(pd.Series([100.0, 200.0, 300.0]), 3)

    assert np.isnan(result.iloc[0])
    assert result.iloc[2] == pytest.approx(200.0)


# --- ADX ----------------------------------------------------------------------
#
# Reference bars. period = 3, so DI smoothing seeds at index 3 and ADX at index 2*3-1 = 5.
#
#   i   high   low   close      up     down    +DM    -DM     TR
#   0   10.0    9.0    9.5       -        -      -      -      -
#   1   10.5    9.5   10.2     0.5     -0.5    0.5    0.0    1.0
#   2   11.2   10.1   11.0     0.7     -0.6    0.7    0.0    1.1
#   3   11.0   10.4   10.6    -0.2     -0.3    0.0    0.0    0.6
#   4   11.8   10.5   11.6     0.8     -0.1    0.8    0.0    1.3
#   5   12.4   11.3   12.1     0.6     -0.8    0.6    0.0    1.1
#   6   12.0   11.0   11.2    -0.4      0.3    0.0    0.3    1.1
#   7   11.5   10.2   10.5    -0.5      0.8    0.0    0.8    1.3
#   8   11.0   10.0   10.8    -0.5      0.2    0.0    0.2    1.0
#   9   11.9   10.6   11.7     0.9     -0.6    0.9    0.0    1.3

HIGH = pd.Series([10.0, 10.5, 11.2, 11.0, 11.8, 12.4, 12.0, 11.5, 11.0, 11.9])
LOW = pd.Series([9.0, 9.5, 10.1, 10.4, 10.5, 11.3, 11.0, 10.2, 10.0, 10.6])
CLOSE = pd.Series([9.5, 10.2, 11.0, 10.6, 11.6, 12.1, 11.2, 10.5, 10.8, 11.7])
PERIOD = 3


def test_adx_warm_up_ends_at_two_periods_minus_one() -> None:
    result = adx(HIGH, LOW, CLOSE, PERIOD)

    assert result.iloc[: 2 * PERIOD - 1].isna().all()
    assert result.notna().iloc[2 * PERIOD - 1 :].all()


def test_adx_is_pinned_at_100_while_no_down_move_has_occurred() -> None:
    """Bars 1-5 never make a lower low against a lower high, so -DM is zero throughout.

    DI- is then 0, and DX = 100 * |DI+ - 0| / (DI+ + 0) = 100 regardless of DI+'s size.
    The first ADX averages DX at bars 3, 4 and 5, so it is mean(100, 100, 100) = 100.
    """
    result = adx(HIGH, LOW, CLOSE, PERIOD)
    assert result.iloc[5] == pytest.approx(100.0, abs=1e-4)


def test_adx_reference_sequence_to_four_decimal_places() -> None:
    """Wilder's recursion continued by hand from the table above.

    Smoothed sums at bar 6 (running-sum form, seeded on bars 1-3 then decayed):
        sum(TR)  = 3.211111   sum(+DM) = 1.111111   sum(-DM) = 0.300000
        DI+ = 100 * 1.111111 / 3.211111 = 34.602076
        DI- = 100 * 0.300000 / 3.211111 =  9.342561
        DX  = 100 * 25.259516 / 43.944637 = 57.480315
        ADX = (100 * 2 + 57.480315) / 3 = 85.826772

    Each later bar repeats the same two steps, giving:
    """
    expected = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        100.0,
        85.826772,
        62.182387,
        50.589831,
        45.743558,
    ]
    result = adx(HIGH, LOW, CLOSE, PERIOD).to_numpy()

    np.testing.assert_allclose(result, expected, atol=1e-4, equal_nan=True)


def test_adx_of_a_flat_market_is_zero_not_undefined() -> None:
    """No range and no direction. 0/0 must resolve to "no trend", never to NaN or inf."""
    flat = pd.Series([1.2345] * 12)
    result = adx(flat, flat, flat, 3)

    assert result.iloc[: 2 * 3 - 1].isna().all()
    assert result.iloc[2 * 3 - 1 :].to_numpy() == pytest.approx([0.0] * 7)


def test_adx_saturates_at_100_in_an_unbroken_trend() -> None:
    """A staircase up: every bar a higher high and a higher low, so -DM is always zero."""
    rising = np.arange(20, dtype="float64")
    high = pd.Series(rising + 1.0)
    low = pd.Series(rising)
    close = pd.Series(rising + 0.5)

    result = adx(high, low, close, 4)
    assert result.dropna().to_numpy() == pytest.approx(100.0, abs=1e-9)


def test_adx_ignores_direction_and_reports_only_strength() -> None:
    """Mirroring the bars top-to-bottom flips DI+ and DI- but must not move ADX."""
    mirrored_high = -LOW
    mirrored_low = -HIGH
    mirrored_close = -CLOSE

    original = adx(HIGH, LOW, CLOSE, PERIOD).to_numpy()
    mirrored = adx(mirrored_high, mirrored_low, mirrored_close, PERIOD).to_numpy()

    np.testing.assert_allclose(original, mirrored, atol=1e-9, equal_nan=True)


def test_adx_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        adx(HIGH, LOW.iloc[:-1], CLOSE, PERIOD)
