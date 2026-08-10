"""Exponential moving average and the average directional index.

ADX is the fiddly one, so the pipeline is spelled out rather than fused:

    +DM / -DM  ->  Wilder-smoothed alongside TR  ->  DI+ / DI-  ->  DX  ->  ADX

Two conventions matter and are both Wilder's. Directional movement is *exclusive*: on any
given bar at most one of +DM and -DM is non-zero, because an inside bar and an ambiguous
bar both count as no directional movement. And DI+ / DI- are ratios of two quantities
smoothed identically, so smoothing as running averages gives exactly the same DI as
Wilder's running sums — the `period` factor cancels.

ADX therefore needs `2 * period` bars before it says anything: `period` to seed the DI
smoothing, then another `period` of DX values to average into the first ADX.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxagent.indicators._common import (
    nan_series,
    require_aligned,
    validate_period,
    wilder_rma,
)
from fxagent.indicators.volatility import true_range

__all__ = ["adx", "ema"]


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, alpha = 2 / (period + 1), seeded on a simple average.

        ema[period-1] = mean(series[0 : period])
        ema[i]        = alpha * series[i] + (1 - alpha) * ema[i-1]

    NaN for the first `period - 1` values. Seeding on the simple average rather than on
    `series[0]` is what makes the warm-up honest: `ewm(adjust=False)` would emit a value
    at index 0 that is just the first price wearing an indicator's name.
    """
    period = validate_period(period)
    name = f"ema_{period}"

    raw = series.to_numpy(dtype="float64", copy=False)
    smoothed = np.full(raw.size, np.nan, dtype="float64")

    real = np.flatnonzero(~np.isnan(raw))
    if real.size == 0:
        return nan_series(series.index, name)
    seed_at = int(real[0]) + period - 1
    if seed_at >= raw.size:
        return nan_series(series.index, name)

    alpha = 2.0 / (period + 1.0)
    smoothed[seed_at] = raw[seed_at - period + 1 : seed_at + 1].mean()
    for i in range(seed_at + 1, raw.size):
        smoothed[i] = alpha * raw[i] + (1.0 - alpha) * smoothed[i - 1]

    return pd.Series(smoothed, index=series.index, name=name)


def _directional_movement(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Wilder's +DM and -DM. At most one is non-zero on any bar; index 0 is NaN.

    The comparison is strict on both sides, so an inside bar (neither extreme extends) and
    an exactly-symmetric outside bar both register as no directional movement.
    """
    up_move = high.diff().to_numpy(dtype="float64", copy=False)
    down_move = (-low.diff()).to_numpy(dtype="float64", copy=False)

    plus = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    # Bar 0 has no predecessor; NaN comparisons above collapsed to 0.0, so restore it.
    undefined = np.isnan(up_move) | np.isnan(down_move)
    plus[undefined] = np.nan
    minus[undefined] = np.nan

    return (
        pd.Series(plus, index=high.index, name="plus_dm"),
        pd.Series(minus, index=high.index, name="minus_dm"),
    )


def _directional_index(
    smoothed_plus: np.ndarray, smoothed_minus: np.ndarray, smoothed_range: np.ndarray
) -> np.ndarray:
    """DX from the smoothed components, keeping warm-up NaN distinct from a real zero.

    Two degenerate cases are decided here rather than left to produce inf or a spurious
    NaN. A zero smoothed true range means the market did not move at all, so both DI are
    zero; and DI+ == DI- == 0 means no net direction, so DX is 0 — an untrended market,
    which is exactly what ADX exists to report.
    """
    size = smoothed_range.size
    di_plus = np.full(size, np.nan, dtype="float64")
    di_minus = np.full(size, np.nan, dtype="float64")

    known = ~np.isnan(smoothed_range)
    moving = known & (smoothed_range > 0.0)
    di_plus[moving] = 100.0 * smoothed_plus[moving] / smoothed_range[moving]
    di_minus[moving] = 100.0 * smoothed_minus[moving] / smoothed_range[moving]

    motionless = known & (smoothed_range == 0.0)
    di_plus[motionless] = 0.0
    di_minus[motionless] = 0.0

    total = di_plus + di_minus
    dx = np.full(size, np.nan, dtype="float64")
    directional = ~np.isnan(total) & (total > 0.0)
    dx[directional] = (
        100.0 * np.abs(di_plus[directional] - di_minus[directional]) / total[directional]
    )
    dx[~np.isnan(total) & (total == 0.0)] = 0.0
    return dx


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average directional index — trend *strength*, indifferent to direction.

    NaN until index `2 * period - 1`, where the first value is the simple mean of the
    first `period` DX values. After that it follows Wilder's recursion. A series shorter
    than that is all NaN rather than an error.
    """
    period = validate_period(period)
    require_aligned(high=high, low=low, close=close)
    name = f"adx_{period}"

    plus_dm, minus_dm = _directional_movement(high, low)
    ranges = true_range(high, low, close)

    smoothed_plus = wilder_rma(plus_dm, period, name="s_plus_dm").to_numpy()
    smoothed_minus = wilder_rma(minus_dm, period, name="s_minus_dm").to_numpy()
    smoothed_range = wilder_rma(ranges, period, name="s_true_range").to_numpy()

    dx = _directional_index(smoothed_plus, smoothed_minus, smoothed_range)

    # DX's own warm-up positions the ADX seed at 2 * period - 1 without further arithmetic:
    # wilder_rma skips DX's leading NaNs and then counts `period` values from the first.
    return wilder_rma(pd.Series(dx, index=high.index), period, name=name)
