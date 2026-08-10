"""True range and average true range.

Bar 0 has no previous close, so its true range is genuinely undefined and is reported as
NaN rather than degraded to `high - low`. That choice propagates: ATR's first value lands
at index `period`, not `period - 1`, because it averages the first `period` *defined*
true ranges. Folding a half-defined bar 0 into the seed would bias the first ATR low, and
Wilder's recursion would carry that bias forward forever.
"""

from __future__ import annotations

import pandas as pd

from fxagent.indicators._common import require_aligned, validate_period, wilder_rma

__all__ = ["atr", "true_range"]


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: the widest of the current bar and the two overnight gaps.

        TR[i] = max(high[i] - low[i], |high[i] - close[i-1]|, |low[i] - close[i-1]|)

    Returns NaN at index 0, where `close[i-1]` does not exist.
    """
    require_aligned(high=high, low=low, close=close)

    previous_close = close.shift(1)
    candidates = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    )
    # skipna=False keeps bar 0 undefined instead of quietly falling back to high - low.
    return candidates.max(axis=1, skipna=False).rename("true_range")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average true range, smoothed the way Wilder specified.

    NaN until index `period`, where the first value is the simple mean of TR[1..period].
    A series shorter than that is all NaN — short input is a data condition, not an error.
    """
    period = validate_period(period)
    ranges = true_range(high, low, close)
    return wilder_rma(ranges, period, name=f"atr_{period}")
