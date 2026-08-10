"""Rolling distribution statistics: where the current value sits in its recent history.

Both functions use a *trailing* window that includes the current bar, so the value at
index `i` is drawn from `series[i - period + 1 : i + 1]` and nothing later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxagent.indicators._common import validate_period

__all__ = ["rolling_percentile", "rolling_zscore"]


def rolling_zscore(series: pd.Series, period: int) -> pd.Series:
    """How many trailing standard deviations the current value sits from its mean.

        z[i] = (series[i] - mean(window)) / stdev(window)

    Uses the sample standard deviation (ddof=1), matching pandas' rolling default, so
    `period` must be at least 2. A window with zero variance yields NaN rather than an
    infinity: a flat window has no scale, so "how many deviations out" has no answer.
    """
    period = validate_period(period, minimum=2)

    window = series.rolling(period)
    deviation = window.std(ddof=1)
    # `where` maps both a zero and a NaN deviation to NaN, so no division ever warns.
    return ((series - window.mean()) / deviation.where(deviation > 0)).rename(f"zscore_{period}")


def _percentile_rank(window: np.ndarray) -> float:
    """Share of the trailing window at or below its final (current) value, as 0-100."""
    return 100.0 * float(np.count_nonzero(window <= window[-1])) / float(window.size)


def rolling_percentile(series: pd.Series, period: int) -> pd.Series:
    """Percentile rank of the current value within its trailing window, on a 0-100 scale.

    Defined as the share of the window less than or equal to the current value, the
    current value included. The window maximum therefore scores 100 and the minimum
    scores `100 / period`, never 0 — a value cannot be below itself. Ties share the
    higher rank.

    Any NaN inside the window makes that window incomplete, so the result is NaN there.
    """
    period = validate_period(period)
    ranked = series.rolling(period).apply(_percentile_rank, raw=True)
    return ranked.rename(f"pctile_{period}")
