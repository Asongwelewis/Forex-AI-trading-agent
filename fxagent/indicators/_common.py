"""Shared plumbing for the indicator layer.

Two rules bind every function in this package, and both live here:

**Warm-up is NaN, never a truncated series.** An indicator returns a Series on the
caller's index, with NaN wherever there was not yet enough history. Dropping the warm-up
instead would silently re-align a strategy's bars against the wrong timestamps.

**Nothing looks ahead.** The value at index `i` is computed from data at indices `<= i`
only. Every recursion here runs forward, and every window is trailing, so evaluating an
indicator on a prefix of a series reproduces that prefix exactly — which is what makes a
backtest's bar-by-bar reading match a live reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["nan_series", "require_aligned", "validate_period", "wilder_rma"]


def validate_period(period: int, *, minimum: int = 1) -> int:
    """Reject a nonsensical lookback.

    A bad `period` is a programming error and raises. A series *shorter* than the period
    is an ordinary data condition and never raises — it yields an all-NaN result.
    """
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)):
        raise TypeError(f"period must be an integer, got {type(period).__name__}")
    if int(period) < minimum:
        raise ValueError(f"period must be >= {minimum}, got {period}")
    return int(period)


def require_aligned(**columns: pd.Series) -> None:
    """Reject OHLC inputs that are not the same series sliced the same way."""
    names = list(columns)
    reference_name = names[0]
    reference = columns[reference_name]
    for name in names[1:]:
        other = columns[name]
        if len(other) != len(reference):
            raise ValueError(
                f"{name} has length {len(other)} but {reference_name} has "
                f"{len(reference)}; OHLC inputs must be aligned"
            )
        if not other.index.equals(reference.index):
            raise ValueError(f"{name} and {reference_name} do not share an index")


def nan_series(index: pd.Index, name: str) -> pd.Series:
    """An all-NaN float Series — the answer whenever there is not enough history."""
    return pd.Series(np.nan, index=index, dtype="float64", name=name)


def _seed_position(values: np.ndarray, period: int) -> int | None:
    """Index of the first output value: `period` observations after the first real one.

    Leading NaNs are skipped rather than poisoning the seed, which is what lets ATR and
    the directional-movement series start at index 1 (bar 0 has no previous close) while
    still producing their first smoothed value exactly `period` bars later.
    """
    real = np.flatnonzero(~np.isnan(values))
    if real.size == 0:
        return None
    position = int(real[0]) + period - 1
    return position if position < values.size else None


def wilder_rma(values: pd.Series, period: int, *, name: str) -> pd.Series:
    """Wilder's smoothing: seed with a simple average, then recurse forward.

        rma[period-1] = mean(values[0 : period])
        rma[i]        = (rma[i-1] * (period - 1) + values[i]) / period

    This is *not* an EMA with span=period. It is an EMA with alpha = 1/period, seeded on a
    simple average — the smoothing Wilder specified for ATR, DI, and ADX. Using pandas'
    `ewm(span=...)` here would shift every value.

    A NaN appearing after the seed propagates through the rest of the output, which is
    honest: the recursion has infinite memory, so a hole in the input really does
    contaminate everything downstream of it.
    """
    period = validate_period(period)
    raw = values.to_numpy(dtype="float64", copy=False)
    smoothed = np.full(raw.size, np.nan, dtype="float64")

    seed_at = _seed_position(raw, period)
    if seed_at is None:
        return pd.Series(smoothed, index=values.index, name=name)

    window_start = seed_at - period + 1
    smoothed[seed_at] = raw[window_start : seed_at + 1].mean()
    for i in range(seed_at + 1, raw.size):
        smoothed[i] = (smoothed[i - 1] * (period - 1) + raw[i]) / period

    return pd.Series(smoothed, index=values.index, name=name)
