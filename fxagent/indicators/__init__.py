"""Deterministic indicators, hand-written against numpy and pandas only.

No TA library sits behind these. Every formula is written out in the module that owns it,
because CLAUDE.md hard rule 4 puts all indicator maths in Python we control and can audit
line by line — an LLM never computes any of it, and neither does an opaque dependency
whose smoothing conventions we would have to reverse-engineer to trust.

Every function here is pure, returns a `pd.Series` on the caller's index, reports its
warm-up as NaN rather than truncating, and never reads a value at index > i when
computing index i. See `_common` for what that guarantees.
"""

from __future__ import annotations

from fxagent.indicators.statistics import rolling_percentile, rolling_zscore
from fxagent.indicators.trend import adx, ema
from fxagent.indicators.volatility import atr, true_range

__all__ = [
    "adx",
    "atr",
    "ema",
    "rolling_percentile",
    "rolling_zscore",
    "true_range",
]
