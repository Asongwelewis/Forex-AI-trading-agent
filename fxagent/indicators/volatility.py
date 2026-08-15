"""True range, average true range, and Bollinger bands.

Bar 0 has no previous close, so its true range is genuinely undefined and is reported as
NaN rather than degraded to `high - low`. That choice propagates: ATR's first value lands
at index `period`, not `period - 1`, because it averages the first `period` *defined*
true ranges. Folding a half-defined bar 0 into the seed would bias the first ATR low, and
Wilder's recursion would carry that bias forward forever.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fxagent.indicators._common import require_aligned, validate_period, wilder_rma

__all__ = ["BollingerBands", "atr", "bollinger_bands", "true_range"]


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


#: Bollinger's own specification, kept as a named constant so a chart overlay and a future
#: strategy cannot quietly disagree about what "the bands" means.
BOLLINGER_PERIOD = 20
BOLLINGER_DEVIATIONS = 2.0


@dataclass(frozen=True)
class BollingerBands:
    """Three aligned series sharing one index. Returned together because they are one object.

    Handing back a tuple of bare Series invites a caller to recompute the middle band with a
    different window than the one the envelope was drawn from, which is how an overlay ends up
    showing bands that do not belong to the mean printed between them.
    """

    upper: pd.Series
    middle: pd.Series
    lower: pd.Series


def bollinger_bands(
    close: pd.Series,
    period: int = BOLLINGER_PERIOD,
    deviations: float = BOLLINGER_DEVIATIONS,
) -> BollingerBands:
    """A simple moving average with a volatility envelope either side of it.

        middle[i] = mean(close[i - period + 1 : i + 1])
        upper[i]  = middle[i] + deviations * stdev(window)
        lower[i]  = middle[i] - deviations * stdev(window)

    NaN for the first `period - 1` values, on all three bands together.

    **The deviation is the population one (ddof=0), which is Bollinger's.** That differs on
    purpose from `rolling_zscore`, which uses the sample deviation (ddof=1) because it answers
    a different question — it estimates how unusual a value is against a population it has
    sampled, while a band is a description of the window it was drawn from and nothing wider.
    The two are within 3% of each other at period 20, which is precisely why the difference has
    to be stated: a discrepancy that size never announces itself, it just makes an overlay
    disagree very slightly with a strategy forever.
    """
    period = validate_period(period, minimum=2)
    if deviations <= 0.0:
        raise ValueError(f"deviations must be positive, got {deviations}")

    window = close.rolling(period)
    middle = window.mean()
    spread = deviations * window.std(ddof=0)

    return BollingerBands(
        upper=(middle + spread).rename(f"bb_upper_{period}"),
        middle=middle.rename(f"bb_middle_{period}"),
        lower=(middle - spread).rename(f"bb_lower_{period}"),
    )
