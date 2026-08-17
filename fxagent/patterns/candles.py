"""The nine detectors, and the scan that runs them over a series.

Each detector is a pure function of at most two candles and a volatility scale. It returns the
numeric criteria that matched, or `None`. No detector reads a clock, a later bar, or anything
about the market other than the bars it was handed — so a scan replayed over stored bars finds
exactly what the live scan found, which is the same property the strategies have and for the
same reason.

**Overlaps are reported, not resolved.** A hammer is usually also a pin bar, and an outside bar
is often also an engulfing bar. Nothing here suppresses one in favour of another. They are
different published definitions of different things — a pin bar is a statement about the bar's
own proportions, a hammer is a statement about its shadow against its body, an engulfing bar is
about two bodies and an outside bar is about two ranges — and picking a winner would mean
inventing a precedence rule that no source states and that a reader could not check.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from fxagent.adapters.base import BarSeries
from fxagent.indicators import atr
from fxagent.patterns.base import (
    DEFAULT_CONFIG,
    Candle,
    PatternConfig,
    PatternHit,
    is_scaled,
)

__all__ = [
    "DETECTORS",
    "PATTERN_NAMES",
    "bearish_engulfing",
    "bullish_engulfing",
    "detect",
    "detect_at",
    "detect_latest",
    "doji",
    "hammer",
    "inside_bar",
    "marubozu",
    "outside_bar",
    "pin_bar",
    "shooting_star",
]

#: What every detector is: (this bar, the one before it or None, ATR here, thresholds) -> the
#: numbers that matched, or None. The name is attached by the registry, not by the function, so
#: a detector cannot report itself under a name the scan does not know about.
type Detector = Callable[[Candle, Candle | None, float, PatternConfig], dict[str, float] | None]


# -- single-bar formations -------------------------------------------------------------------


def doji(
    current: Candle, previous: Candle | None, scale: float, config: PatternConfig
) -> dict[str, float] | None:
    """Open and close within a whisker of each other, on a bar that actually moved."""
    if current.span < config.min_span_atr * scale:
        return None
    if current.body > config.doji_body_atr * scale:
        return None
    return {
        "body": current.body,
        "atr": scale,
        "body_in_atr": current.body / scale,
        "max_body_in_atr": config.doji_body_atr,
        "span_in_atr": current.span / scale,
    }


def marubozu(
    current: Candle, previous: Candle | None, scale: float, config: PatternConfig
) -> dict[str, float] | None:
    """Nearly all body: opens at one extreme, closes at the other."""
    if current.body < config.marubozu_body_atr * scale:
        return None
    shadow_cap = config.marubozu_shadow_atr * scale
    if current.upper_shadow > shadow_cap or current.lower_shadow > shadow_cap:
        return None
    return {
        "body": current.body,
        "atr": scale,
        "body_in_atr": current.body / scale,
        "min_body_in_atr": config.marubozu_body_atr,
        "upper_shadow_in_atr": current.upper_shadow / scale,
        "lower_shadow_in_atr": current.lower_shadow / scale,
        "max_shadow_in_atr": config.marubozu_shadow_atr,
    }


def _shadow_over_body(
    current: Candle,
    scale: float,
    config: PatternConfig,
    *,
    dominant: float,
    opposite: float,
) -> dict[str, float] | None:
    """Shared body of hammer and shooting star, which differ only in which shadow is which."""
    if current.span < config.min_span_atr * scale:
        return None
    if current.body < config.hammer_min_body_atr * scale:
        # A body this small makes the shadow-to-body ratio meaningless — it diverges as the body
        # goes to zero, so without this floor every doji with a wick is a hammer.
        return None
    if dominant < config.hammer_shadow_bodies * current.body:
        return None
    if opposite > config.hammer_opposite_shadow_atr * scale:
        return None
    return {
        "body": current.body,
        "atr": scale,
        "dominant_shadow": dominant,
        "dominant_shadow_in_bodies": dominant / current.body,
        "min_shadow_in_bodies": config.hammer_shadow_bodies,
        "opposite_shadow_in_atr": opposite / scale,
        "max_opposite_shadow_in_atr": config.hammer_opposite_shadow_atr,
    }


def hammer(
    current: Candle, previous: Candle | None, scale: float, config: PatternConfig
) -> dict[str, float] | None:
    """Long lower shadow, small body at the top, almost nothing above.

    Shape only — no check for a preceding downtrend. See `base` on why the trend context is
    deliberately somebody else's measurement.
    """
    return _shadow_over_body(
        current,
        scale,
        config,
        dominant=current.lower_shadow,
        opposite=current.upper_shadow,
    )


def shooting_star(
    current: Candle, previous: Candle | None, scale: float, config: PatternConfig
) -> dict[str, float] | None:
    """The hammer inverted: long upper shadow, small body at the bottom."""
    return _shadow_over_body(
        current,
        scale,
        config,
        dominant=current.upper_shadow,
        opposite=current.lower_shadow,
    )


def pin_bar(
    current: Candle, previous: Candle | None, scale: float, config: PatternConfig
) -> dict[str, float] | None:
    """One shadow is most of the bar.

    Measured against the bar's own span rather than against ATR, which is what makes this a
    different test from `hammer` rather than a looser one: a hammer asks how the shadow compares
    to the body, a pin bar asks how it compares to the whole bar. Both fire on many of the same
    bars and neither is suppressed — see the module docstring.
    """
    if current.span < config.min_span_atr * scale:
        return None
    dominant = max(current.upper_shadow, current.lower_shadow)
    fraction = dominant / current.span
    if fraction < config.pin_shadow_fraction:
        return None
    return {
        "atr": scale,
        "span": current.span,
        "upper_shadow": current.upper_shadow,
        "lower_shadow": current.lower_shadow,
        "dominant_shadow_fraction": fraction,
        "min_shadow_fraction": config.pin_shadow_fraction,
    }


# -- two-bar formations ----------------------------------------------------------------------


def _engulfing(
    current: Candle,
    previous: Candle | None,
    scale: float,
    config: PatternConfig,
    *,
    bullish: bool,
) -> dict[str, float] | None:
    if previous is None:
        return None
    if bullish and not (current.is_bullish and previous.is_bearish):
        return None
    if not bullish and not (current.is_bearish and previous.is_bullish):
        return None
    if current.body < config.engulf_min_body_atr * scale:
        return None

    # Bodies, not ranges. An engulfing bar is defined on open-to-close; a bar that covers the
    # previous *range* is an outside bar, which is a different formation with its own detector.
    top, bottom = max(current.open, current.close), min(current.open, current.close)
    prior_top, prior_bottom = max(previous.open, previous.close), min(previous.open, previous.close)
    if bottom > prior_bottom or top < prior_top:
        return None
    if bottom == prior_bottom and top == prior_top:
        # Identical bodies cover but do not engulf, and with opposite directions this is two
        # bars trading the same range rather than one overwhelming the other.
        return None

    return {
        "body": current.body,
        "previous_body": previous.body,
        "atr": scale,
        "body_in_atr": current.body / scale,
        "min_body_in_atr": config.engulf_min_body_atr,
        "body_over_previous": (current.body / previous.body) if previous.body > 0 else 0.0,
    }


def bullish_engulfing(
    current: Candle, previous: Candle | None, scale: float, config: PatternConfig
) -> dict[str, float] | None:
    """An up bar whose body covers the previous down bar's body."""
    return _engulfing(current, previous, scale, config, bullish=True)


def bearish_engulfing(
    current: Candle, previous: Candle | None, scale: float, config: PatternConfig
) -> dict[str, float] | None:
    """A down bar whose body covers the previous up bar's body."""
    return _engulfing(current, previous, scale, config, bullish=False)


def inside_bar(
    current: Candle, previous: Candle | None, scale: float, config: PatternConfig
) -> dict[str, float] | None:
    """The whole bar sits inside the previous one's high-low range.

    The previous bar has to have been a real bar: an inside bar within a doji is two quiet bars,
    not a compression.
    """
    if previous is None:
        return None
    if previous.span < config.min_span_atr * scale:
        return None
    if current.high > previous.high or current.low < previous.low:
        return None
    if current.high == previous.high and current.low == previous.low:
        return None  # identical range: contained, but nothing has narrowed
    return {
        "atr": scale,
        "span": current.span,
        "previous_span": previous.span,
        "span_over_previous": current.span / previous.span,
        "high_room": previous.high - current.high,
        "low_room": current.low - previous.low,
    }


def outside_bar(
    current: Candle, previous: Candle | None, scale: float, config: PatternConfig
) -> dict[str, float] | None:
    """The bar's range covers the previous one's, on both sides."""
    if previous is None:
        return None
    if current.span < config.min_span_atr * scale:
        return None
    if current.high < previous.high or current.low > previous.low:
        return None
    if current.high == previous.high and current.low == previous.low:
        return None
    return {
        "atr": scale,
        "span": current.span,
        "previous_span": previous.span,
        "span_over_previous": (current.span / previous.span) if previous.span > 0 else 0.0,
        "high_excess": current.high - previous.high,
        "low_excess": previous.low - current.low,
    }


#: The registry. Order fixes the order hits are reported in, so a panel and a stored row list
#: them the same way every time. Adding a formation is an entry here and a line in `DEFINITIONS`.
DETECTORS: tuple[tuple[str, Detector], ...] = (
    ("doji", doji),
    ("hammer", hammer),
    ("shooting_star", shooting_star),
    ("bullish_engulfing", bullish_engulfing),
    ("bearish_engulfing", bearish_engulfing),
    ("inside_bar", inside_bar),
    ("outside_bar", outside_bar),
    ("marubozu", marubozu),
    ("pin_bar", pin_bar),
)

PATTERN_NAMES: tuple[str, ...] = tuple(name for name, _ in DETECTORS)


# -- the scan --------------------------------------------------------------------------------


def _scale_series(bars: BarSeries, config: PatternConfig) -> pd.Series:
    frame = pd.DataFrame(
        {
            "high": [bar.high for bar in bars.bars],
            "low": [bar.low for bar in bars.bars],
            "close": [bar.close for bar in bars.bars],
        }
    )
    return atr(frame["high"], frame["low"], frame["close"], config.atr_period)


def _hits_at(
    bars: BarSeries,
    index: int,
    scales: Sequence[float],
    config: PatternConfig,
) -> tuple[PatternHit, ...]:
    if index == 0:
        # No prior bar, so no volatility to measure this one against. Nothing is reported
        # rather than the first bar being judged against itself.
        return ()

    # ATR *before* this bar, not including it. A bar must not set the yardstick it is measured
    # against: ATR through index `i` incorporates bar `i`'s own true range, so a large bar
    # inflates its own denominator and can suppress its own detection — a marubozu that is
    # marubozu-shaped precisely because it was large becomes too small a fraction of an ATR it
    # just raised. Using the volatility the market had going in also makes the scan strictly
    # backward-looking, which the point-in-time rule wants anyway.
    scale = float(scales[index - 1])
    if not is_scaled(scale):
        # Warm-up, or a market so flat ATR is zero. Reported as no detections rather than as
        # detections measured against a scale of nothing.
        return ()

    current = Candle.from_bar(bars.bars[index])
    previous = Candle.from_bar(bars.bars[index - 1]) if index > 0 else None

    found: list[PatternHit] = []
    for name, detector in DETECTORS:
        criteria = detector(current, previous, scale, config)
        if criteria is not None:
            found.append(
                PatternHit(
                    name=name,
                    bar_index=index,
                    timestamp=bars.bars[index].timestamp,
                    criteria={key: float(value) for key, value in criteria.items()},
                )
            )
    return tuple(found)


def detect_at(
    bars: BarSeries, index: int, *, config: PatternConfig = DEFAULT_CONFIG
) -> tuple[PatternHit, ...]:
    """Every formation present at `index`. Negative indices count from the end, as elsewhere."""
    if not bars.bars:
        return ()
    resolved = index if index >= 0 else len(bars.bars) + index
    if not 0 <= resolved < len(bars.bars):
        raise IndexError(f"bar index {index} is outside a series of {len(bars.bars)} bars")
    return _hits_at(bars, resolved, _scale_series(bars, config).tolist(), config)


def detect_latest(
    bars: BarSeries, *, config: PatternConfig = DEFAULT_CONFIG
) -> tuple[PatternHit, ...]:
    """What is present on the newest bar — the one an evaluation is about."""
    return detect_at(bars, -1, config=config)


def detect(
    bars: BarSeries, *, config: PatternConfig = DEFAULT_CONFIG, last: int | None = None
) -> tuple[PatternHit, ...]:
    """Scan the series, oldest hit first.

    `last` limits the scan to the final N bars, which is what a panel wants: the whole history
    of every formation ever printed is thousands of rows nobody reads. The volatility scale is
    still computed over the full series, so limiting the window never changes a verdict — only
    how many of them are returned.
    """
    if not bars.bars:
        return ()
    if last is not None and last < 0:
        raise ValueError(f"last must not be negative, got {last}")

    scales = _scale_series(bars, config).tolist()
    start = 0 if last is None else max(0, len(bars.bars) - last)
    return tuple(
        hit
        for index in range(start, len(bars.bars))
        for hit in _hits_at(bars, index, scales, config)
    )
