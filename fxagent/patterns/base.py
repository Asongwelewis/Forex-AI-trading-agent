"""What a candle formation is, what a detection reports, and the thresholds that decide.

**Every threshold here is a multiple of ATR at the bar being tested, never a price distance.**
A 10-pip body is a doji on EUR/USD in a dead August hour and an ordinary bar on GBP/JPY during
the New York open, so a fixed cut-off does not mean the same thing twice. Expressing the
thresholds against the local volatility is what makes one config work across every pair and
every session, and it is what makes a formation detected in 2019 comparable to one detected
today. `tests/patterns/test_thresholds_are_volatility_relative.py` scales an entire fixture by
ten and asserts the verdicts are unchanged; a fixed-pip detector fails it immediately.

**Shape only. No trend context, ever.** A textbook hammer requires a preceding downtrend, and
this module does not check for one. That is deliberate: `fxagent.regime.classifier` already
owns what "downtrend" means, and a second definition living here would be free to drift from
it — the exact defect `range_reversion` was fixed for, where a strategy measuring its own ADX
gave two answers to one question. A detector here says "this bar has this shape". Whether the
market around it was trending is a separate measurement a reader can put next to it.

**The scale is the volatility going in, not including the bar itself.** ATR through bar `i`
incorporates bar `i`'s own true range, so a large bar inflates the very denominator it is
measured against and can suppress its own detection. `candles` therefore scales bar `i` by ATR
at `i - 1`, which also makes the scan strictly backward-looking.

**Warm-up is reported as nothing, not as zero.** ATR is NaN until it has enough history, and a
bar with no volatility scale gets no detections rather than being tested against a scale of
zero — which would make every bar a marubozu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from fxagent.adapters.base import Bar, UtcDatetime

__all__ = [
    "CONTEXT_ONLY",
    "DEFAULT_CONFIG",
    "DEFINITIONS",
    "Candle",
    "PatternConfig",
    "PatternHit",
]

#: Stamped on every detection. Two studies found candlestick formations produce no net positive
#: return on EUR/USD after costs (CLAUDE.md, known traps), so these are display context and
#: nothing else. The label travels *on the data* rather than being applied by whatever renders
#: it, because a caption can be styled away and a field cannot.
CONTEXT_ONLY = "CONTEXT ONLY — NOT A SIGNAL"


@dataclass(frozen=True)
class Candle:
    """One bar reduced to the four measurements every formation is defined in terms of.

    A separate type from `Bar` because the detectors should read like their definitions —
    `lower_shadow >= 2 * body` — rather than recomputing `min(open, close) - low` nine times
    with nine chances to get one of them backwards.
    """

    open: float
    high: float
    low: float
    close: float

    @classmethod
    def from_bar(cls, bar: Bar) -> Candle:
        return cls(open=bar.open, high=bar.high, low=bar.low, close=bar.close)

    @property
    def body(self) -> float:
        """Absolute open-to-close distance. Unsigned; `is_bullish` carries the direction."""
        return abs(self.close - self.open)

    @property
    def span(self) -> float:
        """High-to-low. Named `span` rather than `range`, which is a builtin."""
        return self.high - self.low

    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass(frozen=True)
class PatternConfig:
    """Every threshold, as a multiple of ATR or a fraction of the bar's own span.

    The defaults follow the common published definitions where those are numeric (a hammer's
    shadow at twice the body) and are conservative where they are not (how small "small" is).
    They are a starting point for tuning the *rules*, not fitted parameters — CLAUDE.md is
    explicit that tuning parameters is how a system overfits, and these feed a display panel
    rather than a decision, so there is nothing here worth fitting.
    """

    #: Wilder period for the volatility scale. Matches `ClassifierConfig.atr_period` so the
    #: panel's formations and its regime line are measured against the same volatility.
    atr_period: int = 14

    #: A bar whose entire span is under this much ATR is noise, and no formation is reported on
    #: it. Without this the quietest bar of the week is the cleanest doji of the week.
    min_span_atr: float = 0.5

    #: Doji: body at or under this much ATR.
    doji_body_atr: float = 0.1

    #: Marubozu: both shadows at or under this much ATR...
    marubozu_shadow_atr: float = 0.05
    #: ...and a body of at least this much, so a flat bar with no wicks is not one.
    marubozu_body_atr: float = 0.8

    #: Hammer / shooting star: the dominant shadow at least this many multiples of the body...
    hammer_shadow_bodies: float = 2.0
    #: ...the opposite shadow no more than this much ATR...
    hammer_opposite_shadow_atr: float = 0.15
    #: ...and a real body, so a doji with one long wick is a pin bar and not a hammer.
    hammer_min_body_atr: float = 0.1

    #: Pin bar: one shadow is at least this fraction of the whole span. Defined against the
    #: bar's own span rather than against ATR because that is what the published definition
    #: says, and it is the reason a pin bar and a hammer are not the same test.
    pin_shadow_fraction: float = 0.66

    #: Engulfing: the engulfing body must itself be worth something, or a two-pip body swallows
    #: a one-pip body in a dead market and the panel reports a reversal.
    engulf_min_body_atr: float = 0.5

    def __post_init__(self) -> None:
        if self.atr_period < 1:
            raise ValueError(f"atr_period must be positive, got {self.atr_period}")
        if not 0.0 < self.pin_shadow_fraction < 1.0:
            raise ValueError(
                f"pin_shadow_fraction is a fraction of the bar's span and must sit in (0, 1), "
                f"got {self.pin_shadow_fraction}"
            )
        if self.hammer_shadow_bodies <= 1.0:
            raise ValueError(
                f"hammer_shadow_bodies must exceed 1, got {self.hammer_shadow_bodies}; at or "
                "below it the 'long shadow' is no longer than the body and every ordinary bar "
                "is a hammer"
            )
        if self.doji_body_atr >= self.marubozu_body_atr:
            raise ValueError(
                f"doji_body_atr ({self.doji_body_atr}) must sit below marubozu_body_atr "
                f"({self.marubozu_body_atr}); overlapping thresholds would let one bar be both "
                "a body-less doji and a body-only marubozu"
            )
        for name in (
            "min_span_atr",
            "doji_body_atr",
            "marubozu_shadow_atr",
            "marubozu_body_atr",
            "hammer_opposite_shadow_atr",
            "hammer_min_body_atr",
            "engulf_min_body_atr",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} is an ATR multiple and must not be negative, got {value}")


#: The default everywhere, named so "which call sites use stock thresholds" is one grep.
DEFAULT_CONFIG = PatternConfig()


class PatternHit(BaseModel):
    """One formation found at one bar, with the numbers that made it one.

    `criteria` is the whole point. A detection that says only "hammer" is an assertion a reader
    has to take on trust; one that reports the shadow, the body, the ATR it was measured
    against and the threshold it cleared can be checked against the chart it is drawn on. It is
    also what makes a stored detection re-interpretable later under different thresholds
    without re-running the detector over the bars.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    #: Position in the series that was scanned, not a database id.
    bar_index: int = Field(ge=0)
    timestamp: UtcDatetime
    #: Scalars only, so a hit is JSON-safe and reaches the journal without a custom encoder.
    criteria: dict[str, float] = Field(default_factory=dict)
    label: str = CONTEXT_ONLY

    @property
    def definition(self) -> str:
        return DEFINITIONS.get(self.name, "")


#: One sentence per formation, for the panel and for the chartist's prompt. Plain description of
#: the shape — never what it is supposed to predict, because the evidence says it predicts
#: nothing and a definition that smuggled in "signals a reversal" would re-label context as
#: signal in the one place a reader is most likely to believe it.
DEFINITIONS: dict[str, str] = {
    "doji": "Open and close are effectively the same price, so the bar has almost no body.",
    "hammer": (
        "A small body at the top of the bar with a lower shadow several times its length, and "
        "almost nothing above."
    ),
    "shooting_star": (
        "A small body at the bottom of the bar with an upper shadow several times its length, "
        "and almost nothing below."
    ),
    "bullish_engulfing": ("An up bar whose body completely covers the previous down bar's body."),
    "bearish_engulfing": ("A down bar whose body completely covers the previous up bar's body."),
    "inside_bar": "The whole bar, high to low, sits within the previous bar's range.",
    "outside_bar": "The bar's high is above and its low below the previous bar's, covering it.",
    "marubozu": "A bar that is nearly all body: it opens at one extreme and closes at the other.",
    "pin_bar": "One shadow makes up most of the bar's total range, with the body at the far end.",
}


def is_scaled(value: float) -> bool:
    """True when a volatility scale is usable — finite and above zero.

    Guards the warm-up NaN and the flat-market zero in one place. A zero scale would make every
    threshold zero and every bar every formation at once.
    """
    return math.isfinite(value) and value > 0.0
