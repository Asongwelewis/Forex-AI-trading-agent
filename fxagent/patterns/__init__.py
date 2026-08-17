"""Candle formation detection. **CONTEXT ONLY — this package must never reach a decision.**

Two studies found candlestick formations produce no net positive return on EUR/USD after
costs. CLAUDE.md therefore lists them under known traps and the project structure marks this
package "CONTEXT ONLY — must not reach consensus.py". So:

* Every `PatternHit` carries `label = CONTEXT_ONLY` as a field, not as a caption something else
  applies. A renderer can hide a caption; it cannot hide a value it is printing.
* Nothing in `fxagent.regime` imports this package, and
  `tests/patterns/test_patterns_never_reach_consensus.py` walks the transitive import closure
  of `consensus.py` to keep it that way. An import test rather than a review note, because the
  rule is about what the decision path *can* reach, not what it happens to do today.
* `DEFINITIONS` describes each formation's shape and never what it is supposed to predict. A
  definition reading "signals a reversal" would re-label context as signal in the one place a
  reader would be most inclined to believe it.

The detectors are pure functions over OHLC with volatility-relative thresholds — see `base` for
why a fixed pip distance cannot mean the same thing on two pairs, and `candles` for why
overlapping formations are all reported rather than resolved into one.

The two consumers are the dashboard panel and the chartist agent's prompt. Neither can move a
number: hard rule 4 keeps every indicator, signal, size and stop in deterministic Python, and
an agent that reads a formation still only writes prose about it.
"""

from __future__ import annotations

from fxagent.patterns.base import (
    CONTEXT_ONLY,
    DEFAULT_CONFIG,
    DEFINITIONS,
    Candle,
    PatternConfig,
    PatternHit,
)
from fxagent.patterns.candles import (
    DETECTORS,
    PATTERN_NAMES,
    bearish_engulfing,
    bullish_engulfing,
    detect,
    detect_at,
    detect_latest,
    doji,
    hammer,
    inside_bar,
    marubozu,
    outside_bar,
    pin_bar,
    shooting_star,
)

__all__ = [
    "CONTEXT_ONLY",
    "DEFAULT_CONFIG",
    "DEFINITIONS",
    "DETECTORS",
    "PATTERN_NAMES",
    "Candle",
    "PatternConfig",
    "PatternHit",
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
