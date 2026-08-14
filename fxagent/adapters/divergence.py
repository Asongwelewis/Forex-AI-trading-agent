"""Comparing two feeds' bars, so "our data is right" is a measurement rather than a hope.

Kept in the package rather than in the smoke-test script because the arithmetic is the part
that can be quietly wrong, and logic that lives in a script only gets exercised when someone
runs the script by hand.

Two feeds agreeing is the cheapest evidence either is right. What counts as agreement depends
on whose feeds they are:

* **Same broker** (the local MT5 terminal and a cloud terminal on the same account) should
  agree to well under a pip. A visible difference there is an adapter converting something
  wrongly, not the market moving.
* **Independent feeds** must differ. An aggregated feed quotes mid where a broker quotes its
  own bid, so a spread-width gap is expected — and is exactly what makes the comparison a
  useful sanity check rather than a tautology.
* **No shared timestamps at all** is not a price disagreement. It is almost always a timezone
  bug in one adapter, and reporting it as a huge divergence hides that.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime

from fxagent.adapters.base import BarSeries

__all__ = ["Divergence", "compare_series", "interpret"]

#: Feeds reading the same broker's book. Held to a much tighter standard than independent ones.
SAME_BROKER_PAIRS = frozenset({frozenset({"mt5", "metaapi"})})

#: Above this, two feeds on the same broker disagree by more than rounding.
SAME_BROKER_TOLERANCE_PIPS = 1.0

#: Above this, even independent feeds differ by more than any plausible spread.
INDEPENDENT_TOLERANCE_PIPS = 50.0


@dataclass(frozen=True)
class Divergence:
    """How far apart two feeds are across the bars they share."""

    left: str
    right: str
    overlapping: int
    mean_abs_pips: float
    max_abs_pips: float
    worst_at: datetime | None = None

    def render(self) -> str:
        when = f"{self.worst_at:%Y-%m-%d %H:%M} UTC" if self.worst_at else "n/a"
        return (
            f"\n{self.left} vs {self.right}\n"
            f"  overlapping bars {self.overlapping}\n"
            f"  mean |diff|      {self.mean_abs_pips:.2f} pips\n"
            f"  max  |diff|      {self.max_abs_pips:.2f} pips at {when}\n"
        )


def pip_size(price: float) -> float:
    """JPY pairs move in 0.01; everything else in 0.0001."""
    return 0.01 if price > 20 else 0.0001


def compare_series(
    left_name: str, left: BarSeries, right_name: str, right: BarSeries
) -> Divergence:
    """Compare closes on the bars both feeds carry.

    Aligned on timestamp, never by position. The two feeds rarely start at the same bar, and
    an index-to-index comparison would report a large divergence for what is only an offset —
    turning a trivial alignment difference into an alarming number.
    """
    right_by_time = {bar.timestamp: bar for bar in right.bars}
    pairs = [
        (bar, right_by_time[bar.timestamp]) for bar in left.bars if bar.timestamp in right_by_time
    ]

    if not pairs:
        return Divergence(left_name, right_name, 0, 0.0, 0.0, None)

    pip = pip_size(left.bars[-1].close)
    diffs = [(abs(a.close - b.close) / pip, a.timestamp) for a, b in pairs]
    worst_value, worst_time = max(diffs, key=lambda item: item[0])

    return Divergence(
        left=left_name,
        right=right_name,
        overlapping=len(pairs),
        mean_abs_pips=statistics.fmean(value for value, _ in diffs),
        max_abs_pips=worst_value,
        worst_at=worst_time,
    )


def interpret(divergence: Divergence) -> str:
    """A sentence saying whether this divergence is normal, and what it means if not."""
    same_broker = frozenset({divergence.left, divergence.right}) in SAME_BROKER_PAIRS

    if divergence.overlapping == 0:
        return (
            "  NO OVERLAP — the two feeds share no bar timestamps. Almost always a timezone "
            "bug in one adapter rather than a real data difference."
        )
    if same_broker and divergence.max_abs_pips > SAME_BROKER_TOLERANCE_PIPS:
        return (
            "  UNEXPECTED — these read the same broker's book and should agree to well under "
            "a pip. Suspect a conversion error, not the market."
        )
    if not same_broker and divergence.max_abs_pips > INDEPENDENT_TOLERANCE_PIPS:
        return (
            "  SUSPICIOUS — larger than any plausible spread. Check bar alignment before "
            "trusting either series."
        )
    return "  within expectations for these feeds"
