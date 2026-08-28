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

from fxagent.adapters.base import BarSeries, OrderSide

__all__ = ["Divergence", "compare_series", "gap_filler_verdict", "interpret"]

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
    #: Mean absolute delta for open/high/low/close, in that order. Empty for old callers that
    #: construct a Divergence directly rather than through :func:`compare_series`.
    ohlc_mean_abs_pips: tuple[float, ...] = ()
    #: 95th percentile absolute delta for open/high/low/close, in that order.
    ohlc_p95_abs_pips: tuple[float, ...] = ()
    #: Fraction of overlapping bars where the configured barrier touch differs between feeds.
    barrier_touch_flip_share: float | None = None

    def render(self) -> str:
        when = f"{self.worst_at:%Y-%m-%d %H:%M} UTC" if self.worst_at else "n/a"
        return (
            f"\n{self.left} vs {self.right}\n"
            f"  overlapping bars {self.overlapping}\n"
            f"  mean |diff|      {self.mean_abs_pips:.2f} pips\n"
            f"  max  |diff|      {self.max_abs_pips:.2f} pips at {when}\n" + self._render_ohlc()
        )

    def _render_ohlc(self) -> str:
        if len(self.ohlc_mean_abs_pips) != 4:
            return ""
        labels = ("open", "high", "low", "close")
        means = ", ".join(
            f"{label} {value:.2f}"
            for label, value in zip(labels, self.ohlc_mean_abs_pips, strict=True)
        )
        p95 = ", ".join(
            f"{label} {value:.2f}"
            for label, value in zip(labels, self.ohlc_p95_abs_pips, strict=True)
        )
        flips = (
            f"\n  barrier-touch flips {self.barrier_touch_flip_share:.2%} of overlap"
            if self.barrier_touch_flip_share is not None
            else ""
        )
        return f"  mean OHLC |diff|  {means} pips\n  p95  OHLC |diff|  {p95} pips" + flips + "\n"


def pip_size(price: float) -> float:
    """JPY pairs move in 0.01; everything else in 0.0001."""
    return 0.01 if price > 20 else 0.0001


def compare_series(
    left_name: str,
    left: BarSeries,
    right_name: str,
    right: BarSeries,
    *,
    barrier_pips: tuple[float, float] | None = None,
    side: OrderSide = OrderSide.BUY,
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
    fields = ("open", "high", "low", "close")
    deltas = [
        [abs(getattr(a, field) - getattr(b, field)) / pip for a, b in pairs] for field in fields
    ]
    diffs = list(zip(deltas[3], (a.timestamp for a, _ in pairs), strict=True))
    worst_value, worst_time = max(diffs, key=lambda item: item[0])

    flip_share = None
    if barrier_pips is not None:
        stop_pips, target_pips = barrier_pips
        if stop_pips <= 0 or target_pips <= 0:
            raise ValueError("barrier_pips must contain positive stop and target distances")
        flips = sum(
            _barrier_touches(a, side, stop_pips, target_pips, pip)
            != _barrier_touches(b, side, stop_pips, target_pips, pip)
            for a, b in pairs
        )
        flip_share = flips / len(pairs)

    return Divergence(
        left=left_name,
        right=right_name,
        overlapping=len(pairs),
        mean_abs_pips=statistics.fmean(value for value, _ in diffs),
        max_abs_pips=worst_value,
        worst_at=worst_time,
        ohlc_mean_abs_pips=tuple(statistics.fmean(values) for values in deltas),
        ohlc_p95_abs_pips=tuple(_percentile95(values) for values in deltas),
        barrier_touch_flip_share=flip_share,
    )


def _percentile95(values: list[float]) -> float:
    """Small-sample inclusive p95 without a NumPy dependency in the smoke path."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * 0.95
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _barrier_touches(
    bar,
    side: OrderSide,
    stop_pips: float,
    target_pips: float,
    pip: float,
) -> tuple[bool, bool]:
    stop_distance = stop_pips * pip
    target_distance = target_pips * pip
    if side is OrderSide.BUY:
        return bar.low <= bar.open - stop_distance, bar.high >= bar.open + target_distance
    return bar.high >= bar.open + stop_distance, bar.low <= bar.open - target_distance


def gap_filler_verdict(divergence: Divergence) -> str:
    """Return the explicit TwelveData-as-gap-filler decision for a comparison report."""
    if divergence.overlapping == 0:
        return "UNUSABLE: no overlapping timestamps"
    if (
        divergence.barrier_touch_flip_share is not None
        and divergence.barrier_touch_flip_share > 0.01
    ):
        return "UNUSABLE: barrier-touch disagreement exceeds 1%"
    if divergence.max_abs_pips > INDEPENDENT_TOLERANCE_PIPS:
        return "UNUSABLE: OHLC divergence exceeds plausible spread"
    return "USABLE AS GAP-FILLER: retain source tags and cross-check overlaps"


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
