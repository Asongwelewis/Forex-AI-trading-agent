"""Triple-barrier labelling: which of TARGET, STOP or TIME was touched first.

**Intrabar ambiguity resolves to STOP, always.** An H1 bar whose high reaches the target and
whose low reaches the stop touched both, and OHLC does not record which came first. Three
choices exist and only one is defensible:

* Assume TARGET, and every backtest is a fiction — this is the single most common way a
  strategy shows an edge it does not have.
* Split the difference, and the result is a number that describes no trade that could occur.
* Assume STOP, and the backtest understates the strategy by the ambiguity rate.

The third is the only one whose error has a known sign. It is also the only one that cannot be
tuned into a better result, which matters more.

**The ambiguity rate is a reported metric, not a footnote.** A high rate means the stop and
target are too close together for the bar interval — the labels are mostly guesses and the
strategy has not been measured, it has been approximated pessimistically. On H1 a rate above a
few percent says to widen the barriers or drop to M15, not to accept the number and move on.

**Every outcome carries its label span.** `label_span_start` to `label_span_end` is the window
this observation's outcome depended on, and it is what `folds.purged_walk_forward` removes from
a training set that overlaps a test fold. A label without a span cannot be purged, and an
unpurged fold leaks the answer into the question.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from fxagent.adapters.base import BarSeries, OrderSide

__all__ = ["Barrier", "BarrierOutcome", "resolve_barriers"]


class Barrier(StrEnum):
    """Which barrier ended the trade. `TIME` is a real outcome, not a missing one."""

    TARGET = "TARGET"
    STOP = "STOP"
    TIME = "TIME"


@dataclass(frozen=True)
class BarrierOutcome:
    """How one position resolved, and how much of that was inference rather than observation."""

    touched: Barrier
    exit_index: int
    exit_time: datetime
    exit_price: float
    ambiguous_resolution: bool
    label_span_start: datetime
    label_span_end: datetime
    bars_held: int

    @property
    def observed(self) -> bool:
        """Whether the bars actually settled this, or the pessimistic rule did."""
        return not self.ambiguous_resolution


def _touches(
    side: OrderSide, high: float, low: float, stop: float, target: float
) -> tuple[bool, bool]:
    """Whether this bar's range reached the stop and the target, in that order.

    Inclusive comparisons throughout: a low that touches the stop exactly is a stop-out. A
    broker's stop order triggers at the level, not a tick past it, and the strict version of
    this comparison is worth a few basis points of imaginary edge over a long backtest.
    """
    if side is OrderSide.BUY:
        return low <= stop, high >= target
    return high >= stop, low <= target


def resolve_barriers(
    bars: BarSeries,
    entry_index: int,
    *,
    side: OrderSide,
    stop_price: float,
    target_price: float,
    max_bars: int,
) -> BarrierOutcome:
    """Walk forward from the bar after entry until a barrier is touched or time runs out.

    Evaluation starts at `entry_index + 1`. The entry bar itself is excluded because the entry
    fills at that bar's close, so its high and low are already history at the moment the
    position opens — scanning them is a look-ahead of exactly one bar, and it is the kind that
    produces a strategy which appears to exit before it entered.

    `max_bars` is the time barrier and is mandatory. There is no such thing as a position with
    no horizon in a backtest: a trade that never touches either level would otherwise run to the
    end of the sample, and its label span would swallow every fold after it.
    """
    if max_bars < 1:
        raise ValueError(f"max_bars must be at least 1, got {max_bars}")
    if entry_index < 0 or entry_index >= len(bars):
        raise ValueError(f"entry_index {entry_index} is outside a series of {len(bars)} bars")
    if stop_price <= 0 or target_price <= 0:
        raise ValueError("stop_price and target_price must both be positive")

    entry_bar = bars.bars[entry_index]
    last_index = min(entry_index + max_bars, len(bars) - 1)

    for index in range(entry_index + 1, last_index + 1):
        bar = bars.bars[index]
        hit_stop, hit_target = _touches(side, bar.high, bar.low, stop_price, target_price)

        if hit_stop and hit_target:
            # Both inside one bar. The sequence is unknowable from OHLC, so assume the worse
            # one and say so — see the module docstring for why the other options are worse.
            return BarrierOutcome(
                touched=Barrier.STOP,
                exit_index=index,
                exit_time=bar.timestamp,
                exit_price=stop_price,
                ambiguous_resolution=True,
                label_span_start=entry_bar.timestamp,
                label_span_end=bar.timestamp,
                bars_held=index - entry_index,
            )

        if hit_stop or hit_target:
            touched = Barrier.STOP if hit_stop else Barrier.TARGET
            return BarrierOutcome(
                touched=touched,
                exit_index=index,
                exit_time=bar.timestamp,
                exit_price=stop_price if hit_stop else target_price,
                ambiguous_resolution=False,
                label_span_start=entry_bar.timestamp,
                label_span_end=bar.timestamp,
                bars_held=index - entry_index,
            )

    # Time barrier. Exits at the close of the last bar looked at, which is a real price at a
    # real moment — unlike the stop and target levels, this one was actually printed.
    final = bars.bars[last_index]
    return BarrierOutcome(
        touched=Barrier.TIME,
        exit_index=last_index,
        exit_time=final.timestamp,
        exit_price=final.close,
        ambiguous_resolution=False,
        label_span_start=entry_bar.timestamp,
        label_span_end=final.timestamp,
        bars_held=last_index - entry_index,
    )
