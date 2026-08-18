"""Purged, embargoed walk-forward folds.

A plain chronological split is not enough when labels have duration. An observation made at
09:00 whose triple barrier resolves at 14:00 overlaps any test fold beginning before 14:00 —
so training on it is training on the answer. The overlap is invisible in a naive split because
the *decision* timestamp is safely in the past; it is the *label span* that reaches forward.

Two removals, and both are reported per fold:

**Purge.** Drop every training observation whose label span overlaps the test fold's span. This
is the one that matters and it is the one people skip, because without it nothing crashes and
the results merely improve.

**Embargo.** Drop training observations that begin shortly *after* the test fold ends. Serial
correlation does not stop at a fold boundary: the bar after the test window is still correlated
with the bars inside it, and a model trained on it has seen a smeared copy of the test set. One
percent of the sample is the conventional default and is what López de Prado uses.

**Zero purged means one of two very different things, and they must not be confused.**

If the observations have *overlapping* label spans — one decision per bar, each resolving hours
later, which is the López de Prado setting — then a fold that purges nothing is broken, and the
results it produces are inflated. That is the case `assert_purging_is_working` guards.

But a replay that runs one position at a time produces trades whose spans are **disjoint by
construction**: the next entry cannot precede the previous exit. Purging genuinely has nothing
to remove there, and zero is the correct answer rather than a symptom. The embargo is then the
only active mechanism, and it is doing real work, because serial correlation does not care that
no position was open.

So `labels_overlap` reports which regime the observations are in, and the guard and the report
both consult it instead of treating every zero as a defect. A warning that fires on every
correct run is a warning that gets ignored on the run that matters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Protocol

__all__ = [
    "DEFAULT_EMBARGO_FRACTION",
    "Fold",
    "Labelled",
    "assert_purging_is_working",
    "labels_overlap",
    "purged_walk_forward",
]

#: One percent of the sample, the conventional embargo. Expressed as a fraction of the number of
#: observations rather than as a duration, so it scales with the sample instead of with a guess
#: about how long correlation lasts on this particular pair.
DEFAULT_EMBARGO_FRACTION: Final = 0.01


class Labelled(Protocol):
    """What a fold needs from an observation: when it was decided, and when it resolved.

    A protocol rather than a base class, so a `ReplayTrade`, a stored `TradeRecord` and a test
    fixture can all be split by the same code without inheriting from anything.
    """

    @property
    def timestamp(self) -> datetime:
        """When the decision was made — the point-in-time moment."""

    @property
    def label_span_end(self) -> datetime:
        """When the outcome became known. The reason a plain split leaks."""


@dataclass(frozen=True)
class Fold:
    """One train/test split, with an account of everything it removed and why.

    `train` and `test` are positions into the original sequence, not copies of it, so a caller
    can carry whatever payload it likes on the observations without this module knowing about
    them.
    """

    index: int
    train: tuple[int, ...]
    test: tuple[int, ...]
    purged: int
    embargoed: int
    test_start: datetime
    test_end: datetime
    embargo_until: datetime

    @property
    def dropped(self) -> int:
        return self.purged + self.embargoed

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {len(self.train)}, test {len(self.test)}, "
            f"purged {self.purged}, embargoed {self.embargoed} "
            f"({self.test_start:%Y-%m-%d} to {self.test_end:%Y-%m-%d})"
        )


def purged_walk_forward(
    observations: Sequence[Labelled],
    *,
    folds: int = 5,
    embargo_fraction: float = DEFAULT_EMBARGO_FRACTION,
) -> list[Fold]:
    """Split into `folds` contiguous test windows, purging and embargoing each training set.

    Walk-forward rather than shuffled k-fold: the test windows run in time order and training
    uses everything outside the window, which is the arrangement that matches how the system
    will actually be used. Observations must already be sorted by `timestamp`; they are checked
    rather than sorted here, because silently reordering a caller's sequence would detach the
    returned indices from the payload they were meant to index.
    """
    if folds < 2:
        raise ValueError(f"folds must be at least 2, got {folds}")
    if not 0.0 <= embargo_fraction < 1.0:
        raise ValueError(f"embargo_fraction must be in [0, 1), got {embargo_fraction}")

    total = len(observations)
    if total < folds:
        raise ValueError(f"{total} observations cannot be split into {folds} folds")

    times = [observation.timestamp for observation in observations]
    if any(later < earlier for earlier, later in zip(times, times[1:], strict=False)):
        raise ValueError("observations must be sorted by timestamp, oldest first")

    # Embargo is a count of observations converted to a duration, because the gap that matters
    # is wall-clock: a fold boundary landing on a Friday evening is followed by a weekend, and
    # embargoing "the next N observations" there would embargo the whole of Monday morning.
    span = times[-1] - times[0]
    embargo = (
        timedelta(seconds=span.total_seconds() * embargo_fraction) if total > 1 else timedelta(0)
    )

    boundaries = [round(index * total / folds) for index in range(folds + 1)]
    result: list[Fold] = []

    for fold_index in range(folds):
        start, stop = boundaries[fold_index], boundaries[fold_index + 1]
        if start == stop:
            continue
        test = tuple(range(start, stop))
        test_start = times[start]
        # The test fold's reach is its last label's end, not its last decision — a test
        # observation resolving after the window still consumed that future.
        test_end = max(observations[position].label_span_end for position in test)
        embargo_until = test_end + embargo

        train: list[int] = []
        purged = embargoed = 0
        for position in range(total):
            if start <= position < stop:
                continue
            observation = observations[position]
            overlaps = (
                observation.timestamp <= test_end and observation.label_span_end >= test_start
            )
            if overlaps:
                purged += 1
                continue
            if test_end < observation.timestamp <= embargo_until:
                embargoed += 1
                continue
            train.append(position)

        result.append(
            Fold(
                index=fold_index,
                train=tuple(train),
                test=test,
                purged=purged,
                embargoed=embargoed,
                test_start=test_start,
                test_end=test_end,
                embargo_until=embargo_until,
            )
        )

    return result


def labels_overlap(observations: Sequence[Labelled]) -> bool:
    """Whether any observation's label span reaches into a later observation's start.

    The question that decides whether a zero purge count is a defect or a fact. One decision per
    bar, each resolving hours later, overlaps heavily. Trades from a one-position-at-a-time
    replay never overlap, because the next entry cannot precede the previous exit.
    """
    for earlier, later in zip(observations, observations[1:], strict=False):
        if earlier.label_span_end > later.timestamp:
            return True
    return False


def assert_purging_is_working(
    folds: Sequence[Fold], observations: Sequence[Labelled] | None = None
) -> None:
    """Raise if purging removed nothing **and** the labels were overlapping enough to need it.

    Purging that quietly does nothing looks exactly like purging that works, except the results
    are better — so a zero on overlapping labels is a defect, not luck. Pass `observations` and
    the check distinguishes that from the legitimate zero a disjoint set produces; omit them and
    it assumes overlap, which is the conservative reading for the bar-level labels this was
    written for.
    """
    if not folds:
        raise ValueError("no folds to check")
    if sum(fold.purged for fold in folds) > 0:
        return
    if observations is not None and not labels_overlap(observations):
        return
    raise ValueError(
        "no fold purged a single observation, and the labels overlap so at least one should "
        "have. Check that label_span_end is genuinely later than timestamp and that the "
        "observations are sorted. An unpurged fold leaks the answer into the question."
    )
