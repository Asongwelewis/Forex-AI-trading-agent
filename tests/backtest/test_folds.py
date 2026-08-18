"""Purged and embargoed walk-forward splits.

The tests that matter are the two negative ones: `test_an_overlapping_label_is_purged` proves
the removal happens, and `test_purging_that_removes_nothing_is_reported_as_broken` proves that a
silently-zero purge is treated as a defect. Without the second, a bug that made purging a no-op
would look exactly like clean data — with better results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from fxagent.backtest.folds import (
    DEFAULT_EMBARGO_FRACTION,
    Fold,
    assert_purging_is_working,
    labels_overlap,
    purged_walk_forward,
)

START = datetime(2026, 3, 2, tzinfo=UTC)


@dataclass(frozen=True)
class Observation:
    """The minimum a fold needs: when it was decided, and when it resolved."""

    timestamp: datetime
    label_span_end: datetime


def observations(count: int, *, label_hours: int = 4, step_hours: int = 1) -> list[Observation]:
    """`count` decisions an hour apart, each resolving `label_hours` later.

    Overlapping by construction, which is what barrier labels on hourly bars actually look like
    and what makes purging non-trivial.
    """
    return [
        Observation(
            timestamp=START + timedelta(hours=index * step_hours),
            label_span_end=START + timedelta(hours=index * step_hours + label_hours),
        )
        for index in range(count)
    ]


class TestSplitting:
    def test_test_windows_are_contiguous_and_cover_everything(self) -> None:
        folds = purged_walk_forward(observations(100), folds=5)
        covered = [position for fold in folds for position in fold.test]
        assert covered == list(range(100))

    def test_test_windows_run_in_time_order(self) -> None:
        folds = purged_walk_forward(observations(100), folds=5)
        starts = [fold.test_start for fold in folds]
        assert starts == sorted(starts)

    def test_training_never_includes_a_test_observation(self) -> None:
        for fold in purged_walk_forward(observations(100), folds=5):
            assert not set(fold.train) & set(fold.test)

    def test_unsorted_observations_are_refused_rather_than_reordered(self) -> None:
        """Sorting here would detach the returned indices from the caller's payload."""
        jumbled = list(reversed(observations(20)))
        with pytest.raises(ValueError, match="sorted by timestamp"):
            purged_walk_forward(jumbled, folds=4)

    def test_too_few_observations_for_the_folds_asked_for(self) -> None:
        with pytest.raises(ValueError, match="cannot be split"):
            purged_walk_forward(observations(3), folds=5)

    def test_a_single_fold_is_not_a_split(self) -> None:
        with pytest.raises(ValueError, match="folds must be at least 2"):
            purged_walk_forward(observations(20), folds=1)


class TestPurging:
    def test_an_overlapping_label_is_purged(self) -> None:
        """A decision before the test window whose outcome lands inside it is training on it."""
        folds = purged_walk_forward(observations(100, label_hours=4), folds=5)
        assert all(fold.purged > 0 for fold in folds)

    def test_longer_labels_purge_more(self) -> None:
        """The removal scales with how far each label reaches, which is the whole mechanism."""
        short = purged_walk_forward(observations(100, label_hours=2), folds=5)
        long = purged_walk_forward(observations(100, label_hours=20), folds=5)
        assert sum(f.purged for f in long) > sum(f.purged for f in short)

    def test_no_surviving_training_label_overlaps_its_test_window(self) -> None:
        """Stated as the invariant rather than as a count — this is what purging is for."""
        data = observations(120, label_hours=6)
        for fold in purged_walk_forward(data, folds=4):
            for position in fold.train:
                survivor = data[position]
                overlaps = (
                    survivor.timestamp <= fold.test_end
                    and survivor.label_span_end >= fold.test_start
                )
                assert not overlaps

    def test_instantaneous_labels_need_no_purging(self) -> None:
        """The one case where zero is legitimate — and it is not what barrier labels produce."""
        instant = [
            Observation(START + timedelta(days=index), START + timedelta(days=index))
            for index in range(60)
        ]
        folds = purged_walk_forward(instant, folds=5, embargo_fraction=0.0)
        assert sum(fold.purged for fold in folds) == 0


class TestEmbargo:
    def test_observations_just_after_the_test_window_are_embargoed(self) -> None:
        folds = purged_walk_forward(observations(400), folds=4, embargo_fraction=0.05)
        assert sum(fold.embargoed for fold in folds) > 0

    def test_a_zero_embargo_removes_nothing_extra(self) -> None:
        folds = purged_walk_forward(observations(200), folds=4, embargo_fraction=0.0)
        assert sum(fold.embargoed for fold in folds) == 0

    def test_a_larger_embargo_removes_more(self) -> None:
        small = purged_walk_forward(observations(400), folds=4, embargo_fraction=0.01)
        large = purged_walk_forward(observations(400), folds=4, embargo_fraction=0.10)
        assert sum(f.embargoed for f in large) > sum(f.embargoed for f in small)

    def test_the_default_is_one_percent(self) -> None:
        assert DEFAULT_EMBARGO_FRACTION == 0.01

    def test_an_embargo_of_one_would_swallow_the_sample(self) -> None:
        with pytest.raises(ValueError, match="embargo_fraction must be in"):
            purged_walk_forward(observations(50), folds=5, embargo_fraction=1.0)


class TestTheZeroPurgeGuard:
    def test_purging_that_removes_nothing_is_reported_as_broken(self) -> None:
        """A no-op purge looks like clean data, except the results are better."""
        instant = [
            Observation(START + timedelta(days=index), START + timedelta(days=index))
            for index in range(60)
        ]
        folds = purged_walk_forward(instant, folds=5, embargo_fraction=0.0)
        with pytest.raises(ValueError, match="no fold purged a single observation"):
            assert_purging_is_working(folds)

    def test_real_barrier_labels_pass_the_guard(self) -> None:
        assert_purging_is_working(purged_walk_forward(observations(100), folds=5))

    def test_an_empty_fold_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no folds to check"):
            assert_purging_is_working([])


class TestReporting:
    def test_each_fold_reports_what_it_dropped(self) -> None:
        """Zero means purging is not working, so the count has to be visible per fold."""
        for fold in purged_walk_forward(observations(200), folds=5, embargo_fraction=0.05):
            assert fold.dropped == fold.purged + fold.embargoed
            assert "purged" in fold.describe()
            assert "embargoed" in fold.describe()

    def test_a_folds_reach_is_its_last_label_not_its_last_decision(self) -> None:
        """A test observation resolving after its window still consumed that future."""
        data = observations(100, label_hours=6)
        fold = purged_walk_forward(data, folds=5)[0]
        assert fold.test_end == max(data[p].label_span_end for p in fold.test)
        assert fold.test_end > data[fold.test[-1]].timestamp


class TestOverlapDetection:
    """A zero purge count means two different things, and conflating them wastes the warning."""

    def test_bar_level_labels_overlap(self) -> None:
        assert labels_overlap(observations(50, label_hours=4))

    def test_one_position_at_a_time_produces_disjoint_labels(self) -> None:
        """The next entry cannot precede the previous exit, so nothing overlaps."""
        disjoint = [
            Observation(
                START + timedelta(hours=index * 10),
                START + timedelta(hours=index * 10 + 4),
            )
            for index in range(30)
        ]
        assert not labels_overlap(disjoint)

    def test_the_guard_stays_quiet_on_a_legitimate_zero(self) -> None:
        """A warning that fires on every correct run is ignored on the run that matters."""
        disjoint = [
            Observation(
                START + timedelta(hours=index * 10),
                START + timedelta(hours=index * 10 + 4),
            )
            for index in range(30)
        ]
        folds = purged_walk_forward(disjoint, folds=3, embargo_fraction=0.0)
        assert sum(fold.purged for fold in folds) == 0
        assert_purging_is_working(folds, disjoint)  # does not raise

    def test_the_guard_still_fires_when_overlapping_labels_purge_nothing(self) -> None:
        overlapping = observations(60, label_hours=4)
        broken = [
            Fold(
                index=0,
                train=(0,),
                test=(1,),
                purged=0,
                embargoed=0,
                test_start=START,
                test_end=START,
                embargo_until=START,
            )
        ]
        with pytest.raises(ValueError, match="labels overlap"):
            assert_purging_is_working(broken, overlapping)

    def test_omitting_the_observations_assumes_overlap(self) -> None:
        """The conservative reading, for the bar-level labels this was written for."""
        broken = [
            Fold(
                index=0,
                train=(0,),
                test=(1,),
                purged=0,
                embargoed=0,
                test_start=START,
                test_end=START,
                embargo_until=START,
            )
        ]
        with pytest.raises(ValueError, match="no fold purged"):
            assert_purging_is_working(broken)
