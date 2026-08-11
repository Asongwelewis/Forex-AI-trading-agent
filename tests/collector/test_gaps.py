"""Gap detection. The hard part is not reporting the weekend as an outage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.collector.gaps import (
    expected_timestamps,
    find_gaps,
    floor_to_timeframe,
    is_market_open,
)

# 2026-03-09 is a Monday.
MONDAY = datetime(2026, 3, 9, 0, 0, tzinfo=UTC)
FRIDAY = datetime(2026, 3, 13, 0, 0, tzinfo=UTC)
SATURDAY = datetime(2026, 3, 14, 0, 0, tzinfo=UTC)
SUNDAY = datetime(2026, 3, 15, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("moment", "open_"),
    [
        (MONDAY.replace(hour=12), True),
        (FRIDAY.replace(hour=20), True),
        (FRIDAY.replace(hour=21), False),
        (FRIDAY.replace(hour=23), False),
        (SATURDAY.replace(hour=12), False),
        (SUNDAY.replace(hour=20), False),
        (SUNDAY.replace(hour=21), True),
        (SUNDAY.replace(hour=23), True),
    ],
)
def test_market_hours_run_sunday_2100_to_friday_2100(moment: datetime, open_: bool) -> None:
    assert is_market_open(moment) is open_


def test_a_naive_moment_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        is_market_open(datetime(2026, 3, 9, 12, 0))


def test_expected_timestamps_skip_the_weekend() -> None:
    """Friday 21:00 to Sunday 21:00 is 48 hours with no bars anyone can supply."""
    expected = expected_timestamps(
        start=FRIDAY.replace(hour=18), end=SUNDAY.replace(hour=23), timeframe="H1"
    )
    hours = {moment for moment in expected}

    assert FRIDAY.replace(hour=20) in hours
    assert FRIDAY.replace(hour=21) not in hours
    assert SATURDAY.replace(hour=12) not in hours
    assert SUNDAY.replace(hour=20) not in hours
    assert SUNDAY.replace(hour=21) in hours


def test_a_complete_series_has_no_gaps() -> None:
    start, end = MONDAY, MONDAY.replace(hour=12)
    stored = expected_timestamps(start=start, end=end, timeframe="H1")

    assert find_gaps(stored, start=start, end=end, timeframe="H1") == []


def test_the_weekend_is_not_reported_as_a_gap() -> None:
    """The failure that makes a gap report useless: 48 phantom hours every week."""
    start, end = FRIDAY.replace(hour=18), SUNDAY.replace(hour=23)
    stored = expected_timestamps(start=start, end=end, timeframe="H1")

    assert find_gaps(stored, start=start, end=end, timeframe="H1") == []


def test_a_missing_run_is_reported_as_one_gap_not_many() -> None:
    """240 adjacent single-bar holes is unreadable; a backfill fetches ranges anyway."""
    start, end = MONDAY, MONDAY.replace(hour=12)
    full = expected_timestamps(start=start, end=end, timeframe="H1")
    stored = [m for m in full if not (4 <= m.hour <= 7)]

    gaps = find_gaps(stored, start=start, end=end, timeframe="H1")

    assert len(gaps) == 1
    assert gaps[0].missing == 4
    assert gaps[0].start == MONDAY.replace(hour=4)
    assert gaps[0].end == MONDAY.replace(hour=8)


def test_two_separate_outages_stay_separate() -> None:
    start, end = MONDAY, MONDAY.replace(hour=23)
    full = expected_timestamps(start=start, end=end, timeframe="H1")
    stored = [m for m in full if m.hour not in (3, 4, 15, 16)]

    gaps = find_gaps(stored, start=start, end=end, timeframe="H1")

    assert [g.missing for g in gaps] == [2, 2]


def test_an_empty_store_is_one_gap_spanning_the_window() -> None:
    """First run: everything is missing, and it must be one request-shaped range."""
    start, end = MONDAY, MONDAY.replace(hour=6)

    gaps = find_gaps([], start=start, end=end, timeframe="H1")

    assert len(gaps) == 1
    # 00:00..05:00 — six bars. The 06:00 bar is still forming at end and is not expected.
    assert gaps[0].missing == 6


def test_a_gap_either_side_of_the_weekend_is_two_gaps() -> None:
    """They are separate outages; merging them would request 48 hours that do not exist."""
    start, end = FRIDAY.replace(hour=18), SUNDAY.replace(hour=23)
    full = expected_timestamps(start=start, end=end, timeframe="H1")
    stored = [m for m in full if m not in {FRIDAY.replace(hour=19), SUNDAY.replace(hour=22)}]

    gaps = find_gaps(stored, start=start, end=end, timeframe="H1")

    assert len(gaps) == 2


def test_unknown_timeframe_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown timeframe"):
        expected_timestamps(start=MONDAY, end=MONDAY + timedelta(hours=1), timeframe="H2")


def test_reversed_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="before start"):
        expected_timestamps(start=MONDAY.replace(hour=5), end=MONDAY, timeframe="H1")


# -- grid alignment: the bug that made every bar read as missing --------------


@pytest.mark.parametrize(
    ("timeframe", "expected_minute"),
    [("H1", 0), ("M15", 15), ("M5", 20), ("M1", 23)],
)
def test_flooring_snaps_to_the_bar_boundary(timeframe: str, expected_minute: int) -> None:
    moment = MONDAY.replace(hour=9, minute=23, second=47)
    floored = floor_to_timeframe(moment, timeframe)

    assert floored.minute == expected_minute
    assert floored.second == 0
    assert floored <= moment


def test_daily_bars_floor_to_midnight_utc() -> None:
    assert floor_to_timeframe(MONDAY.replace(hour=17, minute=42), "D1") == MONDAY


def test_an_unaligned_start_still_produces_an_aligned_grid() -> None:
    """The real bug: a start of 15:20 gave expected times at :20, which no feed ever supplies."""
    expected = expected_timestamps(
        start=MONDAY.replace(hour=9, minute=20), end=MONDAY.replace(hour=14), timeframe="H1"
    )

    assert all(moment.minute == 0 for moment in expected), expected
    # 09:00 opened before the window began, so the first bar wholly inside it is 10:00.
    assert expected[0] == MONDAY.replace(hour=10)


def test_a_series_stored_on_the_hour_shows_no_gaps_from_an_unaligned_window() -> None:
    """End to end for the same bug: aligned bars, ragged query window, no phantom gaps."""
    stored = [MONDAY.replace(hour=h) for h in range(9, 15)]

    gaps = find_gaps(
        stored,
        start=MONDAY.replace(hour=9, minute=20),
        end=MONDAY.replace(hour=15, minute=20),
        timeframe="H1",
    )

    assert gaps == []


def test_the_forming_bar_is_not_expected() -> None:
    """Demanding it would leave a one-bar gap at the leading edge that never closes."""
    expected = expected_timestamps(
        start=MONDAY.replace(hour=9), end=MONDAY.replace(hour=12, minute=30), timeframe="H1"
    )

    assert MONDAY.replace(hour=11) in expected
    assert MONDAY.replace(hour=12) not in expected, "12:00 has not closed at 12:30"
