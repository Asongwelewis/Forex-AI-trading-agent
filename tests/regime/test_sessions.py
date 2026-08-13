"""Session boundaries, and the DST behaviour that is the whole point of this module.

The headline test is `test_london_bounds_differ_between_january_and_july`. If it ever fails,
the `zoneinfo` conversion has been replaced by fixed UTC constants and every downstream
number — the Asian range, the breakout window, every backtest built on them — is quietly
wrong for half the year.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from fxagent.regime.sessions import (
    LONDON_MORNING,
    Session,
    SessionOpening,
    active_sessions,
    dominant_session,
    is_market_open,
    local_time,
    minutes_until_close,
    session_bounds_utc,
)

WINTER = date(2026, 1, 15)  # Thursday, GMT in London and EST in New York
SUMMER = date(2026, 7, 15)  # Wednesday, BST in London and EDT in New York
#: US clocks sprang forward 2026-03-08; the UK does not until 2026-03-29.
OUT_OF_STEP = date(2026, 3, 12)


def _hhmm(bounds: tuple[datetime, datetime]) -> tuple[str, str]:
    return bounds[0].strftime("%H:%M"), bounds[1].strftime("%H:%M")


class TestDaylightSaving:
    def test_london_bounds_differ_between_january_and_july(self) -> None:
        """The load-bearing test. Same local hours, different UTC instants."""
        winter = session_bounds_utc(Session.LONDON, WINTER)
        summer = session_bounds_utc(Session.LONDON, SUMMER)
        assert winter is not None and summer is not None

        assert _hhmm(winter) == ("08:00", "17:00"), "London in winter is 08:00-17:00 UTC"
        assert _hhmm(summer) == ("07:00", "16:00"), "London in summer is 07:00-16:00 UTC"
        assert winter != summer, (
            "January and July produced identical UTC boundaries, so the zoneinfo conversion "
            "is not happening and every session decision is wrong for half the year"
        )

    def test_new_york_bounds_also_shift(self) -> None:
        winter = session_bounds_utc(Session.NEW_YORK, WINTER)
        summer = session_bounds_utc(Session.NEW_YORK, SUMMER)
        assert winter is not None and summer is not None
        assert _hhmm(winter) == ("13:00", "22:00")
        assert _hhmm(summer) == ("12:00", "21:00")

    def test_tokyo_never_shifts_because_japan_has_no_dst(self) -> None:
        winter = session_bounds_utc(Session.TOKYO, WINTER)
        summer = session_bounds_utc(Session.TOKYO, SUMMER)
        assert winter is not None and summer is not None
        assert _hhmm(winter) == _hhmm(summer) == ("00:00", "09:00")

    def test_london_and_new_york_shift_independently(self) -> None:
        """Mid-March: New York has already sprung forward and London has not.

        A single shared offset, or a "summer starts in April" shortcut, gets this week wrong.
        The overlap is five hours here rather than the usual four.
        """
        london = session_bounds_utc(Session.LONDON, OUT_OF_STEP)
        new_york = session_bounds_utc(Session.NEW_YORK, OUT_OF_STEP)
        overlap = session_bounds_utc(Session.OVERLAP, OUT_OF_STEP)
        assert london is not None and new_york is not None and overlap is not None

        assert _hhmm(london) == ("08:00", "17:00"), "London is still on GMT"
        assert _hhmm(new_york) == ("12:00", "21:00"), "New York is already on EDT"
        assert _hhmm(overlap) == ("12:00", "17:00")

    def test_overlap_is_four_hours_when_the_zones_are_in_step(self) -> None:
        for day in (WINTER, SUMMER):
            overlap = session_bounds_utc(Session.OVERLAP, day)
            assert overlap is not None
            assert overlap[1] - overlap[0] == (
                datetime(2026, 1, 1, 4, tzinfo=UTC) - datetime(2026, 1, 1, tzinfo=UTC)
            )


class TestExactBoundaries:
    @pytest.mark.parametrize(
        ("hour", "minute", "expected"),
        [
            (7, 59, False),  # one minute before the London open
            (8, 0, True),  # the opening minute counts
            (16, 59, True),
            (17, 0, False),  # the closing minute does not
        ],
    )
    def test_london_open_is_half_open_in_winter(
        self, hour: int, minute: int, expected: bool
    ) -> None:
        moment = datetime(2026, 1, 15, hour, minute, tzinfo=UTC)
        assert (Session.LONDON in active_sessions(moment)) is expected

    def test_london_opens_an_hour_earlier_in_utc_during_summer(self) -> None:
        assert Session.LONDON in active_sessions(datetime(2026, 7, 15, 7, 0, tzinfo=UTC))
        assert Session.LONDON not in active_sessions(datetime(2026, 1, 15, 7, 0, tzinfo=UTC))

    def test_bounds_are_none_on_a_local_weekend(self) -> None:
        saturday = date(2026, 1, 17)
        assert session_bounds_utc(Session.LONDON, saturday) is None


class TestWeeklyOpenAndClose:
    @pytest.mark.parametrize(
        ("label", "moment", "expected"),
        [
            ("Friday 20:59", datetime(2026, 1, 16, 20, 59, tzinfo=UTC), True),
            ("Friday 21:01", datetime(2026, 1, 16, 21, 1, tzinfo=UTC), False),
            ("Saturday noon", datetime(2026, 1, 17, 12, 0, tzinfo=UTC), False),
            ("Sunday 20:59", datetime(2026, 1, 18, 20, 59, tzinfo=UTC), False),
            ("Sunday 21:01", datetime(2026, 1, 18, 21, 1, tzinfo=UTC), True),
            ("Monday 09:00", datetime(2026, 1, 19, 9, 0, tzinfo=UTC), True),
        ],
    )
    def test_market_hours(self, label: str, moment: datetime, expected: bool) -> None:
        assert is_market_open(moment) is expected, label

    def test_minutes_until_close_counts_down_to_friday_close(self) -> None:
        assert minutes_until_close(datetime(2026, 1, 16, 20, 59, tzinfo=UTC)) == 1
        assert minutes_until_close(datetime(2026, 1, 16, 20, 0, tzinfo=UTC)) == 60

    def test_minutes_until_close_is_zero_when_shut(self) -> None:
        """Zero means "no time left to hold anything", not "closing right now"."""
        assert minutes_until_close(datetime(2026, 1, 17, 12, 0, tzinfo=UTC)) == 0

    def test_a_full_week_is_measured_from_the_sunday_open(self) -> None:
        just_open = datetime(2026, 1, 18, 21, 1, tzinfo=UTC)
        assert minutes_until_close(just_open) == 5 * 24 * 60 - 1

    def test_no_session_is_active_while_the_market_is_shut(self) -> None:
        assert active_sessions(datetime(2026, 1, 17, 12, 0, tzinfo=UTC)) == frozenset()


class TestActiveSessions:
    def test_overlap_is_additive_not_exclusive(self) -> None:
        """During the overlap, London and New York are still individually active."""
        moment = datetime(2026, 1, 15, 14, 0, tzinfo=UTC)
        assert active_sessions(moment) == frozenset(
            {Session.LONDON, Session.NEW_YORK, Session.OVERLAP}
        )

    def test_tokyo_and_london_share_the_eight_oclock_hour_in_winter(self) -> None:
        moment = datetime(2026, 1, 15, 8, 30, tzinfo=UTC)
        assert active_sessions(moment) == frozenset({Session.TOKYO, Session.LONDON})

    def test_the_hours_after_the_sunday_open_are_deliberately_empty(self) -> None:
        """Market open, nothing active: we do not model Sydney, and we do not pretend to."""
        moment = datetime(2026, 1, 18, 22, 0, tzinfo=UTC)
        assert is_market_open(moment) is True
        assert active_sessions(moment) == frozenset()

    @pytest.mark.parametrize(
        ("active", "expected"),
        [
            (frozenset({Session.LONDON, Session.NEW_YORK, Session.OVERLAP}), Session.OVERLAP),
            (frozenset({Session.TOKYO, Session.LONDON}), Session.LONDON),
            (frozenset({Session.NEW_YORK}), Session.NEW_YORK),
            (frozenset(), None),
        ],
    )
    def test_dominant_session_precedence(
        self, active: frozenset[Session], expected: Session | None
    ) -> None:
        assert dominant_session(active) is expected


class TestSessionOpening:
    """The shared window definition. One object, consulted by both the router and the strategy."""

    def test_the_london_morning_shifts_with_the_london_clock(self) -> None:
        """The property that was violated: 07:00 UTC is inside the window only in summer."""
        assert LONDON_MORNING.permits(datetime(2026, 7, 15, 7, tzinfo=UTC)) is True
        assert LONDON_MORNING.permits(datetime(2026, 1, 15, 7, tzinfo=UTC)) is False

    def test_the_cutoff_is_local_not_utc(self) -> None:
        """11:00 UTC is noon in London in July, so the window has already shut."""
        assert LONDON_MORNING.permits(datetime(2026, 1, 15, 11, tzinfo=UTC)) is True
        assert LONDON_MORNING.permits(datetime(2026, 7, 15, 11, tzinfo=UTC)) is False

    def test_local_hour_reports_the_session_clock(self) -> None:
        assert LONDON_MORNING.local_hour(datetime(2026, 1, 15, 11, tzinfo=UTC)) == 11
        assert LONDON_MORNING.local_hour(datetime(2026, 7, 15, 11, tzinfo=UTC)) == 12

    def test_a_shut_market_permits_nothing(self) -> None:
        assert LONDON_MORNING.permits(datetime(2026, 1, 17, 9, tzinfo=UTC)) is False

    def test_overlap_cannot_anchor_a_local_time_rule(self) -> None:
        with pytest.raises(ValueError, match="no business hours of its own"):
            SessionOpening(Session.OVERLAP, 12)

    def test_the_cutoff_must_be_an_hour_of_the_day(self) -> None:
        with pytest.raises(ValueError, match="must be an hour of the day"):
            SessionOpening(Session.LONDON, 25)


class TestNaiveDatetimesAreRejected:
    @pytest.mark.parametrize(
        "call",
        [
            lambda m: is_market_open(m),
            lambda m: active_sessions(m),
            lambda m: minutes_until_close(m),
            lambda m: local_time(m, "Europe/London"),
        ],
    )
    def test_naive_input_raises(self, call) -> None:  # noqa: ANN001 - parametrised lambda
        with pytest.raises(ValueError, match="timezone-aware"):
            call(datetime(2026, 1, 15, 12, 0))
