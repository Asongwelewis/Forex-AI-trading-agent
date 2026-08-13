"""Which trading session a UTC moment falls in, decided in exchange-local time.

**Session boundaries are never stored as UTC constants.** London trades 08:00-17:00 *local*,
which is 08:00-17:00 UTC in winter and 07:00-16:00 UTC in summer. A hardcoded UTC pair is
therefore wrong for roughly half the year, and wrong silently — the Asian range gets measured
on the wrong bars, the breakout window opens an hour late, and every backtest built on top is
consistently, invisibly skewed. It is the same failure class as reading the MT5 broker clock
as UTC: a number that is correctly labelled and quietly false.

So each session is defined by its local business hours in its own `ZoneInfo`, and the
conversion to UTC happens at evaluation time. DST then handles itself, including the two or
three weeks each spring and autumn when London and New York are not in step.

**OVERLAP is additive, not exclusive.** When London and New York are both trading, the active
set is `{LONDON, NEW_YORK, OVERLAP}` rather than `{OVERLAP}`. Membership tests then read the
way the rules are written — "during LONDON" stays true through the overlap, and "not in
OVERLAP" is a separate question — instead of forcing every caller to remember that OVERLAP
secretly means "and also London and New York".

**The weekly boundary is genuinely UTC.** Sunday 21:00 to Friday 21:00 UTC is a broker
convention, not an exchange's business day, and CLAUDE.md records it in UTC because that is
how it is actually observed. It is deliberately not modelled in a local zone; the DST rule
above applies to *session* hours, which are business hours somewhere, and not to this.

Every function takes the moment it should evaluate. Nothing here reads a clock, so replaying
a bar gives the same answer it gave live.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

__all__ = [
    "LONDON_MORNING",
    "Session",
    "SessionOpening",
    "SessionWindow",
    "WEEKLY_CLOSE_HOUR_UTC",
    "WEEKLY_OPEN_HOUR_UTC",
    "active_sessions",
    "dominant_session",
    "is_market_open",
    "local_time",
    "minutes_until_close",
    "session_bounds_utc",
]


class Session(StrEnum):
    """A trading session. `OVERLAP` is derived from the other two, never scheduled itself."""

    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP = "OVERLAP"


@dataclass(frozen=True)
class SessionWindow:
    """One session's business hours, expressed where that business actually happens.

    `zone` is the IANA name rather than a `ZoneInfo` instance so the window stays trivially
    comparable and printable; the instance is built on demand.
    """

    session: Session
    zone: str
    opens: time
    closes: time

    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.zone)

    def contains(self, moment: datetime) -> bool:
        """Is `moment` inside this window, judged on the local wall clock?

        Half-open: the opening minute counts, the closing minute does not. Weekend local
        days never count — a session is a business day in its own city.
        """
        local = local_time(moment, self.zone)
        if local.weekday() >= 5:
            return False
        return self.opens <= local.timetz().replace(tzinfo=None) < self.closes


#: Business hours in local time. Tokyo runs 09:00-18:00 JST and never observes DST; London and
#: New York both run 08:00-17:00 local and drift against each other twice a year.
SESSION_WINDOWS: dict[Session, SessionWindow] = {
    Session.TOKYO: SessionWindow(Session.TOKYO, "Asia/Tokyo", time(9, 0), time(18, 0)),
    Session.LONDON: SessionWindow(Session.LONDON, "Europe/London", time(8, 0), time(17, 0)),
    Session.NEW_YORK: SessionWindow(Session.NEW_YORK, "America/New_York", time(8, 0), time(17, 0)),
}


@dataclass(frozen=True)
class SessionOpening:
    """The opening slice of a session, expressed in that session's own local time.

    This exists so that "the London morning" has exactly one definition. The router decides
    whether `session_breakout` may speak, and `session_breakout` decides whether it has
    anything to say; before this type they each computed the window from their own constants
    and disagreed by an hour every summer — the router permitting 07:00-10:59 UTC while the
    strategy only spoke from 08:00. Both now call `permits` on the same object, so the two
    windows cannot drift apart without the shared definition moving under both of them.
    """

    session: Session
    #: Exclusive, in local time. 12 means the last permitted bar opens at 11:00 local.
    before_local_hour: int

    def __post_init__(self) -> None:
        if self.session is Session.OVERLAP:
            raise ValueError(
                "OVERLAP has no business hours of its own, so it cannot anchor a local-time "
                "rule; name the session whose clock the rule means"
            )
        if not 0 <= self.before_local_hour <= 24:
            raise ValueError(
                f"before_local_hour must be an hour of the day, got {self.before_local_hour}"
            )
        if _spans_utc_midnight(self.session, self.before_local_hour):
            raise ValueError(
                f"{self.session} before {self.before_local_hour}:00 local spans midnight UTC. "
                "session_breakout groups a day's bars with `_bars_on`, which keys on the UTC "
                "calendar date, so a window straddling midnight UTC would silently drop every "
                "bar on the far side of it — the range would be measured from a partial day "
                "and the first-break rule would not see breaks it should. Refusing here, where "
                "the window is defined, rather than there, where the bars quietly go missing."
            )

    def permits(self, moment: datetime) -> bool:
        """Is `moment` inside this session and still before the local cutoff?

        Takes the instant rather than a precomputed regime so a strategy — which has no
        `Regime` — reaches the same answer through the same code as the router.
        """
        if self.session not in active_sessions(moment):
            return False
        return self.local_hour(moment) < self.before_local_hour

    def local_hour(self, moment: datetime) -> int:
        """The hour `moment` reads as on this session's own clock. 11:00 UTC is 12 in July."""
        return local_time(moment, SESSION_WINDOWS[self.session].zone).hour


#: Probe dates for the midnight check: one either side of the DST line, so a window is
#: judged in both offsets a session can sit at rather than whichever one today happens to be.
_WINTER_PROBE = date(2026, 1, 15)
_SUMMER_PROBE = date(2026, 7, 15)


def _spans_utc_midnight(session: Session, before_local_hour: int) -> bool:
    """Would this window cross midnight UTC in either half of the year?

    An empty window — a cutoff at or before the session's own open — crosses nothing and is
    reported as safe; `permits` simply never returns True for it.
    """
    window = SESSION_WINDOWS[session]
    zone = window.tzinfo()

    for probe in (_WINTER_PROBE, _SUMMER_PROBE):
        opens = datetime.combine(probe, window.opens, tzinfo=zone)
        # Built from local midnight so hour 24 stays expressible, unlike `time(24)`.
        cutoff = datetime.combine(probe, time(0), tzinfo=zone) + timedelta(hours=before_local_hour)
        if cutoff <= opens:
            continue
        # The last permitted instant, not the exclusive bound: a window closing exactly at
        # midnight UTC ends on the previous UTC day and is fine.
        last_instant = cutoff.astimezone(UTC) - timedelta(microseconds=1)
        if opens.astimezone(UTC).date() != last_instant.date():
            return True
    return False


#: The window `session_breakout` trades and the router permits it in. One object, two callers.
LONDON_MORNING = SessionOpening(Session.LONDON, 12)

#: FX opens Sunday 21:00 UTC and closes Friday 21:00 UTC. See the module docstring for why
#: these two, alone, are UTC constants.
WEEKLY_OPEN_HOUR_UTC = 21
WEEKLY_CLOSE_HOUR_UTC = 21
_SUNDAY = 6
_FRIDAY = 4

#: Which session wins when several are active, for the single label a journal line wants.
_DOMINANCE: tuple[Session, ...] = (
    Session.OVERLAP,
    Session.LONDON,
    Session.NEW_YORK,
    Session.TOKYO,
)


def _require_aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError(
            "session logic needs a timezone-aware datetime; a naive one has already lost the "
            "information this module exists to use"
        )
    return moment.astimezone(UTC)


def local_time(moment: datetime, zone: str) -> datetime:
    """`moment` as it reads on the wall clock in `zone`."""
    return _require_aware(moment).astimezone(ZoneInfo(zone))


def is_market_open(moment: datetime) -> bool:
    """Is FX trading at `moment`? Sunday 21:00 UTC through Friday 21:00 UTC.

    **These UTC hour comparisons are correct — do not "fix" them into a local zone.** Every
    other boundary in this module is a business day in a named city and must be converted;
    this one is not. The weekly open and close are a broker convention observed in UTC, they
    belong to no exchange's working hours, and CLAUDE.md records them in UTC because that is
    how they are actually kept. Reading them through `Europe/London` or any other zone would
    move the Friday close by an hour every summer and invent a bug where there is none.
    """
    utc = _require_aware(moment)
    weekday = utc.weekday()

    if weekday == 5:  # Saturday, always shut
        return False
    if weekday == _SUNDAY:
        return utc.hour >= WEEKLY_OPEN_HOUR_UTC
    if weekday == _FRIDAY:
        return utc.hour < WEEKLY_CLOSE_HOUR_UTC
    return True


def _next_weekly_close(moment: datetime) -> datetime:
    """The first Friday 21:00 UTC strictly after `moment`."""
    utc = _require_aware(moment)
    ahead = (_FRIDAY - utc.weekday()) % 7
    candidate = (utc + timedelta(days=ahead)).replace(
        hour=WEEKLY_CLOSE_HOUR_UTC, minute=0, second=0, microsecond=0
    )
    if candidate <= utc:
        candidate += timedelta(days=7)
    return candidate


def minutes_until_close(moment: datetime) -> int:
    """Whole minutes until the weekly close, or 0 when the market is already shut.

    Zero therefore means "no time left to hold anything", which is what a caller deciding
    whether to open a position before the weekend actually wants to branch on.
    """
    if not is_market_open(moment):
        return 0
    remaining = _next_weekly_close(moment) - _require_aware(moment)
    return int(remaining.total_seconds() // 60)


def active_sessions(moment: datetime) -> frozenset[Session]:
    """Every session trading at `moment`, plus OVERLAP when London and New York coincide.

    Empty outside market hours, and empty in the quiet stretch between the Sunday open and
    Tokyo's first bell — which is honest: we do not model Sydney, so nothing is active yet.
    """
    if not is_market_open(moment):
        return frozenset()

    active = {window.session for window in SESSION_WINDOWS.values() if window.contains(moment)}
    if Session.LONDON in active and Session.NEW_YORK in active:
        active.add(Session.OVERLAP)
    return frozenset(active)


def dominant_session(sessions: frozenset[Session]) -> Session | None:
    """The one label that best describes an active set, or None when nothing is trading."""
    for session in _DOMINANCE:
        if session in sessions:
            return session
    return None


def session_bounds_utc(session: Session, on: date) -> tuple[datetime, datetime] | None:
    """The UTC instants bracketing `session` on its local date `on`.

    This is the function that makes the DST behaviour observable: ask for London on a January
    date and a July date and the answers differ by an hour. `on` is a *local* date in the
    session's own zone, so "London on 2026-07-15" means that Wednesday in London.

    Returns None for a local weekend. OVERLAP is the intersection of the other two, and is
    None on any day they do not actually meet.

    No boundary here can land in a DST gap: London and New York both shift at 01:00-02:00
    local, and 08:00, 17:00, 09:00 and 18:00 are all comfortably clear of it.
    """
    if session is Session.OVERLAP:
        london = session_bounds_utc(Session.LONDON, on)
        new_york = session_bounds_utc(Session.NEW_YORK, on)
        if london is None or new_york is None:
            return None
        start, end = max(london[0], new_york[0]), min(london[1], new_york[1])
        return (start, end) if start < end else None

    if on.weekday() >= 5:
        return None

    window = SESSION_WINDOWS[session]
    zone = window.tzinfo()
    opens = datetime.combine(on, window.opens, tzinfo=zone)
    closes = datetime.combine(on, window.closes, tzinfo=zone)
    return opens.astimezone(UTC), closes.astimezone(UTC)
