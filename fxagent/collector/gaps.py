"""Finding the holes in a stored bar series.

Separated from the service so the awkward part — deciding what counts as a gap — is pure and
testable without a database or a network.

The whole difficulty is that **most missing bars are not gaps.** FX closes from Friday 21:00
UTC to Sunday 21:00 UTC, and a quiet pair simply prints no bar in some minutes. A gap detector
that flags every missing interval reports thousands of holes that no backfill can ever fill,
and the real gap — the four hours the collector was down — is lost among them.

So a gap here is a missing interval *inside market hours*, and the market calendar is applied
before anything is reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fxagent.adapters.base import TIMEFRAMES

__all__ = ["Gap", "find_gaps", "is_market_open", "expected_timestamps"]

#: FX opens Sunday 21:00 UTC and closes Friday 21:00 UTC.
_SUNDAY = 6
_FRIDAY = 4
MARKET_OPEN_HOUR = 21
MARKET_CLOSE_HOUR = 21


@dataclass(frozen=True)
class Gap:
    """A contiguous run of missing bars inside market hours."""

    start: datetime
    end: datetime
    timeframe: str
    missing: int

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f"Gap({self.timeframe}, {self.start:%Y-%m-%d %H:%M} -> "
            f"{self.end:%Y-%m-%d %H:%M} UTC, {self.missing} bars)"
        )


def is_market_open(moment: datetime) -> bool:
    """True if the FX market is open at `moment`.

    Weekend only. Public holidays thin liquidity but do not close the market, and a holiday
    calendar here would suppress real gaps on days that genuinely traded.
    """
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    utc = moment.astimezone(UTC)
    weekday = utc.weekday()

    if weekday == 5:  # Saturday
        return False
    if weekday == _FRIDAY and utc.hour >= MARKET_CLOSE_HOUR:
        return False
    return not (weekday == _SUNDAY and utc.hour < MARKET_OPEN_HOUR)


def floor_to_timeframe(moment: datetime, timeframe: str) -> datetime:
    """Round `moment` down to the bar boundary it falls inside.

    Bars open on the boundary — 12:00, 13:00 — never at 12:20. Walking a grid from an
    unaligned `start` produces expected times that no feed will ever supply, so every bar
    reads as missing and the collector re-backfills the same range forever without closing a
    single gap.

    Flooring against the Unix epoch works for every timeframe here because the epoch begins at
    midnight UTC and all of them divide a day evenly.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}")
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    step_seconds = int(TIMEFRAMES[timeframe].total_seconds())
    epoch = int(moment.astimezone(UTC).timestamp())
    return datetime.fromtimestamp(epoch - (epoch % step_seconds), UTC)


def expected_timestamps(*, start: datetime, end: datetime, timeframe: str) -> list[datetime]:
    """Bar-open times in `[start, end]` whose bar has both opened and CLOSED by `end`.

    Two subtleties, and both cause permanent phantom gaps if missed:

    * The grid is aligned to the timeframe, not to `start` — see `floor_to_timeframe`.
    * The bar currently forming is not expected. Its interval has not finished, so no feed can
      supply it, and demanding it leaves a one-bar gap at the leading edge that never closes.

    Every returned time satisfies `start <= open` and `open + step <= end`, so the set matches
    exactly what a `bars_between(start, end)` read can return. A bar that merely *contains*
    `start` is excluded: expecting it while the stored-bar query filters it out produces a
    permanent one-bar gap at the trailing edge — the same phantom-gap failure from the other
    end of the window.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if end < start:
        raise ValueError(f"end {end} is before start {start}")

    step = TIMEFRAMES[timeframe]
    first = start.astimezone(UTC)
    last = end.astimezone(UTC)
    moment = floor_to_timeframe(first, timeframe)

    expected: list[datetime] = []
    while moment + step <= last:
        if moment >= first and is_market_open(moment):
            expected.append(moment)
        moment += step
    return expected


def find_gaps(
    stored: Iterable[datetime], *, start: datetime, end: datetime, timeframe: str
) -> list[Gap]:
    """Runs of expected-but-absent bars in `[start, end]`.

    Consecutive missing bars are merged into one `Gap`, because a backfill fetches ranges and
    reporting 240 adjacent single-bar holes as 240 gaps is unreadable and unactionable.
    """
    have = {moment.astimezone(UTC) for moment in stored}
    want = expected_timestamps(start=start, end=end, timeframe=timeframe)
    step = TIMEFRAMES[timeframe]

    gaps: list[Gap] = []
    run: list[datetime] = []

    def flush() -> None:
        if run:
            gaps.append(
                Gap(start=run[0], end=run[-1] + step, timeframe=timeframe, missing=len(run))
            )
            run.clear()

    previous: datetime | None = None
    for moment in want:
        if moment in have:
            flush()
        else:
            # A weekend between two missing bars breaks the run: they are separate outages.
            if previous is not None and run and moment - previous > step:
                flush()
            run.append(moment)
        previous = moment
    flush()
    return gaps


def merge_adjacent(gaps: Sequence[Gap], *, tolerance: timedelta) -> list[Gap]:
    """Join gaps separated by less than `tolerance`, to cut down backfill requests."""
    if not gaps:
        return []
    ordered = sorted(gaps, key=lambda gap: gap.start)
    merged = [ordered[0]]
    for gap in ordered[1:]:
        last = merged[-1]
        if gap.start - last.end <= tolerance and gap.timeframe == last.timeframe:
            merged[-1] = Gap(
                start=last.start,
                end=max(last.end, gap.end),
                timeframe=last.timeframe,
                missing=last.missing + gap.missing,
            )
        else:
            merged.append(gap)
    return merged
