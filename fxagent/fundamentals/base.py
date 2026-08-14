"""The contract every fundamental source implements, and the record they all produce.

Two rules hold for every source in this package, and they are the reason this module exists
rather than each source rolling its own shape:

**Publication time is not event time.** `Event` carries both and neither defaults to the other.
A COT report references Tuesday and is published on Friday; a calendar release is scheduled
days ahead and its number lands at the scheduled moment. Collapsing the two is hard rule 6's
look-ahead bias, and it is invisible until a backtest quietly reports an edge that was never
tradeable. The store gates every read on `publication_time_utc`; a source that puts the wrong
value there defeats that gate from the outside, where no SQL function can catch it.

**No source may block an analysis cycle.** Forex Factory rate-limits, central bank sites go
down, DNS fails. A fundamentals feed is context, never a precondition — so `fetch_safely`
turns any failure into a warning and an empty list. The analyst then runs with less
information, which is a strictly better outcome than not running.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "IMPORTANCE_VALUES",
    "Event",
    "FundamentalSource",
    "fetch_safely",
]

logger = logging.getLogger(__name__)

#: Mirrors the `events_importance_known` check constraint in migration 0003. Kept in sync by
#: `tests/fundamentals/test_base.py`, which reads the migration rather than trusting this copy.
IMPORTANCE_VALUES = ("High", "Medium", "Low", "Holiday")

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class Event:
    """One fundamental observation, normalised across sources and ready for the store.

    Frozen because an event is a record of something that happened; a source that wants to
    revise one emits a new value rather than mutating the old, and the store's upsert decides
    which fields a revision may overwrite.
    """

    event_time_utc: datetime
    publication_time_utc: datetime
    source: str
    currency: str
    importance: str
    title: str
    category: str | None = None
    body: str | None = None
    forecast: float | None = None
    actual: float | None = None
    previous: float | None = None
    surprise_score: float | None = None
    ingested_at: datetime | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # Validated here rather than at the database boundary so a bad row fails in the source
        # that produced it, naming that source, instead of surfacing as an opaque constraint
        # violation on a bulk upsert of two hundred events.
        for name in ("event_time_utc", "publication_time_utc"):
            value: datetime = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(
                    f"{self.source}: {name} must be timezone-aware; a naive datetime is read "
                    f"in the session timezone and silently shifts the point-in-time gate"
                )
        if not _CURRENCY_RE.match(self.currency):
            raise ValueError(
                f"{self.source}: currency {self.currency!r} is not a 3-letter code "
                f"(the events_currency_shape constraint rejects it)"
            )
        if self.importance not in IMPORTANCE_VALUES:
            raise ValueError(
                f"{self.source}: importance {self.importance!r} is not one of "
                f"{IMPORTANCE_VALUES}; the events_importance_known constraint would reject "
                f"the whole batch"
            )
        if not self.title.strip():
            raise ValueError(f"{self.source}: title is empty, and it is part of the unique key")

    def as_row(self) -> dict[str, Any]:
        """The dict shape `EventRepository.upsert_many` expects."""
        return {
            "event_time_utc": self.event_time_utc.astimezone(UTC),
            "publication_time_utc": self.publication_time_utc.astimezone(UTC),
            "source": self.source,
            "currency": self.currency,
            "category": self.category,
            "importance": self.importance,
            "title": self.title,
            "body": self.body,
            "forecast": self.forecast,
            "actual": self.actual,
            "previous": self.previous,
            "surprise_score": self.surprise_score,
        }


@runtime_checkable
class FundamentalSource(Protocol):
    """A pollable source of fundamental events.

    `since` means *what this source considers new after that instant*, which is deliberately
    the source's own business: an RSS feed compares it against item publication dates and
    returns a true incremental slice, while the Forex Factory calendar has no publication
    metadata at all and treats it as the earliest event time worth reporting. Both are correct
    for their feed, and both still stamp `publication_time_utc` honestly.
    """

    @property
    def name(self) -> str:
        """Short identifier, also written to `events.source`, so it is part of the unique key."""
        ...

    async def fetch(self, since: datetime) -> list[Event]: ...


async def fetch_safely(source: FundamentalSource, since: datetime) -> list[Event]:
    """Run a source, converting any failure into a warning and an empty list.

    Deliberately catches `Exception` rather than an enumerated set. The failure modes here are
    other people's servers — timeouts, TLS errors, malformed XML, an HTML error page served
    with a JSON content type, a schema change overnight — and enumerating them is a guess that
    goes stale silently. The one thing that must never happen is a missing calendar stopping
    the analysis, so the net is wide on purpose.

    `BaseException` is not caught: cancellation and KeyboardInterrupt must still propagate, or
    a shutdown would hang waiting on a source that swallowed its own cancellation.
    """
    try:
        events = await source.fetch(since)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning(
            "fundamental source %s failed, continuing without it: %s: %s",
            source.name,
            type(exc).__name__,
            exc,
        )
        return []

    logger.info("fundamental source %s returned %d event(s)", source.name, len(events))
    return list(events)


async def fetch_all(sources: Sequence[FundamentalSource], since: datetime) -> list[Event]:
    """Poll every source, keeping whatever succeeded. Order follows `sources`."""
    collected: list[Event] = []
    for source in sources:
        collected.extend(await fetch_safely(source, since))
    return collected
