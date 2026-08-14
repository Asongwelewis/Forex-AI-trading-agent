"""The shared contract: event validation, and the fail-soft guarantee.

The validation tests are not ceremony. Every rule checked here mirrors a constraint in
migration 0003, and the point of duplicating it in Python is that a bulk upsert of two hundred
calendar rows fails as one opaque constraint violation naming none of them, whereas `Event`
fails in the source that built the bad row and says which field.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fxagent.fundamentals.base import (
    IMPORTANCE_VALUES,
    Event,
    FundamentalSource,
    fetch_all,
    fetch_safely,
)

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def _event(**overrides: object) -> Event:
    base: dict[str, object] = {
        "event_time_utc": NOW,
        "publication_time_utc": NOW,
        "source": "test",
        "currency": "USD",
        "importance": "High",
        "title": "Some Release",
    }
    base.update(overrides)
    return Event(**base)  # type: ignore[arg-type]


# -- validation ----------------------------------------------------------------------------


def test_naive_event_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _event(event_time_utc=datetime(2026, 8, 14, 12, 0))


def test_naive_publication_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _event(publication_time_utc=datetime(2026, 8, 14, 12, 0))


@pytest.mark.parametrize("currency", ["US", "USDD", "usd", "US1", ""])
def test_currency_must_be_three_upper_letters(currency: str) -> None:
    with pytest.raises(ValueError, match="3-letter code"):
        _event(currency=currency)


def test_unknown_importance_is_rejected() -> None:
    with pytest.raises(ValueError, match="importance"):
        _event(importance="Critical")


def test_empty_title_is_rejected() -> None:
    """Title is part of `events_unique`; an empty one collides with every other empty one."""
    with pytest.raises(ValueError, match="title is empty"):
        _event(title="   ")


def test_importance_values_match_the_migration() -> None:
    """Reads the SQL rather than trusting the Python copy to have been kept in sync."""
    sql = Path("fxagent/store/migrations/0003_events.sql").read_text(encoding="utf-8")
    match = re.search(r"importance in \(([^)]*)\)", sql)
    assert match, "could not find the events_importance_known constraint"
    in_sql = {value.strip().strip("'") for value in match.group(1).split(",")}
    assert in_sql == set(IMPORTANCE_VALUES)


def test_publication_may_precede_or_follow_the_event() -> None:
    """Both orderings are real and neither is an error.

    A scheduled release is published days before it happens; a COT report references Tuesday
    and is published on Friday. A validator that enforced one ordering would reject half the
    world.
    """
    _event(publication_time_utc=NOW - timedelta(days=3))
    _event(publication_time_utc=NOW + timedelta(days=3))


def test_as_row_normalises_to_utc() -> None:
    from datetime import timezone

    tokyo = timezone(timedelta(hours=9))
    row = _event(event_time_utc=NOW.astimezone(tokyo)).as_row()
    assert row["event_time_utc"] == NOW
    assert row["event_time_utc"].tzinfo == UTC


def test_event_is_frozen() -> None:
    event = _event()
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        event.title = "changed"  # type: ignore[misc]


# -- fail soft -----------------------------------------------------------------------------


class _Boom:
    """A source that fails the way real ones do: at request time, not construction."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    @property
    def name(self) -> str:
        return "boom"

    async def fetch(self, since: datetime) -> list[Event]:
        raise self._exc


class _Fine:
    @property
    def name(self) -> str:
        return "fine"

    async def fetch(self, since: datetime) -> list[Event]:
        return [_event(title="Worked")]


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("timed out"),
        ConnectionError("dns failure"),
        ValueError("malformed json"),
        KeyError("schema changed"),
        RuntimeError("something new nobody enumerated"),
    ],
)
async def test_any_failure_becomes_an_empty_list(exc: Exception) -> None:
    """No source may block an analysis cycle — the whole point of the wide catch."""
    assert await fetch_safely(_Boom(exc), NOW) == []


async def test_failure_is_logged_as_a_warning_naming_the_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A silent empty list is indistinguishable from a quiet day."""
    with caplog.at_level(logging.WARNING):
        await fetch_safely(_Boom(TimeoutError("timed out")), NOW)
    assert "boom" in caplog.text
    assert "TimeoutError" in caplog.text


async def test_cancellation_still_propagates() -> None:
    """Swallowing CancelledError would hang shutdown waiting on a source that ignored it."""
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        await fetch_safely(_Boom(asyncio.CancelledError()), NOW)


async def test_keyboard_interrupt_still_propagates() -> None:
    with pytest.raises(KeyboardInterrupt):
        await fetch_safely(_Boom(KeyboardInterrupt()), NOW)


async def test_one_dead_source_does_not_lose_a_live_one() -> None:
    events = await fetch_all([_Boom(TimeoutError()), _Fine()], NOW)
    assert [e.title for e in events] == ["Worked"]


def test_real_sources_satisfy_the_protocol() -> None:
    from fxagent.fundamentals.calendar import ForexFactoryCalendar
    from fxagent.fundamentals.centralbank import CentralBankRss

    assert isinstance(ForexFactoryCalendar(user_agent="t"), FundamentalSource)
    assert isinstance(CentralBankRss(), FundamentalSource)
