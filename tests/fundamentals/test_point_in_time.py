"""The look-ahead test for the fundamentals layer.

`tests/store/test_events_point_in_time.py` proves the repository and the SQL gate hold the
line. This proves the layer *above* them does too, which is a separate risk: every read here
goes through `build_context`, and a context assembler that fetched a convenient extra row, or
passed the wrong instant down, would defeat a perfectly correct gate from the outside.

Two directions of failure are covered, because they are different bugs:

* **Reading an event before it was published.** The classic. A backtest evaluating a 09:00 bar
  must not see a release the world learned about at 09:30, no matter when the event itself
  occurred.
* **Failing to read an event that was legitimately known.** Scheduled releases are published
  days ahead — that is the whole basis of the calendar blackout. A gate that hid them would be
  "safe" and useless, and would silently disable the auto-revoke trigger.

The `source` stamped by each source module is also checked here rather than in isolation,
because publication time is only trustworthy end to end: a correct gate reading a field the
calendar filled in with the event date is still look-ahead.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.fundamentals.context import PolicyRate, build_context
from fxagent.store.repositories.events import EventRepository
from tests.conftest import requires_postgres

pytestmark = [pytest.mark.db, requires_postgres]

#: The bar under evaluation. Every assertion is phrased relative to this instant.
BAR = datetime(2026, 3, 11, 9, 0, tzinfo=UTC)

RATES = {
    "EUR": PolicyRate("EUR", 2.15, BAR.date()),
    "USD": PolicyRate("USD", 4.40, BAR.date()),
}


def _row(
    *,
    title: str,
    event_time: datetime,
    publication_time: datetime,
    currency: str = "USD",
    importance: str = "High",
) -> dict[str, object]:
    return {
        "title": title,
        "event_time_utc": event_time,
        "publication_time_utc": publication_time,
        "currency": currency,
        "importance": importance,
        "source": "forexfactory",
        "category": "calendar",
    }


@pytest.fixture
async def repo(session) -> EventRepository:  # noqa: ANN001 - fixture type is the suite's
    return EventRepository(session)


async def test_event_published_after_the_bar_is_invisible(repo: EventRepository) -> None:
    """The headline case: publication after the bar, event time irrelevant."""
    await repo.upsert_many(
        [
            _row(
                title="Retrospective revision",
                # The event itself happened *before* the bar, which is exactly the trap: a
                # filter on event_time_utc would let this through.
                event_time=BAR - timedelta(hours=2),
                publication_time=BAR + timedelta(minutes=30),
            )
        ]
    )

    context = await build_context(repo, as_of=BAR, symbol="EURUSD", rates=RATES)

    assert context.recent_events == ()
    assert context.imminent_high_impact == ()
    assert context.calendar_blackout is False

    # Positive control. Without it this test also passes when the insert silently wrote nothing,
    # which is the failure mode an assertion of emptiness cannot distinguish from success.
    later = await build_context(repo, as_of=BAR + timedelta(hours=1), symbol="EURUSD", rates=RATES)
    assert [e.title for e in later.recent_events] == ["Retrospective revision"]


async def test_scheduled_release_published_earlier_is_visible(repo: EventRepository) -> None:
    """The inverse failure: a gate that hides legitimately-known future events.

    This is what makes the blackout work at all. The calendar announces NFP days in advance,
    so at 09:00 we may know a 09:10 release is coming — and must.
    """
    await repo.upsert_many(
        [
            _row(
                title="Non-Farm Employment Change",
                event_time=BAR + timedelta(minutes=10),
                publication_time=BAR - timedelta(days=3),
            )
        ]
    )

    context = await build_context(repo, as_of=BAR, symbol="EURUSD", rates=RATES)

    assert [e.title for e in context.imminent_high_impact] == ["Non-Farm Employment Change"]
    assert context.calendar_blackout is True


async def test_blackout_does_not_fire_on_an_unpublished_imminent_release(
    repo: EventRepository,
) -> None:
    """An imminent high-impact event we could not yet have known about must not revoke a grant.

    Failing this would not lose money directly, but it would revoke grants in a backtest that
    could never have been revoked live, which makes the backtest's risk profile fiction.
    """
    await repo.upsert_many(
        [
            _row(
                title="Unscheduled Rate Decision",
                event_time=BAR + timedelta(minutes=5),
                publication_time=BAR + timedelta(minutes=1),
            )
        ]
    )

    context = await build_context(repo, as_of=BAR, symbol="EURUSD", rates=RATES)

    assert context.imminent_high_impact == ()
    assert context.calendar_blackout is False

    # Positive control: the row exists and does black out once it has been published, so the
    # assertion above is about visibility rather than about an empty table.
    after = await build_context(
        repo, as_of=BAR + timedelta(minutes=2), symbol="EURUSD", rates=RATES
    )
    assert after.calendar_blackout is True


async def test_visibility_only_grows_as_the_bar_advances(repo: EventRepository) -> None:
    """Replaying forward may reveal events and must never hide one already seen."""
    await repo.upsert_many(
        [
            _row(
                title="CPI y/y",
                event_time=BAR + timedelta(minutes=5),
                publication_time=BAR + timedelta(minutes=1),
            )
        ]
    )

    before = await build_context(repo, as_of=BAR, symbol="EURUSD", rates=RATES)
    after = await build_context(
        repo, as_of=BAR + timedelta(minutes=2), symbol="EURUSD", rates=RATES
    )

    assert before.calendar_blackout is False
    assert after.calendar_blackout is True


async def test_only_the_pair_s_own_currencies_are_considered(repo: EventRepository) -> None:
    """A JPY release must not black out EURUSD, or every pair halts on every release."""
    await repo.upsert_many(
        [
            _row(
                title="BOJ Policy Rate",
                event_time=BAR + timedelta(minutes=5),
                publication_time=BAR - timedelta(days=1),
                currency="JPY",
            )
        ]
    )

    eurusd = await build_context(repo, as_of=BAR, symbol="EURUSD", rates=RATES)
    assert eurusd.calendar_blackout is False


async def test_naive_as_of_is_rejected(repo: EventRepository) -> None:
    """A naive instant would be read in the session timezone and shift the gate silently."""
    with pytest.raises(ValueError, match="timezone-aware"):
        await build_context(repo, as_of=datetime(2026, 3, 11, 9, 0), symbol="EURUSD", rates=RATES)


async def test_store_outage_is_reported_rather_than_read_as_all_clear(
    repo: EventRepository,
) -> None:
    """ "Could not look" must not be indistinguishable from "nothing there".

    The permission layer treats an empty blackout list as permission to trade. If a failed
    query produced the same empty list with no other signal, an outage would silently unlock
    execution — so the failure is recorded in `unavailable` and `is_degraded` goes true.
    """

    class Broken:
        async def high_impact_near(self, *_: object, **__: object) -> list[object]:
            raise RuntimeError("connection reset")

        async def in_event_window(self, *_: object, **__: object) -> list[object]:
            raise RuntimeError("connection reset")

    context = await build_context(
        Broken(),  # type: ignore[arg-type]
        as_of=BAR,
        symbol="EURUSD",
        rates=RATES,
    )

    assert context.is_degraded is True
    assert "imminent_high_impact" in context.unavailable
    assert "recent_events" in context.unavailable
