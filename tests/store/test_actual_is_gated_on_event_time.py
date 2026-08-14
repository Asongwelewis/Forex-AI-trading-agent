"""Migration 0009: a released value must not be readable before it was released.

The row and the number in it become public at different moments. `test_events_point_in_time`
covers the row; this covers the number, which is the half `upsert_many` can reopen — it updates
`actual` in place while deliberately leaving `publication_time_utc` at first sighting, so a row
published Monday can end up holding Friday's result.

Every test here therefore checks the same row twice, before and after `event_time_utc`. A test
that only looked before would pass just as happily against a column that is always null.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from fxagent.store.repositories.events import EventRepository

from .conftest import requires_postgres

pytestmark = [pytest.mark.db, requires_postgres]

RELEASE = datetime(2026, 3, 6, 13, 30, tzinfo=UTC)
#: Published days ahead, the way a calendar announces a scheduled release.
ANNOUNCED = RELEASE - timedelta(days=3)


async def _seed(repo: EventRepository) -> None:
    """The exact sequence that creates the hazard: announce, then fill in the result."""
    await repo.upsert_many(
        [
            {
                "title": "Non-Farm Employment Change",
                "event_time_utc": RELEASE,
                "publication_time_utc": ANNOUNCED,
                "currency": "USD",
                "importance": "High",
                "source": "forexfactory",
                "category": "calendar",
                "forecast": 125.0,
                "previous": 147.0,
                "actual": None,
            }
        ]
    )
    # Later, the number lands. publication_time_utc is not rewritten — by design.
    await repo.upsert_many(
        [
            {
                "title": "Non-Farm Employment Change",
                "event_time_utc": RELEASE,
                "publication_time_utc": ANNOUNCED,
                "currency": "USD",
                "importance": "High",
                "source": "forexfactory",
                "category": "calendar",
                "forecast": 125.0,
                "previous": 147.0,
                "actual": 189.0,
                "surprise_score": 2.4,
            }
        ]
    )


async def _one(repo: EventRepository, as_of: datetime):  # noqa: ANN202
    events = await repo.visible_at(as_of)
    assert len(events) == 1, f"expected exactly one visible event at {as_of}, got {len(events)}"
    return events[0]


async def test_actual_is_withheld_until_the_release(events_repo: EventRepository) -> None:
    await _seed(events_repo)

    before = await _one(events_repo, RELEASE - timedelta(minutes=1))
    assert before.actual is None
    assert before.surprise_score is None

    # The positive half. Without it this passes against a column that is never populated.
    after = await _one(events_repo, RELEASE)
    assert after.actual == 189.0
    assert after.surprise_score == 2.4


async def test_the_row_and_its_forecast_are_visible_all_along(
    events_repo: EventRepository,
) -> None:
    """Withholding the result must not withhold the schedule — the blackout depends on it."""
    await _seed(events_repo)

    before = await _one(events_repo, RELEASE - timedelta(days=1))
    assert before.title == "Non-Farm Employment Change"
    assert before.forecast == 125.0
    assert before.previous == 147.0
    assert before.event_time_utc == RELEASE


async def test_the_boundary_is_inclusive_at_the_release_instant(
    events_repo: EventRepository,
) -> None:
    """`>=`: the number is public the moment it is released, not a microsecond later."""
    await _seed(events_repo)

    tick = timedelta(microseconds=1)
    assert (await _one(events_repo, RELEASE - tick)).actual is None
    assert (await _one(events_repo, RELEASE)).actual == 189.0


async def test_the_blackout_still_fires_on_an_unreleased_event(
    events_repo: EventRepository,
) -> None:
    """The auto-revoke path reads the same gate and must not be blinded by the new nulling."""
    await _seed(events_repo)

    imminent = await events_repo.high_impact_near(
        RELEASE - timedelta(minutes=5), currencies=("USD", "EUR")
    )
    assert [e.title for e in imminent] == ["Non-Farm Employment Change"]
    assert imminent[0].actual is None


async def test_the_sql_function_gates_independently_of_the_repository(
    events_repo: EventRepository, session
) -> None:  # noqa: ANN001
    """The filter lives in SQL, so an ad-hoc query through the function is safe too."""
    await _seed(events_repo)

    before = (
        await session.execute(
            text("select actual, surprise_score from events_visible_at(:t)"),
            {"t": RELEASE - timedelta(minutes=1)},
        )
    ).one()
    assert before == (None, None)

    after = (
        await session.execute(
            text("select actual, surprise_score from events_visible_at(:t)"),
            {"t": RELEASE},
        )
    ).one()
    assert after == (189.0, 2.4)


async def test_a_result_published_after_its_event_is_still_gated_on_publication(
    events_repo: EventRepository,
) -> None:
    """The COT ordering: event Tuesday, published Friday. Publication still dominates.

    Gating actual on event_time must not accidentally *reveal* a row whose publication has not
    happened — the two filters compose, they do not replace each other.
    """
    event_time = datetime(2026, 3, 3, 20, 0, tzinfo=UTC)
    published = datetime(2026, 3, 6, 20, 30, tzinfo=UTC)
    await events_repo.upsert_many(
        [
            {
                "title": "CFTC Positioning",
                "event_time_utc": event_time,
                "publication_time_utc": published,
                "currency": "USD",
                "importance": "Medium",
                "source": "cftc",
                "actual": 42.0,
            }
        ]
    )

    # After the event, before publication: invisible entirely.
    assert await events_repo.visible_at(published - timedelta(hours=1)) == []
    # After publication: visible, and the value is there because the event already happened.
    visible = await events_repo.visible_at(published)
    assert len(visible) == 1
    assert visible[0].actual == 42.0
