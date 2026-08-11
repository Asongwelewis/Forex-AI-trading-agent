"""The look-ahead test: no events read path may return an unpublished row.

This is the test hard rule 6 exists for. A backtest that can see an event before it was
published produces a beautiful equity curve and a strategy that loses money live, and the
failure is invisible — nothing errors, the numbers are just wrong.

Three distinct ways of getting this wrong are covered:

1. Filtering on `event_time_utc` instead of `publication_time_utc`. Caught by rows whose
   publication trails their event (the COT case) and rows whose event trails their
   publication (a scheduled release).
2. Adding a read method that forgets the filter. Caught by `test_every_read_method_...`,
   which discovers the public read surface by reflection rather than by a hand-kept list, so
   a method added later is covered without anyone remembering to add it here.
3. Bypassing the repository entirely. Caught by exercising the `events_visible_at` SQL
   function directly.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from fxagent.store.repositories.events import EventRepository

from .conftest import requires_postgres

pytestmark = [pytest.mark.db, requires_postgres]

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


def _event(
    *,
    title: str,
    event_time: datetime,
    publication_time: datetime,
    currency: str = "USD",
    importance: str = "High",
    source: str = "forexfactory",
) -> dict[str, object]:
    return {
        "title": title,
        "event_time_utc": event_time,
        "publication_time_utc": publication_time,
        "currency": currency,
        "importance": importance,
        "source": source,
        "category": "calendar",
        "actual": None,
        "forecast": None,
        "previous": None,
    }


async def _seed(repo: EventRepository) -> None:
    """A published event, an unpublished one, and the two awkward orderings."""
    await repo.upsert_many(
        [
            # Published an hour ago, happens in ten minutes. Visible.
            _event(
                title="published-past",
                event_time=NOW + timedelta(minutes=10),
                publication_time=NOW - timedelta(hours=1),
            ),
            # Not published until tomorrow. Must never be visible at NOW, even though its
            # event time is in the past — this is the CFTC COT shape.
            _event(
                title="cot-style-late-publication",
                event_time=NOW - timedelta(days=3),
                publication_time=NOW + timedelta(days=1),
            ),
            # Published exactly at NOW. Boundary: `<=` means visible.
            _event(
                title="published-exactly-now",
                event_time=NOW + timedelta(minutes=5),
                publication_time=NOW,
            ),
            # Published one microsecond after NOW. Boundary: must be invisible.
            _event(
                title="published-just-after",
                event_time=NOW + timedelta(minutes=5),
                publication_time=NOW + timedelta(microseconds=1),
            ),
            # Far future publication, event also far future.
            _event(
                title="fully-future",
                event_time=NOW + timedelta(days=5),
                publication_time=NOW + timedelta(days=4),
            ),
        ]
    )


async def test_visible_at_excludes_unpublished_rows(events_repo: EventRepository) -> None:
    await _seed(events_repo)

    visible = await events_repo.visible_at(NOW)
    titles = {record.title for record in visible}

    assert titles == {"published-past", "published-exactly-now"}
    assert "cot-style-late-publication" not in titles
    assert "published-just-after" not in titles


async def test_publication_boundary_is_inclusive_to_the_microsecond(
    events_repo: EventRepository,
) -> None:
    """`<=`, not `<`: an event published exactly at the query instant is knowable."""
    await _seed(events_repo)

    visible = {record.title for record in await events_repo.visible_at(NOW)}
    assert "published-exactly-now" in visible

    just_before = {
        record.title for record in await events_repo.visible_at(NOW - timedelta(microseconds=1))
    }
    assert "published-exactly-now" not in just_before


async def test_event_window_query_still_respects_publication_time(
    events_repo: EventRepository,
) -> None:
    """A window on event time must not become a way around the publication filter.

    `cot-style-late-publication` sits inside this event-time window and is unpublished. A
    query that filtered only on event time would return it.
    """
    await _seed(events_repo)

    found = await events_repo.in_event_window(
        NOW,
        window_start=NOW - timedelta(days=7),
        window_end=NOW + timedelta(days=7),
    )
    titles = {record.title for record in found}

    assert "cot-style-late-publication" not in titles
    assert "fully-future" not in titles
    assert titles == {"published-past", "published-exactly-now"}


async def test_high_impact_near_cannot_see_an_unpublished_release(
    events_repo: EventRepository,
) -> None:
    """The auto-revoke trigger must not fire on, or be blinded by, unpublished data."""
    await _seed(events_repo)

    near = await events_repo.high_impact_near(NOW, currencies=["USD"])
    titles = {record.title for record in near}

    # Both remaining events fall inside the +15m window and are published.
    assert titles == {"published-past", "published-exactly-now"}


@pytest.mark.parametrize(
    "method_name",
    sorted(
        name
        for name, member in inspect.getmembers(EventRepository, inspect.isfunction)
        if not name.startswith("_") and "as_of" in inspect.signature(member).parameters
    ),
)
async def test_every_read_method_requires_as_of(method_name: str) -> None:
    """`as_of` must be required and positional on every read.

    A default would make it omittable, and the one call site that omitted it would be the one
    that mattered. Discovered by reflection so a method added later is covered automatically.
    """
    signature = inspect.signature(getattr(EventRepository, method_name))
    parameter = signature.parameters["as_of"]

    assert parameter.default is inspect.Parameter.empty, (
        f"{method_name}() gives as_of a default; a point-in-time filter that can be omitted "
        "will eventually be omitted"
    )
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    ordered = [name for name in signature.parameters if name != "self"]
    assert ordered[0] == "as_of", f"{method_name}() must take as_of first, got {ordered}"


async def test_no_public_read_method_lacks_as_of() -> None:
    """Guards the guard: catches a read method added without an `as_of` parameter at all."""
    write_methods = {"upsert_many"}
    missing = [
        name
        for name, member in inspect.getmembers(EventRepository, inspect.isfunction)
        if not name.startswith("_")
        and name not in write_methods
        and "as_of" not in inspect.signature(member).parameters
    ]
    assert not missing, (
        f"EventRepository read methods without an as_of parameter: {missing}. Every read is "
        "point-in-time; add as_of, or add the method to write_methods if it is a write."
    )


async def test_sql_gate_function_filters_independently_of_the_repository(
    events_repo: EventRepository, session: object
) -> None:
    """The filter lives in SQL, so an ad-hoc query through the function is safe too."""
    await _seed(events_repo)
    await session.flush()  # type: ignore[attr-defined]

    result = await session.execute(  # type: ignore[attr-defined]
        text("select count(*) from events_visible_at(:as_of)"), {"as_of": NOW}
    )
    assert result.scalar_one() == 2

    total = await session.execute(text("select count(*) from events"))  # type: ignore[attr-defined]
    assert total.scalar_one() == 5, "all five rows are stored; only visibility differs"


async def test_naive_as_of_is_rejected(events_repo: EventRepository) -> None:
    """A naive datetime would be read in the session timezone and silently shift the gate."""
    with pytest.raises(ValueError, match="timezone-aware"):
        await events_repo.visible_at(datetime(2026, 3, 10, 12, 0))


async def test_visibility_grows_monotonically_with_as_of(
    events_repo: EventRepository,
) -> None:
    """Advancing the clock may reveal events but must never hide one."""
    await _seed(events_repo)

    seen: set[str] = set()
    for offset_days in (0, 1, 2, 5):
        as_of = NOW + timedelta(days=offset_days)
        titles = {record.title for record in await events_repo.visible_at(as_of)}
        assert seen <= titles, (
            f"an event visible earlier disappeared at as_of={as_of}: {seen - titles}"
        )
        seen = titles

    assert len(seen) == 5, "by the last checkpoint every seeded event has been published"
