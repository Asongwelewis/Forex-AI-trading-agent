"""The calendar source: value parsing, surprise arithmetic, and honest publication stamps.

No network. The fixture payload is copied from a real response captured on 14 Aug 2026, keys
and all, so a schema change upstream shows up as these tests passing against a shape the live
feed no longer sends — which is why `test_live_feed_still_has_the_shape_we_parse` exists and is
marked so it can be run deliberately.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from fxagent.fundamentals.calendar import (
    ForexFactoryCalendar,
    parse_value,
    surprise_score,
)

FETCHED_AT = datetime(2026, 8, 14, 10, 0, tzinfo=UTC)

# Verbatim shape from the live feed: six keys, no `actual`.
PAYLOAD = [
    {
        "title": "Bank Lending y/y",
        "country": "JPY",
        "date": "2026-08-09T19:50:00-04:00",
        "impact": "Low",
        "forecast": "5.7%",
        "previous": "5.7%",
    },
    {
        "title": "Non-Farm Employment Change",
        "country": "USD",
        "date": "2026-08-14T08:30:00-04:00",
        "impact": "High",
        "forecast": "125K",
        "previous": "147K",
    },
    {
        "title": "Current Account",
        "country": "JPY",
        "date": "2026-08-09T19:50:00-04:00",
        "impact": "Low",
        "forecast": "2.50T",
        "previous": "3.06T",
    },
    {
        "title": "Bank Holiday",
        "country": "EUR",
        "date": "2026-08-15T00:00:00-04:00",
        "impact": "Holiday",
        "forecast": "",
        "previous": "",
    },
]


def _calendar(payload: object = PAYLOAD, status: int = 200) -> ForexFactoryCalendar:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return ForexFactoryCalendar(
        user_agent="fx-regime-agent/test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        now=lambda: FETCHED_AT,
        max_attempts=1,
    )


# -- value parsing -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5.7%", 5.7),
        ("-1.2%", -1.2),
        ("125K", 125_000.0),
        ("2.50T", 2.5e12),
        ("-1.2B", -1.2e9),
        ("3.06T", 3.06e12),
        ("1,234", 1234.0),
        ("$1.2M", 1.2e6),
        ("<0.1%", 0.1),
        ("0", 0.0),
        ("", None),
        ("   ", None),
        ("N/A", None),
        ("Tentative", None),
        ("Actual > Forecast", None),
        (None, None),
        (4.25, 4.25),
    ],
)
def test_parse_value(raw: object, expected: float | None) -> None:
    assert parse_value(raw) == expected  # type: ignore[arg-type]


def test_percent_is_not_rescaled() -> None:
    """'5.7%' is 5.7, not 0.057.

    Both sides of a surprise come from the same field of the same release, so the unit cancels.
    Dividing one side by 100 and not the other is the bug this pins down.
    """
    assert parse_value("5.7%") == 5.7
    assert parse_value("5.9%") == 5.9


# -- surprise ------------------------------------------------------------------------------


def test_surprise_is_the_forecast_miss_in_standard_deviations() -> None:
    history = [1.0, -1.0, 2.0, -2.0, 1.5, -1.5, 0.5, -0.5]
    score = surprise_score(actual=110.0, forecast=100.0, history=history)
    assert score is not None
    assert score == pytest.approx(10.0 / __import__("statistics").stdev(history))


@pytest.mark.parametrize(
    ("actual", "forecast"),
    [(None, 100.0), (110.0, None), (None, None)],
)
def test_surprise_is_neutral_without_both_sides(
    actual: float | None, forecast: float | None
) -> None:
    """The task's explicit requirement: no forecast means neutral, not zero."""
    assert surprise_score(actual=actual, forecast=forecast, history=[1.0] * 20) is None


def test_surprise_is_neutral_on_thin_history() -> None:
    assert surprise_score(actual=110.0, forecast=100.0, history=[1.0, -1.0, 2.0]) is None


def test_surprise_is_neutral_when_the_release_never_misses() -> None:
    """Zero spread would divide by zero; the honest reading is 'carries no information'."""
    assert surprise_score(actual=110.0, forecast=100.0, history=[0.0] * 12) is None


# -- fetching ------------------------------------------------------------------------------


async def test_publication_time_is_the_fetch_time_not_the_event_date() -> None:
    """The point-in-time guarantee, at the source.

    The feed carries no publication metadata. Stamping an event with its own date would claim
    we knew Friday's number on Friday when the row was first seen today — and the store never
    rewrites publication_time_utc, so a wrong value here is permanent.
    """
    async with _calendar() as calendar:
        events = await calendar.fetch(since=datetime(2026, 8, 1, tzinfo=UTC))

    assert events
    assert all(e.publication_time_utc == FETCHED_AT for e in events)
    # And at least one event's own time is well before that stamp, so the two cannot be
    # coincidentally equal.
    assert any(e.event_time_utc < FETCHED_AT for e in events)


async def test_event_times_are_converted_from_the_feed_offset() -> None:
    async with _calendar() as calendar:
        events = await calendar.fetch(since=datetime(2026, 8, 1, tzinfo=UTC))

    nfp = next(e for e in events if e.title == "Non-Farm Employment Change")
    # 08:30 at -04:00 is 12:30 UTC.
    assert nfp.event_time_utc == datetime(2026, 8, 14, 12, 30, tzinfo=UTC)


async def test_since_filters_on_event_time() -> None:
    async with _calendar() as calendar:
        events = await calendar.fetch(since=datetime(2026, 8, 14, tzinfo=UTC))

    titles = {e.title for e in events}
    assert "Non-Farm Employment Change" in titles
    assert "Bank Lending y/y" not in titles  # dated 09 Aug


async def test_actual_and_surprise_are_none_because_the_feed_has_neither() -> None:
    async with _calendar() as calendar:
        events = await calendar.fetch(since=datetime(2026, 8, 1, tzinfo=UTC))

    assert all(e.actual is None for e in events)
    assert all(e.surprise_score is None for e in events)
    # Forecast and previous *are* present, so this is not an everything-is-None false pass.
    assert any(e.forecast is not None for e in events)


async def test_holiday_importance_survives() -> None:
    """'Holiday' is one of the four the check constraint allows; dropping it loses rows."""
    async with _calendar() as calendar:
        events = await calendar.fetch(since=datetime(2026, 8, 1, tzinfo=UTC))
    assert any(e.importance == "Holiday" for e in events)


async def test_one_bad_entry_does_not_lose_the_others() -> None:
    payload = [*PAYLOAD, {"title": "", "country": "??", "date": "not-a-date", "impact": "???"}]
    async with _calendar(payload) as calendar:
        events = await calendar.fetch(since=datetime(2026, 8, 1, tzinfo=UTC))
    assert len(events) == len(PAYLOAD)


async def test_missing_user_agent_is_refused_at_construction() -> None:
    """faireconomy answers a UA-less request with 429, which reads like rate limiting."""
    with pytest.raises(ValueError, match="user_agent is required"):
        ForexFactoryCalendar(user_agent="   ")


async def test_http_error_raises_so_fetch_safely_can_catch_it() -> None:
    async with _calendar(status=429) as calendar:
        with pytest.raises(httpx.HTTPStatusError):
            await calendar.fetch(since=datetime(2026, 8, 1, tzinfo=UTC))


@pytest.mark.network
async def test_live_feed_still_has_the_shape_we_parse() -> None:
    """Deselected by default; run with `-m network`.

    Everything above tests a payload captured on 14 Aug 2026. That is the right way to test
    parsing and the wrong way to notice that faireconomy renamed a key — the fixture would keep
    passing forever against a shape the live feed no longer sends. This is the only test that
    can tell the difference, which is why it exists despite being unrunnable offline.

    It asserts the *contract*, not the contents: the keys we read, and that `actual` is still
    absent. If `actual` ever appears, the surprise machinery in this module becomes reachable
    and `calendar.py`'s docstring needs revisiting — so that is asserted too, as a tripwire
    rather than a requirement.
    """
    async with ForexFactoryCalendar(user_agent="fx-regime-agent/test") as calendar:
        events = await calendar.fetch(since=datetime(2020, 1, 1, tzinfo=UTC))

    assert events, "the live calendar returned nothing at all"
    assert all(len(e.currency) == 3 for e in events)
    assert {e.importance for e in events} <= {"High", "Medium", "Low", "Holiday"}
    assert any(e.forecast is not None for e in events), "no forecast parsed from any event"
    assert all(e.actual is None for e in events), (
        "the live feed now carries `actual` — surprise scoring is reachable and "
        "calendar.py's docstring is out of date"
    )


async def test_rows_are_ready_for_the_store() -> None:
    async with _calendar() as calendar:
        events = await calendar.fetch(since=datetime(2026, 8, 1, tzinfo=UTC))

    row = events[0].as_row()
    assert set(row) >= {
        "event_time_utc",
        "publication_time_utc",
        "source",
        "currency",
        "importance",
        "title",
    }
    assert row["source"] == "forexfactory"
    assert json.dumps(row, default=str)  # no un-serialisable leftovers
