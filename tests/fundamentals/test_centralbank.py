"""Central bank RSS: real publication times, raw text, and an importance ceiling.

Payloads mirror the four live feeds as sampled on 14 Aug 2026 — including that the ECB's items
carry no `description`, and that the four banks publish in four different UTC offsets.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from fxagent.fundamentals.centralbank import DEFAULT_FEEDS, CentralBankFeed, CentralBankRss

#: Before every fixture item below, so `since` does not silently filter the case under test.
#: The exclusivity of `since` itself is pinned separately by
#: `test_since_is_exclusive_so_repolling_does_not_re_emit`.
SINCE = datetime(2026, 7, 1, tzinfo=UTC)


def _rss(*items: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        + "".join(items)
        + "</channel></rss>"
    )


def _item(title: str, pub: str, description: str | None = None, link: str | None = None) -> str:
    parts = [f"<title>{title}</title>", f"<pubDate>{pub}</pubDate>"]
    if description is not None:
        parts.append(f"<description>{description}</description>")
    if link is not None:
        parts.append(f"<link>{link}</link>")
    return f"<item>{''.join(parts)}</item>"


def _feeds(body: str, *, bank: str = "fed", currency: str = "USD") -> CentralBankRss:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"))

    return CentralBankRss(
        feeds=(CentralBankFeed(bank, currency, "https://example.invalid/feed.xml"),),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_publication_time_comes_from_pubdate_not_fetch_time() -> None:
    """Unlike the calendar, a press release states when it went public. Use that.

    This is what makes a backfill after an outage safe: items land with their true original
    timestamps rather than all appearing to have been published the moment we caught up.
    """
    body = _rss(_item("Federal Reserve issues FOMC statement", "Wed, 29 Jul 2026 18:00:00 GMT"))
    async with _feeds(body) as source:
        events = await source.fetch(datetime(2026, 7, 1, tzinfo=UTC))

    assert len(events) == 1
    assert events[0].publication_time_utc == datetime(2026, 7, 29, 18, 0, tzinfo=UTC)
    # A press release happens when it is published; the two coincide here by nature.
    assert events[0].event_time_utc == events[0].publication_time_utc


@pytest.mark.parametrize(
    ("pub", "expected"),
    [
        ("Wed, 29 Jul 2026 18:00:00 GMT", datetime(2026, 7, 29, 18, 0, tzinfo=UTC)),
        ("Thu, 13 Aug 2026 11:00:00 +0200", datetime(2026, 8, 13, 9, 0, tzinfo=UTC)),
        ("Tue, 11 Aug 2026 14:00:00 +0100", datetime(2026, 8, 11, 13, 0, tzinfo=UTC)),
        ("Fri, 14 Aug 2026 16:00:00 +0900", datetime(2026, 8, 14, 7, 0, tzinfo=UTC)),
    ],
)
async def test_each_bank_s_offset_is_converted(pub: str, expected: datetime) -> None:
    """All four live feeds use a different offset; none of them is UTC by default."""
    async with _feeds(_rss(_item("Something", pub))) as source:
        events = await source.fetch(SINCE)
    assert events[0].publication_time_utc == expected


async def test_since_is_exclusive_so_repolling_does_not_re_emit() -> None:
    body = _rss(_item("Old news", "Wed, 29 Jul 2026 18:00:00 GMT"))
    async with _feeds(body) as source:
        events = await source.fetch(datetime(2026, 7, 29, 18, 0, tzinfo=UTC))
    assert events == []


async def test_body_is_the_raw_description_not_a_summary() -> None:
    """Hard rule 4: interpretation is the agent's job, and it needs the original to audit."""
    raw = "The Committee decided to maintain the target range at 4-1/4 to 4-1/2 percent."
    body = _rss(_item("FOMC statement", "Wed, 29 Jul 2026 18:00:00 GMT", description=raw))
    async with _feeds(body) as source:
        events = await source.fetch(SINCE)
    assert raw in (events[0].body or "")


async def test_link_is_appended_without_replacing_the_text() -> None:
    body = _rss(
        _item(
            "FOMC statement",
            "Wed, 29 Jul 2026 18:00:00 GMT",
            description="Rates held.",
            link="https://example.invalid/a",
        )
    )
    async with _feeds(body) as source:
        events = await source.fetch(SINCE)
    assert "Rates held." in (events[0].body or "")
    assert "https://example.invalid/a" in (events[0].body or "")


async def test_missing_description_is_tolerated() -> None:
    """The ECB feed carries title, link, guid and pubDate only."""
    body = _rss(_item("ECB press release", "Thu, 13 Aug 2026 11:00:00 +0200"))
    async with _feeds(body, bank="ecb", currency="EUR") as source:
        events = await source.fetch(SINCE)
    assert len(events) == 1


@pytest.mark.parametrize(
    "title",
    [
        "Federal Reserve issues FOMC statement",
        "Monetary Policy Summary, August 2026",
        "Bank Rate maintained at 4%",
        "MPC votes to hold",
    ],
)
async def test_policy_items_reach_medium(title: str) -> None:
    async with _feeds(_rss(_item(title, "Wed, 29 Jul 2026 18:00:00 GMT"))) as source:
        events = await source.fetch(SINCE)
    assert events[0].importance == "Medium"


async def test_rss_never_produces_a_high_impact_event() -> None:
    """The ceiling that keeps a press feed away from the auto-revoke trigger.

    `HIGH_IMPACT` is ("High",). If a working paper or a speech could be marked High, it would
    revoke a live grant. Scheduled decisions reach the permission layer via the calendar, which
    is built for it; this feed stays advisory.
    """
    titles = [
        "Federal Reserve issues FOMC statement",
        "(Research Paper) What Prevents Productivity Gains",
        "Results of the Semi-Annual FX Turnover Surveys",
        "Speech by the Governor",
        "Interest rate decision",
    ]
    body = _rss(*(_item(t, "Wed, 29 Jul 2026 18:00:00 GMT") for t in titles))
    async with _feeds(body) as source:
        events = await source.fetch(SINCE)

    assert len(events) == len(titles)
    assert {e.importance for e in events} <= {"Low", "Medium"}


async def test_undated_item_is_dropped_rather_than_guessed() -> None:
    """A missing offset is ambiguous; assuming UTC could shift the gate by up to a day."""
    body = _rss("<item><title>No date here</title></item>")
    async with _feeds(body) as source:
        events = await source.fetch(SINCE)
    assert events == []


async def test_one_failing_bank_does_not_lose_the_others() -> None:
    good = _rss(_item("FOMC statement", "Wed, 29 Jul 2026 18:00:00 GMT"))

    def handler(request: httpx.Request) -> httpx.Response:
        if "broken" in str(request.url):
            return httpx.Response(503)
        return httpx.Response(200, content=good.encode("utf-8"))

    source = CentralBankRss(
        feeds=(
            CentralBankFeed("broken", "EUR", "https://example.invalid/broken.xml"),
            CentralBankFeed("fed", "USD", "https://example.invalid/ok.xml"),
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with source:
        events = await source.fetch(SINCE)

    assert [e.source for e in events] == ["fed"]


def test_default_feeds_cover_the_four_majors() -> None:
    assert {f.currency for f in DEFAULT_FEEDS} == {"USD", "EUR", "GBP", "JPY"}
    assert all(f.url.startswith("https://") for f in DEFAULT_FEEDS)
