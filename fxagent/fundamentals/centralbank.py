"""Official central bank RSS. Stores raw text; interprets nothing.

Feeds, each verified 14 Aug 2026 by fetching it and parsing the result:

| Bank | Currency | Feed | Items |
|---|---|---|---|
| Federal Reserve | USD | `federalreserve.gov/feeds/press_monetary.xml` | 15 |
| ECB | EUR | `ecb.europa.eu/rss/press.html` | 15 |
| Bank of England | GBP | `bankofengland.co.uk/rss/news` | 50 |
| Bank of Japan | JPY | `boj.or.jp/en/rss/whatsnew.xml` | 50 |

Two of those are not the URLs you would guess. The Bank of England's long-published
`boeapps/rss/feeds.aspx?feed=News` now 404s and `/rss/news` is live. The ECB's feed is served
at a `.html` path despite returning `application/rss+xml` — it looks like a mistake and is not.
The Bank of Japan's English feed is `/en/rss/whatsnew.xml`, not `/rss/whatsnew_en.xml`.

All four are RSS 2.0 with `pubDate` in RFC-2822, in four different offsets (GMT, +0200, +0100,
+0900). That field is the whole reason this source is point-in-time safe: unlike the calendar,
a press release states exactly when it became public, so `publication_time_utc` is read from
the feed rather than stamped at fetch time. Re-polling therefore cannot shift an item's
visibility, and a backfill after an outage lands items with their true original timestamps.

**Nothing here summarises, scores or classifies the content.** `body` is the feed's own
description text, unmodified. Reading meaning out of it is agent 3's job, and an agent that
receives pre-digested text cannot be audited against the original (hard rule 4).

**Importance is capped at Medium, deliberately.** `EventRepository.HIGH_IMPACT` is `("High",)`
and drives the auto-revoke trigger, so anything marked High here could revoke a live grant. A
press-release feed is the wrong thing to give that power: it publishes speeches, staff working
papers and survey results on the same channel as rate decisions, and a keyword list that
promotes the wrong one closes positions for a research paper. Scheduled decisions already
reach the permission layer through the calendar, which is built for exactly that. So policy-
looking items become Medium — visible in context, inert for gating — and everything else Low.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from fxagent.fundamentals.base import Event

__all__ = ["DEFAULT_FEEDS", "CentralBankFeed", "CentralBankRss"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CentralBankFeed:
    """One bank's feed and the currency its announcements move."""

    bank: str
    currency: str
    url: str


DEFAULT_FEEDS: tuple[CentralBankFeed, ...] = (
    CentralBankFeed("fed", "USD", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    CentralBankFeed("ecb", "EUR", "https://www.ecb.europa.eu/rss/press.html"),
    CentralBankFeed("boe", "GBP", "https://www.bankofengland.co.uk/rss/news"),
    CentralBankFeed("boj", "JPY", "https://www.boj.or.jp/en/rss/whatsnew.xml"),
)

#: Lower-cased substrings that mark an item as monetary policy rather than general comms.
#: Matched against the title only — a description mentioning "interest rate" in passing is not
#: a rate decision. Promotes to Medium and no further; see the module docstring.
_POLICY_MARKERS = (
    "monetary policy",
    "interest rate",
    "bank rate",
    "policy rate",
    "fomc",
    "mpc",
    "rate decision",
    "statement on monetary",
    "policy statement",
    "asset purchase",
    "quantitative",
)


def _classify(title: str) -> str:
    lowered = title.lower()
    return "Medium" if any(marker in lowered for marker in _POLICY_MARKERS) else "Low"


def _text(item: ET.Element, tag: str) -> str | None:
    """First matching child's text, namespace-agnostic. Feeds vary in whether they declare one."""
    found = item.find(f"{{*}}{tag}")
    if found is None:
        found = item.find(tag)
    if found is None or found.text is None:
        return None
    stripped = found.text.strip()
    return stripped or None


class CentralBankRss:
    """Polls one or more central bank RSS feeds.

    A failure in one feed does not lose the others: each is fetched and parsed independently
    and its exceptions are logged and swallowed here, so three banks still report when the
    fourth is down. `fetch_safely` in `base` is the outer net for a total failure.
    """

    def __init__(
        self,
        *,
        feeds: Sequence[CentralBankFeed] = DEFAULT_FEEDS,
        user_agent: str = "fx-regime-agent/0.1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._feeds = tuple(feeds)
        self._user_agent = user_agent
        self._owns_client = client is None
        # follow_redirects: the BoE and BoJ paths both redirect at least once.
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(20.0), follow_redirects=True
        )

    @property
    def name(self) -> str:
        return "centralbank"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> CentralBankRss:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def fetch(self, since: datetime) -> list[Event]:
        """Items published strictly after `since`, across every configured feed."""
        if since.tzinfo is None:
            raise ValueError("since must be timezone-aware")
        cutoff = since.astimezone(UTC)

        collected: list[Event] = []
        for feed in self._feeds:
            try:
                collected.extend(await self._fetch_feed(feed, cutoff))
            except Exception as exc:  # noqa: BLE001 - one bank down must not lose the others
                logger.warning(
                    "central bank feed %s failed, continuing without it: %s: %s",
                    feed.bank,
                    type(exc).__name__,
                    exc,
                )
        return collected

    async def _fetch_feed(self, feed: CentralBankFeed, cutoff: datetime) -> list[Event]:
        response = await self._client.get(
            feed.url, headers={"User-Agent": self._user_agent, "Accept": "application/rss+xml"}
        )
        response.raise_for_status()

        # from bytes, not text: the declared XML encoding is authoritative and httpx's charset
        # guess is not. The BoJ feed carries non-ASCII in titles.
        root = ET.fromstring(response.content)

        events: list[Event] = []
        skipped = 0
        for item in root.findall(".//{*}item"):
            event = self._to_event(item, feed)
            if event is None:
                skipped += 1
                continue
            if event.publication_time_utc <= cutoff:
                continue
            events.append(event)

        if skipped:
            logger.debug("%s: skipped %d item(s) with no usable date or title", feed.bank, skipped)
        return events

    def _to_event(self, item: ET.Element, feed: CentralBankFeed) -> Event | None:
        title = _text(item, "title")
        raw_date = _text(item, "pubDate") or _text(item, "date")
        if not title or not raw_date:
            return None

        try:
            published = parsedate_to_datetime(raw_date)
        except (TypeError, ValueError):
            return None
        if published.tzinfo is None:
            # A feed that omits the offset is ambiguous; guessing UTC would silently shift the
            # gate by up to a day. Dropping the item is the safe direction.
            return None
        published = published.astimezone(UTC)

        body = _text(item, "description")
        link = _text(item, "link")
        if link:
            # Kept in the body rather than dropped: the agents need somewhere to point, and
            # there is no column for a URL. Appended, never replacing the raw text.
            body = f"{body}\n\n{link}" if body else link

        return Event(
            # A press release happens when it is published. The two times coincide here, unlike
            # the calendar, and that is a property of the feed rather than a simplification.
            event_time_utc=published,
            publication_time_utc=published,
            source=feed.bank,
            currency=feed.currency,
            importance=_classify(title),
            title=title,
            category="central_bank",
            body=body,
        )
