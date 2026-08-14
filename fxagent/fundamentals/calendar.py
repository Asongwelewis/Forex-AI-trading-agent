"""The Forex Factory economic calendar, via faireconomy's keyless weekly JSON.

Endpoint, verified 14 Aug 2026:

    GET https://nfs.faireconomy.media/ff_calendar_thisweek.json

**A User-Agent is mandatory.** The host answers a request without one with HTTP 429, not 401,
which reads like rate limiting and sends you looking at request frequency. Measured directly:
with a UA the same request is 200, with `User-Agent:` blank it is 429.

**It also rate-limits genuine repeats**, so two fetches seconds apart can 429 even with a
User-Agent set. Hourly polling is comfortably inside that, but a retry loop must back off
rather than hammer, hence `_RETRY_STATUSES` and the Retry-After handling below.

Three properties of this feed shape everything here, all confirmed against a live response of
73 events on 14 Aug 2026:

* **There is no `actual` field.** The keys are exactly `title, country, date, impact, forecast,
  previous` — not "usually populated", *absent*. 36 of those 73 events were already in the past
  and still carried no result. So this source can never produce a surprise score, and
  `surprise_score` below will return None for every event it emits. The arithmetic is
  implemented and tested anyway, because it is the contract a result-bearing source plugs into
  (see `docs/ADR-003-fundamentals.md`), and because writing it later against a live feed is how
  you end up debugging both at once.
* **Only the current week exists.** `ff_calendar_lastweek`, `nextweek`, `thismonth` and
  `thisyear` are all 404. There is no backfill and no lookahead past Sunday, which is why
  `EventRepository.latest_publication_time` exists for the permission layer to fail closed on.
* **`country` holds a currency code**, not a country — 'USD', 'JPY'. The store's
  `events_currency_shape` constraint already assumes this.

**Publication time is the fetch time, and that is not a shortcut.** The feed carries no
publication metadata, so the only honest statement is "we learned of this event now". Using the
event's own date would claim we knew Friday's payrolls number on Friday when the row was first
seen on Monday — backwards look-ahead, and precisely what hard rule 6 forbids. The store never
updates `publication_time_utc` on conflict, so the first sighting wins and re-polling cannot
retroactively make an event visible earlier.
"""

from __future__ import annotations

import asyncio
import logging
import re
import statistics
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from fxagent.fundamentals.base import Event

__all__ = ["CALENDAR_URL", "ForexFactoryCalendar", "parse_value", "surprise_score"]

logger = logging.getLogger(__name__)

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

SOURCE = "forexfactory"

#: Worth retrying with a wait. 429 is the documented answer to a missing User-Agent *and* to a
#: genuine repeat, so it is retried rather than treated as fatal.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Suffix multipliers as the feed writes them: "2.50T", "125K", "-1.2B".
_MULTIPLIERS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

#: An optional comparator, an optional currency symbol, the number, an optional multiplier and
#: an optional percent sign. Anchored, so anything unrecognised returns None instead of a
#: partial match — a value silently parsed as the wrong magnitude is worse than a missing one.
_VALUE_RE = re.compile(
    r"""^
    [<>~]?\s*
    [-+]?
    \s*[$€£¥]?\s*
    (?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    \s*(?P<multiplier>[KMBT])?
    \s*%?
    $""",
    re.VERBOSE | re.IGNORECASE,
)

#: Values the feed uses for "nothing here". Checked before the regex so they do not log.
_EMPTY_VALUES = frozenset({"", "-", "--", "n/a", "na", "tentative", "tbd"})


def parse_value(raw: str | float | int | None) -> float | None:
    """Turn a feed value like '2.50T', '-1.2%' or '125K' into a float, or None.

    Percent signs are dropped rather than divided by 100: a forecast of '5.7%' against an
    actual of '5.9%' has a surprise of 0.2 in the units the calendar publishes, and rescaling
    one side only would be worse than not rescaling at all. Every comparison in this module is
    between two values from the same field of the same event, so the unit cancels.
    """
    if raw is None:
        return None
    if isinstance(raw, int | float):
        return float(raw)

    text = raw.strip()
    if text.lower() in _EMPTY_VALUES:
        return None

    match = _VALUE_RE.match(text)
    if match is None:
        # Not an error: the feed carries prose in some fields ("Actual > Forecast"). Log at
        # debug so a systematic parse failure is discoverable without spamming a normal run.
        logger.debug("calendar value %r did not parse as a number", text)
        return None

    number = float(match.group("number").replace(",", ""))
    if text.lstrip("<>~ ").startswith("-"):
        number = -number
    multiplier = match.group("multiplier")
    if multiplier:
        number *= _MULTIPLIERS[multiplier.upper()]
    return number


def surprise_score(
    actual: float | None,
    forecast: float | None,
    history: Sequence[float],
    *,
    minimum_history: int = 8,
) -> float | None:
    """`(actual - forecast) / stdev(history)`, or None when that number would be a fiction.

    Returns None — neutral — in four cases, all of which are "we do not know" rather than
    "no surprise", and conflating those two is how a missing feed becomes a fabricated signal:

    * **No forecast.** Nothing to be surprised against. The task's explicit requirement.
    * **No actual.** The release has not happened, or the source does not carry results — which
      is every event from this calendar.
    * **Too little history.** A standard deviation over three samples is noise with a decimal
      point. Eight is a low bar, and still arbitrary; it is a floor, not a blessing.
    * **Zero spread.** A perfectly-forecast series divides by zero. Some prints genuinely never
      miss, and the honest answer for them is "this release carries no information", not ±inf.

    `history` is the sequence of prior `actual - forecast` values for the *same* release, in the
    same units, and the caller is responsible for that — mixing CPI into a payrolls history
    produces a plausible-looking number that means nothing.
    """
    if actual is None or forecast is None:
        return None
    if len(history) < minimum_history:
        return None

    spread = statistics.stdev(history)
    if spread == 0.0:
        return None
    return (actual - forecast) / spread


class ForexFactoryCalendar:
    """Polls the weekly calendar JSON. Holds no state between calls beyond its HTTP client."""

    def __init__(
        self,
        *,
        user_agent: str,
        client: httpx.AsyncClient | None = None,
        url: str = CALENDAR_URL,
        now: Callable[[], datetime] | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        if not user_agent.strip():
            raise ValueError(
                "user_agent is required: faireconomy answers a request without one with "
                "HTTP 429, which looks like rate limiting and is not. Set CALENDAR_USER_AGENT."
            )
        self._user_agent = user_agent
        self._url = url
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0))
        self._now = now or (lambda: datetime.now(UTC))
        self._max_attempts = max_attempts
        self._backoff = backoff_seconds

    @property
    def name(self) -> str:
        return SOURCE

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> ForexFactoryCalendar:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def fetch(self, since: datetime) -> list[Event]:
        """Every calendar entry at or after `since`, stamped as published now.

        `since` filters on *event* time here, because this feed has no publication time to
        filter on — see the module docstring. That is not a point-in-time violation: the
        publication stamp written to the store is still the moment we learned of the row.
        """
        if since.tzinfo is None:
            raise ValueError("since must be timezone-aware")

        payload = await self._get_json()
        fetched_at = self._now().astimezone(UTC)

        events: list[Event] = []
        skipped = 0
        for entry in payload:
            event = self._to_event(entry, fetched_at=fetched_at)
            if event is None:
                skipped += 1
                continue
            if event.event_time_utc < since.astimezone(UTC):
                continue
            events.append(event)

        if skipped:
            logger.warning(
                "calendar: skipped %d unparseable entr(ies) of %d", skipped, len(payload)
            )
        return events

    async def _get_json(self) -> list[dict[str, Any]]:
        headers = {"User-Agent": self._user_agent, "Accept": "application/json"}
        last_status: int | None = None

        for attempt in range(1, self._max_attempts + 1):
            response = await self._client.get(self._url, headers=headers)
            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload, list):
                    raise ValueError(
                        f"calendar returned {type(payload).__name__}, expected a JSON array"
                    )
                return payload

            last_status = response.status_code
            if response.status_code not in _RETRY_STATUSES or attempt == self._max_attempts:
                break

            # Honour Retry-After when offered; the host does not always send one.
            wait = self._backoff * attempt
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = float(retry_after)
            logger.warning(
                "calendar HTTP %d (attempt %d/%d), retrying in %.1fs",
                response.status_code,
                attempt,
                self._max_attempts,
                wait,
            )
            await asyncio.sleep(wait)

        raise httpx.HTTPStatusError(
            f"calendar returned HTTP {last_status} after {self._max_attempts} attempt(s)",
            request=httpx.Request("GET", self._url),
            response=httpx.Response(last_status or 599),
        )

    def _to_event(self, entry: dict[str, Any], *, fetched_at: datetime) -> Event | None:
        """One feed entry to an `Event`, or None when the entry is unusable.

        Returns None rather than raising so one malformed row cannot cost the other seventy-two.
        """
        try:
            raw_date = str(entry["date"])
            currency = str(entry["country"]).strip().upper()
            title = str(entry["title"]).strip()
            importance = str(entry["impact"]).strip().title()
        except (KeyError, AttributeError):
            return None

        # The feed writes an explicit offset ('2026-08-09T19:50:00-04:00'), so this is exact and
        # needs no assumption about which timezone Forex Factory is rendering in.
        try:
            event_time = datetime.fromisoformat(raw_date)
        except ValueError:
            return None
        if event_time.tzinfo is None:
            return None

        if not title or len(currency) != 3 or not currency.isalpha():
            return None
        if importance not in ("High", "Medium", "Low", "Holiday"):
            return None

        return Event(
            event_time_utc=event_time.astimezone(UTC),
            publication_time_utc=fetched_at,
            source=SOURCE,
            currency=currency,
            importance=importance,
            title=title,
            category="calendar",
            forecast=parse_value(entry.get("forecast")),
            actual=parse_value(entry.get("actual")),
            previous=parse_value(entry.get("previous")),
            # Always None from this feed, which carries no actual. Left explicit so the reason
            # is visible at the call site rather than inferred from an absent keyword.
            surprise_score=None,
        )
