"""CFTC Commitments of Traders positioning, as a crowding percentile.

Endpoint, verified 14 Aug 2026 by fetching it and parsing the result:

    GET https://publicreporting.cftc.gov/resource/6dca-aqww.json

That is the **Legacy Futures-Only** COT report on the CFTC's public Socrata instance, and it is
deliberately not the Traders in Financial Futures report the card named. TFF lives at
`resource/gpe5-46if.json`, is equally live, and has no non-commercial columns at all — it splits
reportable traders into dealer / asset manager / leveraged money / other reportable. Asking it
for `noncomm_positions_long_all` returns `query.soql.no-such-column`, measured, not assumed.
"Non-commercial long and short" is Legacy vocabulary, so the data the card actually asks for
only exists in the Legacy dataset. Flagged rather than silently substituted, because the two
reports partition the same open interest differently and a percentile built on one is not
comparable to a percentile built on the other.

One request covers every configured currency: `cftc_contract_market_code in (...)` with a date
floor returned 1316 rows, 188 weekly observations per contract, back to Jan 2023. History runs
to 1986 for EUR (1450 weeks), so the 156-week window is never the binding constraint in
production — only in a fresh store.

**Contract codes are an explicit map, never string matching.** `CONTRACT_CODES` below is keyed
by currency and holds the six-digit CFTC code. The contract *names* share no naming rule —
'EURO FX', 'BRITISH POUND', 'NZ DOLLAR' — and a substring match for 'DOLLAR' hits four of them.

**Report date is not publication date, and the gap is three days.** The COT references the close
of business Tuesday and is released the following Friday at 15:30 US/Eastern. Filtering on the
reference date would let a backtest read Tuesday's positioning on Tuesday: a 72-hour look-ahead
on a weekly series, which is most of the week. `CotHistory.visible_at` and `cot_visible_at()` in
migration 0011 are the two gates, one for the in-memory path and one for the stored path, and
`positioning_score` cannot be reached without passing through one of them.

**The reference date is not always a Tuesday.** Two of the 188 observations fetched — 2023-07-03
and 2025-11-10 — are Mondays, shifted by a US federal holiday. `publication_time` therefore
computes "the next Friday strictly after the reference date" rather than adding three days, and
`_next_friday` is what makes the holiday weeks land correctly instead of a day early.

**Nothing here is a trigger.** `positioning_score` is a crowding measure: +1 says non-commercial
longs are at a three-year extreme, which is a statement about how much fuel is left, not about
direction. It confirms and it dampens confidence; it never opens a position and never picks a
side. See `carry_divergence` and `regime.consensus` for the two places it is read.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx

from fxagent.fundamentals.base import Event
from fxagent.fundamentals.context import split_symbol

__all__ = [
    "CONTRACT_CODES",
    "COT_DATASET_URL",
    "MIN_HISTORY_WEEKS",
    "NEUTRAL_SCORE",
    "PERCENTILE_WINDOW_WEEKS",
    "CftcCotSource",
    "CotHistory",
    "CotReport",
    "PairPositioning",
    "percentile_rank",
    "poll",
    "publication_time",
    "reference_time",
    "should_fetch",
]

logger = logging.getLogger(__name__)

COT_DATASET_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

SOURCE = "cftc_cot"

#: Currency -> CFTC contract market code, on the CME unless noted. Explicit by construction:
#: every value here was fetched individually on 14 Aug 2026 and returned a current report.
#: Adding a currency means looking its code up, not extending a pattern.
CONTRACT_CODES: dict[str, str] = {
    "EUR": "099741",  # EURO FX
    "GBP": "096742",  # BRITISH POUND
    "JPY": "097741",  # JAPANESE YEN
    "CHF": "092741",  # SWISS FRANC
    "AUD": "232741",  # AUSTRALIAN DOLLAR
    "CAD": "090741",  # CANADIAN DOLLAR
    "NZD": "112741",  # NZ DOLLAR
}

#: Reverse map, built once. Codes are unique by construction; a duplicate would silently drop a
#: currency, so it is asserted rather than assumed.
_CURRENCY_BY_CODE: dict[str, str] = {code: currency for currency, code in CONTRACT_CODES.items()}
if len(_CURRENCY_BY_CODE) != len(CONTRACT_CODES):  # pragma: no cover - a typo guard
    raise RuntimeError("CONTRACT_CODES maps two currencies to one contract code")

#: The ranking window. Three years of weekly observations.
PERCENTILE_WINDOW_WEEKS = 156

#: Below this, the percentile is not reported at all. 52 weekly points is the least that gives a
#: rank any resolution: with 10 observations every value is within 10 percentiles of an extreme,
#: so a fresh store would emit "maximally crowded" for whatever happened to be highest that
#: month. Returning neutral is not a lesser answer here — it is the only true one.
MIN_HISTORY_WEEKS = 52

#: What every "we do not know" path returns. Midpoint of [-1, 1], so it biases nothing.
NEUTRAL_SCORE = 0.0

#: Release schedule. 15:30 in this zone, the Friday after the reference date.
_EASTERN = ZoneInfo("America/New_York")
_RELEASE_TIME = time(15, 30)
#: Positions are measured at the close of business on the reference date.
_REFERENCE_CLOSE = time(16, 30)

#: How long a fetched batch stays fresh. The report changes once a week; anything more frequent
#: than daily is load on a public endpoint for data that provably has not moved.
DEFAULT_MIN_FETCH_INTERVAL = timedelta(days=1)

#: Weeks pulled per request. Comfortably over the 156-week window so a gap-filling backfill does
#: not need a second round trip, and small enough to stay well inside Socrata's default cap.
_FETCH_WEEKS = PERCENTILE_WINDOW_WEEKS + 26

_FRIDAY = 4


# -- schedule ------------------------------------------------------------------


def _next_friday(reference: date) -> date:
    """The first Friday strictly after `reference`.

    Strictly, and computed rather than `+ timedelta(days=3)`. Reference dates are usually
    Tuesdays but not always — a US federal holiday shifts them, and 2023-07-03 and 2025-11-10
    are both Mondays in the fetched history. Adding three days to a Monday lands on Thursday and
    publishes the report a day before it existed, which is the exact bug this module is built to
    prevent, arriving through the back door twice in three years.
    """
    return reference + timedelta(days=((_FRIDAY - reference.weekday()) % 7) or 7)


def publication_time(reference: date) -> datetime:
    """When a report referencing `reference` became public, in UTC.

    The CFTC releases the COT at 15:30 US/Eastern on the Friday following the reference date.
    Built through `zoneinfo` rather than a fixed UTC offset because the release is defined in
    local time: it is 19:30 UTC in summer and 20:30 UTC in winter, and a constant would be an
    hour wrong for half the year — in the permissive direction for one of those halves.

    **Known imprecision, stated rather than hidden.** A federal holiday in the release week
    pushes publication to the following Monday, and the dataset carries no field to detect that
    from (verified: there is no publication column, only `report_date_as_yyyy_mm_dd`). On those
    weeks this returns a time up to three days early, so a backtest could read the report over a
    weekend it was not yet public. The exposure is bounded — a handful of weeks a year, on a
    series that moves weekly — and the alternative, padding every week by three days to be safe,
    would make the freshest data permanently invisible. Recorded here so the choice is auditable
    rather than discovered in a backtest.
    """
    local = datetime.combine(_next_friday(reference), _RELEASE_TIME, tzinfo=_EASTERN)
    return local.astimezone(UTC)


def reference_time(reference: date) -> datetime:
    """Close of business on the reference date, in UTC — when the positions were measured.

    Distinct from `publication_time` by three days. Kept as a real instant rather than a bare
    date so `Event.event_time_utc` has something honest to carry, and so migration 0009's
    withholding rule has a moment to compare against.
    """
    return datetime.combine(reference, _REFERENCE_CLOSE, tzinfo=_EASTERN).astimezone(UTC)


def should_fetch(
    last_fetched_at: datetime | None,
    *,
    now: datetime,
    min_interval: timedelta = DEFAULT_MIN_FETCH_INTERVAL,
) -> bool:
    """Whether a fetch is due. Never fetched -> always due.

    Free function rather than a method so the caller can answer it from the store's newest
    `fetched_at` without constructing a source, which is what makes the cache survive a process
    that only lives for the length of one cron run.
    """
    if last_fetched_at is None:
        return True
    if last_fetched_at.tzinfo is None:
        raise ValueError("last_fetched_at must be timezone-aware")
    return (now - last_fetched_at) >= min_interval


# -- records -------------------------------------------------------------------


class CotObservation(Protocol):
    """The four fields the ranking needs, whatever produced them.

    Structural on purpose. `CotReport` comes off the wire and `CotReportRecord` comes out of the
    store, and `CotHistory` accepts either without the store importing this package or this
    package importing the store — the two directions that would close an import cycle through
    `fundamentals.context`.
    """

    @property
    def currency(self) -> str: ...
    @property
    def report_date(self) -> date: ...
    @property
    def published_at(self) -> datetime: ...
    @property
    def net_position(self) -> int: ...


@dataclass(frozen=True)
class CotReport:
    """One contract's non-commercial positioning for one week.

    Both timestamps are stored, neither derived from the other at read time. `published_at` is
    computed once here, at the boundary, so every downstream consumer compares against the same
    instant instead of re-deriving it and possibly differently.
    """

    currency: str
    contract_code: str
    contract_name: str
    report_date: date
    published_at: datetime
    noncommercial_long: int
    noncommercial_short: int
    open_interest: int | None = None

    def __post_init__(self) -> None:
        if self.published_at.tzinfo is None:
            raise ValueError(f"{self.currency}: published_at must be timezone-aware")
        if self.published_at <= reference_time(self.report_date):
            raise ValueError(
                f"{self.currency}: published_at {self.published_at.isoformat()} is not after "
                f"the {self.report_date} reference close; one timestamp has been filled in "
                f"from the other, which defeats the point-in-time gate"
            )

    @property
    def net_position(self) -> int:
        """Non-commercial long minus short. Positive is a net speculative long."""
        return self.noncommercial_long - self.noncommercial_short

    def as_row(self) -> dict[str, Any]:
        """The dict shape `CotRepository.upsert_many` expects.

        `net_position` is absent deliberately: it is a generated column in Postgres, so the
        database computes it from the legs and cannot be handed a value that disagrees.
        """
        return {
            "currency": self.currency,
            "contract_code": self.contract_code,
            "contract_name": self.contract_name,
            "report_date": self.report_date,
            "published_at": self.published_at.astimezone(UTC),
            "noncommercial_long": self.noncommercial_long,
            "noncommercial_short": self.noncommercial_short,
            "open_interest": self.open_interest,
        }

    def as_event(self) -> Event:
        """The same report as a context row for the dashboard and the agents.

        Low importance, always. `EventRepository.HIGH_IMPACT` drives the auto-revoke trigger, and
        a weekly positioning report is not a release — marking it higher would black out every
        pair for fifteen minutes each Friday afternoon for data everyone already expected.
        """
        return Event(
            event_time_utc=reference_time(self.report_date),
            publication_time_utc=self.published_at,
            source=SOURCE,
            currency=self.currency,
            importance="Low",
            title=f"CFTC COT Non-Commercial Net ({self.contract_name})",
            category="positioning",
            body=(
                f"Non-commercial long {self.noncommercial_long:,}, "
                f"short {self.noncommercial_short:,}, "
                f"net {self.net_position:+,} as of {self.report_date.isoformat()}."
            ),
            actual=float(self.net_position),
        )


@dataclass(frozen=True)
class PairPositioning:
    """A pair's two legs and the difference between them, with the gaps named.

    Returned instead of a bare float because 0.0 has two meanings — "genuinely mid-range" and
    "we have no idea" — and a confidence modifier that cannot tell them apart is one that
    silently treats an outage as a neutral reading.
    """

    symbol: str
    base: str
    quote: str
    base_score: float
    quote_score: float
    unavailable: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        """Base crowding minus quote crowding, clamped to [-1, 1].

        Clamped because each leg is already in [-1, 1] and their difference is in [-2, 2]. A
        cross where both legs sit at opposite three-year extremes is the maximally crowded case
        and reads +1, not +2 — the scale is a statement about crowding, and letting it run past
        its own bounds would break every consumer that assumes them.
        """
        return max(-1.0, min(1.0, self.base_score - self.quote_score))

    @property
    def is_degraded(self) -> bool:
        return bool(self.unavailable)


# -- ranking -------------------------------------------------------------------


def percentile_rank(window: Sequence[float], value: float) -> float:
    """Where `value` sits in `window`, as a fraction in [0, 1].

    The midrank definition — everything strictly below, plus half of everything equal — because
    the alternatives both lie at an end of the scale. "Fraction strictly below" reports 0.0 for
    a series that has never moved, which reads as a record short; "fraction at or below" reports
    1.0 for the same series. Midrank gives 0.5, which is what a flat series means.

    `window` must be sorted ascending; callers sort once and rank repeatedly.
    """
    if not window:
        raise ValueError("percentile_rank needs a non-empty window")
    below = bisect_left(window, value)
    equal = bisect_right(window, value) - below
    return (below + 0.5 * equal) / len(window)


def _score_from_percentile(percentile: float) -> float:
    """[0, 1] -> [-1, 1]. +1 is maximally crowded long, -1 maximally crowded short."""
    return 2.0 * percentile - 1.0


class CotHistory:
    """Weekly positioning per currency, readable only as of a point in time.

    `visible_at` is not a convenience filter, it is the gate. The in-memory twin of
    `cot_visible_at()`: the store path is protected by SQL and the fetch-and-score path — the one
    a backtest and the "does this work at all" check both take — has no SQL to be protected by.
    Both had to exist, or the guarantee would hold for whichever path was not being tested.
    """

    def __init__(self, reports: Iterable[CotObservation]) -> None:
        # Sorted by reference date so `net_series` returns a chronological series and the
        # trailing window is a slice rather than a search.
        self._by_currency: dict[str, list[CotObservation]] = {}
        for report in reports:
            self._by_currency.setdefault(report.currency, []).append(report)
        for series in self._by_currency.values():
            series.sort(key=lambda r: r.report_date)

    def __len__(self) -> int:
        return sum(len(series) for series in self._by_currency.values())

    @property
    def currencies(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_currency))

    def visible_at(self, as_of: datetime) -> CotHistory:
        """A history holding only what had been released at `as_of`.

        Filters on `published_at`. Filtering on `report_date` here would compile, would read
        correctly, and would back-date every observation by three days.
        """
        if as_of.tzinfo is None:
            raise ValueError(
                "as_of must be timezone-aware; a naive value shifts the publication gate"
            )
        checked = as_of.astimezone(UTC)
        return CotHistory(
            report
            for series in self._by_currency.values()
            for report in series
            if report.published_at <= checked
        )

    def reports(self, currency: str) -> tuple[CotObservation, ...]:
        """Every stored week for one currency, oldest first."""
        return tuple(self._by_currency.get(currency.upper(), ()))

    def net_series(self, currency: str) -> tuple[int, ...]:
        """Net non-commercial positions for one currency, oldest first."""
        return tuple(report.net_position for report in self.reports(currency))

    def latest(self, currency: str) -> CotObservation | None:
        series = self._by_currency.get(currency.upper())
        return series[-1] if series else None

    def positioning_score(self, currency: str) -> float:
        """Crowding in [-1, 1] for one currency: +1 maximally crowded long.

        The latest net position ranked against the trailing `PERCENTILE_WINDOW_WEEKS`, mapped
        onto a signed scale. Returns `NEUTRAL_SCORE` — not an approximation of one — whenever
        there are fewer than `MIN_HISTORY_WEEKS` observations. A rank against 12 points is
        arithmetic, not information, and extrapolating to fill the window would manufacture the
        very extremes the measure exists to detect.
        """
        code = currency.upper()
        series = self.net_series(code)
        if len(series) < MIN_HISTORY_WEEKS:
            logger.info(
                "COT history for %s has %d week(s), below the %d-week minimum; returning neutral",
                code,
                len(series),
                MIN_HISTORY_WEEKS,
            )
            return NEUTRAL_SCORE

        window = series[-PERCENTILE_WINDOW_WEEKS:]
        current = window[-1]
        return _score_from_percentile(percentile_rank(sorted(window), current))

    def pair_positioning_score(self, symbol: str) -> PairPositioning:
        """Both legs of a pair, and their difference.

        EURUSD is `eur_score - usd_score`. USD has no entry in `CONTRACT_CODES` and that is not
        an omission: every contract in the map is quoted *against* the dollar, so a record net
        long in EURO FX futures already is a record short dollar position against the euro.
        Adding a separate dollar-index leg would count the same crowding twice and would do it
        with a different basket behind it. So a currency outside the map contributes 0.0, and
        USD-quoted pairs reduce to their base leg — which is the correct reading of the data,
        not a fallback.

        A currency that *is* in the map but has no usable history is a different case, and lands
        in `unavailable` so the caller can distinguish "flat" from "blind".
        """
        base, quote = split_symbol(symbol)
        scores: dict[str, float] = {}
        unavailable: list[str] = []

        for leg in (base, quote):
            if leg not in CONTRACT_CODES:
                # Not a gap: no futures contract exists for this leg in this report.
                scores[leg] = NEUTRAL_SCORE
                continue
            if len(self.net_series(leg)) < MIN_HISTORY_WEEKS:
                unavailable.append(leg)
            scores[leg] = self.positioning_score(leg)

        return PairPositioning(
            symbol=symbol,
            base=base,
            quote=quote,
            base_score=scores[base],
            quote_score=scores[quote],
            unavailable=tuple(unavailable),
        )


# -- source --------------------------------------------------------------------


def _to_int(raw: object) -> int | None:
    """Socrata returns every numeric column as a string. Anything unparseable is None."""
    if raw is None:
        return None
    try:
        return int(float(str(raw).replace(",", "")))
    except (TypeError, ValueError):
        return None


class CftcCotSource:
    """Polls the CFTC Socrata dataset. One request covers every configured currency.

    Implements `FundamentalSource`, so `fetch` returns `Event`s for the context and dashboard
    path. `fetch_reports` returns the numeric records the percentile is built from. Both come
    off the same HTTP response — the two shapes exist because they have different readers and
    different gates, not because the data is fetched twice.
    """

    def __init__(
        self,
        *,
        currencies: Sequence[str] = tuple(CONTRACT_CODES),
        url: str = COT_DATASET_URL,
        user_agent: str = "fx-regime-agent/0.1",
        min_fetch_interval: timedelta = DEFAULT_MIN_FETCH_INTERVAL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        unknown = [c for c in currencies if c.upper() not in CONTRACT_CODES]
        if unknown:
            raise ValueError(
                f"no CFTC contract code for {unknown}; add it to CONTRACT_CODES with the code "
                f"looked up from the dataset, rather than matching on the contract name"
            )
        self._currencies = tuple(c.upper() for c in currencies)
        self._url = url
        self._user_agent = user_agent
        self._min_fetch_interval = min_fetch_interval
        self._last_fetch_at: datetime | None = None
        self._cached: tuple[CotReport, ...] = ()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0), follow_redirects=True
        )

    @property
    def name(self) -> str:
        return SOURCE

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> CftcCotSource:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def fetch(self, since: datetime) -> list[Event]:
        """Reports published strictly after `since`, as context events."""
        reports = await self.fetch_reports(since)
        return [report.as_event() for report in reports]

    async def fetch_reports(self, since: datetime | None = None) -> list[CotReport]:
        """Every configured contract's recent weeks, newest last.

        `since` filters on publication time, matching what `FundamentalSource` means by "new":
        a report referencing an old Tuesday that was only released this Friday is new, and a
        filter on the reference date would drop it.

        Within one process, a second call inside `min_fetch_interval` replays the cached batch
        instead of hitting the endpoint. Across processes the cache is the store, and the caller
        checks `should_fetch` against its newest `fetched_at` — see `__main__`.
        """
        now = datetime.now(UTC)
        if not should_fetch(self._last_fetch_at, now=now, min_interval=self._min_fetch_interval):
            logger.debug(
                "COT fetched %s ago, under the %s interval; replaying %d cached report(s)",
                now - self._last_fetch_at if self._last_fetch_at else "never",
                self._min_fetch_interval,
                len(self._cached),
            )
            reports = list(self._cached)
        else:
            reports = await self._request()
            self._cached = tuple(reports)
            # Stamped only on a completed request. A failure must leave the interval open so the
            # next cycle retries, rather than caching an outage for a day.
            self._last_fetch_at = now

        if since is None:
            return reports
        if since.tzinfo is None:
            raise ValueError("since must be timezone-aware")
        cutoff = since.astimezone(UTC)
        return [report for report in reports if report.published_at > cutoff]

    async def _request(self) -> list[CotReport]:
        codes = [CONTRACT_CODES[currency] for currency in self._currencies]
        floor = datetime.now(UTC).date() - timedelta(weeks=_FETCH_WEEKS)
        quoted = ", ".join(f"'{code}'" for code in codes)

        response = await self._client.get(
            self._url,
            params={
                "$select": (
                    "cftc_contract_market_code,contract_market_name,"
                    "report_date_as_yyyy_mm_dd,noncomm_positions_long_all,"
                    "noncomm_positions_short_all,open_interest_all"
                ),
                "$where": (
                    f"cftc_contract_market_code in ({quoted}) "
                    f"and report_date_as_yyyy_mm_dd >= '{floor.isoformat()}T00:00:00.000'"
                ),
                "$order": "report_date_as_yyyy_mm_dd ASC",
                # Generous: 7 contracts x ~182 weeks. Socrata truncates silently at its own cap
                # rather than erroring, so the ceiling is set well clear of the expected count.
                "$limit": 5000,
            },
            headers={"User-Agent": self._user_agent, "Accept": "application/json"},
        )
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(
                f"expected a JSON array from {self._url}, got {type(payload).__name__}"
            )

        reports: list[CotReport] = []
        skipped = 0
        for row in payload:
            report = _to_report(row)
            if report is None:
                skipped += 1
                continue
            reports.append(report)

        if skipped:
            logger.warning(
                "CFTC COT: skipped %d row(s) with missing or unparseable fields", skipped
            )
        logger.info(
            "CFTC COT: %d report(s) across %d contract(s)",
            len(reports),
            len({r.contract_code for r in reports}),
        )
        return reports


def _to_report(row: dict[str, Any]) -> CotReport | None:
    """One Socrata row to a `CotReport`, or None if anything essential is missing.

    Dropped rather than defaulted. A missing leg defaulted to zero would enter the series as a
    record short and sit inside the three-year window for three years.
    """
    code = str(row.get("cftc_contract_market_code", "")).strip()
    currency = _CURRENCY_BY_CODE.get(code)
    if currency is None:
        return None

    raw_date = row.get("report_date_as_yyyy_mm_dd")
    long_positions = _to_int(row.get("noncomm_positions_long_all"))
    short_positions = _to_int(row.get("noncomm_positions_short_all"))
    if not raw_date or long_positions is None or short_positions is None:
        return None

    try:
        reference = date.fromisoformat(str(raw_date)[:10])
    except ValueError:
        return None

    return CotReport(
        currency=currency,
        contract_code=code,
        contract_name=str(row.get("contract_market_name", "")).strip() or currency,
        report_date=reference,
        published_at=publication_time(reference),
        noncommercial_long=long_positions,
        noncommercial_short=short_positions,
        open_interest=_to_int(row.get("open_interest_all")),
    )


# -- the write path ------------------------------------------------------------


async def poll(
    repository: Any,
    *,
    source: CftcCotSource | None = None,
    now: datetime | None = None,
    min_interval: timedelta = DEFAULT_MIN_FETCH_INTERVAL,
) -> int:
    """Refresh the store if a fetch is due, and return how many rows were written.

    The cache that matters is this one. Every entrypoint in this system is short-lived — a cron
    run, not a daemon (docs/ADR-002-scheduling.md) — so an in-process interval guard is empty on
    every start. Asking the store when it was last fetched is what makes "no more than daily"
    survive the process boundary, and it is why `CotRepository.latest_fetch_at` reads `fetched_at`
    rather than `published_at`.

    Typed as `Any` for the repository because importing `CotRepository` at module level would
    have this module depend on the store, which nothing else in `fundamentals` does at import
    time. The caller has one already.
    """
    moment = now or datetime.now(UTC)
    last = await repository.latest_fetch_at()
    if not should_fetch(last, now=moment, min_interval=min_interval):
        logger.info("COT last fetched %s, inside the %s interval; skipping", last, min_interval)
        return 0

    owned = source is None
    fetcher = source or CftcCotSource()
    try:
        reports = await fetcher.fetch_reports()
    finally:
        if owned:
            await fetcher.aclose()

    if not reports:
        logger.warning("CFTC returned no usable reports; leaving the store untouched")
        return 0

    return await repository.upsert_many(report.as_row() for report in reports)
