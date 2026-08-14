"""CFTC positioning: the percentile, the pair arithmetic, and the refusal to guess.

Payloads mirror the live Socrata rows sampled on 14 Aug 2026 — every numeric column arrives as
a *string*, which is the detail that makes `_to_int` load-bearing rather than defensive.

The point-in-time guarantee is proved separately, in `test_cot_point_in_time.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from fxagent.fundamentals.base import fetch_safely
from fxagent.fundamentals.cot import (
    CONTRACT_CODES,
    MIN_HISTORY_WEEKS,
    NEUTRAL_SCORE,
    PERCENTILE_WINDOW_WEEKS,
    CftcCotSource,
    CotHistory,
    CotReport,
    percentile_rank,
    publication_time,
    reference_time,
    should_fetch,
)

#: A Tuesday, matching the CFTC's usual reference day.
FIRST_TUESDAY = date(2023, 1, 3)


def _report(currency: str, week: int, *, long: int = 0, short: int = 0) -> CotReport:
    """One synthetic week, `week` Tuesdays after `FIRST_TUESDAY`."""
    reference = FIRST_TUESDAY + timedelta(weeks=week)
    return CotReport(
        currency=currency,
        contract_code=CONTRACT_CODES.get(currency, "000000"),
        contract_name=f"{currency} TEST",
        report_date=reference,
        published_at=publication_time(reference),
        noncommercial_long=long,
        noncommercial_short=short,
    )


def _series(currency: str, nets: list[int]) -> list[CotReport]:
    """A history whose net positions are exactly `nets`, oldest first."""
    return [
        _report(currency, week, long=max(net, 0), short=max(-net, 0))
        for week, net in enumerate(nets)
    ]


# -- percentile ranking against a synthetic series -----------------------------


def test_percentile_rank_places_a_value_in_a_known_series() -> None:
    window = [0.0, 1.0, 2.0, 3.0, 4.0]

    # Midrank: everything strictly below, plus half of the one equal value.
    assert percentile_rank(window, 0.0) == pytest.approx(0.1)
    assert percentile_rank(window, 2.0) == pytest.approx(0.5)
    assert percentile_rank(window, 4.0) == pytest.approx(0.9)


def test_percentile_rank_of_a_flat_series_is_the_midpoint() -> None:
    """The case both naive definitions get wrong, in opposite directions.

    "Fraction strictly below" would say 0.0 — a record short. "Fraction at or below" would say
    1.0 — a record long. A series that has never moved is neither.
    """
    assert percentile_rank([7.0] * 20, 7.0) == pytest.approx(0.5)


def test_percentile_rank_handles_values_outside_the_window() -> None:
    window = [10.0, 20.0, 30.0]
    assert percentile_rank(window, -5.0) == pytest.approx(0.0)
    assert percentile_rank(window, 99.0) == pytest.approx(1.0)


def test_positioning_score_ranks_the_latest_week_against_a_rising_series() -> None:
    """A monotonically rising net ends at its own maximum, so the score pins near +1."""
    history = CotHistory(_series("EUR", list(range(MIN_HISTORY_WEEKS))))

    score = history.positioning_score("EUR")

    # Midrank of the maximum over n points is (n - 0.5) / n, mapped onto [-1, 1].
    expected = 2.0 * ((MIN_HISTORY_WEEKS - 0.5) / MIN_HISTORY_WEEKS) - 1.0
    assert score == pytest.approx(expected)
    assert 0.95 < score < 1.0


def test_positioning_score_pins_near_minus_one_at_a_record_short() -> None:
    history = CotHistory(_series("EUR", list(range(MIN_HISTORY_WEEKS, 0, -1))))

    score = history.positioning_score("EUR")

    assert -1.0 < score < -0.95


def test_positioning_score_is_neutral_mid_range() -> None:
    """A sawtooth ending in the middle of its own range must not read as an extreme."""
    nets = [(-1) ** week * (week % 40) for week in range(MIN_HISTORY_WEEKS)]
    nets[-1] = 0

    score = CotHistory(_series("EUR", nets)).positioning_score("EUR")

    assert abs(score) < 0.35


def test_the_window_is_trailing_so_ancient_extremes_stop_counting() -> None:
    """A record from four years ago must leave the window, or nothing is ever extreme again.

    Without the `[-PERCENTILE_WINDOW_WEEKS:]` slice this reads as merely mid-range, because the
    old spike still sits above the current value.
    """
    ancient_spike = [1_000_000] + [0] * 10
    recent = list(range(PERCENTILE_WINDOW_WEEKS))
    history = CotHistory(_series("EUR", ancient_spike + recent))

    assert history.positioning_score("EUR") > 0.95


def test_net_position_is_long_minus_short() -> None:
    report = _report("EUR", 0, long=201_847, short=259_938)
    assert report.net_position == -58_091


# -- neutral on insufficient history -------------------------------------------


def test_fewer_than_fifty_two_weeks_returns_neutral() -> None:
    """One week short of the minimum. The boundary, not a comfortable distance from it."""
    history = CotHistory(_series("EUR", list(range(MIN_HISTORY_WEEKS - 1))))

    assert history.positioning_score("EUR") == NEUTRAL_SCORE


def test_exactly_fifty_two_weeks_is_enough() -> None:
    """The other side of the same boundary, so the test pins a threshold rather than a range."""
    history = CotHistory(_series("EUR", list(range(MIN_HISTORY_WEEKS))))

    assert history.positioning_score("EUR") != NEUTRAL_SCORE


def test_an_unknown_currency_is_neutral_rather_than_an_error() -> None:
    assert CotHistory([]).positioning_score("ZAR") == NEUTRAL_SCORE


def test_short_history_is_not_extrapolated_even_from_a_screaming_extreme() -> None:
    """Ten weeks at a record high is still ten weeks. Returning +1 here would be a fabrication.

    This is the failure that matters most on a cold store: the first month of collection would
    otherwise emit maximum crowding for whatever happened to be the highest of four points.
    """
    history = CotHistory(_series("EUR", list(range(10))))

    assert history.positioning_score("EUR") == NEUTRAL_SCORE


# -- the two-leg pair combination ----------------------------------------------


def _two_leg_history(base_nets: list[int], quote_nets: list[int]) -> CotHistory:
    return CotHistory(_series("EUR", base_nets) + _series("GBP", quote_nets))


def test_a_pair_is_its_base_leg_minus_its_quote_leg() -> None:
    rising = list(range(MIN_HISTORY_WEEKS))
    falling = list(range(MIN_HISTORY_WEEKS, 0, -1))
    history = _two_leg_history(rising, falling)

    pair = history.pair_positioning_score("EURGBP")

    assert pair.base == "EUR"
    assert pair.quote == "GBP"
    assert pair.base_score == pytest.approx(history.positioning_score("EUR"))
    assert pair.quote_score == pytest.approx(history.positioning_score("GBP"))
    assert pair.score == pytest.approx(max(-1.0, min(1.0, pair.base_score - pair.quote_score)))
    # Crowded long the base, crowded short the quote: the maximally crowded case for the pair.
    assert pair.score == pytest.approx(1.0)


def test_two_equally_crowded_legs_cancel() -> None:
    """Both currencies at the same percentile means the *pair* is not crowded at all."""
    identical = list(range(MIN_HISTORY_WEEKS))

    pair = _two_leg_history(identical, identical).pair_positioning_score("EURGBP")

    assert pair.score == pytest.approx(0.0)


def test_the_pair_score_is_clamped_to_the_unit_interval() -> None:
    """Each leg spans [-1, 1], so their difference spans [-2, 2] and must be brought back."""
    pair = _two_leg_history(
        list(range(MIN_HISTORY_WEEKS)), list(range(MIN_HISTORY_WEEKS, 0, -1))
    ).pair_positioning_score("EURGBP")

    assert -1.0 <= pair.score <= 1.0


def test_usd_contributes_nothing_because_every_contract_is_already_priced_in_dollars() -> None:
    """EURUSD reduces to the EUR leg. Not a fallback — adding a dollar leg would double-count."""
    history = CotHistory(_series("EUR", list(range(MIN_HISTORY_WEEKS))))

    pair = history.pair_positioning_score("EURUSD")

    assert pair.quote_score == NEUTRAL_SCORE
    assert pair.score == pytest.approx(history.positioning_score("EUR"))
    # USD has no contract, so it is not a missing input and must not be reported as one.
    assert pair.unavailable == ()


def test_an_inverted_pair_flips_the_sign() -> None:
    """USDJPY and JPYUSD must be mirror images, or one of them is being read backwards."""
    history = CotHistory(_series("JPY", list(range(MIN_HISTORY_WEEKS))))

    assert history.pair_positioning_score("USDJPY").score == pytest.approx(
        -history.pair_positioning_score("JPYUSD").score
    )


def test_a_leg_with_too_little_history_is_reported_as_degraded_not_as_flat() -> None:
    """0.0 means "mid-range" everywhere else, so "we cannot say" needs its own channel."""
    history = _two_leg_history(list(range(MIN_HISTORY_WEEKS)), list(range(10)))

    pair = history.pair_positioning_score("EURGBP")

    assert pair.quote_score == NEUTRAL_SCORE
    assert pair.unavailable == ("GBP",)
    assert pair.is_degraded is True


def test_a_broker_suffix_does_not_break_the_split() -> None:
    """Exness quotes EURUSDm; a mis-split pair would invert the sign of the whole measure."""
    history = CotHistory(_series("EUR", list(range(MIN_HISTORY_WEEKS))))

    assert history.pair_positioning_score("EURUSDm").base == "EUR"


# -- the release schedule ------------------------------------------------------


def test_publication_is_the_friday_after_a_tuesday_reference() -> None:
    published = publication_time(date(2026, 8, 4))  # a Tuesday

    assert published.date() == date(2026, 8, 7)  # the Friday
    # 15:30 US/Eastern, which is 19:30 UTC while daylight time is in force.
    assert published == datetime(2026, 8, 7, 19, 30, tzinfo=UTC)


def test_publication_follows_eastern_time_across_the_dst_boundary() -> None:
    """A fixed UTC offset would be an hour wrong for half the year, permissively in winter."""
    summer = publication_time(date(2026, 8, 4))
    winter = publication_time(date(2026, 1, 6))

    assert summer.hour == 19
    assert winter.hour == 20


@pytest.mark.parametrize("reference", [date(2023, 7, 3), date(2025, 11, 10)])
def test_a_holiday_shifted_monday_reference_still_publishes_on_friday(reference: date) -> None:
    """Both of these are real, taken from the fetched history. Two of 188 observations.

    `reference + timedelta(days=3)` — the obvious implementation — lands on Thursday here and
    publishes the report a day before it existed.
    """
    assert reference.weekday() == 0
    published = publication_time(reference)

    assert published.weekday() == 4
    assert (published.date() - reference).days == 4


def test_publication_is_always_after_the_reference_close() -> None:
    for offset in range(21):
        reference = FIRST_TUESDAY + timedelta(days=offset)
        assert publication_time(reference) > reference_time(reference)


def test_a_report_whose_timestamps_are_collapsed_is_rejected() -> None:
    """The mistake this module exists to prevent, refused at construction."""
    with pytest.raises(ValueError, match="filled in from the other"):
        CotReport(
            currency="EUR",
            contract_code="099741",
            contract_name="EURO FX",
            report_date=FIRST_TUESDAY,
            # Publication set to the reference date: the classic collapse.
            published_at=datetime(2023, 1, 3, tzinfo=UTC),
            noncommercial_long=10,
            noncommercial_short=5,
        )


def test_a_naive_publication_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        CotReport(
            currency="EUR",
            contract_code="099741",
            contract_name="EURO FX",
            report_date=FIRST_TUESDAY,
            published_at=datetime(2023, 1, 6, 15, 30),
            noncommercial_long=10,
            noncommercial_short=5,
        )


# -- the contract map ----------------------------------------------------------


def test_every_configured_currency_has_an_explicit_six_digit_code() -> None:
    assert set(CONTRACT_CODES) == {"EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"}
    for currency, code in CONTRACT_CODES.items():
        assert code.isdigit() and len(code) == 6, f"{currency} -> {code!r}"


def test_contract_codes_are_unique() -> None:
    """A duplicate would silently drop a currency from every fetch."""
    assert len(set(CONTRACT_CODES.values())) == len(CONTRACT_CODES)


def test_a_currency_with_no_code_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="no CFTC contract code"):
        CftcCotSource(currencies=("EUR", "SEK"))


# -- the wire ------------------------------------------------------------------


def _row(code: str, name: str, reference: str, long: int, short: int) -> dict[str, str]:
    """A Socrata row. Every numeric column is a string, exactly as the live endpoint sends it."""
    return {
        "cftc_contract_market_code": code,
        "contract_market_name": name,
        "report_date_as_yyyy_mm_dd": f"{reference}T00:00:00.000",
        "noncomm_positions_long_all": str(long),
        "noncomm_positions_short_all": str(short),
        "open_interest_all": "799909",
    }


def _source(rows: list[dict[str, str]], *, status: int = 200) -> CftcCotSource:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=json.dumps(rows).encode())

    return CftcCotSource(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_string_numerics_are_parsed_and_the_contract_code_maps_to_a_currency() -> None:
    async with _source([_row("099741", "EURO FX", "2026-08-04", 201_847, 259_938)]) as source:
        reports = await source.fetch_reports()

    assert len(reports) == 1
    report = reports[0]
    assert report.currency == "EUR"
    assert report.noncommercial_long == 201_847
    assert report.net_position == -58_091
    assert report.report_date == date(2026, 8, 4)
    assert report.published_at == datetime(2026, 8, 7, 19, 30, tzinfo=UTC)


async def test_an_unrecognised_contract_code_is_dropped_not_guessed() -> None:
    """The dataset carries every commodity there is. A gold contract must not become a currency."""
    async with _source(
        [
            _row("099741", "EURO FX", "2026-08-04", 10, 5),
            _row("088691", "GOLD", "2026-08-04", 999, 111),
        ]
    ) as source:
        reports = await source.fetch_reports()

    assert [r.currency for r in reports] == ["EUR"]


async def test_a_row_missing_a_leg_is_dropped_rather_than_defaulted_to_zero() -> None:
    """A missing long defaulted to 0 enters as a record short and sits in the window for years."""
    broken = _row("099741", "EURO FX", "2026-08-04", 10, 5)
    broken["noncomm_positions_long_all"] = ""

    async with _source([broken]) as source:
        assert await source.fetch_reports() == []


async def test_since_filters_on_publication_time_not_on_the_reference_date() -> None:
    """A report referencing an old Tuesday but released this Friday is new, and must survive."""
    rows = [
        _row("099741", "EURO FX", "2026-07-28", 10, 5),
        _row("099741", "EURO FX", "2026-08-04", 20, 5),
    ]
    # After the first report's reference date but before its publication. A filter keyed on the
    # reference date would drop it; a filter keyed on publication keeps it.
    since = datetime(2026, 7, 29, tzinfo=UTC)

    async with _source(rows) as source:
        reports = await source.fetch_reports(since)

    assert [r.report_date for r in reports] == [date(2026, 7, 28), date(2026, 8, 4)]


async def test_a_second_call_inside_the_interval_does_not_hit_the_endpoint() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, content=json.dumps([_row("099741", "EURO FX", "2026-08-04", 10, 5)]).encode()
        )

    source = CftcCotSource(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with source:
        first = await source.fetch_reports()
        second = await source.fetch_reports()

    assert calls == 1
    assert [r.report_date for r in first] == [r.report_date for r in second]


async def test_a_failed_fetch_does_not_start_the_cache_clock() -> None:
    """Caching an outage for a day would turn a blip into a day without positioning."""
    responses = [httpx.Response(503), httpx.Response(200, content=json.dumps([]).encode())]

    def handler(_: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    source = CftcCotSource(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with source:
        with pytest.raises(httpx.HTTPStatusError):
            await source.fetch_reports()
        # The retry must actually go out rather than replaying an empty cached batch.
        await source.fetch_reports()

    assert responses == []


def test_should_fetch_is_true_when_nothing_has_ever_been_fetched() -> None:
    assert should_fetch(None, now=datetime(2026, 8, 14, tzinfo=UTC)) is True


def test_should_fetch_is_false_inside_the_interval_and_true_outside_it() -> None:
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)

    assert should_fetch(now - timedelta(hours=23), now=now) is False
    assert should_fetch(now - timedelta(hours=25), now=now) is True


# -- fail soft -----------------------------------------------------------------


async def test_an_unreachable_endpoint_yields_nothing_rather_than_raising() -> None:
    """Hard requirement: positioning is context, and must never stop an analysis cycle."""

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("name resolution failed")

    source = CftcCotSource(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with source:
        events = await fetch_safely(source, datetime(2026, 1, 1, tzinfo=UTC))

    assert events == []


async def test_a_five_hundred_yields_nothing_rather_than_raising() -> None:
    async with _source([], status=500) as source:
        assert await fetch_safely(source, datetime(2026, 1, 1, tzinfo=UTC)) == []


async def test_the_source_emits_low_importance_events_only() -> None:
    """High importance drives auto-revoke. A weekly survey must never close a live position."""
    async with _source([_row("099741", "EURO FX", "2026-08-04", 201_847, 259_938)]) as source:
        events = await source.fetch(datetime(2026, 1, 1, tzinfo=UTC))

    assert len(events) == 1
    event = events[0]
    assert event.importance == "Low"
    assert event.source == "cftc_cot"
    assert event.currency == "EUR"
    assert event.actual == pytest.approx(-58_091.0)
    # The two timestamps stay apart on the way into the store.
    assert event.publication_time_utc > event.event_time_utc
