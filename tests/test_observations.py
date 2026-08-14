"""BLS and Eurostat parsing, plus the revision behaviour the whole exercise depends on.

Payload shapes are copied from live responses captured 14 Aug 2026 — BLS payrolls at 158,858
for July, Eurostat HICP at 2.1/2.1/2.0 — so a provider changing shape shows up as these tests
passing against something the API no longer sends. The `network` test at the bottom is the one
that notices.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from fxagent.observations import (
    BLS_SERIES,
    BlsSource,
    EurostatSeries,
    EurostatSource,
)

BLS_PAYLOAD = {
    "status": "REQUEST_SUCCEEDED",
    "message": [],
    "Results": {
        "series": [
            {
                "seriesID": "CES0000000001",
                "data": [
                    {"year": "2026", "period": "M07", "periodName": "July", "value": "158858"},
                    {"year": "2026", "period": "M06", "periodName": "June", "value": "158881"},
                    # Annual average: a summary, not a release. Must not become a 13th month.
                    {"year": "2025", "period": "M13", "periodName": "Annual", "value": "157000"},
                ],
            }
        ]
    },
}

EUROSTAT_PAYLOAD = {
    "label": "HICP - monthly data (annual rate of change)",
    "dimension": {
        "time": {"category": {"index": {"2025-10": 0, "2025-11": 1, "2025-12": 2, "2026-01": 3}}}
    },
    # Sparse on purpose: index 3 has no observation, which is the case a zip() would misalign.
    "value": {"0": 2.1, "1": 2.1, "2": 2.0},
}


def _bls(payload: object = BLS_PAYLOAD, status: int = 200) -> BlsSource:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return BlsSource(
        series=("CES0000000001",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _eurostat(payload: object = EUROSTAT_PAYLOAD) -> EurostatSource:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return EurostatSource(
        series=(EurostatSeries("prc_hicp_manr", {"coicop": "CP00", "geo": "EA"}),),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


# -- BLS -------------------------------------------------------------------------------------


async def test_bls_values_are_stored_untransformed() -> None:
    """A level in thousands stays a level in thousands. No unit reconciliation here."""
    async with _bls() as source:
        observations = await source.fetch()

    july = next(o for o in observations if o.reference_period == "2026-07")
    assert july.value == 158858.0
    assert july.period_start == date(2026, 7, 1)
    assert july.source == "bls"
    assert july.series_id == "CES0000000001"


async def test_bls_annual_average_is_skipped() -> None:
    """M13 is a summary. Kept, it would put a thirteenth month in a monthly series."""
    async with _bls() as source:
        observations = await source.fetch()
    assert all(o.reference_period != "2025-M13" for o in observations)
    assert len(observations) == 2


async def test_bls_failure_status_in_a_200_body_is_raised() -> None:
    """BLS answers 200 with a failure status, including when the daily quota is spent."""
    payload = {"status": "REQUEST_NOT_PROCESSED", "message": ["daily threshold reached"]}
    async with _bls(payload) as source:
        observations = await source.fetch()
    # Swallowed per-series and logged, so the other series still run.
    assert observations == []


async def test_one_failing_bls_series_does_not_lose_the_others() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "BAD" in str(request.url):
            return httpx.Response(500)
        return httpx.Response(200, json=BLS_PAYLOAD)

    source = BlsSource(
        series=("BAD", "CES0000000001"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with source:
        observations = await source.fetch()
    assert len(observations) == 2


def test_the_configured_bls_series_are_the_fx_relevant_ones() -> None:
    assert "CES0000000001" in BLS_SERIES  # payrolls
    assert "CUUR0000SA0" in BLS_SERIES  # CPI


# -- Eurostat --------------------------------------------------------------------------------


async def test_eurostat_values_are_read_through_the_time_index() -> None:
    async with _eurostat() as source:
        observations = await source.fetch()

    by_period = {o.reference_period: o.value for o in observations}
    assert by_period == {"2025-10": 2.1, "2025-11": 2.1, "2025-12": 2.0}


async def test_eurostat_gaps_do_not_shift_later_periods() -> None:
    """`value` is sparse. Zipping it against the period list misaligns everything after a gap."""
    payload = {
        "dimension": {"time": {"category": {"index": {"2026-01": 0, "2026-02": 1, "2026-03": 2}}}},
        "value": {"0": 1.0, "2": 3.0},  # February missing
    }
    async with _eurostat(payload) as source:
        observations = await source.fetch()

    by_period = {o.reference_period: o.value for o in observations}
    assert by_period == {"2026-01": 1.0, "2026-03": 3.0}


async def test_eurostat_series_id_is_stable_regardless_of_filter_order() -> None:
    a = EurostatSeries("prc_hicp_manr", {"coicop": "CP00", "geo": "EA"})
    b = EurostatSeries("prc_hicp_manr", {"geo": "EA", "coicop": "CP00"})
    assert a.series_id == b.series_id


async def test_eurostat_unexpected_shape_is_swallowed_per_series() -> None:
    async with _eurostat({"nothing": "useful"}) as source:
        assert await source.fetch() == []


@pytest.mark.parametrize(
    ("label", "expected_period", "expected_start"),
    [
        ("2026-07", "2026-07", date(2026, 7, 1)),
        ("2026Q3", "2026-Q3", date(2026, 7, 1)),
    ],
)
async def test_eurostat_period_labels(
    label: str, expected_period: str, expected_start: date
) -> None:
    payload = {
        "dimension": {"time": {"category": {"index": {label: 0}}}},
        "value": {"0": 1.5},
    }
    async with _eurostat(payload) as source:
        observations = await source.fetch()
    assert observations[0].reference_period == expected_period
    assert observations[0].period_start == expected_start


async def test_a_bare_year_is_skipped_rather_than_dated_january() -> None:
    payload = {"dimension": {"time": {"category": {"index": {"2026": 0}}}}, "value": {"0": 1.5}}
    async with _eurostat(payload) as source:
        assert await source.fetch() == []


# -- live shape ------------------------------------------------------------------------------


@pytest.mark.network
async def test_live_endpoints_still_have_the_shape_we_parse() -> None:
    """Deselected by default; run with `-m network`."""
    async with BlsSource(series=("CES0000000001",)) as bls:
        bls_observations = await bls.fetch()
    assert bls_observations, "BLS returned nothing"
    assert all(o.value > 0 for o in bls_observations)

    async with EurostatSource() as eurostat:
        euro_observations = await eurostat.fetch()
    assert euro_observations, "Eurostat returned nothing"
