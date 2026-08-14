"""The two COT gates must be one gate. Tested as an equivalence, not as two separate proofs.

`CotHistory.visible_at` and `cot_visible_at()` are independent implementations of the same rule
in two languages. That is the duplicate-definition shape this codebase has been bitten by three
times — `range_reversion` measuring its own ADX beside the classifier's, `UtcDatetime` declared
twice, `IMPORTANCE_VALUES` copied out of a check constraint — and the lesson each time was that
testing the two halves separately is what let them drift. Both suites pass right up until the
day the definitions disagree, because neither one ever looks at the other.

So nothing here asserts what either gate returns. It asserts they return **the same thing**,
over one shared fixture, at instants chosen to sit exactly on the boundaries where an off-by-one
would live: the microsecond before a release, the release instant itself, and the microsecond
after. A `<` where the other has `<=` survives every test in the other two files and dies here.

The comparison covers the derived score as well as the row set, because equal rows read in a
different order, or truncated by a different limit, would still produce a different percentile —
and the percentile is what a strategy actually consumes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from fxagent.fundamentals.cot import (
    CONTRACT_CODES,
    MIN_HISTORY_WEEKS,
    CotHistory,
    CotReport,
    publication_time,
)
from fxagent.store.repositories.cot import CotRepository
from tests.conftest import requires_postgres

pytestmark = [pytest.mark.db, requires_postgres]

#: Enough weeks that the percentile is defined, across three currencies so an ordering or
#: grouping difference between the two implementations has somewhere to show up.
WEEKS = MIN_HISTORY_WEEKS + 20
CURRENCIES = ("EUR", "GBP", "JPY")

#: The most recent reference date in the fixture. A Tuesday.
LAST_TUESDAY = date(2026, 8, 4)


def _fixture() -> list[CotReport]:
    """One shared history. Both gates see exactly these rows and nothing else.

    Net positions vary per currency and per week — deliberately not flat — so two gates that
    disagreed about *which* rows are visible produce different percentiles rather than the same
    neutral 0.0 by luck.
    """
    reports: list[CotReport] = []
    for index, currency in enumerate(CURRENCIES):
        for back in range(WEEKS):
            reference = LAST_TUESDAY - timedelta(weeks=back)
            swing = (back * 7919 + index * 104_729) % 50_000
            reports.append(
                CotReport(
                    currency=currency,
                    contract_code=CONTRACT_CODES[currency],
                    contract_name=f"{currency} FIXTURE",
                    report_date=reference,
                    published_at=publication_time(reference),
                    noncommercial_long=100_000 + swing,
                    noncommercial_short=100_000 + (swing * 3) % 40_000,
                )
            )
    return reports


def _instants(reports: list[CotReport]) -> list[datetime]:
    """Every release instant in the fixture, plus a microsecond either side of it.

    The boundaries are the whole point. An implementation using `<` instead of `<=` is correct
    everywhere except at exactly one instant per week, which no randomly chosen `as_of` would
    ever land on.
    """
    releases = sorted({report.published_at for report in reports})
    # The full sweep is ~220 releases x 3; trimmed to the newest 12 plus the oldest two, which
    # covers the interesting end (a full window behind it) and the cold-start end.
    sampled = releases[:2] + releases[-12:]

    instants: list[datetime] = [datetime(2000, 1, 1, tzinfo=UTC)]
    for release in sampled:
        instants.extend(
            [
                release - timedelta(microseconds=1),
                release,
                release + timedelta(microseconds=1),
            ]
        )
    instants.append(datetime(2030, 1, 1, tzinfo=UTC))
    return instants


def _key(report: object) -> tuple[str, date, int]:
    """Identity plus payload, so a row that differs in its net is not counted as a match."""
    return (report.currency, report.report_date, report.net_position)  # type: ignore[attr-defined]


@pytest.fixture
async def both(session):  # noqa: ANN001, ANN201 - fixture types are the suite's
    """The same reports, loaded into Postgres and held in memory. One source, two gates."""
    reports = _fixture()
    repository = CotRepository(session)
    await repository.upsert_many(report.as_row() for report in reports)
    await session.commit()
    return reports, CotHistory(reports), repository


async def test_the_two_gates_admit_exactly_the_same_rows(both) -> None:  # noqa: ANN001
    """Row-for-row equality at every boundary instant, in both directions."""
    reports, history, repository = both

    for as_of in _instants(reports):
        in_memory = {
            _key(report)
            for currency in CURRENCIES
            for report in history.visible_at(as_of).reports(currency)
        }
        in_sql = {_key(record) for record in await repository.visible_at(as_of, limit=5000)}

        assert in_memory == in_sql, (
            f"the gates disagree at {as_of.isoformat()}: "
            f"only in Python {sorted(in_memory - in_sql)[:3]}, "
            f"only in SQL {sorted(in_sql - in_memory)[:3]}"
        )


async def test_the_two_gates_produce_the_same_score(both) -> None:  # noqa: ANN001
    """Equal row sets are not enough — the number a strategy consumes has to match too.

    Different ordering or a different limit could leave the sets equal while the trailing
    window slices differently, and the percentile is the only part of this anything trades on.
    """
    reports, history, repository = both

    for as_of in _instants(reports):
        from_memory = history.visible_at(as_of)
        from_sql = CotHistory(await repository.visible_at(as_of, limit=5000))

        for currency in CURRENCIES:
            assert from_memory.positioning_score(currency) == pytest.approx(
                from_sql.positioning_score(currency)
            ), f"{currency} scores diverge at {as_of.isoformat()}"


async def test_the_two_gates_agree_on_the_release_instant_itself(both) -> None:  # noqa: ANN001
    """Pinned separately because it is the single instant an inclusive/exclusive slip changes.

    The generic sweep above would catch this, but only as one failure among many; naming it
    means a `<` vs `<=` regression reports itself rather than being inferred from a list of
    diverging timestamps.
    """
    reports, history, repository = both
    release = max(report.published_at for report in reports)

    for offset, expected_newest in (
        (timedelta(microseconds=-1), LAST_TUESDAY - timedelta(weeks=1)),
        (timedelta(0), LAST_TUESDAY),
    ):
        as_of = release + offset
        memory_latest = history.visible_at(as_of).latest("EUR")
        sql_latest = await repository.latest_report_date(as_of, "EUR")

        assert memory_latest is not None
        assert memory_latest.report_date == sql_latest == expected_newest


async def test_the_equivalence_would_actually_notice_a_divergence(both) -> None:  # noqa: ANN001
    """A negative control for the tests above.

    An equivalence test is only worth its runtime if unequal inputs make it fail. Here the
    in-memory history is deliberately given an extra week the database was never told about,
    and the comparison must reject it — otherwise `test_the_two_gates_admit_exactly_the_same_rows`
    could be passing because `_key` collapses everything to a constant.
    """
    reports, _, repository = both
    extra = CotReport(
        currency="EUR",
        contract_code=CONTRACT_CODES["EUR"],
        contract_name="EUR FIXTURE",
        report_date=LAST_TUESDAY + timedelta(weeks=1),
        published_at=publication_time(LAST_TUESDAY + timedelta(weeks=1)),
        noncommercial_long=999_999,
        noncommercial_short=0,
    )
    tampered = CotHistory([*reports, extra])
    as_of = extra.published_at

    in_memory = {_key(report) for report in tampered.visible_at(as_of).reports("EUR")}
    in_sql = {
        _key(record)
        for record in await repository.visible_at(as_of, currencies=["EUR"], limit=5000)
    }

    assert in_memory != in_sql
    assert _key(extra) in in_memory - in_sql
