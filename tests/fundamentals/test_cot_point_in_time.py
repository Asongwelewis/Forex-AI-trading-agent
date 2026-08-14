"""The look-ahead test for COT positioning: a Wednesday bar cannot see Tuesday's report.

This is the failure the whole module is shaped around, and it is worth stating precisely. The
CFTC measures positions at the close of business on Tuesday and releases the report on Friday at
15:30 US/Eastern. A backtest evaluating Wednesday's bar therefore has *no* knowledge of that
week's positioning — not partial knowledge, none — and will not have any until Friday evening.

The gap is three days on a series that only moves once a week, so getting this wrong does not
produce a subtle edge. It hands the strategy the answer for most of the period it is being
scored over, and it does so through a column named `report_date` that is real, correct, and
exactly the wrong one to filter on.

Two gates therefore exist and both are tested here:

* **`CotHistory.visible_at`**, in Python, for the fetch-and-score path — a backtest holding a
  fetched history in memory has no SQL to be protected by.
* **`cot_visible_at()`**, in SQL, for the stored path, so the guarantee holds for an ad-hoc
  query as well as for traffic through `CotRepository` (hard rule 6).

The Python half runs everywhere. The SQL half is marked `db` and skips without a container,
which is exactly why the Python half is not optional.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from fxagent.fundamentals.context import PolicyRate, build_context
from fxagent.fundamentals.cot import (
    MIN_HISTORY_WEEKS,
    NEUTRAL_SCORE,
    CotHistory,
    CotReport,
    publication_time,
)
from fxagent.store.repositories.cot import CotRepository
from fxagent.store.repositories.events import EventRepository
from tests.conftest import requires_postgres

#: The week under test. Tuesday's report, released the following Friday at 15:30 ET = 19:30 UTC.
TUESDAY = date(2026, 8, 4)
FRIDAY_RELEASE = datetime(2026, 8, 7, 19, 30, tzinfo=UTC)

#: The bar a backtest is evaluating. Two days after the reference date, two days before release.
WEDNESDAY_BAR = datetime(2026, 8, 5, 9, 0, tzinfo=UTC)


def _report(reference: date, *, long: int, short: int, currency: str = "EUR") -> CotReport:
    return CotReport(
        currency=currency,
        contract_code="099741",
        contract_name="EURO FX",
        report_date=reference,
        published_at=publication_time(reference),
        noncommercial_long=long,
        noncommercial_short=short,
    )


def _history_through(reference: date, *, weeks: int = MIN_HISTORY_WEEKS) -> list[CotReport]:
    """`weeks` of flat, unremarkable history ending the week before `reference`.

    Flat on purpose: it makes the score of the final week the only thing that can move the
    measure, so a leak shows up as a large number rather than as a rounding difference.
    """
    return [
        _report(reference - timedelta(weeks=back), long=100_000, short=100_000)
        for back in range(weeks, 0, -1)
    ]


# -- the guarantee, in Python --------------------------------------------------


def test_a_wednesday_bar_cannot_see_that_weeks_report() -> None:
    """The headline case. The report references Tuesday; the bar is Wednesday; it is invisible."""
    reports = [*_history_through(TUESDAY), _report(TUESDAY, long=500_000, short=0)]
    history = CotHistory(reports)

    visible = history.visible_at(WEDNESDAY_BAR)

    assert history.latest("EUR").report_date == TUESDAY
    assert visible.latest("EUR").report_date == TUESDAY - timedelta(weeks=1)
    assert TUESDAY not in {r.report_date for r in visible.reports("EUR")}


def test_the_withheld_report_would_have_moved_the_score_if_it_had_leaked() -> None:
    """A positive control. Without it, the test above also passes on an empty history.

    The withheld week is a record net long against a perfectly flat three years, so if it were
    visible the score would pin at the top of the range. Seeing neutral instead is the proof
    that the filter is doing work, not that there was nothing to filter.
    """
    reports = [*_history_through(TUESDAY), _report(TUESDAY, long=500_000, short=0)]
    history = CotHistory(reports)

    assert history.visible_at(WEDNESDAY_BAR).positioning_score("EUR") == pytest.approx(0.0)
    # ...and once released, the same data ranks at the extreme. Not 1.0: the midrank of the
    # maximum over n points is (n - 0.5) / n, which is 0.981 for the 53 weeks visible here.
    assert history.visible_at(FRIDAY_RELEASE).positioning_score("EUR") > 0.95


def test_the_report_becomes_visible_at_the_release_instant_and_not_a_minute_before() -> None:
    reports = [*_history_through(TUESDAY), _report(TUESDAY, long=500_000, short=0)]
    history = CotHistory(reports)

    just_before = history.visible_at(FRIDAY_RELEASE - timedelta(minutes=1))
    exactly = history.visible_at(FRIDAY_RELEASE)

    assert just_before.latest("EUR").report_date != TUESDAY
    assert exactly.latest("EUR").report_date == TUESDAY


@pytest.mark.parametrize(
    ("label", "offset"),
    [
        ("tuesday, the reference date itself", timedelta(0)),
        ("wednesday", timedelta(days=1)),
        ("thursday", timedelta(days=2)),
        ("friday morning, hours before the release", timedelta(days=3, hours=6)),
    ],
)
def test_no_instant_before_the_release_can_see_the_report(label: str, offset: timedelta) -> None:
    """Every day of the gap, not just the one the card named. The bug is a range, not a point."""
    reference_midnight = datetime.combine(TUESDAY, datetime.min.time(), tzinfo=UTC)
    history = CotHistory([_report(TUESDAY, long=500_000, short=0)])

    visible = history.visible_at(reference_midnight + offset)

    assert visible.reports("EUR") == (), f"leaked on {label}"


def test_visibility_only_grows_as_the_replay_advances() -> None:
    """Replaying forward may reveal reports and must never hide one already seen."""
    history = CotHistory(
        [_report(TUESDAY - timedelta(weeks=w), long=10, short=0) for w in range(6)]
    )

    seen = 0
    for day in range(60):
        count = len(history.visible_at(WEDNESDAY_BAR - timedelta(days=45) + timedelta(days=day)))
        assert count >= seen
        seen = count


def test_a_naive_as_of_is_rejected_rather_than_guessed() -> None:
    """A naive instant would be read in the local zone and shift the gate by hours."""
    with pytest.raises(ValueError, match="timezone-aware"):
        CotHistory([]).visible_at(datetime(2026, 8, 5, 9, 0))


def test_the_score_falls_back_to_neutral_when_the_gate_leaves_too_little_history() -> None:
    """Early in a replay the gate legitimately empties the window. That is neutral, not zero-ish.

    Worth its own test because it is the one case where the point-in-time filter and the
    minimum-history rule interact, and the safe answer has to come out of both at once.
    """
    history = CotHistory(_history_through(TUESDAY))

    early = history.visible_at(WEDNESDAY_BAR - timedelta(weeks=MIN_HISTORY_WEEKS - 10))

    assert len(early.net_series("EUR")) < MIN_HISTORY_WEEKS
    assert early.positioning_score("EUR") == NEUTRAL_SCORE


def test_the_pair_score_inherits_the_gate() -> None:
    """`pair_positioning_score` must be reached through a gated history, never around it."""
    reports = [
        *_history_through(TUESDAY),
        _report(TUESDAY, long=500_000, short=0),
    ]
    history = CotHistory(reports)

    assert history.visible_at(WEDNESDAY_BAR).pair_positioning_score("EURUSD").score == (
        pytest.approx(0.0)
    )
    assert history.visible_at(FRIDAY_RELEASE).pair_positioning_score("EURUSD").score > 0.95


# -- the same guarantee, in SQL ------------------------------------------------


@pytest.fixture
async def cot_repo(session) -> CotRepository:  # noqa: ANN001 - fixture type is the suite's
    return CotRepository(session)


@pytest.mark.db
@requires_postgres
async def test_the_sql_gate_hides_an_unreleased_report(cot_repo: CotRepository) -> None:
    """The store half. `cot_visible_at()` filters on published_at, in the database."""
    await cot_repo.upsert_many([_report(TUESDAY, long=500_000, short=0).as_row()])

    assert await cot_repo.count_visible(WEDNESDAY_BAR) == 0
    # Positive control: the row exists and does appear once released.
    assert await cot_repo.count_visible(FRIDAY_RELEASE) == 1


@pytest.mark.db
@requires_postgres
async def test_net_position_is_generated_by_the_database(cot_repo: CotRepository) -> None:
    """Stored net can never disagree with its own legs, because nothing writes it."""
    await cot_repo.upsert_many([_report(TUESDAY, long=201_847, short=259_938).as_row()])

    records = await cot_repo.visible_at(FRIDAY_RELEASE)

    assert [r.net_position for r in records] == [-58_091]


@pytest.mark.db
@requires_postgres
async def test_a_revision_cannot_move_publication_time_earlier(cot_repo: CotRepository) -> None:
    """Otherwise a correction issued today retroactively becomes visible two days ago.

    The revision claims it was published on the Wednesday — after the reference close, so the
    table's own CHECK constraint has nothing to say about it, and squarely inside the window the
    real report was still secret. The conflict clause is the only thing standing between that
    claim and a backtest that can read Tuesday's positioning on Wednesday.
    """
    original = _report(TUESDAY, long=100, short=50)
    await cot_repo.upsert_many([original.as_row()])

    revised = dict(original.as_row())
    revised["noncommercial_long"] = 999
    revised["published_at"] = WEDNESDAY_BAR
    await cot_repo.upsert_many([revised])

    assert await cot_repo.count_visible(WEDNESDAY_BAR) == 0
    records = await cot_repo.visible_at(FRIDAY_RELEASE)
    assert records[0].published_at == FRIDAY_RELEASE
    # The revision itself did land — only its timestamp was refused.
    assert records[0].noncommercial_long == 999


@pytest.mark.db
@requires_postgres
async def test_the_table_refuses_a_row_whose_timestamps_are_collapsed(
    cot_repo: CotRepository,
) -> None:
    """The Python guard has a SQL twin, so a direct insert cannot bypass it.

    `CotReport.__post_init__` catches this for anything built through the source. This catches
    it for a migration, a backfill script, or an ad-hoc insert — the paths that do not go
    through the dataclass and are exactly where a hand-written row lands.
    """
    row = _report(TUESDAY, long=100, short=50).as_row()
    row["published_at"] = datetime(2026, 8, 3, tzinfo=UTC)  # before the reference date

    with pytest.raises(Exception, match="cot_published_after_reference"):
        await cot_repo.upsert_many([row])


@pytest.mark.db
@requires_postgres
async def test_build_context_reads_positioning_through_the_gate(session) -> None:  # noqa: ANN001
    """End to end: the assembler must not defeat a correct gate from the outside."""
    cot_repo = CotRepository(session)
    rows = [r.as_row() for r in _history_through(TUESDAY)]
    rows.append(_report(TUESDAY, long=500_000, short=0).as_row())
    await cot_repo.upsert_many(rows)

    rates = {
        "EUR": PolicyRate("EUR", 2.15, TUESDAY),
        "USD": PolicyRate("USD", 4.40, TUESDAY),
    }
    events = EventRepository(session)

    wednesday = await build_context(
        events, as_of=WEDNESDAY_BAR, symbol="EURUSD", rates=rates, cot_repository=cot_repo
    )
    friday = await build_context(
        events, as_of=FRIDAY_RELEASE, symbol="EURUSD", rates=rates, cot_repository=cot_repo
    )

    assert wednesday.market.positioning_score == pytest.approx(0.0)
    assert friday.market.positioning_score > 0.95


@pytest.mark.db
@requires_postgres
async def test_a_cot_outage_is_reported_rather_than_read_as_neutral(session) -> None:  # noqa: ANN001
    """ "Could not look" must land in `unavailable`, not silently become a mid-range reading."""

    class Broken:
        async def visible_at(self, *_: object, **__: object) -> list[object]:
            raise RuntimeError("connection reset")

    context = await build_context(
        EventRepository(session),
        as_of=WEDNESDAY_BAR,
        symbol="EURUSD",
        rates={"EUR": PolicyRate("EUR", 2.15, TUESDAY), "USD": PolicyRate("USD", 4.40, TUESDAY)},
        cot_repository=Broken(),  # type: ignore[arg-type]
    )

    assert "positioning" in context.unavailable
    assert context.is_degraded is True
    assert context.positioning is None
    assert context.market.positioning_score == 0.0


# -- the write path ------------------------------------------------------------


@pytest.mark.db
@requires_postgres
async def test_poll_writes_reports_and_then_declines_to_refetch(session) -> None:  # noqa: ANN001
    """The cache that matters, because every entrypoint here is a cron run and starts cold."""
    from fxagent.fundamentals.cot import poll

    class Recording:
        """A source that counts fetches and never touches the network."""

        calls = 0

        async def fetch_reports(self, since: datetime | None = None) -> list[CotReport]:
            Recording.calls += 1
            return [*_history_through(TUESDAY, weeks=3), _report(TUESDAY, long=10, short=5)]

        async def aclose(self) -> None:
            return None

    repo = CotRepository(session)
    source = Recording()

    result = await poll(repo, source=source)  # type: ignore[arg-type]
    await session.commit()
    assert result.written == 4
    assert result.fetched == 4
    assert result.skipped is False
    assert Recording.calls == 1

    # A second run the same day must read the store's `fetched_at` and stand down. Skipped is
    # not a failure: the workflow would otherwise alert every time the cache did its job.
    second = await poll(repo, source=source)  # type: ignore[arg-type]
    assert second.skipped is True
    assert second.is_failure is False
    assert Recording.calls == 1

    # --force is the escape hatch for a human re-running a failed job the same day.
    await poll(repo, source=source, force=True)  # type: ignore[arg-type]
    assert Recording.calls == 2

    # A day later it is due again without being asked twice.
    tomorrow = datetime.now(UTC) + timedelta(days=1, minutes=1)
    await poll(repo, source=source, now=tomorrow)  # type: ignore[arg-type]
    assert Recording.calls == 3


@pytest.mark.db
@requires_postgres
async def test_poll_leaves_the_store_untouched_when_the_fetch_comes_back_empty(
    session,  # noqa: ANN001
) -> None:
    """An outage must not be able to empty a history that took three years to accumulate."""
    from fxagent.fundamentals.cot import poll

    class Empty:
        async def fetch_reports(self, since: datetime | None = None) -> list[CotReport]:
            return []

        async def aclose(self) -> None:
            return None

    repo = CotRepository(session)
    await repo.upsert_many([_report(TUESDAY, long=10, short=5).as_row()])
    await session.commit()

    result = await poll(repo, source=Empty(), force=True)  # type: ignore[arg-type]

    assert result.written == 0
    assert await repo.count_visible(FRIDAY_RELEASE) == 1
    # An empty fetch is the one outcome the scheduled job must alert on: it means the
    # accumulation clock has stopped, which is otherwise discovered years later.
    assert result.is_failure is True
