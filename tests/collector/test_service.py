"""The collector against a real Postgres and a scripted data source.

The headline case is idempotency: running the collector twice must leave the store exactly as
one run did. The collector re-fetches overlapping windows constantly — every poll, every
restart, every backfill — so if a second pass duplicated rows the table would grow without
bound and every series read would break on `BarSeries`'s strictly-increasing check.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.adapters.base import Bar, BarSeries
from fxagent.adapters.credits import CreditLimitExceeded
from fxagent.collector.service import SERVICE_NAME, CollectorConfig, CollectorService
from fxagent.store.engine import Database
from fxagent.store.repositories import BarRepository, HeartbeatRepository
from tests.conftest import requires_postgres

pytestmark = [pytest.mark.db, requires_postgres]

# A Wednesday, comfortably inside market hours.
NOW = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)


class ScriptedSource:
    """A `DataSource` that serves generated bars and counts what was asked of it."""

    def __init__(self, *, source: str = "twelvedata", fail_after: int | None = None) -> None:
        self._source = source
        self.calls: list[tuple[str, str, int, datetime | None]] = []
        self._fail_after = fail_after

    @property
    def source(self) -> str:
        return self._source

    async def bars_ending_at(
        self, symbol: str, timeframe: str, count: int, *, end: datetime | None
    ) -> BarSeries:
        self.calls.append((symbol, timeframe, count, end))
        if self._fail_after is not None and len(self.calls) > self._fail_after:
            raise CreditLimitExceeded("scripted exhaustion")

        last = end or NOW
        last = last.replace(minute=0, second=0, microsecond=0)
        bars = tuple(
            Bar(
                timestamp=last - timedelta(hours=count - 1 - i),
                open=1.0800 + i * 0.0001,
                high=1.0830 + i * 0.0001,
                low=1.0790 + i * 0.0001,
                close=1.0820 + i * 0.0001,
                volume=0,
            )
            for i in range(count)
        )
        return BarSeries(symbol=symbol, timeframe=timeframe, bars=bars)


def _service(database: Database, source: ScriptedSource, **overrides: object) -> CollectorService:
    defaults: dict[str, object] = {
        "symbols": ("EURUSD",),
        "timeframes": ("H1",),
        "poll_overlap_bars": 6,
        "backfill_days": 1,
        "backfill_batch": 24,
    }
    defaults.update(overrides)
    config = CollectorConfig(**defaults)  # type: ignore[arg-type]
    return CollectorService(database=database, source=source, config=config, now=lambda: NOW)


async def _stored_count(database: Database, source: str = "twelvedata") -> int:
    async with database.session() as session:
        series = await BarRepository(session).bars_between(
            "EURUSD", "H1", start=NOW - timedelta(days=7), end=NOW, source=source
        )
    return len(series)


# -- idempotency: the requirement --------------------------------------------


async def test_backfill_is_idempotent_across_two_runs(database: Database) -> None:
    """Two runs, one result. This is what makes a restart safe."""
    first = _service(database, ScriptedSource())
    await first.backfill()
    after_one = await _stored_count(database)

    assert after_one > 0, "the first run must actually store something"

    second = _service(database, ScriptedSource())
    await second.backfill()
    after_two = await _stored_count(database)

    assert after_two == after_one


async def test_polling_repeatedly_does_not_duplicate_rows(database: Database) -> None:
    """Every poll deliberately overlaps the last, so overlap must be free."""
    service = _service(database, ScriptedSource())

    await service.poll_once()
    first = await _stored_count(database)
    for _ in range(3):
        await service.poll_once()

    assert await _stored_count(database) == first


async def test_a_second_run_finds_no_gaps_left(database: Database) -> None:
    service = _service(database, ScriptedSource())
    await service.backfill()

    remaining = await service.gaps_for("EURUSD", "H1", start=NOW - timedelta(hours=12), end=NOW)

    assert remaining == [], f"backfill left gaps behind: {remaining}"


# -- source correctness -------------------------------------------------------


async def test_rows_are_written_with_the_adapter_s_source(database: Database) -> None:
    await _service(database, ScriptedSource(source="twelvedata")).poll_once()

    async with database.session() as session:
        assert await BarRepository(session).sources_for("EURUSD") == ["twelvedata"]


async def test_two_providers_do_not_land_in_one_series(database: Database) -> None:
    """The done-when condition: no series mixes sources."""
    await _service(database, ScriptedSource(source="twelvedata")).poll_once()
    await _service(database, ScriptedSource(source="metaapi")).poll_once()

    async with database.session() as session:
        repo = BarRepository(session)
        assert sorted(await repo.sources_for("EURUSD")) == ["metaapi", "twelvedata"]

        # Each series is single-source, and reads them separately without collapsing either.
        td = await repo.bars_between(
            "EURUSD", "H1", start=NOW - timedelta(days=1), end=NOW, source="twelvedata"
        )
        meta = await repo.bars_between(
            "EURUSD", "H1", start=NOW - timedelta(days=1), end=NOW, source="metaapi"
        )

    assert len(td) == len(meta) > 0
    timestamps = [bar.timestamp for bar in td.bars]
    assert timestamps == sorted(set(timestamps)), "a single-source series has no duplicate times"


# -- resilience ---------------------------------------------------------------


async def test_credit_exhaustion_stops_cleanly_without_losing_what_was_written(
    database: Database,
) -> None:
    """A clean stop with a consistent store, not a crash part-way through a write."""
    # A batch small enough that the gap needs several requests, so exhaustion lands part-way.
    source = ScriptedSource(fail_after=1)
    service = _service(database, source, backfill_batch=6)

    await service.backfill()

    assert service.stats.credit_refusals == 1, "the refusal must be recorded, not swallowed"
    stored = await _stored_count(database)
    assert stored > 0, "bars fetched before exhaustion must still be committed"


async def test_a_failing_symbol_does_not_stop_the_poll(database: Database) -> None:
    class Broken(ScriptedSource):
        async def bars_ending_at(
            self, symbol: str, timeframe: str, count: int, *, end: datetime | None
        ) -> BarSeries:
            if symbol == "GBPUSD":
                raise RuntimeError("provider hiccup")
            return await super().bars_ending_at(symbol, timeframe, count, end=end)

    service = CollectorService(
        database=database,
        source=Broken(),
        config=CollectorConfig(
            symbols=("GBPUSD", "EURUSD"), timeframes=("H1",), poll_overlap_bars=6
        ),
        now=lambda: NOW,
    )

    await service.poll_once()

    assert service.stats.errors == 1
    assert "provider hiccup" in service.stats.last_error
    assert await _stored_count(database) > 0, "the healthy symbol was still collected"


# -- liveness -----------------------------------------------------------------


async def test_a_heartbeat_is_recorded_and_counts_up(database: Database) -> None:
    service = _service(database, ScriptedSource())

    await service._heartbeat()  # noqa: SLF001
    await service._heartbeat()  # noqa: SLF001

    async with database.session() as session:
        beat = await HeartbeatRepository(session).get(SERVICE_NAME)

    assert beat is not None
    assert beat.beats == 2
    assert beat.detail is not None
    assert beat.detail["source"] == "twelvedata"


async def test_started_at_stays_fixed_so_crash_looping_is_visible(
    database: Database,
) -> None:
    """A restart moves started_at; a healthy run does not. That is how uptime is proved."""
    service = _service(database, ScriptedSource())
    await service._heartbeat()  # noqa: SLF001

    async with database.session() as session:
        first = await HeartbeatRepository(session).get(SERVICE_NAME)

    await service._heartbeat()  # noqa: SLF001

    async with database.session() as session:
        second = await HeartbeatRepository(session).get(SERVICE_NAME)

    assert first is not None and second is not None
    assert second.started_at_utc == first.started_at_utc
    assert second.beats > first.beats


async def test_stale_services_are_detectable(database: Database) -> None:
    async with database.begin() as session:
        await HeartbeatRepository(session).beat(
            "collector", now=NOW - timedelta(hours=2), started_at=NOW - timedelta(hours=3)
        )

    async with database.session() as session:
        stale = await HeartbeatRepository(session).stale_services(
            tolerance=timedelta(minutes=5), as_of=NOW
        )

    assert [record.service for record in stale] == ["collector"]


async def test_stop_request_is_observed_by_the_run_loop(database: Database) -> None:
    service = _service(database, ScriptedSource(), poll_interval=timedelta(seconds=0.05))
    service.request_stop()

    stats = await service.run()

    assert stats.polls <= 1, "a stop before the first sleep must end the loop promptly"


# -- regressions --------------------------------------------------------------


async def test_a_single_bar_gap_is_actually_filled(database: Database) -> None:
    """Gap.end is exclusive; requesting bars ending at it fetches the bar AFTER the gap.

    The original off-by-one left exactly one bar missing at the start of every gap, and no
    number of re-runs closed it — each run re-detected the same hole and re-fetched the same
    wrong bar.
    """
    source = ScriptedSource()
    service = _service(database, source)

    # Seed everything except one bar in the middle.
    async with database.begin() as session:
        full = await source.bars_ending_at("EURUSD", "H1", 8, end=NOW)
        kept = tuple(bar for bar in full.bars if bar.timestamp != NOW - timedelta(hours=3))
        await BarRepository(session).upsert_series(
            BarSeries(symbol="EURUSD", timeframe="H1", bars=kept), source="twelvedata"
        )

    window_start = NOW - timedelta(hours=7)
    before = await service.gaps_for("EURUSD", "H1", start=window_start, end=NOW)
    assert [gap.missing for gap in before] == [1], f"expected one single-bar gap, got {before}"

    for gap in before:
        await service._fill_gap("EURUSD", "H1", gap)  # noqa: SLF001

    after = await service.gaps_for("EURUSD", "H1", start=window_start, end=NOW)
    assert after == [], f"the single-bar gap survived the fill: {after}"


async def test_backfill_leaves_no_gaps_from_a_ragged_start_time(database: Database) -> None:
    """The end-to-end form of the grid-alignment bug.

    `backfill` computes its window as `now - backfill_days`, which is almost never on a bar
    boundary. Before the fix that produced an expected grid offset from every real bar, so
    every bar read as missing and the backfill never converged.
    """
    ragged = NOW.replace(minute=23, second=47)
    service = CollectorService(
        database=database,
        source=ScriptedSource(),
        config=CollectorConfig(
            symbols=("EURUSD",), timeframes=("H1",), backfill_days=1, backfill_batch=24
        ),
        now=lambda: ragged,
    )

    await service.backfill()
    remaining = await service.gaps_for("EURUSD", "H1", start=ragged - timedelta(days=1), end=ragged)

    assert remaining == [], f"backfill from a ragged start left gaps: {remaining}"
