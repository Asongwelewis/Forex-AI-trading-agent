"""The collector: an always-on service that does one job and nothing else.

It fetches bars and writes them to the store. It does **not** compute indicators, run
strategies, or call an LLM — `tests/collector/test_collector_is_data_only.py` enforces that by
inspecting this package's imports, because the discipline is what makes the split worth having:
analysis can always be re-run over stored data, while a collection window that was missed is
gone permanently.

That asymmetry drives the design:

* **Writes are idempotent.** Every poll re-fetches an overlapping window, and re-ingesting the
  same range must be a no-op. The `bars_unique` key does the work; this only has to not fight it.
* **`source` is set on every row, always.** It is part of the bar's identity, so a row written
  without it — or with the wrong one — silently forks the series.
* **Shutdown is graceful.** SIGTERM finishes the poll in flight and records a final heartbeat,
  so a container restart is distinguishable from a crash in the uptime record.
* **Credit exhaustion is a clean stop, not a crash.** The service keeps running and keeps
  beating; it just stops fetching until the quota rolls over.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from fxagent.adapters.base import BarSeries
from fxagent.adapters.credits import CreditLimitExceeded
from fxagent.collector.gaps import Gap, find_gaps, merge_adjacent
from fxagent.store.engine import Database
from fxagent.store.repositories import BarRepository, HeartbeatRepository

__all__ = ["CollectorConfig", "CollectorService", "DataSource"]

logger = logging.getLogger(__name__)

SERVICE_NAME = "collector"


class DataSource(Protocol):
    """What the collector needs from a data adapter. Narrower than `BrokerAdapter` on purpose.

    A collector that could place an order is a collector that might. Requiring only these two
    members means an execution-capable adapter can be passed, but nothing here can reach the
    parts of it that trade.
    """

    @property
    def source(self) -> str: ...

    async def bars_ending_at(
        self, symbol: str, timeframe: str, count: int, *, end: datetime | None
    ) -> BarSeries: ...


@dataclass(frozen=True)
class CollectorConfig:
    """What to collect and how often."""

    symbols: tuple[str, ...] = ("EURUSD", "GBPUSD", "EURGBP")
    timeframes: tuple[str, ...] = ("H1",)
    poll_interval: timedelta = timedelta(minutes=15)
    #: Re-fetched every poll. Overlap is what makes a brief outage self-healing without a
    #: separate repair path — the next poll simply covers the hole.
    poll_overlap_bars: int = 12
    #: How far back the startup backfill reaches.
    backfill_days: int = 30
    #: Bars per backfill request. Twelve Data caps a response at 5000.
    backfill_batch: int = 500
    heartbeat_interval: timedelta = timedelta(minutes=1)
    #: Wait after a credit refusal before trying again.
    credit_cooldown: timedelta = timedelta(minutes=5)


@dataclass
class CollectorStats:
    """Counters for the heartbeat detail payload."""

    polls: int = 0
    bars_written: int = 0
    gaps_filled: int = 0
    credit_refusals: int = 0
    errors: int = 0
    last_error: str = ""
    per_symbol: dict[str, int] = field(default_factory=dict)

    def as_detail(self) -> dict[str, object]:
        return {
            "polls": self.polls,
            "bars_written": self.bars_written,
            "gaps_filled": self.gaps_filled,
            "credit_refusals": self.credit_refusals,
            "errors": self.errors,
            "last_error": self.last_error[:500],
            "per_symbol": dict(self.per_symbol),
        }


class CollectorService:
    """Polls a data source and writes bars. Owns no analysis of any kind."""

    def __init__(
        self,
        *,
        database: Database,
        source: DataSource,
        config: CollectorConfig | None = None,
        now: object = None,
    ) -> None:
        self._database = database
        self._source = source
        self._config = config or CollectorConfig()
        self._now = now or (lambda: datetime.now(UTC))  # type: ignore[assignment]
        self._stats = CollectorStats()
        self._stopping = asyncio.Event()
        self._started_at = self._clock()

    def _clock(self) -> datetime:
        return self._now()  # type: ignore[operator,no-any-return]

    @property
    def stats(self) -> CollectorStats:
        return self._stats

    def __repr__(self) -> str:
        return (
            f"CollectorService(source={self._source.source!r}, "
            f"symbols={list(self._config.symbols)}, polls={self._stats.polls})"
        )

    # -- lifecycle -------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the loop to finish the current poll and exit."""
        if not self._stopping.is_set():
            logger.info("collector stop requested; finishing the poll in flight")
            self._stopping.set()

    def install_signal_handlers(self) -> None:
        """SIGTERM and SIGINT ask for a graceful stop rather than killing mid-write.

        SIGTERM is what a container runtime sends first. Without this the process dies between
        the write and the heartbeat, and the uptime record shows a crash where there was an
        orderly redeploy.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                # Windows' proactor loop has no add_signal_handler; fall back.
                signal.signal(sig, lambda *_: self.request_stop())

    async def run(self) -> CollectorStats:
        """Backfill, then poll until stopped."""
        logger.info(
            "collector starting: source=%s symbols=%s timeframes=%s",
            self._source.source,
            list(self._config.symbols),
            list(self._config.timeframes),
        )
        await self._heartbeat()
        await self.backfill()  # absorbs credit exhaustion itself; see its docstring

        while not self._stopping.is_set():
            await self.poll_once()
            await self._heartbeat()
            await self._sleep_until_next_poll()

        await self._heartbeat(final=True)
        logger.info("collector stopped cleanly after %d polls", self._stats.polls)
        return self._stats

    async def _sleep_until_next_poll(self) -> None:
        """Wait, but wake immediately on a stop request."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self._stopping.wait(), timeout=self._config.poll_interval.total_seconds()
            )

    # -- collection ------------------------------------------------------------

    async def poll_once(self) -> int:
        """Fetch the recent window for every symbol and timeframe. Returns rows written."""
        written = 0
        for timeframe in self._config.timeframes:
            for symbol in self._config.symbols:
                if self._stopping.is_set():
                    break
                try:
                    written += await self._fetch_and_store(
                        symbol, timeframe, count=self._config.poll_overlap_bars, end=None
                    )
                except CreditLimitExceeded as exc:
                    self._stats.credit_refusals += 1
                    logger.warning("poll paused on provider credits: %s", exc)
                    await self._cooldown()
                    return written
                except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the loop
                    self._stats.errors += 1
                    self._stats.last_error = f"{type(exc).__name__}: {exc}"
                    logger.exception("failed to collect %s %s", symbol, timeframe)

        self._stats.polls += 1
        return written

    async def backfill(self) -> int:
        """Fill gaps back to `backfill_days`, oldest first. Returns rows written.

        Running out of provider credits ends the backfill cleanly rather than raising: whatever
        was fetched is already committed, the remaining gaps are still gaps, and the next run
        finds them again. That is the whole reason the ledger refuses locally — the alternative
        is a 429 part-way through a write with no record of how far it got.
        """
        end = self._clock()
        start = end - timedelta(days=self._config.backfill_days)
        filled = 0

        try:
            for timeframe in self._config.timeframes:
                for symbol in self._config.symbols:
                    if self._stopping.is_set():
                        return filled
                    gaps = await self.gaps_for(symbol, timeframe, start=start, end=end)
                    if not gaps:
                        logger.info("no gaps for %s %s", symbol, timeframe)
                        continue

                    logger.info(
                        "backfilling %s %s: %d gap(s), %d bars missing",
                        symbol,
                        timeframe,
                        len(gaps),
                        sum(gap.missing for gap in gaps),
                    )
                    for gap in gaps:
                        filled += await self._fill_gap(symbol, timeframe, gap)
        except CreditLimitExceeded as exc:
            self._stats.credit_refusals += 1
            logger.warning(
                "backfill stopped on provider credits after %d bars; remaining gaps will be "
                "picked up on the next run: %s",
                filled,
                exc,
            )

        self._stats.gaps_filled += filled
        return filled

    async def gaps_for(
        self, symbol: str, timeframe: str, *, start: datetime, end: datetime
    ) -> list[Gap]:
        """Market-hours gaps in what is already stored for this source."""
        async with self._database.session() as session:
            stored = await BarRepository(session).bars_between(
                symbol, timeframe, start=start, end=end, source=self._source.source
            )
        gaps = find_gaps(
            (bar.timestamp for bar in stored.bars), start=start, end=end, timeframe=timeframe
        )
        return merge_adjacent(gaps, tolerance=timedelta(hours=6))

    async def _fill_gap(self, symbol: str, timeframe: str, gap: Gap) -> int:
        """Walk a gap newest-first, since the API pages backwards from `end_date`."""
        written = 0
        step = _step_for(timeframe)
        # `Gap.end` is exclusive — it is the open time of the first bar that is NOT missing.
        # `end_date` on the request is inclusive, so asking for gap.end fetches the bar after
        # the gap and never its last missing bar, leaving a one-bar hole that no amount of
        # re-running closes.
        cursor = gap.end - step
        remaining = gap.missing

        while remaining > 0 and not self._stopping.is_set():
            batch = min(self._config.backfill_batch, remaining)
            try:
                written += await self._fetch_and_store(symbol, timeframe, count=batch, end=cursor)
            except CreditLimitExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                self._stats.errors += 1
                self._stats.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "gap fill failed for %s %s at %s: %s", symbol, timeframe, cursor, exc
                )
                break

            remaining -= batch
            cursor = gap.start if remaining <= 0 else cursor - step * batch
            if cursor <= gap.start:
                break
        return written

    async def _fetch_and_store(
        self, symbol: str, timeframe: str, *, count: int, end: datetime | None
    ) -> int:
        series = await self._source.bars_ending_at(symbol, timeframe, count, end=end)
        if not len(series):
            return 0

        async with self._database.begin() as session:
            # `source` comes from the adapter, never from config: a row is identified by where
            # it came from, and letting the two disagree forks the series.
            written = await BarRepository(session).upsert_series(series, source=self._source.source)

        self._stats.bars_written += written
        self._stats.per_symbol[symbol] = self._stats.per_symbol.get(symbol, 0) + written
        logger.debug(
            "stored %d %s %s bars from %s", written, symbol, timeframe, self._source.source
        )
        return written

    async def _cooldown(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self._stopping.wait(), timeout=self._config.credit_cooldown.total_seconds()
            )

    # -- liveness --------------------------------------------------------------

    async def _heartbeat(self, *, final: bool = False) -> None:
        detail = self._stats.as_detail()
        detail["source"] = self._source.source
        detail["final"] = final
        try:
            async with self._database.begin() as session:
                await HeartbeatRepository(session).beat(
                    SERVICE_NAME,
                    now=self._clock(),
                    started_at=self._started_at,
                    detail=detail,
                )
        except Exception:  # noqa: BLE001 - liveness reporting must never stop collection
            logger.exception("failed to record heartbeat")


def _step_for(timeframe: str) -> timedelta:
    from fxagent.adapters.base import TIMEFRAMES

    return TIMEFRAMES[timeframe]
