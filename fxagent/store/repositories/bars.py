"""Price bars: bulk upsert on ingest, range reads for the strategies."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from fxagent.adapters.base import TIMEFRAMES, Bar, BarSeries
from fxagent.store.repositories.base import Repository, require_utc
from fxagent.store.schema import bars

__all__ = ["BarRepository"]

logger = logging.getLogger(__name__)


class BarRepository(Repository):
    """Reads and writes OHLC bars."""

    async def upsert_series(self, series: BarSeries, *, source: str) -> int:
        """Store a `BarSeries` from any adapter. Re-ingesting the same range is a no-op.

        The collector re-fetches overlapping ranges constantly (a restart, a backfill, a
        catch-up after a dropped connection), so this has to be idempotent. The last bar of a
        fetch is usually still forming, which is why values are updated rather than ignored:
        the same timestamp legitimately has a different close a minute later.
        """
        rows = [
            {
                "symbol": series.symbol,
                "timeframe": series.timeframe,
                "ts_utc": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "bid_close": None,
                "ask_close": None,
                "source": source,
                "ingested_at": datetime.now(UTC),
            }
            for bar in series.bars
        ]
        return await self.upsert_rows(rows)

    async def upsert_rows(self, rows: Iterable[dict[str, Any]]) -> int:
        """Upsert raw bar dicts, for sources that carry bid/ask which `Bar` does not model."""
        prepared = [self._prepare(row) for row in rows]
        if not prepared:
            return 0

        statement = insert(bars).values(prepared)
        statement = statement.on_conflict_do_update(
            constraint="bars_unique",
            set_={
                "open": statement.excluded.open,
                "high": statement.excluded.high,
                "low": statement.excluded.low,
                "close": statement.excluded.close,
                "volume": statement.excluded.volume,
                "bid_close": statement.excluded.bid_close,
                "ask_close": statement.excluded.ask_close,
                "ingested_at": statement.excluded.ingested_at,
            },
        )
        result = await self._session.execute(statement)
        return result.rowcount or 0

    @staticmethod
    def _prepare(row: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(row)
        prepared["ts_utc"] = require_utc(prepared["ts_utc"], "ts_utc")
        timeframe = prepared.get("timeframe")
        if timeframe not in TIMEFRAMES:
            raise ValueError(
                f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}"
            )
        if not prepared.get("source"):
            raise ValueError("source is required; it is part of a bar's identity")
        prepared.setdefault("ingested_at", datetime.now(UTC))
        prepared.setdefault("bid_close", None)
        prepared.setdefault("ask_close", None)
        return prepared

    async def latest_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        *,
        source: str,
        as_of: datetime | None = None,
    ) -> BarSeries:
        """The `count` most recent bars at or before `as_of`, returned oldest first.

        `source` is required, not optional. A bar's identity includes its source, so an
        unfiltered read returns the same timestamp once per provider — which `BarSeries`
        rejects outright, and which would otherwise feed an indicator a series interleaving
        OANDA and MT5 prices. There is no meaningful "any source" series.

        `as_of` is optional here, unlike on the events repository: a bar's timestamp IS its
        publication time, so filtering on `ts_utc` is already point-in-time correct. Pass it in
        a backtest to cut the series at the bar being replayed.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        if timeframe not in TIMEFRAMES:
            raise ValueError(
                f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}"
            )

        statement = (
            select(bars)
            .where(
                bars.c.symbol == symbol,
                bars.c.timeframe == timeframe,
                bars.c.source == source,
            )
            .order_by(bars.c.ts_utc.desc())
            .limit(count)
        )
        if as_of is not None:
            statement = statement.where(bars.c.ts_utc <= require_utc(as_of, "as_of"))

        result = await self._session.execute(statement)
        rows = list(result)
        rows.reverse()  # query is newest-first for the LIMIT; BarSeries wants oldest-first
        return BarSeries(
            symbol=symbol,
            timeframe=timeframe,
            bars=tuple(self._to_bar(row._mapping) for row in rows),  # noqa: SLF001
        )

    async def bars_between(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime,
        end: datetime,
        source: str,
    ) -> BarSeries:
        """Every bar with `start <= ts_utc <= end`, oldest first. One source, as above."""
        first = require_utc(start, "start")
        last = require_utc(end, "end")
        if last < first:
            raise ValueError(f"end {last} is before start {first}")

        statement = (
            select(bars)
            .where(
                bars.c.symbol == symbol,
                bars.c.timeframe == timeframe,
                bars.c.ts_utc >= first,
                bars.c.ts_utc <= last,
                bars.c.source == source,
            )
            .order_by(bars.c.ts_utc.asc())
        )

        result = await self._session.execute(statement)
        return BarSeries(
            symbol=symbol,
            timeframe=timeframe,
            bars=tuple(self._to_bar(row._mapping) for row in result),  # noqa: SLF001
        )

    async def latest_timestamp(
        self, symbol: str, timeframe: str, *, source: str | None = None
    ) -> datetime | None:
        """Newest stored bar time, so the collector knows where to resume."""
        statement = select(func.max(bars.c.ts_utc)).where(
            bars.c.symbol == symbol, bars.c.timeframe == timeframe
        )
        if source is not None:
            statement = statement.where(bars.c.source == source)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def sources_for(self, symbol: str) -> Sequence[str]:
        """Which sources have data for this symbol."""
        result = await self._session.execute(
            select(bars.c.source).where(bars.c.symbol == symbol).distinct().order_by(bars.c.source)
        )
        return [row.source for row in result]

    @staticmethod
    def _to_bar(mapping: Any) -> Bar:
        return Bar(
            timestamp=mapping["ts_utc"],
            open=mapping["open"],
            high=mapping["high"],
            low=mapping["low"],
            close=mapping["close"],
            volume=mapping["volume"],
        )
