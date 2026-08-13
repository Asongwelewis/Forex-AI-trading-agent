"""One-time backfill of Exness H1 history from the local MT5 terminal into Supabase.

Windows-only, and needs the terminal running and logged in. Run it once per symbol/range;
re-running is safe and writes nothing new.

Why this exists rather than the collector: Twelve Data's free tier caps a response at 5000
bars and does not reach back to 2024 reliably, and its prices are not the venue we would
execute on. The terminal already holds the whole window. These rows land under
``source='mt5_exness'`` and are a *separate series* from anything the collector writes — hard
rule 7, `bars_unique` on (symbol, timeframe, ts_utc, source). Never mix the two in one read.

Idempotent by construction: every write goes through `BarRepository.upsert_series`, which
upserts on that key. Batched by calendar month so a dropped connection costs one month rather
than the run, and so progress is visible on a two-year pull.

    uv run python scripts/backfill_mt5.py
    uv run python scripts/backfill_mt5.py --symbol GBPUSD --from 2024-01-01 --to 2024-06-30
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from fxagent.adapters.base import Bar, BarSeries
from fxagent.adapters.mt5_local import MT5LocalAdapter, _bar_from_rate
from fxagent.regime.sessions import is_market_open
from fxagent.store import BarRepository, Database

logger = logging.getLogger("backfill")

SOURCE = "mt5_exness"
TIMEFRAME = "H1"
#: Gaps at or above this many missing in-market hours are listed individually.
GAP_THRESHOLD_HOURS = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="EURUSD", help="canonical, no broker suffix")
    parser.add_argument("--from", dest="start", default="2024-01-01")
    parser.add_argument("--to", dest="end", default="2025-12-31")
    parser.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    return parser.parse_args()


def _months(start: date, end: date) -> Iterator[tuple[datetime, datetime]]:
    """Half-open [month_start, next_month_start) pairs covering the range, in UTC."""
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        nxt = date(cursor.year + (cursor.month == 12), (cursor.month % 12) + 1, 1)
        window_start = max(cursor, start)
        window_end = min(nxt, end + timedelta(days=1))
        yield (
            datetime.combine(window_start, datetime.min.time(), tzinfo=UTC),
            datetime.combine(window_end, datetime.min.time(), tzinfo=UTC),
        )
        cursor = nxt


def _fetch_month(
    adapter: MT5LocalAdapter, broker_symbol: str, start: datetime, end: datetime
) -> list[Bar]:
    """Bars opening in [start, end), converted to our model with the server offset removed."""
    import MetaTrader5 as mt5

    offset = adapter.server_utc_offset
    rates = mt5.copy_rates_range(
        broker_symbol,
        mt5.TIMEFRAME_H1,
        (start + offset).replace(tzinfo=None),
        (end + offset - timedelta(seconds=1)).replace(tzinfo=None),
    )
    if rates is None:
        logger.warning("no rates for %s %s: %s", broker_symbol, start.date(), mt5.last_error())
        return []
    return [_bar_from_rate(rate, offset) for rate in rates]


def _missing_in_market_hours(previous: datetime, following: datetime) -> list[datetime]:
    """Hourly slots between two bars that the FX week says should have traded."""
    slot = previous + timedelta(hours=1)
    missing = []
    while slot < following:
        if is_market_open(slot):
            missing.append(slot)
        slot += timedelta(hours=1)
    return missing


def _report_gaps(stamps: list[datetime]) -> None:
    """Print every gap, not a count. A count cannot be sanity-checked against a calendar."""
    listed: list[tuple[datetime, datetime, int]] = []
    small = 0

    for previous, following in zip(stamps, stamps[1:], strict=False):
        if following - previous <= timedelta(hours=1):
            continue
        missing = _missing_in_market_hours(previous, following)
        if not missing:
            continue  # the whole gap sits in the weekend; not a hole in the data
        if len(missing) >= GAP_THRESHOLD_HOURS:
            listed.append((previous, following, len(missing)))
        else:
            small += 1

    print(f"\n  gaps of >= {GAP_THRESHOLD_HOURS}h inside FX trading hours: {len(listed)}")
    if listed:
        print(f"  {'after':<18} {'before':<18} {'missing':>8}")
        for previous, following, count in listed:
            print(f"    {previous:%Y-%m-%d %H:%M}   {following:%Y-%m-%d %H:%M}   {count:>6}h")
    print(f"  shorter in-market gaps (1-{GAP_THRESHOLD_HOURS - 1}h): {small}")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args = _parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    database = Database.from_env()
    collected: list[Bar] = []
    written = 0

    with MT5LocalAdapter.from_env() as adapter:
        broker_symbol = adapter._select(args.symbol)  # noqa: SLF001 - one-off tool
        print(f"  terminal offset {adapter.server_utc_offset} | symbol {broker_symbol}")
        print(f"  {args.symbol} {TIMEFRAME} {start} .. {end} -> source={SOURCE!r}\n")

        for window_start, window_end in _months(start, end):
            bars = _fetch_month(adapter, broker_symbol, window_start, window_end)
            if not bars:
                print(f"  {window_start:%Y-%m}  {0:>5} bars")
                continue
            collected.extend(bars)

            if args.dry_run:
                print(f"  {window_start:%Y-%m}  {len(bars):>5} bars  (dry run, not written)")
                continue

            series = BarSeries(symbol=args.symbol, timeframe=TIMEFRAME, bars=tuple(bars))
            async with database.begin() as session:
                written += await BarRepository(session).upsert_series(series, source=SOURCE)
            print(f"  {window_start:%Y-%m}  {len(bars):>5} bars  written")

    if not collected:
        raise SystemExit("no bars returned at all; is the terminal logged in?")

    stamps = sorted({bar.timestamp for bar in collected})
    print(f"\n  fetched {len(collected)} bars ({len(stamps)} distinct timestamps)")
    print(f"  upserted {written} rows")
    print(f"  first {stamps[0]:%Y-%m-%d %H:%M} UTC   last {stamps[-1]:%Y-%m-%d %H:%M} UTC")
    _report_gaps(stamps)


if __name__ == "__main__":
    asyncio.run(main())
