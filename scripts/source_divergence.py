"""Read-only TwelveData versus Exness divergence report over stored bars.

This is the historical counterpart to ``smoke_test.py``. It never fetches or writes data: both
series must already be in the journal, and source names remain visible in the output.

    uv run python scripts/source_divergence.py --symbol EURUSD --from 2024-01-01 --to 2025-12-31
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from fxagent.adapters.base import OrderSide
from fxagent.adapters.divergence import compare_series, gap_filler_verdict, interpret
from fxagent.backtest.replay import DEFAULT_SOURCE
from fxagent.store.engine import Database
from fxagent.store.repositories.bars import BarRepository


def _day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


async def run(args: argparse.Namespace) -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    database = Database.from_env()
    try:
        async with database.session() as session:
            repository = BarRepository(session)
            exness = await repository.bars_between(
                args.symbol,
                args.timeframe,
                start=args.start,
                end=args.end,
                source=args.exness_source,
            )
            twelve = await repository.bars_between(
                args.symbol,
                args.timeframe,
                start=args.start,
                end=args.end,
                source=args.twelvedata_source,
            )
    finally:
        await database.dispose()

    barrier = (args.stop_pips, args.target_pips)
    report = compare_series(
        args.exness_source,
        exness,
        args.twelvedata_source,
        twelve,
        barrier_pips=barrier,
        side=OrderSide(args.side),
    )
    if not report.overlapping:
        for name, series in ((args.exness_source, exness), (args.twelvedata_source, twelve)):
            if series.bars:
                print(
                    f"{name}: {len(series)} bars from {series.bars[0].timestamp.isoformat()} "
                    f"to {series.bars[-1].timestamp.isoformat()}"
                )
            else:
                print(f"{name}: no bars in requested range")
    print(report.render(), end="")
    print(interpret(report))
    print(f"  gap-filler verdict: {gap_filler_verdict(report)}")
    return 0 if report.overlapping else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare stored Exness and TwelveData bars.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--from", dest="start", required=True, type=_day)
    parser.add_argument("--to", dest="end", required=True, type=_day)
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--exness-source", default=DEFAULT_SOURCE)
    parser.add_argument("--twelvedata-source", default="twelvedata")
    parser.add_argument("--side", choices=[side.value for side in OrderSide], default="BUY")
    parser.add_argument("--stop-pips", type=float, default=20.0)
    parser.add_argument("--target-pips", type=float, default=40.0)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
