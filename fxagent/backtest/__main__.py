"""`uv run python -m fxagent.backtest --symbol EUR/USD --from ... --to ...`

Summary to stdout, trade-by-trade to CSV. The CSV is the artefact: it carries the label spans,
the barrier that was touched, whether the exit was inferred, and which spread source each fill
used — everything needed to re-derive the summary or to argue with it.

Reads only. A backtest that wrote to the trades table would put simulated fills beside real
ones, and the `mode` column would be the only thing keeping them apart.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from fxagent.backtest.folds import purged_walk_forward
from fxagent.backtest.replay import DEFAULT_SOURCE, ReplayConfig, ReplayResult, replay
from fxagent.backtest.report import build_report
from fxagent.costs import CostConfig
from fxagent.risk.sizing import RiskConfig
from fxagent.risk.symbols import SymbolSpec
from fxagent.store.engine import Database
from fxagent.store.repositories.bars import BarRepository

logger = logging.getLogger(__name__)


def _canonical(symbol: str) -> str:
    """`EUR/USD` -> `EURUSD`. The store keys on the unslashed form."""
    return symbol.strip().upper().replace("/", "").replace("_", "").replace("-", "")


def _day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fxagent.backtest",
        description="Replay stored bars through the live pipeline and report with intervals.",
    )
    parser.add_argument("--symbol", required=True, help="EUR/USD or EURUSD")
    parser.add_argument("--from", dest="start", required=True, type=_day, metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="end", required=True, type=_day, metavar="YYYY-MM-DD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--csv", type=Path, help="Trade-by-trade output path")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--spread-pips",
        type=float,
        default=1.0,
        help="Fixed spread used where the feed stored no bid/ask",
    )
    parser.add_argument("--slippage-pips", type=float, default=0.5)
    parser.add_argument(
        "--spread-ceiling-pips",
        type=float,
        help="Measured symbol-hour p90; refuse quoted entries above it",
    )
    parser.add_argument("--swap-long", type=float, default=0.0, help="Account ccy per lot/night")
    parser.add_argument("--swap-short", type=float, default=0.0)
    parser.add_argument("--equity", type=float, default=10_000.0, help="Stated reference equity")
    parser.add_argument("--max-bars-held", type=int, default=24, help="The time barrier, in bars")
    parser.add_argument("--history-bars", type=int, default=300)
    parser.add_argument("--log-level", default="WARNING")
    return parser


def write_csv(result: ReplayResult, path: Path) -> None:
    """One row per trade, every field flattened. Enums render as their value, times as ISO."""
    rows = [dataclasses.asdict(trade) for trade in result.trades]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: value.isoformat()
                    if isinstance(value, datetime)
                    else ",".join(value)
                    if isinstance(value, tuple)
                    else value
                    for key, value in row.items()
                }
            )


async def run(arguments: argparse.Namespace) -> int:
    symbol = _canonical(arguments.symbol)
    database = Database.from_env()
    try:
        async with database.session() as session:
            repository = BarRepository(session)
            bars = await repository.bars_between(
                symbol,
                arguments.timeframe,
                start=arguments.start,
                end=arguments.end,
                source=arguments.source,
            )
            quotes = await repository.quotes_between(
                symbol,
                arguments.timeframe,
                start=arguments.start,
                end=arguments.end,
                source=arguments.source,
            )
    finally:
        await database.dispose()

    if len(bars) == 0:
        print(
            f"No {arguments.timeframe} bars for {symbol} from source {arguments.source!r} in "
            f"that range. Check the collector has run, and that the source name is right.",
            file=sys.stderr,
        )
        return 1

    config = ReplayConfig(
        spec=SymbolSpec.forex(symbol),
        risk=RiskConfig(reference_equity=arguments.equity),
        costs=CostConfig(
            fixed_spread_pips=arguments.spread_pips,
            slippage_pips=arguments.slippage_pips,
            swap_long_per_lot=arguments.swap_long,
            swap_short_per_lot=arguments.swap_short,
        ),
        history_bars=arguments.history_bars,
        max_bars_held=arguments.max_bars_held,
        spread_ceiling_pips=arguments.spread_ceiling_pips,
    )

    result = replay(bars, config, quotes=quotes)
    print(result.describe())
    print()

    if not result.trades:
        print("No trades, so no report. The ledger above is the finding.")
        return 0

    folds = None
    if len(result.trades) >= arguments.folds:
        folds = purged_walk_forward(list(result.trades), folds=arguments.folds)
        print("\n".join(fold.describe() for fold in folds))
        print()

    report = build_report(
        result,
        config.costs,
        risk_fraction=config.risk.risk_fraction,
        starting_equity=arguments.equity,
        folds=folds,
    )
    print(report.describe())

    if arguments.csv:
        write_csv(result, arguments.csv)
        print(f"\n{len(result.trades)} trades written to {arguments.csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(level=arguments.log_level, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(run(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
