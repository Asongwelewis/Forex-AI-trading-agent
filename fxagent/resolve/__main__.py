"""`uv run python -m fxagent.resolve --once`

Reads open paper trades, closes the ones the stored bars have settled, and reports. It needs no
MT5 terminal — it is arithmetic over bars that are already in the store — so it can run beside
the trader or on its own.

Exit code 0 whenever the run completed, including when it closed nothing. "No trade was ready"
is the normal answer, not a failure; a non-zero exit there would make the alerting useless.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from fxagent.adapters.mt5_local import SOURCE as MT5_SOURCE
from fxagent.costs import CostConfig
from fxagent.resolve.service import DEFAULT_MAX_BARS_HELD, ResolveConfig, resolve_open_trades
from fxagent.risk.symbols import SymbolSpec
from fxagent.store.engine import Database
from fxagent.store.repositories import TradeRepository

logger = logging.getLogger("fxagent.resolve")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fxagent.resolve",
        description="Close out paper trades against the bars that have since arrived.",
    )
    parser.add_argument("--source", default=MT5_SOURCE)
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--max-bars-held", type=int, default=DEFAULT_MAX_BARS_HELD)
    parser.add_argument(
        "--fixed-spread-pips",
        type=float,
        default=1.0,
        help="Used only where the stored bar carries no bid/ask. Reported per fill either way.",
    )
    parser.add_argument("--slippage-pips", type=float, default=0.5)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Accepted for symmetry with the other entrypoints; this always runs once.",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser


async def _run(args: argparse.Namespace) -> int:
    database = Database.from_env()
    try:
        # Specs are derived from the symbols that actually have open trades, so a symbol the
        # config forgot cannot silently leave its trades unresolved forever.
        async with database.session() as session:
            symbols = {trade.symbol for trade in await TradeRepository(session).open_trades()}

        config = ResolveConfig(
            source=args.source,
            timeframe=args.timeframe.upper(),
            max_bars_held=args.max_bars_held,
            costs=CostConfig(
                fixed_spread_pips=args.fixed_spread_pips, slippage_pips=args.slippage_pips
            ),
            specs={symbol: SymbolSpec.forex(symbol) for symbol in symbols},
        )

        stats = await resolve_open_trades(database, config)
        logger.info("resolver finished: %s", stats.as_detail())
        # Errors are the only failure. Closing nothing is the normal answer.
        return 1 if stats.errors else 0
    finally:
        await database.dispose()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.info("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
