"""`uv run --extra mt5 python -m fxagent.trader --dry-run`

The one long-lived local process. It reads bars the collector has stored, decides, journals,
and sends an advisory card. **It cannot place an order** — see `service.py` and GATE A in
CLAUDE.md.

`--dry-run` runs exactly one pass, does the whole decision, and writes nothing. It is the
command to run first on a new machine: it proves the store is reachable, the series is there
under the source you think it is, and the pipeline produces a verdict — without putting a row
in the ledger from a machine that was only being tested.

`--once` runs one pass for real. With neither flag it loops until interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import timedelta

from fxagent.adapters.mt5_local import SOURCE as MT5_SOURCE
from fxagent.alerts.telegram import TelegramNotifier
from fxagent.risk.symbols import SymbolSpec
from fxagent.store.engine import Database
from fxagent.trader.service import TraderConfig, TraderService

logger = logging.getLogger("fxagent.trader")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fxagent.trader",
        description="Evaluate every configured symbol on each bar. Advisory only.",
    )
    parser.add_argument(
        "--symbols",
        default=os.environ.get("MT5_SYMBOLS", "EURUSD"),
        help="Comma-separated, e.g. EURUSD,GBPUSD. Defaults to MT5_SYMBOLS.",
    )
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument(
        "--source",
        default=MT5_SOURCE,
        help=(
            "Which stored series to read. Defaults to the MT5/Exness feed, because that is "
            "the book that would fill the order. Reading another one is a different experiment."
        ),
    )
    parser.add_argument(
        "--history-bars",
        type=int,
        default=300,
        help=(
            "Must match ReplayConfig.history_bars, or live and backtest "
            "measure different systems."
        ),
    )
    parser.add_argument("--interval-minutes", type=float, default=5.0)
    parser.add_argument(
        "--equity",
        type=float,
        default=float(os.environ.get("FX_REFERENCE_EQUITY", "1000")),
        help="Stated equity to size against.",
    )
    parser.add_argument(
        "--no-daily-bias",
        action="store_true",
        help="Skip the D1 read. The bias then suppresses nothing, and the ledger records that.",
    )
    parser.add_argument("--once", action="store_true", help="One pass, then exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="One pass, full decision, no writes and no notification.",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser


def _symbols(raw: str) -> tuple[str, ...]:
    return tuple(part.strip().upper() for part in raw.split(",") if part.strip())


async def _run(args: argparse.Namespace) -> int:
    symbols = _symbols(args.symbols)
    if not symbols:
        logger.error("no symbols given")
        return 2

    config = TraderConfig(
        symbols=symbols,
        timeframe=args.timeframe.upper(),
        source=args.source,
        history_bars=args.history_bars,
        poll_interval=timedelta(minutes=args.interval_minutes),
        # `SymbolSpec.forex` derives the contract size and lot step from the currency legs.
        # Override per symbol here when a broker's spec differs; do not guess a missing one.
        specs={symbol: SymbolSpec.forex(symbol) for symbol in symbols},
        reference_equity=args.equity,
        use_daily_bias=not args.no_daily_bias,
    )

    database = Database.from_env()
    # `None` when Telegram is not configured, which is a supported mode: the ledger is the
    # record, and a card is a convenience on top of it.
    notifier = None if args.dry_run else TelegramNotifier.from_env()
    try:
        service = TraderService(database=database, config=config, notifier=notifier)

        if args.dry_run or args.once:
            results = await service.pass_once(dry_run=args.dry_run)
            fired = sum(1 for result in results if result.fired)
            actionable = sum(1 for result in results if result.actionable)
            logger.info(
                "%s: %d symbol(s) evaluated, %d fired, %d actionable",
                "dry run" if args.dry_run else "one pass",
                len(results),
                fired,
                actionable,
            )
            # A pass that evaluated nothing is not success. It usually means the series is
            # empty under this `--source`, which looks identical to a quiet market in the logs
            # and is the single most likely thing to be wrong on a new machine.
            return 0 if results else 1

        service.install_signal_handlers()
        await service.run()
        return 0
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
