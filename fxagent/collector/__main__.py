"""Entry point: `python -m fxagent.collector`.

Migrates, health-checks, then runs until SIGTERM. Refuses to start on an unhealthy database
rather than collecting into a store that is missing a table — a service that starts anyway has
already written a partial hour before anyone notices.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import timedelta

from dotenv import load_dotenv

from fxagent.adapters.twelvedata import TwelveDataAdapter
from fxagent.collector.service import CollectorConfig, CollectorService
from fxagent.store import Database, apply_migrations, check_health

logger = logging.getLogger("fxagent.collector")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m fxagent.collector",
        description="Always-on bar collector. Writes to Supabase; computes nothing.",
    )
    parser.add_argument(
        "--symbols",
        default="EURUSD,GBPUSD,EURGBP",
        help="comma-separated, canonical form without the broker suffix",
    )
    parser.add_argument("--timeframes", default="H1", help="comma-separated, e.g. H1,H4")
    parser.add_argument("--poll-minutes", type=float, default=15.0)
    parser.add_argument("--backfill-days", type=int, default=30)
    parser.add_argument(
        "--once",
        action="store_true",
        help="backfill and run a single poll, then exit. For smoke tests and cron.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    database = Database.from_env()
    try:
        async with database.connect() as connection:
            applied = await apply_migrations(connection)
        if applied:
            logger.info("applied %d migration(s)", len(applied))

        report = await check_health(database)
        if not report:
            logger.error("refusing to start: %s", report.summary())
            return 2

        adapter = TwelveDataAdapter.from_env()
        config = CollectorConfig(
            symbols=tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip()),
            timeframes=tuple(t.strip().upper() for t in args.timeframes.split(",") if t.strip()),
            poll_interval=timedelta(minutes=args.poll_minutes),
            backfill_days=args.backfill_days,
        )
        service = CollectorService(database=database, source=adapter, config=config)

        try:
            if args.once:
                await service.backfill()
                await service.poll_once()
                logger.info("single pass complete: %s", service.stats.as_detail())
            else:
                service.install_signal_handlers()
                await service.run()
        finally:
            await adapter.aclose()
        return 0
    finally:
        await database.dispose()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("collector failed to start: %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
