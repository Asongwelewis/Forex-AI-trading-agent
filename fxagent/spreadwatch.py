"""Sample Exness bid/ask once a minute during the London open, and write it to Supabase.

**Why this exists.** The 2024–25 backtest fills all 217 orders at a configured 1-pip spread,
because no feed in `bars` carries a two-sided quote — `bid_close` and `ask_close` are null on
every one of the 12,456 rows. That constant is the single largest unmeasured assumption in the
result, and the run's expectancy interval is `[-0.15, +0.15]R`: a spread wider than modelled by
half a pip moves the whole distribution by more than the point estimate is worth.

**A distribution, not an average.** The mean spread across a quiet hour is not the number that
matters. `session_breakout` fires on the bar that breaks the Asian range at the London open,
which is when a market-maker widens — so the fill that decides the trade is drawn from the tail,
not the middle. This records every sample so the percentiles can be computed later; it computes
nothing itself.

**Local, and deliberately dumb.** MT5 is Windows-only and needs a running terminal, so this runs
on the execution machine rather than on the ARM box. It reads `symbol_info_tick` and writes
rows. It holds no opinion, computes no indicator, and is never imported by the analysis path —
`tests/test_spreadwatch_is_measurement_only.py` asserts as much.

Run it for a week:

    uv run --extra mt5 python -m fxagent.spreadwatch --symbols EURUSD,GBPUSD

It sleeps outside the window and samples inside it, so it can be left running. Nothing here
places an order, and the adapter it connects through asserts the account is a demo before it
returns.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import Any, Final

from sqlalchemy.dialects.postgresql import insert

from fxagent.store.engine import Database
from fxagent.store.schema import spread_samples

__all__ = [
    "DEFAULT_SYMBOLS",
    "LONDON_WINDOW_END_UTC",
    "LONDON_WINDOW_START_UTC",
    "SpreadSample",
    "in_window",
    "sample_once",
    "subscribe",
]

logger = logging.getLogger(__name__)

#: 07:00–11:00 UTC. Wider than the London open itself on purpose: the hour before gives a
#: baseline to measure the open against, and a widening with nothing to compare it to is just a
#: number. Note these are fixed UTC bounds rather than `zoneinfo` — this is an instrument, not a
#: session rule, and it must sample the same clock hours all year so summer and winter samples
#: are comparable.
LONDON_WINDOW_START_UTC: Final = time(7, 0)
LONDON_WINDOW_END_UTC: Final = time(11, 0)

DEFAULT_SYMBOLS: Final = ("EURUSD", "GBPUSD", "EURGBP")

#: One sample a minute. Fast enough to catch a news widening that lasts a few minutes, slow
#: enough that a week of it is ~1,200 rows per symbol rather than a million.
SAMPLE_SECONDS: Final = 60


@dataclass(frozen=True)
class SpreadSample:
    """One two-sided quote, as the terminal reported it.

    `spread_points` is the terminal's own figure rather than one derived from bid and ask. The
    two should agree; where they do not, the symbol's `point` is misconfigured and storing both
    is what makes that visible instead of reconciling it away.
    """

    symbol: str
    broker_symbol: str
    sampled_at: datetime
    bid: float
    ask: float
    spread_points: int
    point: float
    spread_float: bool
    source: str = "mt5_exness"

    @property
    def derived_points(self) -> float:
        return (self.ask - self.bid) / self.point if self.point else 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "broker_symbol": self.broker_symbol,
            "sampled_at": self.sampled_at,
            "bid": self.bid,
            "ask": self.ask,
            "spread_points": self.spread_points,
            "point": self.point,
            "spread_float": self.spread_float,
            "source": self.source,
            "ingested_at": datetime.now(UTC),
        }


def in_window(moment: datetime) -> bool:
    """Whether `moment` falls inside the sampling window on a weekday.

    Weekends are excluded rather than sampled and filtered later: the terminal still reports a
    stale quote when the market is shut, and a few hundred frozen Saturday rows would sit in the
    middle of the distribution looking like an unusually tight London morning.
    """
    if moment.weekday() >= 5:
        return False
    return LONDON_WINDOW_START_UTC <= moment.astimezone(UTC).time() < LONDON_WINDOW_END_UTC


def subscribe(adapter: Any, symbols: tuple[str, ...], mt5: Any) -> list[str]:
    """Add every symbol to Market Watch once, at startup. Returns the broker names selected.

    **Not done per sample.** A symbol absent from Market Watch returns no tick at all, and
    selecting it does not make one appear in the same instant — the terminal has to receive the
    first quote from the server. Measured: EURGBPm returned nothing when selected and read
    together, and ticked normally a few seconds later. Doing this in the sampling loop would
    therefore lose the first reading of every symbol, and lose every reading of a symbol whose
    first tick is slow, which is the illiquid pair the study most wants.
    """
    selected: list[str] = []
    for symbol in symbols:
        broker_symbol = adapter._broker_symbol(symbol)  # noqa: SLF001 - measurement path
        if mt5.symbol_select(broker_symbol, True):
            selected.append(broker_symbol)
        else:
            logger.warning(
                "could not add %s to Market Watch; it will not be sampled", broker_symbol
            )
    return selected


def sample_once(adapter: Any, symbols: tuple[str, ...], mt5: Any) -> list[SpreadSample]:
    """Read one quote per symbol. Returns what it got; a symbol that failed is skipped and logged.

    A missing tick is a fact about that instant, not a reason to abandon the run — the poller has
    to survive a week unattended, and a week that dies on the first null tick measures nothing.
    """
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    samples: list[SpreadSample] = []

    for symbol in symbols:
        broker_symbol = adapter._broker_symbol(symbol)  # noqa: SLF001 - measurement path
        tick = mt5.symbol_info_tick(broker_symbol)
        info = mt5.symbol_info(broker_symbol)
        if tick is None or info is None or tick.bid <= 0 or tick.ask <= 0:
            logger.warning("no usable tick for %s at %s", broker_symbol, now)
            continue
        samples.append(
            SpreadSample(
                symbol=symbol,
                broker_symbol=broker_symbol,
                sampled_at=now,
                bid=float(tick.bid),
                ask=float(tick.ask),
                spread_points=int(info.spread),
                point=float(info.point),
                spread_float=bool(info.spread_float),
            )
        )
    return samples


async def store(database: Database, samples: list[SpreadSample]) -> int:
    """Upsert, ignoring duplicates. A restart mid-minute must not double-count.

    Double-counting would pull the measured distribution toward whatever the market was doing at
    the moment of the restart, which is the one moment correlated with the operator noticing
    something was wrong.
    """
    if not samples:
        return 0
    rows = [sample.as_row() for sample in samples]
    statement = (
        insert(spread_samples)
        .values(rows)
        .on_conflict_do_nothing(constraint="spread_samples_unique")
    )
    async with database.session() as session:
        result = await session.execute(statement)
        await session.commit()
    return result.rowcount or 0


async def run(symbols: tuple[str, ...], *, once: bool = False) -> int:
    """Sample until interrupted. Sleeps through everything outside the window."""
    import MetaTrader5 as mt5  # noqa: PLC0415 - Windows-only, imported at the call site

    from fxagent.adapters.mt5_local import MT5LocalAdapter  # noqa: PLC0415

    database = Database.from_env()
    written = 0
    try:
        with MT5LocalAdapter.from_env() as adapter:
            account = adapter.get_account()
            subscribe(adapter, symbols, mt5)
            logger.info(
                "sampling %s on a %s account, window %s-%s UTC",
                ",".join(symbols),
                "DEMO" if account.is_demo else "LIVE",
                LONDON_WINDOW_START_UTC,
                LONDON_WINDOW_END_UTC,
            )
            while True:
                now = datetime.now(UTC)
                # `--once` samples whatever the clock says. It exists to prove the terminal,
                # the schema and the credentials line up before a week-long run is started, and
                # a smoke test that silently does nothing outside 07:00-11:00 proves the
                # opposite of what it was run for. The loop below still respects the window.
                if once or in_window(now):
                    written += await store(database, sample_once(adapter, symbols, mt5))
                else:
                    logger.debug("outside the window at %s; sleeping", now)
                if once:
                    break
                await asyncio.sleep(SAMPLE_SECONDS)
    finally:
        await database.dispose()
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fxagent.spreadwatch",
        description="Sample Exness bid/ask during the London open into Supabase.",
    )
    parser.add_argument(
        "--symbols",
        default=os.environ.get("MT5_SYMBOLS", ",".join(DEFAULT_SYMBOLS)),
        help="Comma-separated, unsuffixed (EURUSD, not EURUSDm)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Take one sample now, whatever the clock says, and exit. Smoke test.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Imported here rather than at module scope, matching the other entrypoints: reading a
    # developer's .env is an application concern, and a library import that mutates the
    # environment as a side effect is a surprise in a test run.
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv()
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(level=arguments.log_level, format="%(asctime)s %(levelname)s %(message)s")
    symbols = tuple(s.strip().upper() for s in arguments.symbols.split(",") if s.strip())
    if not symbols:
        raise SystemExit("no symbols given")
    written = asyncio.run(run(symbols, once=arguments.once))
    logger.info("wrote %d sample(s)", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
