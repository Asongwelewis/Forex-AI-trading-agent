"""What the broker's spread actually was, per symbol and per hour, from the stored bars.

This is card L2.1's deliverable, and it exists as a script rather than a notebook because the
number it produces feeds `CostConfig` and the L4.3 veto threshold — it has to be re-runnable
and diffable, not remembered.

**Percentiles, not a mean.** The mean spread across a quiet hour is not the number that
decides a trade. `session_breakout` fires on the bar that breaks the Asian range at the London
open, which is exactly when a market-maker widens — so the fill that decides the trade is
drawn from the upper tail. A cost model built on the mean is a cost model that is right about
the bars nobody trades.

**One number per bar is a real limitation, and it is stated in the output.** MT5's per-bar
`spread` field is *a* spread within the bar; the platform does not document whether it is the
minimum, the last tick's, or an average, and brokers differ. So an H1 reading cannot separate
"quiet hour with one spike at the close" from "uniformly wide". Running `fxagent.spreadwatch`
concurrently for a few sessions is what calibrates that, and the M15 series here is the cheap
approximation in the meantime: if M15 percentiles sit materially above H1 for the same hour,
the H1 field is smoothing something away.

    uv run python scripts/spread_report.py --symbols EURUSD,GBPUSD --timeframe H1
    uv run python scripts/spread_report.py --symbols EURUSD --timeframe M15 --hours 7,8,9,10

Reads only.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from sqlalchemy import select

from fxagent.adapters.mt5_local import SOURCE
from fxagent.risk.symbols import SymbolSpec
from fxagent.store import Database
from fxagent.store.schema import bars

logger = logging.getLogger("spread_report")

#: The hours the London-open breakout lives in, in UTC. `session_breakout` derives its window
#: from `regime.sessions.LONDON_MORNING` and it moves with DST; 7-11 covers it in both halves
#: of the year, which is what a report wants — a window that shifted mid-sample would make two
#: distributions look like one.
DEFAULT_HOURS: Final = (7, 8, 9, 10)

#: Reported for every hour. p99 needs a few thousand observations to mean anything, and the
#: output says so rather than printing a number that looks as solid as the others.
PERCENTILES: Final = (50, 75, 90, 99)

#: Below this many samples in a bucket, a p99 is one or two bars and is not a percentile.
P99_MIN_SAMPLES: Final = 500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="EURUSD,GBPUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument(
        "--hours",
        default="",
        help="Comma-separated UTC hours to break out. Defaults to the London-open window.",
    )
    parser.add_argument("--from", dest="start", default="2024-01-01")
    parser.add_argument("--to", dest="end", default="2025-12-31")
    return parser


def _day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


def _percentile(values: list[float], pct: int) -> float:
    """Nearest-rank percentile. No interpolation — a spread is a price the broker quoted.

    Interpolating between two observed spreads invents a quote that was never offered, which is
    the same category of error as filling at the mid.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, int(round(pct / 100 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


def _fmt(value: float) -> str:
    return "   n/a" if value != value else f"{value:6.2f}"


async def _load(
    database: Database, symbol: str, timeframe: str, source: str, start: datetime, end: datetime
) -> list[tuple[datetime, float, float]]:
    statement = (
        select(bars.c.ts_utc, bars.c.bid_close, bars.c.ask_close)
        .where(
            bars.c.symbol == symbol,
            bars.c.timeframe == timeframe,
            bars.c.source == source,
            bars.c.ts_utc >= start,
            bars.c.ts_utc <= end,
        )
        .order_by(bars.c.ts_utc.asc())
    )
    async with database.session() as session:
        result = await session.execute(statement)
        rows = []
        for ts, bid, ask in result:
            stamp = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
            rows.append((stamp, bid, ask))
        return rows


def _report(symbol: str, rows: list[tuple[datetime, float, float]], hours: tuple[int, ...]) -> None:
    spec = SymbolSpec.forex(symbol)
    pip = spec.pip

    quoted = [(ts, ask - bid) for ts, bid, ask in rows if bid is not None and ask is not None]
    coverage = len(quoted) / len(rows) if rows else 0.0

    print(f"\n{'=' * 72}")
    print(f"  {symbol}   {len(rows)} bars, {len(quoted)} quoted ({coverage:.0%})")
    if coverage < 1.0:
        # Unquoted bars fall back to the configured spread, so a partial range is a range whose
        # fills are partly assumption. Said here rather than left to be noticed.
        print("  WARNING: unquoted bars fall back to CostConfig.fixed_spread_pips")
    print(f"{'=' * 72}")

    if not quoted:
        print("  nothing to report")
        return

    by_hour: dict[int, list[float]] = defaultdict(list)
    for ts, spread in quoted:
        by_hour[ts.hour].append(spread / pip)

    everything = [spread / pip for _, spread in quoted]
    print(f"\n  All hours ({len(everything)} bars), spread in pips:")
    _print_row("  all", everything)

    print(f"\n  By UTC hour{' (London-open window highlighted)' if hours else ''}:")
    print(f"    {'hour':>5}  {'n':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}  {'p99':>6}  {'max':>6}")
    for hour in sorted(by_hour):
        marker = "*" if hour in hours else " "
        _print_row(f"  {marker}{hour:02d}", by_hour[hour])

    if hours:
        window = [value for hour in hours for value in by_hour.get(hour, [])]
        if window:
            print(f"\n  London-open window {hours}, pooled ({len(window)} bars):")
            _print_row("  win", window)
            print(
                f"\n  --> The p90 here is the number L4.3's veto uses, and the one\n"
                f"      TriggerConfig should compare a live spread against.\n"
                f"      p90 = {_percentile(window, 90):.2f} pips "
                f"({_percentile(window, 90) * pip / 1e-5:.0f} points on a 5-digit quote)"
            )


def _print_row(label: str, values: list[float]) -> None:
    p99 = _percentile(values, 99) if len(values) >= P99_MIN_SAMPLES else float("nan")
    print(
        f"    {label:>5}  {len(values):>6}  "
        f"{_fmt(_percentile(values, 50))}  {_fmt(_percentile(values, 75))}  "
        f"{_fmt(_percentile(values, 90))}  {_fmt(p99)}  {_fmt(max(values))}"
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    args = build_parser().parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    hours = (
        tuple(int(h) for h in args.hours.split(",") if h.strip())
        if args.hours
        else DEFAULT_HOURS
    )
    start, end = _day(args.start), _day(args.end)

    database = Database.from_env()
    try:
        print(f"\n  source={args.source!r}  timeframe={args.timeframe}  {args.start} .. {args.end}")
        for symbol in symbols:
            rows = await _load(database, symbol, args.timeframe, args.source, start, end)
            if not rows:
                print(f"\n  {symbol}: no bars stored for this source and timeframe")
                continue
            _report(symbol, rows, hours)

        print(
            "\n  Note: one spread reading per bar. MT5 does not document whether that is the\n"
            "  minimum, the last tick's, or an average within the bar. Run\n"
            "  `python -m fxagent.spreadwatch` for a few sessions and compare against these\n"
            "  numbers before treating the percentiles as the spread at the moment of a fill.\n"
        )
    finally:
        await database.dispose()


if __name__ == "__main__":
    asyncio.run(main())
