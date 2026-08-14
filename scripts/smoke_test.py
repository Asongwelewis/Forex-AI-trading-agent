"""Read-only connection check, and a cross-provider divergence report.

    uv run python scripts/smoke_test.py --adapter mt5
    uv run python scripts/smoke_test.py --adapter twelvedata
    uv run python scripts/smoke_test.py --compare mt5,twelvedata

THIS SCRIPT NEVER PLACES AN ORDER. It only reads.

The comparison exists because two feeds agreeing is the only cheap evidence that either is
right. Expect them to differ:

* **MT5 and MetaApi should agree almost exactly** — same broker, same book. A material
  difference there means one of the adapters is mis-converting, not that the market moved.
* **Twelve Data will differ** from both, and should. It is an independent aggregated feed
  quoting mid rather than a broker's bid, so a spread-width discrepancy is expected and is
  precisely what makes it useful as a sanity check. A divergence far larger than a plausible
  spread means one series is wrong — usually a timestamp misalignment rather than a price error.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import timedelta

from dotenv import load_dotenv

from fxagent.adapters.base import BarSeries
from fxagent.adapters.divergence import compare_series, interpret

BAR_COUNT = 100
TIMEFRAME = "H1"

ADAPTERS = ("mt5", "twelvedata", "metaapi")


# -- adapter loading -----------------------------------------------------------


async def _load_series(name: str, symbol: str, timeframe: str, count: int) -> BarSeries:
    """Fetch bars from one adapter. Each is imported lazily so a missing SDK is not fatal."""
    if name == "mt5":
        from fxagent.adapters.mt5_local import MT5LocalAdapter

        adapter = MT5LocalAdapter.from_env()
        with adapter as connected:
            return connected.get_bars(symbol, timeframe, count)

    if name == "twelvedata":
        from fxagent.adapters.twelvedata import TwelveDataAdapter

        async with TwelveDataAdapter.from_env() as adapter:
            return await adapter.get_bars(symbol, timeframe, count)

    if name == "metaapi":
        raise NotImplementedError(
            "the MetaApi adapter is not built yet — it was deferred in Phase 7 pending a paid "
            "plan. Use --adapter twelvedata for data or mt5 for the local terminal."
        )

    raise ValueError(f"unknown adapter {name!r}; expected one of {ADAPTERS}")


# -- single-adapter report -----------------------------------------------------


def _describe(name: str, series: BarSeries) -> None:
    print(f"\n{name}: {series.timeframe} bars for {series.symbol}")
    print(f"  count      {len(series)}")
    print(f"  oldest     {series.bars[0].timestamp:%Y-%m-%d %H:%M} UTC")
    print(f"  newest     {series.bars[-1].timestamp:%Y-%m-%d %H:%M} UTC")
    print(f"  last close {series.bars[-1].close}")
    if all(bar.volume == 0 for bar in series.bars):
        print("  volume     not reported by this feed (0 does not mean no trading)")


# -- entry point ---------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    names = (
        [n.strip() for n in args.compare.split(",") if n.strip()]
        if args.compare
        else [args.adapter]
    )
    if len(names) == 1 and args.compare:
        print("--compare needs at least two adapters", file=sys.stderr)
        return 2

    loaded: dict[str, BarSeries] = {}
    for name in names:
        try:
            series = await _load_series(name, args.symbol, args.timeframe, args.count)
        except NotImplementedError as exc:
            print(f"\n{name}: skipped — {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"\n{name}: FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)
            if len(names) == 1:
                return 1
            continue
        loaded[name] = series
        _describe(name, series)

    if not loaded:
        print("\nno adapter returned data.", file=sys.stderr)
        return 1

    if len(loaded) > 1:
        print("\n--- divergence ---")
        ordered = list(loaded.items())
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                (left_name, left), (right_name, right) = ordered[i], ordered[j]
                divergence = compare_series(left_name, left, right_name, right)
                print(divergence.render(), end="")
                print(interpret(divergence))

    print("\nno orders were placed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only adapter smoke test. Never trades.")
    parser.add_argument("--adapter", choices=ADAPTERS, default="twelvedata")
    parser.add_argument(
        "--compare",
        default="",
        help="comma-separated adapters to fetch and cross-check, e.g. mt5,twelvedata",
    )
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--count", type=int, default=BAR_COUNT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    return asyncio.run(_run(args))


def _format_offset(offset: timedelta) -> str:
    """Render a server offset as UTC+03:00 rather than as a raw timedelta."""
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
