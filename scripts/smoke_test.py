"""Read-only connection check against the local MetaTrader 5 terminal.

Connects, prints the account summary, fetches 100 H1 bars for the first configured
symbol, prints the current spread, and exits.

THIS SCRIPT NEVER PLACES AN ORDER. It only reads.

    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta

from dotenv import load_dotenv

from fxagent.adapters.mt5_local import MT5Error, MT5LocalAdapter

BAR_COUNT = 100
TIMEFRAME = "H1"


def _format_offset(offset: timedelta) -> str:
    """Render a server offset as UTC+03:00 rather than as a raw timedelta."""
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()

    try:
        adapter = MT5LocalAdapter.from_env()
    except MT5Error as exc:
        print(f"configuration problem:\n  {exc}", file=sys.stderr)
        return 2

    try:
        with adapter as mt5_adapter:
            symbol = mt5_adapter.reference_symbol
            print(f"\nconnected: {mt5_adapter!r}")
            print(f"broker clock runs at {_format_offset(mt5_adapter.server_utc_offset)}")

            account = mt5_adapter.get_account()
            print("\naccount")
            print(f"  balance  {account.balance:.2f} {account.currency}")
            print(f"  equity   {account.equity:.2f} {account.currency}")
            print(f"  margin   {account.margin:.2f} {account.currency}")
            print(f"  demo     {account.is_demo}")

            series = mt5_adapter.get_bars(symbol, TIMEFRAME, BAR_COUNT)
            print(f"\n{TIMEFRAME} bars for {symbol}")
            print(f"  count    {len(series)}")
            print(f"  oldest   {series.bars[0].timestamp:%Y-%m-%d %H:%M} UTC")
            print(f"  newest   {series.bars[-1].timestamp:%Y-%m-%d %H:%M} UTC")
            print(f"  last close {series.bars[-1].close}")

            tick = mt5_adapter.get_tick(symbol)
            print(f"\nquote for {symbol}")
            print(f"  bid      {tick.bid}")
            print(f"  ask      {tick.ask}")
            print(f"  spread   {tick.spread_points} points")
            print(f"  as of    {tick.timestamp:%Y-%m-%d %H:%M:%S} UTC")

            positions = mt5_adapter.get_positions()
            print(f"\nopen positions: {len(positions)}")
    except MT5Error as exc:
        print(f"\nconnection failed:\n  {exc}", file=sys.stderr)
        return 1

    print("\nno orders were placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
