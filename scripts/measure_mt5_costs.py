"""Read-only Exness cost facts for the cost-calibration runbook.

This script intentionally does not place an order. It reports the terminal's swap mode/rates,
tick conversion fields, current spread, and the configured deviation budget. Real slippage remains
an observation from a watched demo fill (Gate A), never an estimate invented here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv


def _value(info: object, name: str, default: object = None) -> object:
    return getattr(info, name, default)


def run(symbols: tuple[str, ...]) -> int:
    import MetaTrader5 as mt5  # noqa: PLC0415 - Windows-only measurement tool

    from fxagent.adapters.mt5_local import MT5LocalAdapter  # noqa: PLC0415

    with MT5LocalAdapter.from_env() as adapter:
        print("read-only cost facts; no order_send call is made")
        print(f"server={adapter._server} offset={adapter.server_utc_offset}")  # noqa: SLF001
        for symbol in symbols:
            broker_symbol = adapter._select(symbol)  # noqa: SLF001 - measurement path
            info = mt5.symbol_info(broker_symbol)
            tick = mt5.symbol_info_tick(broker_symbol)
            if info is None or tick is None:
                print(f"{symbol}: unavailable ({broker_symbol})")
                continue
            point = float(_value(info, "point", 0.0) or 0.0)
            tick_size = float(_value(info, "trade_tick_size", 0.0) or 0.0)
            tick_value = float(_value(info, "trade_tick_value", 0.0) or 0.0)
            point_value = tick_value * point / tick_size if tick_size else 0.0
            print(
                f"{symbol} ({broker_symbol}): bid={tick.bid} ask={tick.ask} "
                f"spread_points={_value(info, 'spread')} "
                f"spread_float={_value(info, 'spread_float')}"
            )
            print(
                f"  swap_mode={_value(info, 'swap_mode')} swap_long={_value(info, 'swap_long')} "
                f"swap_short={_value(info, 'swap_short')} "
                f"rollover3={_value(info, 'swap_rollover3days')}"
            )
            print(
                f"  point={point} tick_size={tick_size} tick_value={tick_value} "
                f"point_value={point_value} stops_level={_value(info, 'trade_stops_level')}"
            )
        print("slippage: NOT MEASURED (requires a watched demo fill under Gate A)")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description="Read-only MT5/Exness cost facts.")
    parser.add_argument("--symbols", default="EURUSD,GBPUSD,EURGBP")
    args = parser.parse_args(argv)
    symbols = tuple(value.strip().upper() for value in args.symbols.split(",") if value.strip())
    if not symbols:
        raise SystemExit("no symbols supplied")
    return run(symbols)


if __name__ == "__main__":
    raise SystemExit(main())
