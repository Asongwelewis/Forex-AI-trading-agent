"""A `BrokerAdapter` over synthetic data, so strategies can be tested without MT5.

Fully deterministic: the same `(seed, symbol, timeframe, count, now)` always yields the
same bars, and `now` defaults to a fixed instant rather than the wall clock. Tests that
depend on a real clock fail on a Sunday.

This models price, not economics: `equity == balance` and margin is always zero, because
a synthetic P/L would be a number that looks authoritative while meaning nothing. Phase 7's
backtest engine is where fills, spread, swap and slippage get modelled honestly.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from datetime import UTC, datetime

from fxagent.adapters.base import (
    TIMEFRAMES,
    AccountState,
    Bar,
    BarSeries,
    OrderRequest,
    OrderResult,
    OrderSide,
    Position,
    Tick,
)

logger = logging.getLogger(__name__)

#: Default synthetic instruments and their starting mid price.
DEFAULT_SYMBOLS: Mapping[str, float] = {
    "EURUSD": 1.08,
    "GBPUSD": 1.27,
    "EURGBP": 0.85,
    "USDJPY": 157.0,
}

#: Fixed default "now" — a Monday inside the London session. Never the wall clock.
DEFAULT_NOW = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def _point_for(base_price: float) -> float:
    """Point size implied by the quote convention: JPY pairs quote to 3 digits, others to 5."""
    return 1e-3 if base_price > 20 else 1e-5


class MockAdapter:
    """In-memory `BrokerAdapter` implementation backed by a seeded random walk."""

    def __init__(
        self,
        *,
        symbols: Mapping[str, float] | None = None,
        now: datetime | None = None,
        seed: int = 7,
        spread_points: int = 10,
        balance: float = 1000.0,
        currency: str = "USD",
    ) -> None:
        self._symbols = dict(symbols if symbols is not None else DEFAULT_SYMBOLS)
        self._now = now if now is not None else DEFAULT_NOW
        self._seed = seed
        self._spread_points = spread_points
        self._balance = balance
        self._currency = currency
        self._positions: dict[int, Position] = {}
        self._next_ticket = 1

        if self._now.tzinfo is None:
            raise ValueError("now must be timezone-aware")

    # -- helpers ---------------------------------------------------------------

    def _require_symbol(self, symbol: str) -> float:
        try:
            return self._symbols[symbol]
        except KeyError:
            raise ValueError(
                f"unknown symbol {symbol!r}; MockAdapter knows {sorted(self._symbols)}"
            ) from None

    def _digits(self, symbol: str) -> int:
        return 3 if self._symbols[symbol] > 20 else 5

    def _generate_bars(self, symbol: str, timeframe: str, count: int) -> list[Bar]:
        """Seeded random walk ending at `now`, oldest bar first."""
        base = self._require_symbol(symbol)
        step = TIMEFRAMES[timeframe]
        digits = self._digits(symbol)
        rng = random.Random(f"{self._seed}|{symbol}|{timeframe}")
        volatility = base * 0.0008

        bars: list[Bar] = []
        price = base
        start = self._now - step * count
        for index in range(count):
            open_ = price
            close = open_ + rng.gauss(0.0, volatility)
            high = max(open_, close) + abs(rng.gauss(0.0, volatility)) * 0.5
            low = min(open_, close) - abs(rng.gauss(0.0, volatility)) * 0.5
            bars.append(
                Bar(
                    timestamp=start + step * index,
                    open=round(open_, digits),
                    high=round(high, digits),
                    low=round(low, digits),
                    close=round(close, digits),
                    volume=rng.randint(100, 5_000),
                )
            )
            price = close
        return bars

    # -- BrokerAdapter ---------------------------------------------------------

    def get_bars(self, symbol: str, timeframe: str, count: int) -> BarSeries:
        self._require_symbol(symbol)
        if timeframe not in TIMEFRAMES:
            raise ValueError(
                f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}"
            )
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        return BarSeries(
            symbol=symbol,
            timeframe=timeframe,
            bars=tuple(self._generate_bars(symbol, timeframe, count)),
        )

    def get_tick(self, symbol: str) -> Tick:
        base = self._require_symbol(symbol)
        digits = self._digits(symbol)
        point = _point_for(base)
        last_close = self._generate_bars(symbol, "M1", 1)[-1].close
        bid = round(last_close, digits)
        ask = round(bid + self._spread_points * point, digits)
        return Tick(symbol=symbol, timestamp=self._now, bid=bid, ask=ask, point=point)

    def get_account(self) -> AccountState:
        return AccountState(
            balance=self._balance,
            equity=self._balance,
            margin=0.0,
            currency=self._currency,
            is_demo=True,
        )

    def get_positions(self) -> list[Position]:
        return [self._positions[ticket] for ticket in sorted(self._positions)]

    def place_order(self, order: OrderRequest) -> OrderResult:
        """Fill at ask for a buy and bid for a sell — never at mid."""
        self._require_symbol(order.symbol)
        tick = self.get_tick(order.symbol)
        fill = tick.ask if order.side is OrderSide.BUY else tick.bid

        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions[ticket] = Position(
            ticket=ticket,
            symbol=order.symbol,
            side=order.side,
            volume=order.volume,
            entry_price=fill,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            opened_at=self._now,
        )
        logger.debug("mock fill: ticket=%s %s %s @ %s", ticket, order.side, order.symbol, fill)
        return OrderResult(
            success=True,
            timestamp=self._now,
            ticket=ticket,
            retcode=0,
            message="filled",
            filled_price=fill,
            filled_volume=order.volume,
        )

    def close_position(self, ticket: int) -> OrderResult:
        position = self._positions.pop(ticket, None)
        if position is None:
            return OrderResult(
                success=False,
                timestamp=self._now,
                retcode=1,
                message=f"no open position with ticket {ticket}",
            )
        tick = self.get_tick(position.symbol)
        fill = tick.bid if position.side is OrderSide.BUY else tick.ask
        return OrderResult(
            success=True,
            timestamp=self._now,
            ticket=ticket,
            retcode=0,
            message="closed",
            filled_price=fill,
            filled_volume=position.volume,
        )
