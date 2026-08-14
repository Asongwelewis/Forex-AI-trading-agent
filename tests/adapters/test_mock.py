"""Tests for MockAdapter, including that it genuinely satisfies BrokerAdapter.

`isinstance` against a runtime-checkable Protocol only compares method NAMES, so it
would pass for an implementation with completely wrong signatures. The signature
comparison below is the assertion that actually has teeth.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from fxagent.adapters.base import (
    AccountState,
    BarSeries,
    BrokerAdapter,
    OrderRequest,
    OrderSide,
    Tick,
)
from fxagent.adapters.mock import MockAdapter

PROTOCOL_METHODS = (
    "get_bars",
    "get_tick",
    "get_account",
    "get_positions",
    "place_order",
    "close_position",
)


def _buy(symbol: str = "EURUSD") -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY,
        volume=0.01,
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
    )


# --- protocol conformance -----------------------------------------------------


def test_mock_adapter_satisfies_protocol() -> None:
    adapter: BrokerAdapter = MockAdapter()
    assert isinstance(adapter, BrokerAdapter)


@pytest.mark.parametrize("name", PROTOCOL_METHODS)
def test_mock_adapter_signature_matches_protocol(name: str) -> None:
    expected = inspect.signature(getattr(BrokerAdapter, name))
    actual = inspect.signature(getattr(MockAdapter, name))
    assert actual == expected, f"{name} signature drifted from BrokerAdapter"


@pytest.mark.parametrize("name", PROTOCOL_METHODS)
def test_protocol_method_returns_declared_type(name: str) -> None:
    adapter = MockAdapter()
    returns = {
        "get_bars": (("EURUSD", "H1", 5), BarSeries),
        "get_tick": (("EURUSD",), Tick),
        "get_account": ((), AccountState),
        "get_positions": ((), list),
        "place_order": ((_buy(),), object),
        "close_position": ((1,), object),
    }
    args, expected_type = returns[name]
    assert isinstance(getattr(adapter, name)(*args), expected_type)


# --- determinism --------------------------------------------------------------


def test_bars_are_deterministic_across_instances() -> None:
    first = MockAdapter().get_bars("EURUSD", "H1", 50)
    second = MockAdapter().get_bars("EURUSD", "H1", 50)
    assert first == second


def test_different_seeds_produce_different_series() -> None:
    first = MockAdapter(seed=1).get_bars("EURUSD", "H1", 50)
    second = MockAdapter(seed=2).get_bars("EURUSD", "H1", 50)
    assert first != second


def test_now_defaults_to_a_fixed_instant_not_the_wall_clock() -> None:
    assert MockAdapter().get_tick("EURUSD").timestamp == datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


# --- generated data is well formed --------------------------------------------


def test_generated_series_is_ordered_and_the_requested_length() -> None:
    series = MockAdapter().get_bars("EURUSD", "H1", 120)
    assert len(series) == 120
    timestamps = [bar.timestamp for bar in series.bars]
    assert timestamps == sorted(timestamps)
    assert all(bar.timestamp.tzinfo is UTC for bar in series.bars)


def test_jpy_pair_uses_three_digit_point_convention() -> None:
    tick = MockAdapter().get_tick("USDJPY")
    assert tick.point == pytest.approx(1e-3)
    assert tick.spread_points == 10


@pytest.mark.parametrize(
    ("symbol", "timeframe", "count"),
    [("XAUUSD", "H1", 10), ("EURUSD", "H3", 10), ("EURUSD", "H1", 0)],
    ids=["unknown-symbol", "unknown-timeframe", "non-positive-count"],
)
def test_get_bars_rejects_bad_input(symbol: str, timeframe: str, count: int) -> None:
    with pytest.raises(ValueError):
        MockAdapter().get_bars(symbol, timeframe, count)


# --- account and order lifecycle ----------------------------------------------


def test_account_reports_demo() -> None:
    assert MockAdapter().get_account().is_demo is True


def test_buy_fills_at_ask_and_sell_fills_at_bid_never_mid() -> None:
    adapter = MockAdapter()
    tick = adapter.get_tick("EURUSD")

    buy = adapter.place_order(_buy())
    sell = adapter.place_order(
        OrderRequest(
            symbol="EURUSD",
            side=OrderSide.SELL,
            volume=0.01,
            entry_price=1.1000,
            stop_loss=1.1050,
            take_profit=1.0900,
        )
    )

    assert buy.filled_price == tick.ask
    assert sell.filled_price == tick.bid


def test_placing_an_order_opens_a_position_carrying_its_protection() -> None:
    adapter = MockAdapter()
    order = _buy()
    result = adapter.place_order(order)

    (position,) = adapter.get_positions()
    assert result.success
    assert position.ticket == result.ticket
    assert position.stop_loss == order.stop_loss
    assert position.take_profit == order.take_profit


def test_tickets_are_unique() -> None:
    adapter = MockAdapter()
    tickets = {adapter.place_order(_buy()).ticket for _ in range(5)}
    assert len(tickets) == 5


def test_closing_removes_the_position() -> None:
    adapter = MockAdapter()
    ticket = adapter.place_order(_buy()).ticket
    assert ticket is not None

    result = adapter.close_position(ticket)
    assert result.success
    assert adapter.get_positions() == []


def test_closing_an_unknown_ticket_fails_without_raising() -> None:
    result = MockAdapter().close_position(99_999)
    assert result.success is False
    assert "99999" in result.message


def test_placing_an_order_for_an_unknown_symbol_raises() -> None:
    with pytest.raises(ValueError, match="unknown symbol"):
        MockAdapter().place_order(_buy(symbol="XAUUSD"))
