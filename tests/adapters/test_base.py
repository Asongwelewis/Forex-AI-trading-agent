"""Contract tests for the adapter boundary models.

The load-bearing test in this file is that an order without protection cannot be
constructed. Everything above this layer assumes that, so it is asserted directly
rather than inferred from a broker rejection.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from fxagent.adapters.base import (
    AccountState,
    Bar,
    BarSeries,
    BrokerAdapter,
    OrderRequest,
    OrderResult,
    OrderSide,
    Position,
    Tick,
)

T0 = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def _buy(**overrides: object) -> OrderRequest:
    kwargs: dict[str, object] = {
        "symbol": "EURUSD",
        "side": OrderSide.BUY,
        "volume": 0.01,
        "entry_price": 1.1000,
        "stop_loss": 1.0950,
        "take_profit": 1.1100,
    }
    kwargs.update(overrides)
    return OrderRequest(**kwargs)  # type: ignore[arg-type]


# --- OrderRequest cannot exist without protection ----------------------------


@pytest.mark.parametrize("missing", ["stop_loss", "take_profit"])
def test_order_cannot_be_constructed_without_protection(missing: str) -> None:
    kwargs = {
        "symbol": "EURUSD",
        "side": OrderSide.BUY,
        "volume": 0.01,
        "entry_price": 1.1000,
        "stop_loss": 1.0950,
        "take_profit": 1.1100,
    }
    del kwargs[missing]

    with pytest.raises(ValidationError, match=missing):
        OrderRequest(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["stop_loss", "take_profit"])
def test_order_rejects_explicit_none_for_protection(field: str) -> None:
    with pytest.raises(ValidationError):
        _buy(**{field: None})


def test_protection_cannot_be_stripped_after_construction() -> None:
    """Frozen model: a valid order cannot be mutated into an unprotected one."""
    order = _buy()
    with pytest.raises(ValidationError):
        order.stop_loss = None  # type: ignore[misc]


# --- the stop/target side validator ------------------------------------------


@pytest.mark.parametrize(
    ("side", "stop_loss"),
    [
        (OrderSide.BUY, 1.1050),
        (OrderSide.BUY, 1.1000),
        (OrderSide.SELL, 1.0950),
        (OrderSide.SELL, 1.1000),
    ],
    ids=[
        "buy-stop-above-entry",
        "buy-stop-exactly-at-entry",
        "sell-stop-below-entry",
        "sell-stop-exactly-at-entry",
    ],
)
def test_stop_loss_on_wrong_side_is_rejected(side: OrderSide, stop_loss: float) -> None:
    take_profit = 1.1100 if side is OrderSide.BUY else 1.0900
    with pytest.raises(ValidationError, match="stop_loss"):
        _buy(side=side, stop_loss=stop_loss, take_profit=take_profit)


@pytest.mark.parametrize(
    ("side", "take_profit"),
    [
        (OrderSide.BUY, 1.0900),
        (OrderSide.BUY, 1.1000),
        (OrderSide.SELL, 1.1100),
        (OrderSide.SELL, 1.1000),
    ],
)
def test_take_profit_on_wrong_side_is_rejected(side: OrderSide, take_profit: float) -> None:
    stop_loss = 1.0950 if side is OrderSide.BUY else 1.1050
    with pytest.raises(ValidationError, match="take_profit"):
        _buy(side=side, stop_loss=stop_loss, take_profit=take_profit)


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_correctly_sided_order_is_accepted(side: OrderSide) -> None:
    if side is OrderSide.BUY:
        order = _buy(side=side, stop_loss=1.0950, take_profit=1.1100)
    else:
        order = _buy(side=side, stop_loss=1.1050, take_profit=1.0900)
    assert order.risk_distance == pytest.approx(0.0050)


def test_risk_distance_is_never_zero() -> None:
    """Phase 6 divides by this; zero distance must be unconstructable."""
    with pytest.raises(ValidationError):
        _buy(stop_loss=1.1000)


# --- timestamps ---------------------------------------------------------------


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Tick(
            symbol="EURUSD", timestamp=datetime(2026, 1, 5, 12, 0), bid=1.1, ask=1.1001, point=1e-5
        )


def test_non_utc_timestamps_are_normalised_to_utc() -> None:
    plus_two = datetime(2026, 1, 5, 14, 0, tzinfo=UTC).astimezone(
        __import__("zoneinfo").ZoneInfo("Europe/Berlin")
    )
    tick = Tick(symbol="EURUSD", timestamp=plus_two, bid=1.1, ask=1.1001, point=1e-5)
    assert tick.timestamp.tzinfo is UTC
    assert tick.timestamp == datetime(2026, 1, 5, 14, 0, tzinfo=UTC)


# --- Tick ---------------------------------------------------------------------


def test_spread_is_derived_in_points() -> None:
    tick = Tick(symbol="EURUSD", timestamp=T0, bid=1.10000, ask=1.10010, point=1e-5)
    assert tick.spread_points == 10


def test_inverted_quote_is_rejected() -> None:
    with pytest.raises(ValidationError, match="below bid"):
        Tick(symbol="EURUSD", timestamp=T0, bid=1.10010, ask=1.10000, point=1e-5)


# --- Bar / BarSeries ----------------------------------------------------------


@pytest.mark.parametrize(
    ("high", "low"),
    [(1.0990, 1.0980), (1.1010, 1.1005)],
    ids=["high-below-close", "low-above-open"],
)
def test_impossible_ohlc_is_rejected(high: float, low: float) -> None:
    with pytest.raises(ValidationError):
        Bar(timestamp=T0, open=1.1000, high=high, low=low, close=1.1000, volume=10)


def test_bars_must_be_strictly_increasing_in_time() -> None:
    def bar(offset_hours: int) -> Bar:
        return Bar(
            timestamp=T0 + timedelta(hours=offset_hours),
            open=1.1,
            high=1.11,
            low=1.09,
            close=1.1,
            volume=1,
        )

    with pytest.raises(ValidationError, match="strictly increasing"):
        BarSeries(symbol="EURUSD", timeframe="H1", bars=(bar(1), bar(0)))


def test_unknown_timeframe_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown timeframe"):
        BarSeries(symbol="EURUSD", timeframe="H3", bars=())


# --- other contracts ----------------------------------------------------------


def test_position_permits_absent_stop_loss() -> None:
    """Broker state we do not control may lack a stop; that is unbounded risk, not zero."""
    position = Position(
        ticket=1, symbol="EURUSD", side=OrderSide.BUY, volume=0.01, entry_price=1.1, opened_at=T0
    )
    assert position.stop_loss is None


def test_successful_result_must_carry_a_ticket() -> None:
    with pytest.raises(ValidationError, match="must carry a ticket"):
        OrderResult(success=True, timestamp=T0)


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AccountState(
            balance=1.0, equity=1.0, margin=0.0, currency="USD", is_demo=True, leverage=500
        )  # type: ignore[call-arg]


def test_protocol_is_runtime_checkable() -> None:
    class NotAnAdapter:
        pass

    assert not isinstance(NotAnAdapter(), BrokerAdapter)
