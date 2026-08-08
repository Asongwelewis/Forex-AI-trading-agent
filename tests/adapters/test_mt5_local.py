"""Tests for MT5LocalAdapter that do not need a terminal.

The conversion layer is where MT5 bugs actually live — server-time offsets, numpy
scalars, and 0.0-means-absent sentinels — so it is written as pure functions and
tested directly. Connection behaviour is exercised against a fake `MetaTrader5`
module injected onto the adapter.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from fxagent.adapters.base import BrokerAdapter, OrderSide
from fxagent.adapters.mt5_local import (
    MT5AccountGuardError,
    MT5ConnectionError,
    MT5LocalAdapter,
    _bar_from_rate,
    _filling_mode,
    _mt5_timeframe,
    _optional_price,
    _position_from_mt5,
    _snap_offset,
    _utc_from_server_epoch,
)

PROTOCOL_METHODS = (
    "get_bars",
    "get_tick",
    "get_account",
    "get_positions",
    "place_order",
    "close_position",
)

#: The dtype MT5 actually returns from copy_rates_from_pos.
RATE_DTYPE = np.dtype(
    [
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<u8"),
        ("spread", "<i4"),
        ("real_volume", "<u8"),
    ]
)

ENV = {
    "MT5_LOGIN": "12345678",
    "MT5_PASSWORD": "hunter2",
    "MT5_SERVER": "Exness-MT5Trial9",
    "MT5_SYMBOLS": "EURUSD,GBPUSD",
}


# --- protocol conformance -----------------------------------------------------


@pytest.mark.parametrize("name", PROTOCOL_METHODS)
def test_signature_matches_protocol(name: str) -> None:
    expected = inspect.signature(getattr(BrokerAdapter, name))
    actual = inspect.signature(getattr(MT5LocalAdapter, name))
    assert actual == expected, f"{name} signature drifted from BrokerAdapter"


def test_every_supported_timeframe_maps_to_an_mt5_constant() -> None:
    import MetaTrader5 as mt5

    from fxagent.adapters.base import TIMEFRAMES

    for timeframe in TIMEFRAMES:
        assert isinstance(_mt5_timeframe(mt5, timeframe), int)


# --- server time is not UTC ---------------------------------------------------


def test_server_epoch_is_shifted_back_to_true_utc() -> None:
    """A bar stamped 12:00 on a GMT+3 broker clock is really 09:00 UTC."""
    server_noon = datetime(2026, 1, 5, 12, 0, tzinfo=UTC).timestamp()
    result = _utc_from_server_epoch(server_noon, timedelta(hours=3))
    assert result == datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    assert result.tzinfo is UTC


def test_zero_offset_leaves_timestamps_untouched() -> None:
    epoch = datetime(2026, 1, 5, 12, 0, tzinfo=UTC).timestamp()
    assert _utc_from_server_epoch(epoch, timedelta(0)) == datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("raw_minutes", "expected_minutes"),
    [(179, 180), (181, 180), (150, 150), (-121, -120), (7, 0)],
)
def test_detected_offset_snaps_to_the_half_hour_grid(
    raw_minutes: int, expected_minutes: int
) -> None:
    assert _snap_offset(timedelta(minutes=raw_minutes)) == timedelta(minutes=expected_minutes)


# --- numpy coercion -----------------------------------------------------------


def _rate(epoch: float) -> np.void:
    return np.array(
        [(int(epoch), 1.10000, 1.10250, 1.09800, 1.10100, 1234, 12, 0)], dtype=RATE_DTYPE
    )[0]


def test_bar_fields_are_native_python_types_not_numpy_scalars() -> None:
    bar = _bar_from_rate(_rate(datetime(2026, 1, 5, 12, 0, tzinfo=UTC).timestamp()), timedelta(0))

    assert type(bar.open) is float
    assert type(bar.volume) is int
    assert not isinstance(bar.open, np.floating)
    assert not isinstance(bar.volume, np.integer)


def test_bar_serialises_to_json() -> None:
    """numpy int64 does not serialise; this is the assertion that proves coercion worked."""
    bar = _bar_from_rate(_rate(datetime(2026, 1, 5, 12, 0, tzinfo=UTC).timestamp()), timedelta(0))
    assert '"volume":1234' in bar.model_dump_json().replace(" ", "")


def test_bar_timestamp_has_the_server_offset_removed() -> None:
    bar = _bar_from_rate(
        _rate(datetime(2026, 1, 5, 12, 0, tzinfo=UTC).timestamp()), timedelta(hours=2)
    )
    assert bar.timestamp == datetime(2026, 1, 5, 10, 0, tzinfo=UTC)


# --- 0.0 means absent, not zero ----------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [(0.0, None), (1.0950, 1.0950)])
def test_absent_stop_is_none_not_zero(raw: float, expected: float | None) -> None:
    assert _optional_price(raw) == expected


def test_position_without_protection_converts_to_none() -> None:
    raw = SimpleNamespace(
        ticket=42,
        type=0,
        volume=0.01,
        price_open=1.1,
        sl=0.0,
        tp=0.0,
        time=datetime(2026, 1, 5, 12, 0, tzinfo=UTC).timestamp(),
        symbol="EURUSDm",
    )
    position = _position_from_mt5(raw, "EURUSD", timedelta(0))

    assert position.stop_loss is None
    assert position.take_profit is None
    assert position.side is OrderSide.BUY
    assert position.symbol == "EURUSD"


def test_position_type_one_is_a_sell() -> None:
    raw = SimpleNamespace(
        ticket=43,
        type=1,
        volume=0.02,
        price_open=1.1,
        sl=1.11,
        tp=1.09,
        time=datetime(2026, 1, 5, 12, 0, tzinfo=UTC).timestamp(),
        symbol="EURUSD",
    )
    assert _position_from_mt5(raw, "EURUSD", timedelta(0)).side is OrderSide.SELL


# --- filling mode -------------------------------------------------------------


@pytest.mark.parametrize(
    ("allowed", "expected_attr"),
    [(2, "ORDER_FILLING_IOC"), (3, "ORDER_FILLING_IOC"), (1, "ORDER_FILLING_FOK")],
    ids=["ioc-only", "both-prefers-ioc", "fok-only"],
)
def test_filling_mode_respects_what_the_symbol_supports(allowed: int, expected_attr: str) -> None:
    import MetaTrader5 as mt5

    chosen = _filling_mode(mt5, SimpleNamespace(filling_mode=allowed))
    assert chosen == int(getattr(mt5, expected_attr))


# --- configuration ------------------------------------------------------------


def test_from_env_parses_configuration() -> None:
    adapter = MT5LocalAdapter.from_env(
        {**ENV, "MT5_SYMBOL_SUFFIX": "m", "MT5_SERVER_UTC_OFFSET_HOURS": "3"}
    )
    assert adapter.reference_symbol == "EURUSD"
    assert adapter._broker_symbol("EURUSD") == "EURUSDm"
    assert adapter._configured_offset == timedelta(hours=3)


@pytest.mark.parametrize("missing", ["MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"])
def test_from_env_names_the_missing_variable(missing: str) -> None:
    env = {key: value for key, value in ENV.items() if key != missing}
    with pytest.raises(MT5ConnectionError, match=missing):
        MT5LocalAdapter.from_env(env)


def test_from_env_rejects_a_non_numeric_login() -> None:
    with pytest.raises(MT5ConnectionError, match="numeric account id"):
        MT5LocalAdapter.from_env({**ENV, "MT5_LOGIN": "my-account"})


def test_password_never_appears_in_repr() -> None:
    adapter = MT5LocalAdapter.from_env(ENV)
    assert "hunter2" not in repr(adapter)
    assert "12345678" in repr(adapter)


def test_operations_before_connecting_fail_loudly() -> None:
    adapter = MT5LocalAdapter.from_env(ENV)
    with pytest.raises(MT5ConnectionError, match="not connected"):
        adapter.get_account()


def test_suffix_is_stripped_from_broker_symbols() -> None:
    adapter = MT5LocalAdapter.from_env({**ENV, "MT5_SYMBOL_SUFFIX": "m"})
    assert adapter._strip_suffix("EURUSDm") == "EURUSD"
    assert adapter._strip_suffix("EURUSD") == "EURUSD"


# --- account guards -----------------------------------------------------------


class FakeMT5:
    """Minimal stand-in for the MetaTrader5 module."""

    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_REAL = 2

    def __init__(self, *, login: int, trade_mode: int, trade_allowed: bool = True) -> None:
        self._login = login
        self._trade_mode = trade_mode
        self._trade_allowed = trade_allowed
        self.shutdown_calls = 0

    def terminal_info(self) -> SimpleNamespace:
        return SimpleNamespace(trade_allowed=self._trade_allowed)

    def account_info(self) -> SimpleNamespace:
        return SimpleNamespace(
            login=self._login,
            trade_mode=self._trade_mode,
            balance=1000.0,
            equity=1000.0,
            margin=0.0,
            currency="USD",
        )

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _adapter_with(fake: FakeMT5) -> MT5LocalAdapter:
    adapter = MT5LocalAdapter.from_env(ENV)
    adapter._mt5 = fake  # type: ignore[assignment]
    return adapter


def test_real_money_account_is_rejected() -> None:
    fake = FakeMT5(login=12345678, trade_mode=FakeMT5.ACCOUNT_TRADE_MODE_REAL)
    with pytest.raises(MT5AccountGuardError, match="not a DEMO account"):
        _adapter_with(fake)._verify_account()


def test_wrong_account_is_rejected_even_when_demo() -> None:
    fake = FakeMT5(login=99999999, trade_mode=FakeMT5.ACCOUNT_TRADE_MODE_DEMO)
    with pytest.raises(MT5AccountGuardError, match="Refusing to operate"):
        _adapter_with(fake)._verify_account()


def test_matching_demo_account_is_accepted() -> None:
    fake = FakeMT5(login=12345678, trade_mode=FakeMT5.ACCOUNT_TRADE_MODE_DEMO)
    _adapter_with(fake)._verify_account()


def test_disabled_algo_trading_is_rejected_with_an_actionable_message() -> None:
    fake = FakeMT5(login=12345678, trade_mode=0, trade_allowed=False)
    with pytest.raises(MT5ConnectionError, match="Algo Trading"):
        _adapter_with(fake)._verify_terminal()


def test_account_state_reports_demo_from_trade_mode() -> None:
    fake = FakeMT5(login=12345678, trade_mode=FakeMT5.ACCOUNT_TRADE_MODE_DEMO)
    assert _adapter_with(fake).get_account().is_demo is True


def test_disconnect_shuts_the_terminal_down_and_is_idempotent() -> None:
    fake = FakeMT5(login=12345678, trade_mode=0)
    adapter = _adapter_with(fake)

    adapter.disconnect()
    adapter.disconnect()

    assert fake.shutdown_calls == 1
