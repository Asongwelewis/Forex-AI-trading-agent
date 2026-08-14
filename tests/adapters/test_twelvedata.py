"""The Twelve Data adapter, against a stubbed transport — no network, no API key needed."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from fxagent.adapters.base import BrokerAdapter
from fxagent.adapters.credits import CreditLedger, CreditLimitExceeded
from fxagent.adapters.twelvedata import (
    SOURCE,
    TwelveDataAdapter,
    TwelveDataResponseError,
    TwelveDataUnsupported,
    from_provider_symbol,
    to_provider_symbol,
)


def _values(count: int = 3) -> list[dict[str, str]]:
    """Newest-first, exactly as the API returns them."""
    base = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
    return [
        {
            "datetime": (base - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": f"{1.0800 + i * 0.001:.5f}",
            "high": f"{1.0830 + i * 0.001:.5f}",
            "low": f"{1.0790 + i * 0.001:.5f}",
            "close": f"{1.0820 + i * 0.001:.5f}",
        }
        for i in range(count)
    ]


def _adapter(handler: Any, **kwargs: Any) -> TwelveDataAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return TwelveDataAdapter(api_key="test-key", client=client, **kwargs)


def _ok(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, content=json.dumps(payload), headers={"content-type": "application/json"}
    )


# -- symbols ------------------------------------------------------------------


@pytest.mark.parametrize(("canonical", "provider"), [("EURUSD", "EUR/USD"), ("USDJPY", "USD/JPY")])
def test_symbols_translate_both_ways(canonical: str, provider: str) -> None:
    assert to_provider_symbol(canonical) == provider
    assert from_provider_symbol(provider) == canonical


def test_an_already_slashed_symbol_passes_through() -> None:
    assert to_provider_symbol("EUR/USD") == "EUR/USD"


def test_an_uninferable_symbol_is_rejected_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="cannot infer"):
        to_provider_symbol("EURUSDm")


# -- protocol -----------------------------------------------------------------


def test_the_adapter_satisfies_the_broker_protocol() -> None:
    """Structurally a BrokerAdapter; the execution half refuses at runtime."""
    adapter = _adapter(lambda request: _ok({"values": _values()}))
    assert isinstance(adapter, BrokerAdapter)


def test_the_source_name_is_stable() -> None:
    """It is part of the bars unique key: changing it would double every historical row."""
    assert _adapter(lambda r: _ok({})).source == SOURCE == "twelvedata"


# -- bars ---------------------------------------------------------------------


async def test_bars_are_returned_oldest_first() -> None:
    """The API sends newest-first and BarSeries rejects that ordering outright."""
    adapter = _adapter(lambda request: _ok({"values": _values(3), "status": "ok"}))

    series = await adapter.get_bars("EURUSD", "H1", 3)

    assert len(series) == 3
    timestamps = [bar.timestamp for bar in series.bars]
    assert timestamps == sorted(timestamps)
    assert series.symbol == "EURUSD", "canonical, not EUR/USD"


async def test_the_request_asks_for_utc_explicitly() -> None:
    """Without timezone=UTC the API answers in exchange-local time and does not say so."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return _ok({"values": _values(), "status": "ok"})

    await _adapter(handler).get_bars("EURUSD", "H1", 3)

    assert seen["timezone"] == "UTC"
    assert seen["symbol"] == "EUR/USD"
    assert seen["interval"] == "1h"


async def test_timestamps_come_back_timezone_aware_utc() -> None:
    adapter = _adapter(lambda request: _ok({"values": _values(), "status": "ok"}))
    series = await adapter.get_bars("EURUSD", "H1", 3)

    for bar in series.bars:
        assert bar.timestamp.tzinfo is not None
        assert bar.timestamp.utcoffset() == timedelta(0)


async def test_missing_fx_volume_becomes_zero_not_an_error() -> None:
    """FX pairs report no volume. 0 means 'not reported' and must not be read as a quantity."""
    adapter = _adapter(lambda request: _ok({"values": _values(), "status": "ok"}))
    series = await adapter.get_bars("EURUSD", "H1", 3)

    assert all(bar.volume == 0 for bar in series.bars)


async def test_the_api_key_never_appears_in_the_repr() -> None:
    adapter = _adapter(lambda request: _ok({}))
    assert "test-key" not in repr(adapter)


# -- the HTTP 200 error trap --------------------------------------------------


async def test_an_error_payload_with_http_200_is_raised_not_parsed() -> None:
    """Twelve Data reports a blown quota as HTTP 200 with status=error in the body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(
            {
                "code": 429,
                "message": "You have run out of API credits",
                "status": "error",
            }
        )

    with pytest.raises(TwelveDataResponseError, match="run out of API credits") as caught:
        await _adapter(handler).get_bars("EURUSD", "H1", 3)

    assert caught.value.code == 429


async def test_an_empty_value_list_is_an_error_not_an_empty_series() -> None:
    adapter = _adapter(lambda request: _ok({"values": [], "status": "ok"}))
    with pytest.raises(TwelveDataResponseError, match="no H1 bars"):
        await adapter.get_bars("EURUSD", "H1", 3)


async def test_a_malformed_bar_is_reported_with_its_row() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(
            {"values": [{"datetime": "2026-03-10 12:00:00", "open": "1.08"}], "status": "ok"}
        )

    with pytest.raises(TwelveDataResponseError, match="missing field"):
        await _adapter(handler).get_bars("EURUSD", "H1", 1)


async def test_a_real_429_is_distinguished_from_a_local_refusal() -> None:
    """If the provider throttles while the ledger says there is room, they disagree — say so."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content="{}", headers={"content-type": "application/json"})

    with pytest.raises(TwelveDataResponseError, match="ledger and the provider disagree"):
        await _adapter(handler).get_bars("EURUSD", "H1", 3)


# -- execution is refused ------------------------------------------------------


def test_place_order_raises_rather_than_returning_a_failed_result() -> None:
    """A failed OrderResult would read as 'the broker rejected it', which is a different fact."""
    adapter = _adapter(lambda request: _ok({}))

    with pytest.raises(NotImplementedError, match="read-only data feed"):
        adapter.place_order(None)  # type: ignore[arg-type]


def test_close_position_raises() -> None:
    with pytest.raises(NotImplementedError):
        _adapter(lambda request: _ok({})).close_position(1)


@pytest.mark.parametrize("method", ["get_account", "get_positions"])
def test_account_shaped_methods_raise(method: str) -> None:
    adapter = _adapter(lambda request: _ok({}))
    with pytest.raises(TwelveDataUnsupported):
        getattr(adapter, method)()


# -- credits ------------------------------------------------------------------


async def test_a_call_over_the_daily_limit_never_reaches_the_network() -> None:
    """The point of the ledger: refuse locally rather than be throttled mid-backfill."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok({"values": _values(), "status": "ok"})

    ledger = CreditLedger(daily_limit=2, per_minute_limit=100)
    adapter = _adapter(handler, ledger=ledger)

    await adapter.get_bars("EURUSD", "H1", 3)
    await adapter.get_bars("EURUSD", "H1", 3)
    assert calls["n"] == 2

    with pytest.raises(CreditLimitExceeded):
        await adapter.get_bars("EURUSD", "H1", 3)

    assert calls["n"] == 2, "the refused call must not have been sent"


async def test_every_request_costs_exactly_one_credit() -> None:
    adapter = _adapter(lambda request: _ok({"values": _values(), "status": "ok"}))

    await adapter.get_bars("EURUSD", "H1", 3)
    await adapter.get_bars("GBPUSD", "H1", 3)

    assert adapter.ledger.spent_today == 2


# -- guards -------------------------------------------------------------------


async def test_an_oversized_request_is_rejected_before_it_is_sent() -> None:
    adapter = _adapter(lambda request: _ok({}))
    with pytest.raises(ValueError, match="response cap"):
        await adapter.get_bars("EURUSD", "H1", 6000)


async def test_an_unknown_timeframe_is_rejected() -> None:
    adapter = _adapter(lambda request: _ok({}))
    with pytest.raises(ValueError, match="unknown timeframe"):
        await adapter.get_bars("EURUSD", "H2", 10)


async def test_a_naive_end_date_is_rejected() -> None:
    adapter = _adapter(lambda request: _ok({"values": _values(), "status": "ok"}))
    with pytest.raises(ValueError, match="timezone-aware"):
        await adapter.bars_ending_at("EURUSD", "H1", 3, end=datetime(2026, 3, 10, 12, 0))


def test_a_missing_api_key_is_rejected_with_a_pointer() -> None:
    with pytest.raises(ValueError, match="TWELVEDATA_API_KEY"):
        TwelveDataAdapter.from_env({})
