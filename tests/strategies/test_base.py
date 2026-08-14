"""The Signal contract, and the deliberate gap between domain and broker vocabulary."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from fxagent.adapters.base import OrderSide
from fxagent.strategies import (
    MarketContext,
    Signal,
    SignalDirection,
    bars_to_frame,
    order_side_for,
)
from tests.strategies.builders import WEEK_START, flat_run, h1_series

WHEN = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


def _signal(**overrides: object) -> Signal:
    fields: dict[str, object] = {
        "symbol": "EURUSD",
        "direction": SignalDirection.LONG,
        "confidence": 0.5,
        "entry_price": 1.1000,
        "stop_loss": 1.0980,
        "take_profit": 1.1040,
        "strategy_name": "test",
        "timestamp": WHEN,
    }
    return Signal(**{**fields, **overrides})  # type: ignore[arg-type]


# --- the two enums stay apart --------------------------------------------------


def test_signal_direction_and_order_side_share_no_vocabulary() -> None:
    """If these ever overlap, someone has started merging the two enums."""
    assert {member.value for member in SignalDirection}.isdisjoint(
        {member.value for member in OrderSide}
    )


def test_long_maps_to_buy_and_short_to_sell() -> None:
    assert order_side_for(SignalDirection.LONG) is OrderSide.BUY
    assert order_side_for(SignalDirection.SHORT) is OrderSide.SELL


def test_flat_has_no_broker_equivalent_and_refuses_to_guess() -> None:
    """The whole point of the split: FLAT must never silently become a BUY."""
    with pytest.raises(ValueError, match="no broker equivalent"):
        order_side_for(SignalDirection.FLAT)


# --- Signal validation ---------------------------------------------------------


def test_long_signal_accepts_a_stop_below_and_target_above() -> None:
    signal = _signal()
    assert signal.risk_distance == pytest.approx(0.0020)
    assert signal.reward_risk == pytest.approx(2.0)


def test_long_signal_rejects_a_stop_above_entry() -> None:
    with pytest.raises(ValidationError, match="stop_loss .* must be below entry"):
        _signal(stop_loss=1.1010)


def test_long_signal_rejects_a_target_below_entry() -> None:
    with pytest.raises(ValidationError, match="take_profit .* must be above entry"):
        _signal(take_profit=1.0990)


def test_short_signal_rejects_a_stop_below_entry() -> None:
    with pytest.raises(ValidationError, match="stop_loss .* must be above entry"):
        _signal(direction=SignalDirection.SHORT, stop_loss=1.0980, take_profit=1.0960)


def test_short_signal_accepts_the_mirrored_arrangement() -> None:
    signal = _signal(direction=SignalDirection.SHORT, stop_loss=1.1020, take_profit=1.0960)
    assert signal.risk_distance == pytest.approx(0.0020)
    assert signal.reward_risk == pytest.approx(2.0)


def test_a_directional_signal_cannot_be_built_without_protection() -> None:
    with pytest.raises(ValidationError, match="must carry both a stop_loss and a take_profit"):
        _signal(stop_loss=None)


def test_a_flat_signal_carries_no_protection() -> None:
    signal = _signal(direction=SignalDirection.FLAT, stop_loss=None, take_profit=None)
    assert signal.risk_distance == 0.0
    assert signal.reward_risk == 0.0


def test_a_flat_signal_rejects_a_stop() -> None:
    with pytest.raises(ValidationError, match="FLAT signal carries no position"):
        _signal(direction=SignalDirection.FLAT, take_profit=None)


def test_confidence_is_bounded_to_the_unit_interval() -> None:
    with pytest.raises(ValidationError):
        _signal(confidence=1.5)
    with pytest.raises(ValidationError):
        _signal(confidence=-0.1)


def test_a_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _signal(timestamp=datetime(2026, 1, 5, 9, 0))


def test_a_signal_is_frozen_after_construction() -> None:
    signal = _signal()
    with pytest.raises(ValidationError):
        signal.stop_loss = 1.0  # type: ignore[misc]


def test_reasoning_defaults_to_empty_and_survives_json() -> None:
    signal = _signal(reasoning={"atr": 0.002, "gate": "passed", "bars": 48, "fresh": True})
    assert '"gate":"passed"' in signal.model_dump_json().replace(" ", "")
    assert _signal().reasoning == {}


# --- MarketContext -------------------------------------------------------------


def test_market_context_defaults_are_neutral() -> None:
    context = MarketContext.neutral()
    assert context.rate_differential == 0.0
    assert context.macro_bias == 0.0
    assert MarketContext() == context


def test_macro_bias_is_bounded() -> None:
    with pytest.raises(ValidationError):
        MarketContext(macro_bias=1.5)


def test_market_context_is_frozen() -> None:
    with pytest.raises(ValidationError):
        MarketContext().macro_bias = 0.5  # type: ignore[misc]


# --- frame conversion ----------------------------------------------------------


def test_bars_to_frame_preserves_order_and_uses_bar_timestamps() -> None:
    bars = h1_series(flat_run(end=WEEK_START, count=5))
    frame = bars_to_frame(bars)

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 5
    assert frame.index[0] == bars.bars[0].timestamp
    assert frame.index[-1] == bars.bars[-1].timestamp
    assert frame.index.is_monotonic_increasing
