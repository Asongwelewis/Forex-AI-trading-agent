"""Properties that hold for all three strategies: purity, determinism, and no clock.

A strategy that reads the wall clock produces one answer live and a different one in a
backtest replaying the same bars — and the backtest is then measuring something that never
happened. These tests are cheap insurance against that drifting in later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fxagent.adapters.base import BarSeries
from fxagent.adapters.mock import MockAdapter
from fxagent.regime.classifier import Regime, RegimeClassifier
from fxagent.strategies import (
    CarryDivergence,
    MarketContext,
    RangeReversion,
    SessionBreakout,
    Signal,
    Strategy,
)

STRATEGY_MODULES = sorted(
    path for path in (Path(__file__).resolve().parents[2] / "fxagent" / "strategies").glob("*.py")
)

#: Anything that would make `generate` depend on when it ran, or on the outside world.
FORBIDDEN_SOURCE = (
    "datetime.now",
    "utcnow",
    "time.time",
    "date.today",
    "random",
    "requests",
    "urllib",
    "httpx",
    "open(",
    "print(",
)

CONTEXT = MarketContext(rate_differential=1.0, macro_bias=0.3)


def _regime_for(bars: BarSeries) -> Regime:
    """The regime the classifier really measures for these bars.

    Passed to all three strategies even though only `range_reversion` reads it, so these
    properties are checked against the arguments the live caller actually supplies.
    """
    return RegimeClassifier().classify(bars)


#: Each strategy paired with the timeframe it is built to read.
CASES = [
    pytest.param(SessionBreakout(), "H1", id="session_breakout"),
    pytest.param(RangeReversion(), "H1", id="range_reversion"),
    pytest.param(CarryDivergence(), "D1", id="carry_divergence"),
]


def _mock_bars(timeframe: str, count: int = 120) -> BarSeries:
    """Synthetic but well-formed bars, from the adapter the whole suite shares."""
    return MockAdapter(now=datetime(2026, 1, 5, 12, tzinfo=UTC)).get_bars(
        "EURUSD", timeframe, count
    )


# --- shape ---------------------------------------------------------------------


@pytest.mark.parametrize(("strategy", "timeframe"), CASES)
def test_every_strategy_implements_the_base_class(strategy: Strategy, timeframe: str) -> None:
    assert isinstance(strategy, Strategy)
    assert strategy.name
    assert strategy.required_bars > 0


def test_the_three_strategies_have_distinct_names() -> None:
    """Names key the journal and the router's per-strategy weights; a collision merges them."""
    names = {strategy.name for strategy in (SessionBreakout(), RangeReversion(), CarryDivergence())}
    assert names == {"session_breakout", "range_reversion", "carry_divergence"}


def test_the_base_class_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


# --- purity --------------------------------------------------------------------


@pytest.mark.parametrize(("strategy", "timeframe"), CASES)
def test_real_shaped_bars_never_raise(strategy: Strategy, timeframe: str) -> None:
    """Against 120 bars of a random walk, the answer may be None — but never an exception."""
    bars = _mock_bars(timeframe)
    result = strategy.generate(bars, CONTEXT, _regime_for(bars))
    assert result is None or isinstance(result, Signal)


@pytest.mark.parametrize(("strategy", "timeframe"), CASES)
def test_the_same_bars_always_give_the_same_answer(strategy: Strategy, timeframe: str) -> None:
    bars = _mock_bars(timeframe)
    regime = _regime_for(bars)
    assert strategy.generate(bars, CONTEXT, regime) == strategy.generate(bars, CONTEXT, regime)


@pytest.mark.parametrize(("strategy", "timeframe"), CASES)
def test_generate_does_not_mutate_the_bars_it_was_given(strategy: Strategy, timeframe: str) -> None:
    bars = _mock_bars(timeframe)
    before = bars.model_dump()

    strategy.generate(bars, CONTEXT, _regime_for(bars))

    assert bars.model_dump() == before


@pytest.mark.parametrize(("strategy", "timeframe"), CASES)
def test_a_signal_timestamp_is_always_the_last_bars_timestamp(
    strategy: Strategy, timeframe: str
) -> None:
    bars = _mock_bars(timeframe)
    signal = strategy.generate(bars, CONTEXT, _regime_for(bars))

    if signal is not None:
        assert signal.timestamp == bars.bars[-1].timestamp


@pytest.mark.parametrize(("strategy", "timeframe"), CASES)
def test_a_signal_is_always_for_the_symbol_it_was_given(strategy: Strategy, timeframe: str) -> None:
    bars = MockAdapter(now=datetime(2026, 1, 5, 12, tzinfo=UTC)).get_bars("GBPUSD", timeframe, 120)
    signal = strategy.generate(bars, CONTEXT, _regime_for(bars))

    if signal is not None:
        assert signal.symbol == "GBPUSD"


# --- no clock, no I/O ----------------------------------------------------------


def test_the_strategy_package_was_actually_found() -> None:
    """Guards the scan below against silently passing on an empty file list."""
    assert len(STRATEGY_MODULES) >= 5


@pytest.mark.parametrize("module", STRATEGY_MODULES, ids=lambda path: path.name)
@pytest.mark.parametrize("forbidden", FORBIDDEN_SOURCE)
def test_no_strategy_module_reads_a_clock_or_the_outside_world(
    module: Path, forbidden: str
) -> None:
    source = module.read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert forbidden not in code, f"{module.name} contains {forbidden!r}"
