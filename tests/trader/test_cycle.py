"""The assembly, end to end: bars in, a verdict and a ledger row out.

These are the tests the project did not have, and their absence is why nobody noticed that
nothing connected the components. Each one below fails if the pipeline is disconnected, which
no component test can do.

The strategies are stubbed rather than real. Their behaviour is exhaustively tested in
`tests/strategies/`, and using the real ones here would make an assertion about the assembly
depend on whether a hand-built bar sequence happens to trigger a breakout — a test that goes
red for the wrong reason.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from fxagent.adapters.base import BarSeries
from fxagent.regime.classifier import Regime
from fxagent.regime.router import CARRY_DIVERGENCE, RANGE_REVERSION, SESSION_BREAKOUT
from fxagent.risk.symbols import SymbolSpec
from fxagent.strategies.base import MarketContext, Signal, SignalDirection, Strategy
from fxagent.trader.cycle import CycleConfig, ledger_row, run_cycle
from tests.strategies.builders import flat_bar, h1_series

SYMBOL = "EURUSD"
#: A Monday inside the London morning, so the router has something to permit.
LONDON = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


class StubStrategy(Strategy):
    """Returns whatever it was handed. The assembly is under test, not the strategy."""

    def __init__(self, name: str, signal: Signal | None) -> None:
        self._name = name
        self._signal = signal
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_bars(self) -> int:
        return 1

    def generate(
        self, bars: BarSeries, context: MarketContext, regime: Regime | None = None
    ) -> Signal | None:
        self.calls += 1
        return self._signal


def _bars(count: int = 300, *, end: datetime = LONDON) -> BarSeries:
    """A flat H1 run ending at `end`. ADX is 0 on this series, so the market is ranging."""
    start = end - timedelta(hours=count - 1)
    return h1_series([flat_bar(start + timedelta(hours=i)) for i in range(count)])


def _long(strategy: str, *, at: datetime = LONDON, confidence: float = 0.7) -> Signal:
    return Signal(
        symbol=SYMBOL,
        direction=SignalDirection.LONG,
        confidence=confidence,
        entry_price=1.1000,
        stop_loss=1.0980,
        take_profit=1.1040,
        strategy_name=strategy,
        timestamp=at,
        reasoning={"stub": True},
    )


def _config(**overrides: object) -> CycleConfig:
    return CycleConfig(spec=SymbolSpec.forex(SYMBOL), history_bars=300, **overrides)


def _strategies(**signals: Signal | None) -> dict[str, Strategy]:
    return {name: StubStrategy(name, signal) for name, signal in signals.items()}


# --- the pipeline actually runs -----------------------------------------------


def test_a_silent_bar_still_produces_a_full_ledger_row() -> None:
    """The refusal record is the product as much as the trades are.

    This is the property that found the consensus failure: over 12,341 decisions the trade log
    said nothing at all, and the rejection records said the router had never permitted two
    strategies on a single bar. Losing it would lose the ability to notice the next such thing.
    """
    result = run_cycle(
        _bars(),
        config=_config(),
        equity=1000.0,
        strategies=_strategies(
            **{SESSION_BREAKOUT: None, RANGE_REVERSION: None, CARRY_DIVERGENCE: None}
        ),
    )

    assert not result.fired
    assert result.plan is None

    row = ledger_row(result)
    assert row["fired"] is False
    assert row["reason"], "a silent evaluation with no recorded reason is not training data"
    votes = row["votes"]["votes"]
    assert {vote["strategy"] for vote in votes} == {
        SESSION_BREAKOUT,
        RANGE_REVERSION,
        CARRY_DIVERGENCE,
    }, "every strategy in the router's slate gets a line, including the silent ones"


def test_every_strategy_in_the_slate_is_asked() -> None:
    strategies = _strategies(
        **{SESSION_BREAKOUT: None, RANGE_REVERSION: None, CARRY_DIVERGENCE: None}
    )
    run_cycle(_bars(), config=_config(), equity=1000.0, strategies=strategies)

    assert all(stub.calls == 1 for stub in strategies.values())  # type: ignore[attr-defined]


def test_a_strategy_the_caller_forgot_still_appears_in_the_ledger() -> None:
    """Silent and absent are different facts, and conflating them hid a year-long bug."""
    result = run_cycle(
        _bars(),
        config=_config(),
        equity=1000.0,
        strategies=_strategies(**{RANGE_REVERSION: None}),
    )

    names = {vote["strategy"] for vote in result.selection.diagnostics["votes"]}
    assert SESSION_BREAKOUT in names and CARRY_DIVERGENCE in names


# --- firing, and sizing --------------------------------------------------------


def test_a_permitted_sleeve_fires_and_is_sized() -> None:
    """The whole path: classify, route, select, size. Ranging, so range_reversion is permitted."""
    result = run_cycle(
        _bars(),
        config=_config(),
        equity=100_000.0,
        strategies=_strategies(
            **{
                SESSION_BREAKOUT: None,
                RANGE_REVERSION: _long(RANGE_REVERSION),
                CARRY_DIVERGENCE: None,
            }
        ),
    )

    assert result.fired, result.selection.diagnostics["reason"]
    assert result.actionable
    assert result.plan is not None
    assert result.plan.volume > 0
    assert result.plan.risk_fraction <= result.plan.max_risk_per_trade


def test_size_never_exceeds_the_half_percent_cap() -> None:
    """Hard rule 8, asserted at the assembly rather than only at the sizing function.

    The cap is already enforced two ways inside `risk.sizing`. This checks the wiring did not
    hand it a different number on the way past — which is the only new way it could break.
    """
    result = run_cycle(
        _bars(),
        config=_config(),
        equity=1_000_000.0,
        strategies=_strategies(**{RANGE_REVERSION: _long(RANGE_REVERSION)}),
    )

    assert result.plan is not None
    assert result.plan.risk_fraction <= 0.005


def test_an_unsizeable_setup_is_reported_not_silently_dropped() -> None:
    """Fired-but-not-sizeable and never-fired are different states.

    A tiny account cannot take the minimum lot at 0.5% risk. That is not a rejection of the
    setup — the signal stands and the ledger says why no order followed.
    """
    result = run_cycle(
        _bars(),
        config=_config(),
        equity=1.0,
        strategies=_strategies(**{RANGE_REVERSION: _long(RANGE_REVERSION)}),
    )

    assert result.fired
    assert result.plan is None
    assert not result.actionable
    assert result.sizing_note
    assert ledger_row(result)["votes"]["sizing_note"] == result.sizing_note


# --- the daily view -------------------------------------------------------------


def test_an_absent_daily_series_is_recorded_not_treated_as_neutral() -> None:
    """A filter that was never consulted must not look like one that had no opinion.

    The two produce the same trades and a different system, so the difference has to be in the
    record rather than only in the caller's memory.
    """
    result = run_cycle(
        _bars(),
        config=_config(),
        equity=100_000.0,
        strategies=_strategies(**{RANGE_REVERSION: _long(RANGE_REVERSION)}),
        daily_bars=None,
    )

    assert result.bias.direction is None
    assert "no daily view was supplied" in result.selection.diagnostics["bias"]["reason"]


# --- guards -----------------------------------------------------------------------


def test_an_empty_series_is_refused_rather_than_evaluated() -> None:
    with pytest.raises(ValueError, match="empty series"):
        run_cycle(
            BarSeries(symbol=SYMBOL, timeframe="H1", bars=()),
            config=_config(),
            equity=1000.0,
            strategies={},
        )


def test_the_ledger_row_is_json_safe() -> None:
    """It goes into a JSONB column. A stray datetime or Decimal fails at the driver, at 3am."""
    result = run_cycle(
        _bars(),
        config=_config(),
        equity=100_000.0,
        strategies=_strategies(**{RANGE_REVERSION: _long(RANGE_REVERSION)}),
    )

    row = ledger_row(result)
    json.dumps(row["votes"])
    json.dumps(row["regime"])


def test_the_cycle_reads_no_clock() -> None:
    """The timestamp comes from the last bar, which is what lets the replay share this code.

    A cycle that read `datetime.now()` would give a different answer replayed than live, and
    the backtest would stop being a measurement of the live system.
    """
    bars = _bars(end=datetime(2024, 3, 7, 9, 0, tzinfo=UTC))
    result = run_cycle(bars, config=_config(), equity=1000.0, strategies={})

    assert result.timestamp == datetime(2024, 3, 7, 9, 0, tzinfo=UTC)


def test_two_runs_over_the_same_bars_agree() -> None:
    """Deterministic, so a re-run after a dropped connection converges rather than diverging."""
    strategies = _strategies(**{RANGE_REVERSION: _long(RANGE_REVERSION)})
    first = run_cycle(_bars(), config=_config(), equity=100_000.0, strategies=strategies)
    second = run_cycle(_bars(), config=_config(), equity=100_000.0, strategies=strategies)

    assert first.selection.diagnostics["reason"] == second.selection.diagnostics["reason"]
    assert (first.plan is None) == (second.plan is None)
    if first.plan and second.plan:
        assert first.plan.volume == second.plan.volume
