"""The paper run and the backtest must measure the same system, or comparing them is theatre.

`fxagent.costs` sits at the top level so both charge the same spread, slippage and swap — that
is asserted in `tests/test_costs_are_shared.py`. This file covers the other three numbers that
have to agree and are easy to let drift, because each is written down in more than one place
for a good local reason:

* the **time barrier**, which decides when a trade is over;
* the **history window**, which decides what an indicator was seeded on;
* the **source**, which decides whose book the bars came from.

A drift in any of them shows up as a divergence between live and backtest results, and it would
be read as alpha decay rather than as the arithmetic difference it is.
"""

from __future__ import annotations

from fxagent.adapters.mt5_local import SOURCE as ADAPTER_SOURCE
from fxagent.backtest.replay import DEFAULT_SOURCE, ReplayConfig
from fxagent.resolve.service import DEFAULT_MAX_BARS_HELD
from fxagent.risk.symbols import SymbolSpec
from fxagent.trader.cycle import MAX_BARS_HELD as TRADER_MAX_BARS_HELD
from fxagent.trader.cycle import CycleConfig
from fxagent.trader.service import TraderConfig


def _replay_config() -> ReplayConfig:
    return ReplayConfig(spec=SymbolSpec.forex("EURUSD"))


def test_the_time_barrier_is_the_same_everywhere() -> None:
    """Three writers: the replay decides it, the trader writes the label span from it, and the
    resolver measures against it. A trader writing a 24-bar span while the resolver used 48
    would purge the wrong window out of every training fold that overlapped it.
    """
    horizons = {
        "replay": _replay_config().max_bars_held,
        "trader label span": TRADER_MAX_BARS_HELD,
        "resolver": DEFAULT_MAX_BARS_HELD,
    }
    assert len(set(horizons.values())) == 1, f"the time barrier has drifted: {horizons}"


def test_the_history_window_is_the_same_in_both() -> None:
    """Wilder smoothing has infinite memory, so an ADX seeded on 300 bars is not the ADX seeded
    on 5,000. A live run reading a different window from the backtest is a different system
    reporting the same name.
    """
    spec = SymbolSpec.forex("EURUSD")
    assert CycleConfig(spec=spec).history_bars == _replay_config().history_bars
    assert TraderConfig().history_bars == _replay_config().history_bars


def test_the_default_source_is_the_venues_own_book() -> None:
    """ADR-005. A backtest on one feed and fills on another are different experiments, and the
    replay's constant is now re-exported from the adapter that writes the rows rather than
    restated — two literals that happen to match are one rename from silent disagreement.
    """
    assert DEFAULT_SOURCE == ADAPTER_SOURCE == "mt5_exness"


def test_the_paper_trade_span_runs_to_the_horizon_not_to_the_exit() -> None:
    """The label span is the window this observation's outcome *could* depend on.

    `folds.purged_walk_forward` removes exactly that window from any training set overlapping a
    test fold. A span cut to the realised exit would be shorter than what was knowable at
    decision time, and purging on it would leak the answer into the question.
    """
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from fxagent.adapters.base import TIMEFRAMES
    from fxagent.agents.schemas import Briefing, ExecutionPlan
    from fxagent.regime.bias import DirectionalBias
    from fxagent.regime.selection import Contribution, SelectedSignal, SelectionResult
    from fxagent.risk.exposure import MAX_TOTAL_RISK
    from fxagent.risk.sizing import MAX_RISK_PER_TRADE
    from fxagent.strategies.base import Signal, SignalDirection
    from fxagent.trader.cycle import CycleResult, paper_trade

    at = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    signal = Signal(
        symbol="EURUSD",
        direction=SignalDirection.LONG,
        confidence=0.7,
        entry_price=1.1000,
        stop_loss=1.0980,
        take_profit=1.1040,
        strategy_name="range_reversion",
        timestamp=at,
        reasoning={},
    )
    selection = SelectionResult(
        signal=SelectedSignal(
            symbol="EURUSD",
            direction=SignalDirection.LONG,
            timestamp=at,
            total_weight=1.0,
            confidence=0.7,
            contributions=(Contribution(signal=signal, weight=1.0),),
        ),
        diagnostics={"reason": "test"},
    )
    plan = ExecutionPlan(
        volume=0.1,
        risk_fraction=0.005,
        risk_amount=500.0,
        stop_distance=0.002,
        total_open_risk=0.005,
        max_risk_per_trade=MAX_RISK_PER_TRADE,
        max_total_risk=MAX_TOTAL_RISK,
    )
    result = CycleResult(
        cycle_id=uuid4(),
        symbol="EURUSD",
        timeframe="H1",
        timestamp=at,
        selection=selection,
        briefing=Briefing(symbol="EURUSD", timestamp=at),
        bias=DirectionalBias.none("none"),
        plan=plan,
    )

    row = paper_trade(result, timeframe="H1")

    assert row["label_span_start"] == at
    assert row["label_span_end"] == at + TIMEFRAMES["H1"] * TRADER_MAX_BARS_HELD
    assert row["label_span_end"] - row["label_span_start"] == timedelta(
        hours=TRADER_MAX_BARS_HELD
    )


def test_a_paper_trade_can_never_claim_to_be_live() -> None:
    """`mode` is a constant with no argument that changes it, and Postgres refuses LIVE too.

    `trades_no_live_mode` is a database constraint, so this is belt and braces rather than the
    only defence — but the constant is what makes the intent unambiguous at the call site.
    """
    from fxagent.store.repositories.trades import PERMITTED_MODES
    from fxagent.trader.cycle import PAPER_MODE

    assert PAPER_MODE == "ADVISORY"
    assert PAPER_MODE in PERMITTED_MODES
    assert "LIVE" not in PERMITTED_MODES
