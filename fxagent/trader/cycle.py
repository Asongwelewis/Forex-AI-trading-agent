"""One symbol, one bar, one decision — the assembly that was missing.

Every part of this pipeline already existed and was tested. Nothing connected them: the
workflow named an `fxagent.analyst` that had never been written, and roughly nineteen thousand
lines of components had no code path from a stored bar to a recommendation. A green suite over
disconnected parts stays green, which is exactly why this went unnoticed.

**Nothing here decides anything.** The classifier measures, the router permits, the selector
chooses, `fxagent.risk` sizes. This module calls them in order and records what they said. Any
threshold, gate or comparison appearing in this file would be a second definition of a rule
that already has one — the failure that
`tests/regime/test_gate_coupling.py` exists to prevent, one layer up.

**The ledger is written on every path, and first.** A cycle that fired, a cycle where the
router permitted nothing, a cycle where the daily bias suppressed the only candidate — all
three produce an `evaluations` row with the full diagnostics. That record is what found the
consensus failure described in `fxagent.regime.selection`, and it is the training set for
anything that later learns which refusals were right. It is written before narration and before
any execution decision, so a Telegram outage or an LLM timeout cannot cost us the observation.

**Advisory is not a flag on the execution path; it is the absence of one.** `run_cycle`
produces a `CycleResult` and never places an order. Execution is a separate concern that will
sit behind `fxagent/permission/`, which does not exist yet — see GATE A in CLAUDE.md. There is
deliberately no parameter here that could be set to make this trade.

**Point-in-time by construction, like the replay.** The bars handed to the pipeline end at the
bar being decided, because that is the whole series that was fetched. There is no full-history
frame to forget to truncate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fxagent.adapters.base import BarSeries
from fxagent.agents.schemas import Briefing, ExecutionPlan
from fxagent.indicators import adx, atr
from fxagent.regime.bias import DirectionalBias, carry_bias
from fxagent.regime.classifier import RegimeClassifier
from fxagent.regime.router import RegimeRouter
from fxagent.regime.selection import SelectionResult, SleeveSelector
from fxagent.risk.exposure import MAX_TOTAL_RISK
from fxagent.risk.sizing import MAX_RISK_PER_TRADE, NOT_SIZEABLE, RiskConfig, position_size
from fxagent.risk.symbols import SymbolSpec
from fxagent.strategies.base import MarketContext, Signal, Strategy, bars_to_frame

__all__ = ["CycleConfig", "CycleResult", "run_cycle"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleConfig:
    """What one evaluation needs that is not the bars.

    `history_bars` must match `ReplayConfig.history_bars`. Wilder smoothing has infinite memory,
    so an ADX seeded on 300 bars is not the ADX seeded on 5,000 — and a live run that reads a
    different window from the backtest is measuring a different system while reporting the same
    name. This is the shared number; there is no reason for the two to differ and every reason
    for them not to.
    """

    spec: SymbolSpec
    history_bars: int = 300
    risk: RiskConfig = field(default_factory=RiskConfig)
    #: Injected, never fetched here. `neutral()` means no rate differential, which makes the
    #: daily view structurally absent rather than wrong — and the ledger says so.
    context: MarketContext = field(default_factory=MarketContext.neutral)

    def __post_init__(self) -> None:
        if self.history_bars < 1:
            raise ValueError(f"history_bars must be positive, got {self.history_bars}")


@dataclass(frozen=True)
class CycleResult:
    """What one evaluation produced, decided and unexecuted.

    `plan` is `None` on two quite different paths and the distinction matters: no signal fired,
    or a signal fired and could not be sized. `sizing_note` separates them, because "the router
    permitted nothing" and "the stop is too wide for this account" call for different responses
    and a single `None` would conflate them.
    """

    cycle_id: UUID
    symbol: str
    timeframe: str
    timestamp: datetime
    selection: SelectionResult
    briefing: Briefing
    bias: DirectionalBias
    plan: ExecutionPlan | None = None
    sizing_note: str = ""
    #: Filled in by the caller once the row is written. Carried so a notifier can cite it.
    evaluation_id: int | None = None

    @property
    def fired(self) -> bool:
        return self.selection.fired

    @property
    def actionable(self) -> bool:
        """Fired *and* sizeable. The only state a permission layer would ever be asked about."""
        return self.fired and self.plan is not None


def _indicator_readings(bars: BarSeries) -> dict[str, float | None]:
    """Named readings at the decided bar, for the briefing and the panel.

    `None` is a warm-up, and it is rendered as such rather than as zero, for the same reason the
    classifier reports it that way: an ADX of zero and an ADX that cannot yet be computed are
    different facts, and a chart that draws the second as the first invents a trendless market.
    """
    frame = bars_to_frame(bars)
    high, low, close = frame["high"], frame["low"], frame["close"]

    readings: dict[str, float | None] = {}
    for name, series in (
        ("adx_14", adx(high, low, close, 14)),
        ("atr_14", atr(high, low, close, 14)),
    ):
        value = series.iloc[-1] if len(series) else float("nan")
        # NaN is the warm-up, and `value != value` is the only NaN test that does not need
        # numpy imported here for one comparison.
        readings[name] = None if value != value else float(value)
    return readings


def _sized_plan(
    signal: Signal,
    *,
    equity: float,
    config: CycleConfig,
    open_risk: float,
) -> tuple[ExecutionPlan | None, str]:
    """Turn the selected signal into a sized order, or say why it is not one.

    Returns `(None, reason)` rather than raising. A stop too wide for the account at 0.5% is not
    an error and not a rejection of the setup — the signal stands, and the same setup on a larger
    account sizes fine. `fxagent.risk.sizing` already names that state; this carries the name
    through rather than inventing a second vocabulary for it.
    """
    if signal.stop_loss is None or signal.take_profit is None:
        # `Signal` forbids this for a directional signal. Belt and braces: if it ever became
        # reachable, sizing a trade with no stop is the one failure worth being loud about.
        return None, "the selected signal carries no stop; refusing to size it"

    size = position_size(
        equity,
        config.risk.risk_fraction,
        signal.entry_price,
        signal.stop_loss,
        config.spec,
        account_currency=config.risk.account_currency,
    )
    if size is None:
        return None, NOT_SIZEABLE

    total = open_risk + size.risk_fraction
    return (
        ExecutionPlan(
            volume=size.volume,
            risk_fraction=size.risk_fraction,
            risk_amount=size.risk_amount,
            stop_distance=size.stop_distance,
            total_open_risk=min(total, 1.0),
            max_risk_per_trade=MAX_RISK_PER_TRADE,
            max_total_risk=MAX_TOTAL_RISK,
        ),
        "",
    )


def run_cycle(
    bars: BarSeries,
    *,
    config: CycleConfig,
    equity: float,
    strategies: dict[str, Strategy],
    classifier: RegimeClassifier | None = None,
    router: RegimeRouter | None = None,
    selector: SleeveSelector | None = None,
    daily_bars: BarSeries | None = None,
    open_risk: float = 0.0,
    cycle_id: UUID | None = None,
) -> CycleResult:
    """Classify, route, select, size — and explain, whatever the answer.

    Pure and synchronous. No clock is read (the timestamp comes from the last bar), no socket is
    opened, and nothing is written. That is what lets the replay and the live run share this
    code path rather than each having their own, which is the property that made the 2024–25
    counterfactual believable in the first place.

    `daily_bars` is the D1 series behind the directional bias. Passing `None` is legitimate — the
    bias then reports "no daily view was supplied" and filters nothing — but it is a materially
    different system from one with the filter on, so the ledger records which it was rather than
    letting an absent daily fetch look like a neutral daily view.
    """
    measure = classifier or RegimeClassifier()
    slate = router or RegimeRouter()
    choose = selector or SleeveSelector()
    identifier = cycle_id or uuid4()

    if not len(bars):
        raise ValueError("cannot evaluate an empty series")

    regime = measure.classify(bars)
    weights = slate.weights(regime)

    signals: dict[str, Signal | None] = {}
    for name in weights:
        strategy = strategies.get(name)
        if strategy is None:
            # Not silently skipped: the selector's ledger iterates `weights`, so a strategy the
            # caller did not construct still gets a line saying it was never asked. That is the
            # difference between "silent" and "absent", and conflating them is how a
            # two-of-three rule ran on one strategy for a year.
            continue
        signals[name] = strategy.generate(bars, config.context, regime)

    bias = (
        carry_bias(daily_bars, config.context)
        if daily_bars is not None and len(daily_bars)
        else DirectionalBias.none("no daily view was supplied")
    )

    selection = choose.select(regime, signals, weights, bias=bias)

    plan: ExecutionPlan | None = None
    sizing_note = ""
    if selection.signal is not None:
        plan, sizing_note = _sized_plan(
            selection.signal.primary, equity=equity, config=config, open_risk=open_risk
        )

    briefing = Briefing.from_selection(
        regime,
        selection,
        indicators=_indicator_readings(bars),
        execution=plan,
    )

    if plan is None and selection.fired:
        logger.info(
            "%s %s fired but produced no order: %s", bars.symbol, bars.timeframe, sizing_note
        )

    return CycleResult(
        cycle_id=identifier,
        symbol=bars.symbol,
        timeframe=bars.timeframe,
        timestamp=regime.timestamp,
        selection=selection,
        briefing=briefing,
        bias=bias,
        plan=plan,
        sizing_note=sizing_note,
    )


def ledger_row(result: CycleResult) -> dict[str, Any]:
    """The `evaluations` row for this cycle, fired or not.

    `consensus_score` keeps its column name and now carries the selected sleeve's router weight.
    Renaming the column would have meant a migration whose only effect was to make every stored
    row before it unreadable by the same query — and the number it holds has always been "how
    strongly was this permitted", which is what the weight is.
    """
    diagnostics = dict(result.selection.diagnostics)
    diagnostics["sizing_note"] = result.sizing_note
    diagnostics["plan"] = result.plan.model_dump(mode="json") if result.plan else None
    diagnostics["actionable"] = result.actionable

    return {
        "cycle_id": result.cycle_id,
        "ts_utc": result.timestamp,
        "symbol": result.symbol,
        "regime": result.briefing.model_dump(mode="json", include=_REGIME_FIELDS),
        "votes": diagnostics,
        "consensus_score": (
            result.selection.signal.total_weight if result.selection.signal else 0.0
        ),
        "fired": result.fired,
        "reason": result.selection.diagnostics.get("reason") or "no reason was recorded",
    }


#: The measurement half of the briefing, stored in the `regime` column. Listed rather than
#: computed so that adding a field to `Briefing` does not silently change the shape of every
#: row written afterwards.
_REGIME_FIELDS = {
    "session",
    "sessions",
    "market_open",
    "minutes_until_weekly_close",
    "trend_strength",
    "volatility_percentile",
    "is_trending",
    "is_ranging",
    "indicators",
}


def utcnow() -> datetime:
    """The one clock read in this package, so tests can point at it."""
    return datetime.now(UTC)
