"""Fixtures for the agent tests.

Every briefing here is built by running the *real* `Consensus` over real `Signal` objects at a
real instant, then handing the result to `Briefing.from_consensus`. Hand-writing a `Briefing`
literal would test the narration against a shape nobody produces — and the assembly step, where
a diagnostics key gets renamed and quietly stops reaching the prompt, is exactly where this
layer breaks.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fxagent.agents.schemas import (
    AnalogueBrief,
    Briefing,
    CalendarEventBrief,
    ExecutionPlan,
    PatternBrief,
    SpreadState,
)
from fxagent.patterns import PatternHit
from fxagent.regime.consensus import Consensus
from fxagent.strategies.base import SignalDirection
from tests.regime.builders import SYMBOL, regime_at, signal

#: A Monday inside the London session, so the session fields are populated rather than empty.
MOMENT = datetime(2026, 1, 12, 9, 0, tzinfo=UTC)

BREAKOUT = "session_breakout"
REVERSION = "range_reversion"
CARRY = "carry_divergence"

#: The router's full slate. Reversion is weighted 0.0 — gated, not silent, which is a
#: distinction the narration has to keep.
WEIGHTS = {BREAKOUT: 1.0, CARRY: 0.6, REVERSION: 0.0}


def analogue(
    identifier: int = 1,
    *,
    similarity: float = 0.93,
    outcome: str = "TAKE_PROFIT",
    outcome_r: float = 2.0,
) -> AnalogueBrief:
    """One retrieved window whose outcome resolved well before `MOMENT`."""
    return AnalogueBrief(
        id=identifier,
        symbol=SYMBOL,
        timestamp=datetime(2025, 3, 4, 9, 0, tzinfo=UTC),
        similarity=similarity,
        outcome=outcome,
        outcome_r=outcome_r,
        resolved_at=datetime(2025, 3, 6, 9, 0, tzinfo=UTC),
    )


def _briefing(signals: dict[str, object], **kwargs: object) -> Briefing:
    regime = regime_at(MOMENT, trend_strength=30.0, volatility_percentile=62.0)
    result = Consensus().evaluate(regime, signals, WEIGHTS)  # type: ignore[arg-type]
    return Briefing.from_consensus(regime, result, **kwargs)  # type: ignore[arg-type]


def fired_briefing(
    *,
    analogues: tuple[AnalogueBrief, ...] = (),
    patterns: tuple[PatternBrief, ...] = (),
    indicators: dict[str, float | None] | None = None,
    execution: ExecutionPlan | None = None,
    spread: SpreadState | None = None,
    events: tuple[CalendarEventBrief, ...] = (),
) -> Briefing:
    """Two strategies agree LONG on summed weight 1.60, clearing both thresholds.

    Reversion disagrees *and* is weighted 0.0, so the briefing carries a gated dissent. A
    firing bar where every strategy happened to agree would not show whether the narration can
    report a vote that was never asked for.
    """
    return _briefing(
        {
            BREAKOUT: signal(BREAKOUT, SignalDirection.LONG, timestamp=MOMENT, confidence=0.7),
            CARRY: signal(CARRY, SignalDirection.LONG, timestamp=MOMENT, confidence=0.5),
            REVERSION: signal(REVERSION, SignalDirection.SHORT, timestamp=MOMENT),
        },
        analogues=analogues,
        patterns=patterns,
        indicators=indicators,
        execution=execution,
        spread=spread,
        events=events,
    )


def declined_briefing() -> Briefing:
    """One strategy votes, one is gated, one is silent — so nothing qualifies."""
    return _briefing(
        {
            BREAKOUT: signal(BREAKOUT, SignalDirection.LONG, timestamp=MOMENT, confidence=0.7),
            CARRY: None,
            REVERSION: signal(REVERSION, SignalDirection.SHORT, timestamp=MOMENT),
        }
    )


def pattern(name: str = "doji", *, bar_index: int = 40) -> PatternBrief:
    """One detected formation, shaped as `fxagent.patterns` produces it.

    Built through `PatternBrief.from_hit` so the projection the chartist actually sees is the
    one under test, rather than a hand-written dict that agrees with it today.
    """
    return PatternBrief.from_hit(
        PatternHit(
            name=name,
            bar_index=bar_index,
            timestamp=MOMENT,
            criteria={"body": 0.00005, "atr": 0.002, "body_in_atr": 0.025},
        )
    )


#: Named indicator readings, including one still warming up. `None` must render as unknown.
INDICATORS: dict[str, float | None] = {"adx_14": 30.0, "atr_14": 0.002, "ema_200": None}


def execution(
    *,
    risk_fraction: float = 0.004,
    total_open_risk: float = 0.008,
    volume: float = 0.04,
) -> ExecutionPlan:
    """A sized order inside both of hard rule 8's caps, unless a test asks otherwise."""
    return ExecutionPlan(
        volume=volume,
        risk_fraction=risk_fraction,
        risk_amount=40.0,
        stop_distance=0.002,
        total_open_risk=total_open_risk,
        max_risk_per_trade=0.005,
        max_total_risk=0.02,
    )


def spread(*, over_median: float = 1.0) -> SpreadState:
    """Dealing conditions at a chosen multiple of their own median."""
    return SpreadState(spread=0.0001 * over_median, median_spread=0.0001)


def event(
    *,
    minutes_until: int = 30,
    importance: str = "HIGH",
    title: str = "Non-Farm Payrolls",
    currency: str = "USD",
) -> CalendarEventBrief:
    """One scheduled release near the bar. `actual` is absent, as it is for anything upcoming."""
    return CalendarEventBrief(
        title=title,
        currency=currency,
        importance=importance,
        minutes_until=minutes_until,
        forecast=0.2,
        previous=0.3,
    )
