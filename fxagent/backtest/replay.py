"""Replay stored bars through the real pipeline, one bar at a time.

**Nothing here reimplements a decision.** The classifier, the three strategies, the router,
consensus and `fxagent.risk` are imported and called, not modelled. A backtest that computes its
own ADX or applies its own consensus rule measures a system that does not exist, and it will
agree with the analyst right up until one of them is changed. The only thing this module owns is
the loop, the clock, and the costs.

**Point-in-time by construction.** Each iteration builds a `BarSeries` ending at the current bar
and hands that to the pipeline. There is no filtering of a full-history frame and no "drop the
future" step that could be forgotten — the future is not in the object. The point-in-time test
asserts it by replaying a truncated series and getting the same decisions back.

**The history window matches what the analyst will fetch, deliberately.** Wilder smoothing has
infinite memory, so an indicator seeded on 250 bars differs slightly from one seeded on 5,000.
The temptation is to give the backtest all available history and call it more accurate. It is
not: the analyst will fetch a bounded window from Supabase on every run, and a backtest that
sees more history than the live system is measuring a strategy nobody will run.
`ReplayConfig.history_bars` is that shared number.

**One position at a time.** A signal arriving while a position is open is recorded as skipped
rather than pyramided. Stacking positions on one symbol would breach the 2% total-open-risk cap
by construction and would make every R multiple depend on how many other trades happened to be
running, which is not something a strategy's edge should be judged on.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from fxagent.adapters.base import BarSeries, OrderSide
from fxagent.backtest.barriers import Barrier, BarrierOutcome, resolve_barriers
from fxagent.costs import CostConfig, Quote, SpreadSource, fill, swap_cost
from fxagent.regime.classifier import RegimeClassifier
from fxagent.regime.consensus import Consensus
from fxagent.regime.router import RegimeRouter
from fxagent.risk.sizing import RiskConfig, position_size
from fxagent.risk.symbols import SymbolSpec
from fxagent.stats.returns import r_multiple
from fxagent.strategies.base import MarketContext, SignalDirection, Strategy, order_side_for
from fxagent.strategies.carry_divergence import TIMEFRAME as CARRY_TIMEFRAME
from fxagent.strategies.carry_divergence import CarryDivergence
from fxagent.strategies.range_reversion import RangeReversion
from fxagent.strategies.session_breakout import TIMEFRAME as BREAKOUT_TIMEFRAME
from fxagent.strategies.session_breakout import SessionBreakout

__all__ = [
    "DEFAULT_SOURCE",
    "STRATEGY_TIMEFRAMES",
    "ReplayConfig",
    "ReplayResult",
    "ReplayTrade",
    "default_strategies",
    "replay",
]

logger = logging.getLogger(__name__)

#: The MT5/Exness feed. Named because a backtest on one source and a live run on another are
#: different experiments — Exness quotes its own book, and its H1 bars are not TwelveData's.
DEFAULT_SOURCE: Final = "mt5_exness"


#: Which timeframe each strategy reads, taken from the strategy's own constant rather than
#: restated here. `None` means it does not care. `range_reversion` declares nothing and works on
#: any intraday series; the other two raise on the wrong one, which is the behaviour this table
#: exists to respect rather than to trip over.
STRATEGY_TIMEFRAMES: Final[dict[str, str | None]] = {
    "session_breakout": BREAKOUT_TIMEFRAME,
    "range_reversion": None,
    "carry_divergence": CARRY_TIMEFRAME,
}


def default_strategies(timeframe: str | None = None) -> dict[str, Strategy]:
    """The strategies that read `timeframe`, constructed as the analyst will construct them.

    **Filtered, not caught.** `carry_divergence` raises on anything but D1 and
    `session_breakout` raises on anything but H1 — deliberately, because a carry view judged on
    an intraday series is a different strategy wearing the same name. So a replay asks only the
    strategies whose timeframe it is running, and `ReplayResult.excluded_strategies` names the
    ones it did not ask. Wrapping the calls in `except ValueError` instead would turn a loud
    contract violation into a strategy that silently never votes, and a two-of-three consensus
    quietly running on one strategy is the failure this is here to make visible.

    Passing `None` returns all three, which is only useful for asserting what the full set is.

    A function rather than a module constant: strategies are cheap to build, and a shared mutable
    instance across two backtest runs is the kind of state that makes a sweep irreproducible.
    """
    built: dict[str, Strategy] = {}
    for strategy in (SessionBreakout(), RangeReversion(), CarryDivergence()):
        declared = STRATEGY_TIMEFRAMES.get(strategy.name)
        if timeframe is None or declared is None or declared == timeframe:
            built[strategy.name] = strategy
    return built


@dataclass(frozen=True)
class ReplayConfig:
    """Everything a run needs that is not the bars themselves."""

    spec: SymbolSpec
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    #: Bars of history handed to the pipeline at each step. Must match the analyst's fetch.
    history_bars: int = 300
    #: The time barrier, in bars. 24 on H1 is a day — past which an intraday setup is not what
    #: it was, whatever the price is doing.
    max_bars_held: int = 24
    #: Injected rather than fetched. `MarketContext.neutral()` means no rate differential, which
    #: makes `carry_divergence` structurally silent — reported, not hidden, in `ReplayResult`.
    context: MarketContext = field(default_factory=MarketContext.neutral)

    def __post_init__(self) -> None:
        if self.history_bars < 1:
            raise ValueError(f"history_bars must be positive, got {self.history_bars}")
        if self.max_bars_held < 1:
            raise ValueError(f"max_bars_held must be positive, got {self.max_bars_held}")


@dataclass(frozen=True)
class ReplayTrade:
    """One completed round trip, with every number that produced it.

    `gross_r` and `r_multiple` are both carried. The first is the trade the strategy asked for;
    the second is what the account got after spread, slippage and swap. Their difference is the
    cost of trading, and a report that showed only one of them could not tell you whether a
    strategy died of a bad edge or of a wide spread.
    """

    symbol: str
    direction: SignalDirection
    primary_strategy: str
    strategies: tuple[str, ...]
    confidence: float
    timestamp: datetime
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    stop_price: float
    target_price: float
    volume: float
    barrier: Barrier
    ambiguous_resolution: bool
    spread_source: SpreadSource
    entry_cost: float
    exit_cost: float
    swap: float
    pnl: float
    risk_amount: float
    gross_r: float
    r_multiple: float
    bars_held: int
    label_span_start: datetime
    label_span_end: datetime

    @property
    def cost_r(self) -> float:
        """What trading it cost, in R. `gross_r` minus this is `r_multiple`."""
        return self.gross_r - self.r_multiple


@dataclass(frozen=True)
class ReplayResult:
    """The trades, and an account of every bar that did not produce one.

    The rejections are as much the product as the trades. "42 trades" says nothing about whether
    the router was gating everything, whether consensus never agreed, or whether every setup was
    too small to size — and those three call for completely different fixes.
    """

    symbol: str
    trades: tuple[ReplayTrade, ...]
    bars_replayed: int
    decisions: int
    fired: int
    skipped_in_position: int
    not_sizeable: int
    spread_sources: Counter[str]
    excluded_strategies: tuple[str, ...]
    carry_is_inert: bool
    first_bar: datetime | None
    last_bar: datetime | None

    @property
    def ambiguous(self) -> int:
        return sum(1 for trade in self.trades if trade.ambiguous_resolution)

    @property
    def ambiguity_rate(self) -> float:
        """Share of trades whose exit was inferred pessimistically rather than observed."""
        return self.ambiguous / len(self.trades) if self.trades else 0.0

    @property
    def r_multiples(self) -> tuple[float, ...]:
        return tuple(trade.r_multiple for trade in self.trades)

    def describe(self) -> str:
        stored = self.spread_sources.get(SpreadSource.STORED, 0)
        fixed = self.spread_sources.get(SpreadSource.FIXED, 0)
        lines = [
            f"{self.symbol}: {self.bars_replayed:,} bars replayed, {self.decisions:,} decisions",
            f"  {self.fired} fired, {self.skipped_in_position} skipped in position, "
            f"{self.not_sizeable} not sizeable",
            f"  spread: {stored} stored, {fixed} fixed",
            f"  intrabar ambiguity: {self.ambiguous}/{len(self.trades)} "
            f"({self.ambiguity_rate:.1%}), all resolved as STOP",
        ]
        if self.excluded_strategies:
            lines.append(
                f"  not asked on this timeframe: {', '.join(self.excluded_strategies)} — "
                "consensus ran on the rest"
            )
        if self.carry_is_inert:
            lines.append(
                "  carry_divergence is inert: no rate differential was supplied, so it never "
                "voted. Two-of-three consensus ran on two strategies."
            )
        return "\n".join(lines)


def _window(bars: BarSeries, end_index: int, size: int) -> BarSeries:
    """Bars up to and including `end_index`, at most `size` of them. Never reaches past."""
    start = max(0, end_index + 1 - size)
    return BarSeries(
        symbol=bars.symbol,
        timeframe=bars.timeframe,
        bars=bars.bars[start : end_index + 1],
    )


def replay(
    bars: BarSeries,
    config: ReplayConfig,
    *,
    quotes: dict[datetime, tuple[float | None, float | None]] | None = None,
    strategies: dict[str, Strategy] | None = None,
    classifier: RegimeClassifier | None = None,
    router: RegimeRouter | None = None,
    consensus: Consensus | None = None,
) -> ReplayResult:
    """Walk the series bar by bar, running the real pipeline at each step.

    `quotes` maps a bar timestamp to its stored `(bid, ask)`. Where a timestamp is absent, or
    its quote is half-populated, the fill falls back to the configured fixed spread and is
    marked as such — see `fxagent.costs`.

    The four collaborators default to the production objects and exist as arguments so a test
    can drive the loop without first constructing bars that satisfy every strategy's gate — a
    test that fails because a gate moved has said nothing about the loop. They are injection
    points for *configuration*, not for reimplementation: whatever is passed still has to be a
    real `Strategy`, `RegimeRouter` and `Consensus`, and
    `tests/backtest/test_replay.py::test_the_defaults_are_the_production_pipeline` asserts the
    defaults are the three strategies the analyst will run.
    """
    engine_strategies = default_strategies(bars.timeframe) if strategies is None else strategies
    excluded = tuple(name for name in default_strategies() if name not in engine_strategies)
    classifier = classifier or RegimeClassifier()
    router = router or RegimeRouter()
    consensus = consensus or Consensus()
    quote_map = quotes or {}

    warmup = max([classifier.required_bars, *(s.required_bars for s in engine_strategies.values())])

    trades: list[ReplayTrade] = []
    spread_sources: Counter[str] = Counter()
    decisions = fired = skipped = not_sizeable = 0
    carry_voted = False
    open_until_index = -1

    for index in range(warmup, len(bars)):
        bar = bars.bars[index]

        if index <= open_until_index:
            # A position is running. Still count the bar so the rejection ledger adds up.
            window = _window(bars, index, config.history_bars)
            regime = classifier.classify(window)
            weights = router.weights(regime)
            signals = {
                name: strategy.generate(window, config.context, regime)
                for name, strategy in engine_strategies.items()
            }
            decisions += 1
            if consensus.evaluate(regime, signals, weights).fired:
                skipped += 1
            continue

        window = _window(bars, index, config.history_bars)
        regime = classifier.classify(window)
        weights = router.weights(regime)
        signals = {
            name: strategy.generate(window, config.context, regime)
            for name, strategy in engine_strategies.items()
        }
        carry_voted = carry_voted or signals.get("carry_divergence") is not None
        decisions += 1

        outcome = consensus.evaluate(regime, signals, weights)
        if not outcome.fired or outcome.signal is None:
            continue

        agreed = outcome.signal
        primary = agreed.primary
        if primary.stop_loss is None or primary.take_profit is None:
            # `Signal` validation forbids this for a directional signal; belt and braces.
            continue

        side = order_side_for(agreed.direction)
        bid, ask = quote_map.get(bar.timestamp, (None, None))
        entry_fill = fill(bar.close, side, config.spec, config.costs, Quote(bid=bid, ask=ask))

        size = position_size(
            config.risk.reference_equity,
            config.risk.risk_fraction,
            entry_fill.price,
            primary.stop_loss,
            config.spec,
            account_currency=config.risk.account_currency,
        )
        if size is None:
            not_sizeable += 1
            continue

        resolved = resolve_barriers(
            bars,
            index,
            side=side,
            stop_price=primary.stop_loss,
            target_price=primary.take_profit,
            max_bars=config.max_bars_held,
        )

        trades.append(
            _settle(agreed, primary, entry_fill, size, resolved, bar, config, quote_map, side)
        )
        spread_sources[entry_fill.spread_source] += 1
        fired += 1
        open_until_index = resolved.exit_index

    return ReplayResult(
        symbol=bars.symbol,
        trades=tuple(trades),
        bars_replayed=max(0, len(bars) - warmup),
        decisions=decisions,
        fired=fired,
        skipped_in_position=skipped,
        not_sizeable=not_sizeable,
        spread_sources=spread_sources,
        excluded_strategies=excluded,
        carry_is_inert="carry_divergence" in engine_strategies and not carry_voted,
        first_bar=bars.bars[0].timestamp if len(bars) else None,
        last_bar=bars.bars[-1].timestamp if len(bars) else None,
    )


def _settle(
    agreed,  # ConsensusSignal
    primary,  # Signal
    entry_fill,  # Fill
    size,  # PositionSize
    resolved: BarrierOutcome,
    entry_bar,  # Bar
    config: ReplayConfig,
    quote_map: dict[datetime, tuple[float | None, float | None]],
    side: OrderSide,
) -> ReplayTrade:
    """Turn a resolved barrier into a costed trade. Exit pays the spread again, on the way out."""
    closing_side = OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY
    exit_bid, exit_ask = quote_map.get(resolved.exit_time, (None, None))
    exit_fill = fill(
        resolved.exit_price,
        closing_side,
        config.spec,
        config.costs,
        Quote(bid=exit_bid, ask=exit_ask),
    )

    move = exit_fill.price - entry_fill.price
    if side is OrderSide.SELL:
        move = -move

    swap = swap_cost(side, size.volume, entry_bar.timestamp, resolved.exit_time, config.costs)
    pnl = move * size.volume * config.spec.contract_size * size.quote_to_account_rate + swap

    return ReplayTrade(
        symbol=agreed.symbol,
        direction=agreed.direction,
        primary_strategy=primary.strategy_name,
        strategies=agreed.strategy_names,
        confidence=agreed.confidence,
        timestamp=agreed.timestamp,
        entry_time=entry_bar.timestamp,
        entry_price=entry_fill.price,
        exit_time=resolved.exit_time,
        exit_price=exit_fill.price,
        stop_price=primary.stop_loss or 0.0,
        target_price=primary.take_profit or 0.0,
        volume=size.volume,
        barrier=resolved.touched,
        ambiguous_resolution=resolved.ambiguous_resolution,
        spread_source=entry_fill.spread_source,
        entry_cost=entry_fill.total_cost,
        exit_cost=exit_fill.total_cost,
        swap=swap,
        pnl=pnl,
        risk_amount=size.risk_amount,
        gross_r=r_multiple(primary.entry_price, resolved.exit_price, primary.stop_loss or 0.0),
        r_multiple=pnl / size.risk_amount,
        bars_held=resolved.bars_held,
        label_span_start=resolved.label_span_start,
        label_span_end=resolved.label_span_end,
    )
