"""Turn a replay into a report, through `fxagent.stats` and nothing else.

**No bare point estimates.** Every metric here is an `Estimate` with its interval, and the
drawdown is a bootstrap distribution rather than the single number the realised path happened to
produce. One equity curve is one draw; quoting its max drawdown as *the* max drawdown is a
sample of size one presented as a property of the strategy.

**Both equity modes, labelled, never overlaid.** Compounded and additive-R are different
quantities and neither is universally the conservative one, so the report prints both and names
each. A drawdown budget reasoned without the distinction — card 22's `MAX_ACCEPTABLE_DRAWDOWN`
— should be read against the additive figure, because the shape a ruin budget guards against is
the sustained losing run and additive is the deeper measure of that shape.

**`INSUFFICIENT_DATA` is a headline, not a footnote.** Below 100 trades the metrics are still
computed, because a wide interval is informative about how little is known, but the verdict line
says so first and the numbers are not to be ranked on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fxagent.backtest.folds import Fold, labels_overlap
from fxagent.backtest.replay import ReplayResult
from fxagent.costs import CostConfig, SpreadSource
from fxagent.stats.performance import (
    Estimate,
    expectancy_r,
    judgement,
    profit_factor,
    sharpe,
    sortino,
    win_rate,
)
from fxagent.stats.resample import Resampler, monte_carlo
from fxagent.stats.returns import EquityMode

__all__ = ["AMBIGUITY_CONCERN", "BacktestReport", "build_report"]

#: Above this share of trades resolved pessimistically, the labels are mostly inference and the
#: report says so. Not a gate — a stated concern, because the fix is to widen the barriers or
#: drop to a finer timeframe, and that is a decision for a human.
AMBIGUITY_CONCERN: Final = 0.05

_PATHS: Final = 5_000


@dataclass(frozen=True)
class BacktestReport:
    """Every metric with an interval, both equity modes, and the caveats that qualify them."""

    symbol: str
    trades: int
    verdict: str | None
    expectancy_r: Estimate
    profit_factor: Estimate
    sharpe: Estimate
    sortino: Estimate
    win_rate: Estimate
    compounded_drawdown: object
    additive_drawdown: object
    ambiguity_rate: float
    spread_note: str
    swap_note: str
    carry_note: str | None
    fold_note: str | None

    def describe(self) -> str:
        lines: list[str] = [f"=== {self.symbol}: {self.trades} trades ==="]

        if self.verdict:
            lines.append(
                f"{self.verdict} — fewer than 100 trades. The intervals below are honest about "
                "how little is known; the point estimates are not to be ranked on."
            )

        lines.append("")
        lines.append("Metric            estimate [95% interval]")
        for name, estimate in (
            ("expectancy (R)", self.expectancy_r),
            ("profit factor", self.profit_factor),
            ("sharpe/trade", self.sharpe),
            ("sortino/trade", self.sortino),
        ):
            lines.append(f"  {name:<16} {estimate.describe()}")
        lines.append(
            f"  {'win rate':<16} {self.win_rate.describe()}  "
            "(information only — gates nothing, see CLAUDE.md)"
        )

        lines.append("")
        lines.append("Max drawdown, bootstrapped — the two modes are different quantities:")
        for label, distribution in (
            ("compounded", self.compounded_drawdown),
            ("additive-R", self.additive_drawdown),
        ):
            lines.append(f"  {label:<12} {distribution.describe()}")
        lines.append(
            "  A drawdown budget reasoned without this distinction should be read against "
            "additive-R."
        )

        lines.append("")
        lines.append("Caveats:")
        lines.append(f"  spread: {self.spread_note}")
        lines.append(f"  swap:   {self.swap_note}")
        lines.append(
            f"  intrabar ambiguity: {self.ambiguity_rate:.1%} of exits inferred as STOP"
            + (
                "  <-- above 5%: stops are too tight for this timeframe"
                if self.ambiguity_rate > AMBIGUITY_CONCERN
                else ""
            )
        )
        if self.carry_note:
            lines.append(f"  {self.carry_note}")
        if self.fold_note:
            lines.append(f"  {self.fold_note}")
        return "\n".join(lines)


def _spread_note(result: ReplayResult, costs: CostConfig) -> str:
    stored = result.spread_sources.get(SpreadSource.STORED, 0)
    fixed = result.spread_sources.get(SpreadSource.FIXED, 0)
    if stored and not fixed:
        return f"all {stored} entries filled on stored bid/ask"
    if fixed and not stored:
        return (
            f"all {fixed} entries filled on a configured {costs.fixed_spread_pips:g}-pip "
            "spread — the feed carried no quotes, so this result is only as good as that guess"
        )
    return (
        f"{stored} entries on stored bid/ask, {fixed} on a configured "
        f"{costs.fixed_spread_pips:g}-pip spread — a mixed run, and the two are not equivalent"
    )


def build_report(
    result: ReplayResult,
    costs: CostConfig,
    *,
    risk_fraction: float,
    starting_equity: float = 10_000.0,
    folds: list[Fold] | None = None,
    paths: int = _PATHS,
    seed: int = 8_675_309,
) -> BacktestReport:
    """Assemble the report. Raises on an empty run rather than printing zeros.

    A backtest that produced no trades is a finding — the router gated everything, or consensus
    never agreed — and `ReplayResult.describe` is where that story is told. Manufacturing a
    report full of NaN intervals would bury it.
    """
    r_multiples = result.r_multiples
    if not r_multiples:
        raise ValueError(
            f"{result.symbol} produced no trades, so there is nothing to measure. Read the "
            "replay ledger instead — it says how many decisions were made and why each was "
            "rejected."
        )

    common = {"samples": 2_000, "seed": seed}
    drawdowns = {
        mode: monte_carlo(
            r_multiples,
            risk_fraction=risk_fraction,
            starting_equity=starting_equity,
            resampler=Resampler.BLOCK,
            mode=mode,
            paths=paths,
            seed=seed,
        ).max_drawdown
        for mode in (EquityMode.COMPOUNDED, EquityMode.ADDITIVE)
    }

    fold_note = None
    if folds:
        purged = sum(fold.purged for fold in folds)
        embargoed = sum(fold.embargoed for fold in folds)
        overlapping = labels_overlap(list(result.trades))
        fold_note = (
            f"walk-forward: {len(folds)} folds, {purged} observations purged, {embargoed} embargoed"
        )
        if purged == 0 and overlapping:
            fold_note += "  <-- ZERO PURGED on overlapping labels: purging is not working"
        elif purged == 0:
            fold_note += (
                "  (zero is correct here: one position at a time, so trade label spans are "
                "disjoint and there is nothing to purge — the embargo is the active mechanism)"
            )

    return BacktestReport(
        symbol=result.symbol,
        trades=len(r_multiples),
        verdict=judgement(len(r_multiples)),
        expectancy_r=expectancy_r(r_multiples, **common),
        profit_factor=profit_factor(r_multiples, **common),
        sharpe=sharpe(r_multiples, **common),
        sortino=sortino(r_multiples, **common),
        win_rate=win_rate(r_multiples, **common),
        compounded_drawdown=drawdowns[EquityMode.COMPOUNDED],
        additive_drawdown=drawdowns[EquityMode.ADDITIVE],
        ambiguity_rate=result.ambiguity_rate,
        spread_note=_spread_note(result, costs),
        swap_note=(
            "configured and charged"
            if costs.swap_is_configured
            else "NOT configured — no rollover cost was charged, so any multi-day result is "
            "optimistic by however much the broker actually charges"
        ),
        carry_note=(
            "carry_divergence never voted: no rate differential was supplied to the replay"
            if result.carry_is_inert
            else None
        ),
        fold_note=fold_note,
    )
