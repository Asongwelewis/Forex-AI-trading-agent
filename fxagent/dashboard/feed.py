"""Build the right pane: one entry per evaluation, newest first.

Pure, like `chart`. It takes the rows and returns the panel.

**Every evaluation appears, not only the ones that fired.** On a normal London session the
agent will decline dozens of times and trade once or twice, and "why did nothing happen" is the
question the panel is for. The rejections carry the same vote lines, the same regime and the
same narration as an acceptance, because `Consensus` writes its diagnostics on both paths —
this module would have to work to throw that away, and it does not.

**Nothing here can rank, filter or summarise an agent's opinion.** The narrations are passed
through as text. There is no scoring of them, no "the chartist agrees with the historian"
derived field, and no place where an agent's words become a number — which is the only
arrangement in which CLAUDE.md's rule that agents never decide anything survives contact with
a screen that puts them next to the decision.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from fxagent.dashboard.contract import read_agents, read_patterns, read_regime, read_votes
from fxagent.dashboard.models import (
    FeedEntry,
    FeedPayload,
    GrantSnapshot,
    TradeView,
)
from fxagent.store.repositories.evaluations import EvaluationRecord
from fxagent.store.repositories.trades import TradeRecord

__all__ = ["AGENT_LAYER_ABSENT", "build_feed"]

logger = logging.getLogger(__name__)

#: Shown once at the top of the feed while no evaluation carries an `agents` block. A section
#: that is empty because nothing wrote it must not look like a section that is empty because
#: the agents had nothing to say.
#:
#: `fxagent/agents/` exists now and always produces a block — it templates whatever the
#: providers cannot serve — so this note no longer means "unbuilt". It means these particular
#: rows were written by a pass that did not call it, which is what a stored evaluation from
#: before the layer landed looks like.
AGENT_LAYER_ABSENT = (
    "No evaluation in this window carries agent narration. The agent layer "
    "(fxagent/agents/) always writes a block when it runs — a deterministic template when no "
    "provider is reachable — so these rows were written by a pass that did not call it, rather "
    "than by agents that had nothing to say."
)


def _trade_views(trades: Sequence[TradeRecord]) -> tuple[TradeView, ...]:
    return tuple(
        TradeView(
            trade_id=trade.id,
            direction=trade.direction,
            volume=trade.volume,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time_utc.isoformat(),
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            exit_price=trade.exit_price,
            exit_time=trade.exit_time_utc.isoformat() if trade.exit_time_utc else None,
            barrier_touched=trade.barrier_touched,
            r_multiple=trade.r_multiple,
            mode=trade.mode,
        )
        for trade in trades
    )


def build_feed(
    symbol: str,
    evaluations: Sequence[EvaluationRecord],
    trades: Sequence[TradeRecord],
    grant: GrantSnapshot,
) -> FeedPayload:
    """Assemble the panel. `evaluations` may be in any order; the feed is sorted newest first.

    Trades are attached to the evaluation that produced them through `evaluation_id`, which is
    the foreign key the store already enforces. A trade whose evaluation is outside this window
    simply does not appear in the feed — it is still on the chart, where it has a price and a
    time to be drawn at.
    """
    by_evaluation: dict[int, list[TradeRecord]] = {}
    for trade in trades:
        by_evaluation.setdefault(trade.evaluation_id, []).append(trade)

    entries: list[FeedEntry] = []
    saw_agents = False

    for record in sorted(evaluations, key=lambda row: (row.ts_utc, row.id), reverse=True):
        agents = read_agents(record.votes)
        patterns, pattern_discards = read_patterns(record.votes)
        saw_agents = saw_agents or any(
            (agents.chartist, agents.historian, agents.risk_officer, agents.discarded)
        )

        entries.append(
            FeedEntry(
                evaluation_id=record.id,
                cycle_id=str(record.cycle_id),
                timestamp=record.ts_utc.isoformat(),
                symbol=record.symbol,
                fired=record.fired,
                reason=record.reason,
                consensus_score=record.consensus_score,
                regime=read_regime(record.regime),
                votes=read_votes(record.votes),
                chartist=agents.chartist,
                historian=agents.historian,
                analogues=agents.analogues,
                risk_officer=agents.risk_officer,
                patterns=patterns,
                trades=_trade_views(by_evaluation.get(record.id, ())),
                discarded=agents.discarded + pattern_discards,
            )
        )

    notes: list[str] = []
    if not entries:
        notes.append(
            f"No evaluations stored for {symbol} in this window. The analyst writes one row per "
            "symbol per cycle, including cycles that fired nothing, so an empty feed means the "
            "analyst has not run rather than that it had nothing to say."
        )
    elif not saw_agents:
        notes.append(AGENT_LAYER_ABSENT)

    return FeedPayload(symbol=symbol, entries=tuple(entries), grant=grant, notes=tuple(notes))
