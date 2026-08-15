"""How an `evaluations` row is read into the panel — the one place that knows the JSONB layout.

The store keeps two JSONB documents per evaluation: `regime`, the classifier's measurement, and
`votes`, the diagnostics dict `Consensus.evaluate` returns. Both are written by the analyst and
read here. Nothing else in the dashboard touches raw JSON, so the layout is one file's problem.

**Three of the panel's sections have no producer yet.** The agent narrations, the retrieved
analogues and the candle formations are read from reserved keys inside the `votes` document that
nothing currently writes. This file therefore does two jobs: it reads what exists today, and it
states the contract the agent layer must satisfy to light the rest of the panel up. The reserved
layout is::

    votes = {
        ...                                   # the diagnostics Consensus.evaluate produced
        "agents": {
            "chartist":     {"text": str, "provider": str?, "model": str?, "generated_at": str?},
            "historian":    {"text": str, ..., "analogues": [
                                {"timestamp": str, "symbol": str, "similarity": float,
                                 "outcome": str?, "outcome_r": float?, "resolved_at": str?}]},
            "risk_officer": {"text": str, ..., "proceed_recommendation": str?},
        },
        "patterns": [{"name": str, "definition": str, "bar_time": str?}],
    }

Until those keys appear the panel says so in as many words. An empty section that looks the same
as a quiet one is how a broken agent goes unnoticed for a week.

**A block that fails validation is discarded whole.** Not repaired, not partially rendered — the
block is dropped, its name is added to `discarded`, and the panel prints that the narration was
discarded. This mirrors hard rule 5 on the read side: the rule exists because a half-parsed LLM
response is more dangerous than an absent one, and that is no less true when the half-parsing
happens in a renderer than when it happens in a validator.

**Extra keys are ignored rather than rejected.** The analyst will grow fields the dashboard has
not been taught yet, and a reader that discarded an entire narration because it gained a
`token_count` would turn every forward-compatible change into an outage. Missing *required*
fields still discard, because those are the ones the panel would have to invent.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from fxagent.dashboard.models import (
    AgentNarration,
    Analogue,
    PatternNote,
    RegimeView,
    RiskOfficerNote,
    VoteView,
)

__all__ = [
    "AGENTS_KEY",
    "AgentBlocks",
    "PATTERNS_KEY",
    "VOTES_KEY",
    "read_agents",
    "read_patterns",
    "read_regime",
    "read_votes",
]

logger = logging.getLogger(__name__)

#: Reserved keys inside the `votes` JSONB document. Named constants because the analyst will
#: eventually import them to write the same keys this reads.
AGENTS_KEY = "agents"
PATTERNS_KEY = "patterns"
#: The per-strategy vote list, inside the diagnostics document under its own key.
VOTES_KEY = "votes"

#: Agents named in CLAUDE.md, in the order the panel shows them.
CHARTIST = "chartist"
HISTORIAN = "historian"
RISK_OFFICER = "risk_officer"


class _Incoming(BaseModel):
    """Tolerant of new fields, strict about the ones the panel would otherwise invent."""

    model_config = ConfigDict(extra="ignore")


class _NarrationIn(_Incoming):
    text: str
    provider: str | None = None
    model: str | None = None
    generated_at: str | None = None


class _AnalogueIn(_Incoming):
    timestamp: str
    symbol: str
    similarity: float
    outcome: str | None = None
    outcome_r: float | None = None
    resolved_at: str | None = None


class _HistorianIn(_NarrationIn):
    analogues: tuple[_AnalogueIn, ...] = ()


class _RiskOfficerIn(_NarrationIn):
    proceed_recommendation: str | None = None


class _PatternIn(_Incoming):
    name: str
    definition: str
    bar_time: str | None = None


class AgentBlocks:
    """The three narrations, the analogues, and the names of anything discarded.

    A plain object rather than a tuple because five return values in a fixed order is a
    call site nobody can read six months later.
    """

    __slots__ = ("analogues", "chartist", "discarded", "historian", "risk_officer")

    def __init__(
        self,
        *,
        chartist: AgentNarration | None = None,
        historian: AgentNarration | None = None,
        analogues: tuple[Analogue, ...] = (),
        risk_officer: RiskOfficerNote | None = None,
        discarded: tuple[str, ...] = (),
    ) -> None:
        self.chartist = chartist
        self.historian = historian
        self.analogues = analogues
        self.risk_officer = risk_officer
        self.discarded = discarded


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def read_regime(regime: Any) -> RegimeView:
    """A `RegimeView` from the stored `regime` document, tolerating an older or partial one.

    Every field defaults, because a regime row written before a field existed is ordinary
    history and not a reason to refuse to draw the rest of the panel. `session` is recomputed
    from `sessions` when absent rather than left blank: it is a derived label, and deriving it
    here keeps a stored row that predates the computed field readable.
    """
    document = _as_mapping(regime)
    sessions = tuple(str(name) for name in document.get("sessions", ()) if name)

    session = document.get("session")
    if session is None and sessions:
        # Same precedence as `dominant_session`, applied to the stored strings.
        for candidate in ("OVERLAP", "LONDON", "NEW_YORK", "TOKYO"):
            if candidate in sessions:
                session = candidate
                break

    minutes = document.get("minutes_until_weekly_close")
    return RegimeView(
        session=str(session) if session is not None else None,
        sessions=sessions,
        market_open=bool(document.get("market_open", False)),
        minutes_until_weekly_close=int(minutes) if isinstance(minutes, (int, float)) else None,
        trend_strength=_as_float(document.get("trend_strength")),
        volatility_percentile=_as_float(document.get("volatility_percentile")),
        is_trending=bool(document.get("is_trending", False)),
        is_ranging=bool(document.get("is_ranging", False)),
    )


def read_votes(votes: Any) -> tuple[VoteView, ...]:
    """The per-strategy vote list, in router-key order as `Consensus` wrote it.

    Accepts either the whole diagnostics document or the bare list inside it. The bare form is
    not speculative generality — it is what a caller holding `diagnostics["votes"]` already has,
    and demanding it be re-wrapped would produce exactly one bug, once, silently.
    """
    rows = votes.get(VOTES_KEY) if isinstance(votes, Mapping) else votes
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()

    parsed: list[VoteView] = []
    for row in rows:
        entry = _as_mapping(row)
        name = entry.get("strategy")
        if not name:
            continue  # a vote with no strategy names nothing; there is no panel row to draw
        parsed.append(
            VoteView(
                strategy=str(name),
                weight=_as_float(entry.get("weight")) or 0.0,
                direction=_as_str(entry.get("direction")),
                confidence=_as_float(entry.get("confidence")),
                participated=bool(entry.get("participated", False)),
                reason=str(entry.get("reason", "")),
            )
        )
    return tuple(parsed)


def read_agents(votes: Any) -> AgentBlocks:
    """The three narrations from the reserved `agents` key. Absent is not an error."""
    agents = _as_mapping(_as_mapping(votes).get(AGENTS_KEY))
    if not agents:
        return AgentBlocks()

    discarded: list[str] = []

    chartist = _parse(agents.get(CHARTIST), _NarrationIn, CHARTIST, discarded)
    historian = _parse(agents.get(HISTORIAN), _HistorianIn, HISTORIAN, discarded)
    risk = _parse(agents.get(RISK_OFFICER), _RiskOfficerIn, RISK_OFFICER, discarded)

    return AgentBlocks(
        chartist=_narration(CHARTIST, chartist),
        historian=_narration(HISTORIAN, historian),
        analogues=(
            tuple(Analogue(**item.model_dump()) for item in historian.analogues)
            if historian is not None
            else ()
        ),
        risk_officer=(
            RiskOfficerNote(
                text=risk.text,
                proceed_recommendation=risk.proceed_recommendation,
                provider=risk.provider,
                model=risk.model,
                generated_at=risk.generated_at,
            )
            if risk is not None
            else None
        ),
        discarded=tuple(discarded),
    )


def read_patterns(votes: Any) -> tuple[tuple[PatternNote, ...], tuple[str, ...]]:
    """Detected candle formations and the names of any that failed validation.

    Each note is stamped with its CONTEXT ONLY label by `PatternNote`'s default, not by the
    template, so a formation cannot reach the screen without it.
    """
    raw = _as_mapping(votes).get(PATTERNS_KEY)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return (), ()

    notes: list[PatternNote] = []
    discarded: list[str] = []
    for index, item in enumerate(raw):
        parsed = _parse(item, _PatternIn, f"{PATTERNS_KEY}[{index}]", discarded)
        if parsed is not None:
            notes.append(
                PatternNote(
                    name=parsed.name, definition=parsed.definition, bar_time=parsed.bar_time
                )
            )
    return tuple(notes), tuple(discarded)


def _parse[T: BaseModel](raw: Any, model: type[T], name: str, discarded: list[str]) -> T | None:
    """Validate one block, or discard it whole and record that it was discarded."""
    if raw is None:
        return None
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        # The content is deliberately not logged: a narration can quote account state, and a
        # log line is the one place it would outlive the request.
        logger.warning("discarded %s block: %s", name, error.error_count())
        discarded.append(name)
        return None


def _narration(agent: str, parsed: _NarrationIn | None) -> AgentNarration | None:
    if parsed is None:
        return None
    return AgentNarration(
        agent=agent,
        text=parsed.text,
        provider=parsed.provider,
        model=parsed.model,
        generated_at=parsed.generated_at,
    )


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_str(value: Any) -> str | None:
    return str(value) if value is not None else None
