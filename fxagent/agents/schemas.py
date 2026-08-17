"""The structured input the agents are given, and the validated shapes they may return.

Two directions, one file, because they are one contract: an agent may only say things about
the document it was handed, and the check that enforces that needs both halves in scope.

**`Briefing` is the whole of what an agent sees.** No bars, no raw price series, no account
state — the regime measurement, the vote lines the router and consensus produced, and the plan
levels the deterministic core chose. `payload()` renders it, and `grounding()` derives the
permitted number set *from that same rendering*, so the two cannot drift: whatever an agent was
not shown, it may not quote.

**Any response failing validation is discarded entirely** (CLAUDE.md hard rule 5). There is no
partial parsing, no regex repair and no retry-until-it-parses. `validate_note` returns `None`
and the caller falls back to `fxagent.agents.templates`, which always works. That fallback is
what makes strictness affordable: a discarded narration costs a paragraph of prose, never a
decision, so the rule can be enforced rather than negotiated with.

**"An agent must never output a number that was not in its input" is checked, not requested.**
`Grounding.ungrounded` extracts every numeric literal from the response — from the prose and
from any structured field alike, by re-reading the response's own JSON — and refuses any that
does not match a number in the payload at the precision it was written to. An agent that
invents a level, an ADX reading or an R multiple is discarded whole. Instructing the model not
to do it is not a control; a model that ignores the instruction produces a number that looks
exactly like one it was given.

The check is deliberately strict, and the consequence of strictness is a fallback rather than a
failure. Where a legitimate count would otherwise be refused — "all 3 strategies agreed" — the
count is *added to the payload* rather than the check being loosened. Putting the number in the
input is the fix; widening what counts as grounded is a hole with a rationale attached.

**Dates are omissible.** `include_dates=False` drops every timestamp from the payload, for
backtest-mode prompts where the current date is look-ahead (CLAUDE.md, known traps). Because
grounding is computed from the payload actually rendered, dropping the dates also removes their
digits from the permitted set, so a backtest-mode narration cannot quote a year either.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator

from fxagent.adapters.base import UtcDatetime
from fxagent.patterns import CONTEXT_ONLY, PatternHit
from fxagent.regime.classifier import Regime
from fxagent.regime.consensus import ConsensusResult
from fxagent.strategies.base import SignalDirection

__all__ = [
    "GROUNDING_KEY",
    "MAX_NOTE_CHARS",
    "MAX_PLAN_SUMMARY_CHARS",
    "MAX_READ_CHARS",
    "AgentEcho",
    "AgentNote",
    "AnalogueBrief",
    "Briefing",
    "CalendarEventBrief",
    "ChartistNote",
    "ExecutionPlan",
    "Grounding",
    "HistorianNote",
    "PatternBrief",
    "PatternSeen",
    "RiskFlag",
    "RiskOfficerNote",
    "SpreadState",
    "TermExplained",
    "TradePlan",
    "VoteLine",
    "validate_note",
]

logger = logging.getLogger(__name__)

#: Key under which `Grounding` travels in Pydantic's validation context. A note validated
#: without it raises rather than passing, so there is no path on which the number check is
#: silently skipped — forgetting to pass the context is a loud error, not a quiet exemption.
GROUNDING_KEY = "grounding"

#: Upper bound on a narration. The panel gives each agent a card, not a page, and an
#: unbounded text field is an unbounded token bill on a free tier.
MAX_NOTE_CHARS = 700

#: The risk officer's `plan_summary`. Tighter again: it sits directly above the levels it is
#: summarising, so anything longer is the reader being asked to check prose against numbers
#: already on the screen.
MAX_PLAN_SUMMARY_CHARS = 300

#: The chartist's `read` is tighter still. Its card sits beside the regime line and the vote
#: list, both of which the reader can already see — so a reading that needs more than this is
#: restating the input rather than reading it.
MAX_READ_CHARS = 400

#: Decimal places every float in the payload is rounded to before rendering. Enough for any FX
#: price this system quotes, and it keeps 1.0999999999999999 out of a prompt — which matters
#: beyond tidiness, because the payload is also the grounding set and an agent quoting the
#: clean value of a noisy float would be discarded for saying the right thing.
_PAYLOAD_DECIMALS = 6

#: One numeric literal. The lookbehind keeps the `-` in `2026-08-16` from being read as a sign,
#: which would ground -8 and -16 instead of 8 and 16; a leading sign is only a sign when what
#: precedes it is not itself part of a number.
_NUMBER = re.compile(r"(?<![\d.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), _PAYLOAD_DECIMALS)


def _literals(text: str) -> list[tuple[float, int]]:
    """Every numeric literal in `text`, as (value, decimal places as written).

    The decimal count is carried because it is what makes rounding legitimate: an agent shown
    27.34567 and writing "27.3" has quoted its input, and an agent writing "27.9" has not.
    """
    found: list[tuple[float, int]] = []
    for match in _NUMBER.finditer(text):
        raw = match.group().replace(",", "")
        _, _, fraction = raw.partition(".")
        try:
            found.append((float(raw), len(fraction)))
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
    return found


@dataclass(frozen=True)
class Grounding:
    """The numbers an agent is permitted to use, derived from the payload it was shown."""

    values: frozenset[float]
    #: Retrieved analogue identifiers. Checked by identity rather than by rounding, because
    #: "which window did you mean" has no nearest match.
    analogue_ids: frozenset[int] = frozenset()
    #: Formations the detector actually found on this bar. Names carry no digits, so the
    #: numeric check cannot police them and this set has to exist separately.
    pattern_names: frozenset[str] = frozenset()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Grounding:
        rendered = json.dumps(payload, sort_keys=True, default=str)
        analogues = payload.get("analogues") or ()
        patterns = payload.get("patterns") or ()
        return cls(
            values=frozenset(value for value, _ in _literals(rendered)),
            analogue_ids=frozenset(
                int(item["id"]) for item in analogues if isinstance(item, dict) and "id" in item
            ),
            pattern_names=frozenset(
                str(item["name"]) for item in patterns if isinstance(item, dict) and "name" in item
            ),
        )

    def permits(self, value: float, decimals: int) -> bool:
        """True if `value`, written to `decimals` places, is one of the payload's numbers."""
        return any(round(known, decimals) == value for known in self.values)

    def ungrounded(self, text: str) -> tuple[str, ...]:
        """Every literal in `text` that was not in the payload, as written."""
        return tuple(
            _format(value, decimals)
            for value, decimals in _literals(text)
            if not self.permits(value, decimals)
        )


def _format(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}" if decimals else f"{int(value)}"


# -- the input ----------------------------------------------------------------------------


class _Document(BaseModel):
    """Frozen and closed. A briefing assembled with a typo in a field name is a wiring bug."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class VoteLine(_Document):
    """One strategy's vote, as `Consensus` recorded it — including why it did not count.

    `reason` is carried through verbatim rather than re-worded here. It is the sentence the
    deterministic core wrote, and an agent asked to explain a decision should be reading the
    core's own words rather than a paraphrase assembled for its benefit.
    """

    strategy: str = Field(min_length=1)
    weight: float = Field(ge=0.0)
    direction: str | None = None
    confidence: float | None = None
    participated: bool = False
    reason: str = ""


class TradePlan(_Document):
    """The levels the deterministic core chose. Nothing here is negotiable by an agent."""

    direction: str
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    reward_risk: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    #: Whose levels these are. Consensus takes one contributor's plan whole rather than
    #: averaging, so this names the strategy that supplied them.
    primary_strategy: str = Field(min_length=1)


class AnalogueBrief(_Document):
    """One retrieved historical window and how it actually resolved.

    `resolved_at` travels because it is the point-in-time claim itself: a window whose outcome
    landed after the bar under analysis must never have been retrieved, and carrying the
    resolution time means the assertion is visible in the prompt, the panel and the journal
    rather than only in the SQL that enforced it.
    """

    id: int
    symbol: str = Field(min_length=1)
    timestamp: UtcDatetime
    similarity: float
    outcome: str | None = None
    outcome_r: float | None = None
    resolved_at: UtcDatetime | None = None


class PatternBrief(_Document):
    """One detected formation, as the chartist is shown it.

    A thin projection of `fxagent.patterns.PatternHit` rather than the hit itself, so the
    agent layer's prompt shape is not pinned to the detector's internals — and so `label`
    travels into the prompt, where the model reads the CONTEXT stamp before it writes about
    the formation rather than after.
    """

    name: str = Field(min_length=1)
    definition: str = ""
    bar_index: int = Field(ge=0)
    timestamp: UtcDatetime | None = None
    criteria: dict[str, float] = Field(default_factory=dict)
    label: str = CONTEXT_ONLY

    @classmethod
    def from_hit(cls, hit: PatternHit) -> PatternBrief:
        return cls(
            name=hit.name,
            definition=hit.definition,
            bar_index=hit.bar_index,
            timestamp=hit.timestamp,
            criteria=dict(hit.criteria),
            label=hit.label,
        )


#: The three things the risk officer may recommend. **Advisory, all three of them.**
PROCEED_RECOMMENDATIONS = ("PROCEED", "CAUTION", "WAIT")

#: How loud a flag is. Ordered, so a caller can sort by it — and deliberately *only* orderable,
#: with no numeric weight, because a severity that could be summed is a severity that could
#: become a score.
SEVERITIES = ("INFO", "WARN", "CRITICAL")


class RiskFlag(_Document):
    """One condition worth a human's attention, and how loud it is.

    Prose plus a severity label. There is no score, no weight and no threshold on this class:
    the deterministic layer already raised the conditions that matter and sized against them,
    and a flag that carried a number something could total up would be a second risk model
    sitting beside the real one.
    """

    flag: str = Field(min_length=1, max_length=200)
    severity: str = Field(default="INFO")

    @model_validator(mode="after")
    def _severity_is_one_of_three(self) -> Self:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")
        return self


class ExecutionPlan(_Document):
    """The sized order the deterministic core produced, and the caps it was sized against.

    Every number here was computed by `fxagent.risk` before any agent existed. The caps travel
    beside the values deliberately: "0.4% risked" means nothing on its own, and an agent given
    the number without the limit has to be told the limit in prose — which is a number in a
    prompt that nothing checks.

    A plan **over** a cap is carried rather than rejected. That is a fact about the sizing, and
    the risk officer's job is to say so loudly; a briefing that refused to be built would hide
    the one condition most worth surfacing behind a stack trace.
    """

    #: Lots, already rounded down to the broker's volume step.
    volume: float = Field(gt=0)
    #: Fraction of equity at risk on this trade. Hard rule 8 caps it at 0.005.
    risk_fraction: float = Field(ge=0.0, le=1.0)
    #: The same figure in account currency, because that is what a human checks it against.
    risk_amount: float = Field(ge=0.0)
    #: Entry-to-stop distance in price terms — the denominator the size came from.
    stop_distance: float = Field(gt=0)
    #: Fraction of equity at risk across every open position, this one included.
    total_open_risk: float = Field(ge=0.0, le=1.0)
    max_risk_per_trade: float = Field(gt=0.0, le=1.0)
    max_total_risk: float = Field(gt=0.0, le=1.0)

    @property
    def over_trade_cap(self) -> bool:
        return self.risk_fraction > self.max_risk_per_trade

    @property
    def over_total_cap(self) -> bool:
        return self.total_open_risk > self.max_total_risk

    @property
    def total_risk_used(self) -> float:
        """Open risk as a fraction of the total cap. 1.0 means the book is full."""
        return self.total_open_risk / self.max_total_risk


class SpreadState(_Document):
    """The spread now against its own recent median.

    A ratio rather than a pip count, for the same reason the pattern thresholds are ATR
    multiples: two pips is nothing on EUR/USD and a blowout on EUR/GBP, and a fixed number
    would mean different things on different rows of the same panel. Exness is a market-maker
    CFD broker and its spreads widen on news, so this is the input that most often makes an
    otherwise-fine plan a bad one.
    """

    spread: float = Field(ge=0.0)
    median_spread: float = Field(gt=0.0)

    @property
    def over_median(self) -> float:
        return self.spread / self.median_spread


class CalendarEventBrief(_Document):
    """One scheduled release near this bar.

    `actual` is carried and is **not** a look-ahead: `events_visible_at()` nulls it until
    `event_time_utc` has passed (migration 0009), so a non-null value here is a number that was
    genuinely public. Hard rule 6 wants that enforced in SQL rather than in a prompt, and it is
    — this model repeats no part of the gate, which is exactly why it cannot disagree with it.

    `minutes_until` is negative for a release that has already happened. Kept signed rather than
    split into two fields because the spread damage from a release outlasts the release, and
    "eight minutes ago" is as much a reason for caution as "in eight minutes".
    """

    title: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    importance: str = Field(min_length=1)
    minutes_until: int
    forecast: float | None = None
    previous: float | None = None
    actual: float | None = None


class AgentEcho(_Document):
    """What another agent said, as the agent reading it is shown it.

    Text and attribution only. No agent is shown another's structured fields, so nothing an
    agent scores can be read back in and turned into agreement — and there is no path by which
    two narrations become a number.
    """

    agent: str = Field(min_length=1)
    text: str
    provider: str = ""


class Briefing(_Document):
    """Everything an agent is shown about one evaluation of one symbol.

    Built by `from_consensus` from the two objects the deterministic core already produced, so
    there is no third description of a decision to keep in step with the other two.

    `indicators` and `patterns` are passed in rather than computed here. The briefing is a
    projection, not a pipeline stage: a class that reached for bars and ran a detector would be
    a second place market state is measured, and the point of everything upstream is that there
    is exactly one.
    """

    symbol: str = Field(min_length=1)
    timestamp: UtcDatetime

    session: str | None = None
    sessions: tuple[str, ...] = ()
    market_open: bool = True
    minutes_until_weekly_close: int = Field(default=0, ge=0)
    trend_strength: float | None = None
    volatility_percentile: float | None = None
    is_trending: bool = False
    is_ranging: bool = False

    votes: tuple[VoteLine, ...] = ()
    fired: bool = False
    reason: str = ""
    winning_direction: str | None = None
    long_weight: float = 0.0
    long_votes: int = 0
    short_weight: float = 0.0
    short_votes: int = 0
    min_total_weight: float = 0.0
    min_agreeing: int = 0

    plan: TradePlan | None = None
    analogues: tuple[AnalogueBrief, ...] = ()

    #: Named indicator readings at this bar — `{"adx_14": 27.3, "atr_14": 0.0021}`. `None` is
    #: a warm-up and is rendered as such, never as zero, for the same reason the classifier
    #: reports it that way: a zero ADX and an unmeasurable one are different facts.
    indicators: dict[str, float | None] = Field(default_factory=dict)
    #: Detected candle formations. CONTEXT ONLY — they reach the chartist and the panel and
    #: nothing else. `tests/patterns/test_patterns_never_reach_consensus.py` holds the other
    #: end of that.
    patterns: tuple[PatternBrief, ...] = ()

    #: The sized order and the exposure it sits inside. `None` until `fxagent.risk` has run,
    #: and `None` on every bar that fired nothing — there is no order to size.
    execution: ExecutionPlan | None = None
    #: Dealing conditions right now against their own recent median.
    spread: SpreadState | None = None
    #: Scheduled releases near this bar, in the window the caller asked for.
    events: tuple[CalendarEventBrief, ...] = ()
    #: What the agents already asked have said. Populated by `narrate` for the agents whose
    #: spec opts in, so ordering within one pass is the only thing that decides who sees whom.
    agent_notes: tuple[AgentEcho, ...] = ()
    #: Conditions the deterministic record already establishes — each one a comparison between
    #: two numbers on this briefing. Derived by the agent that reasons about them (see
    #: `AgentSpec.augment`) rather than computed here, so the briefing stays a projection.
    risk_flags: tuple[RiskFlag, ...] = ()

    @classmethod
    def from_consensus(
        cls,
        regime: Regime,
        result: ConsensusResult,
        *,
        analogues: Sequence[AnalogueBrief] = (),
        indicators: Mapping[str, float | None] | None = None,
        patterns: Sequence[PatternHit | PatternBrief] = (),
        execution: ExecutionPlan | None = None,
        spread: SpreadState | None = None,
        events: Sequence[CalendarEventBrief] = (),
    ) -> Briefing:
        """Assemble a briefing from a classified regime and the consensus it produced.

        Reads `result.diagnostics` with `.get` defaults throughout. The diagnostics document is
        the analyst's own output and its shape is stable, but it is also what a stored
        evaluation row replays through, and a briefing that refused to build from a row written
        before a key existed would make old evaluations unexplainable rather than sparse.
        """
        diagnostics = result.diagnostics
        signal = result.signal

        plan: TradePlan | None = None
        if signal is not None:
            primary = signal.primary
            if primary.stop_loss is not None and primary.take_profit is not None:
                plan = TradePlan(
                    direction=str(signal.direction),
                    entry_price=primary.entry_price,
                    stop_loss=primary.stop_loss,
                    take_profit=primary.take_profit,
                    reward_risk=primary.reward_risk,
                    confidence=signal.confidence,
                    primary_strategy=primary.strategy_name,
                )

        return cls(
            symbol=regime.symbol,
            timestamp=regime.timestamp,
            session=str(regime.session) if regime.session is not None else None,
            sessions=tuple(str(name) for name in regime.sessions),
            market_open=regime.market_open,
            minutes_until_weekly_close=regime.minutes_until_weekly_close,
            trend_strength=regime.trend_strength,
            volatility_percentile=regime.volatility_percentile,
            is_trending=regime.is_trending,
            is_ranging=regime.is_ranging,
            votes=tuple(
                VoteLine(
                    strategy=str(vote.get("strategy", "")),
                    weight=float(vote.get("weight", 0.0)),
                    direction=vote.get("direction"),
                    confidence=vote.get("confidence"),
                    participated=bool(vote.get("participated", False)),
                    reason=str(vote.get("reason", "")),
                )
                for vote in diagnostics.get("votes", ())
                if vote.get("strategy")
            ),
            fired=bool(diagnostics.get("fired", result.fired)),
            reason=str(diagnostics.get("reason", "")),
            winning_direction=diagnostics.get("winning_direction"),
            long_weight=float(diagnostics.get("long_weight", 0.0)),
            long_votes=int(diagnostics.get("long_votes", 0)),
            short_weight=float(diagnostics.get("short_weight", 0.0)),
            short_votes=int(diagnostics.get("short_votes", 0)),
            min_total_weight=float(diagnostics.get("min_total_weight", 0.0)),
            min_agreeing=int(diagnostics.get("min_agreeing", 0)),
            plan=plan,
            analogues=tuple(analogues),
            indicators=dict(indicators or {}),
            patterns=tuple(
                item if isinstance(item, PatternBrief) else PatternBrief.from_hit(item)
                for item in patterns
            ),
            execution=execution,
            spread=spread,
            events=tuple(events),
        )

    # -- rendering -------------------------------------------------------------

    @property
    def participating(self) -> tuple[VoteLine, ...]:
        return tuple(vote for vote in self.votes if vote.participated)

    def payload(self, *, include_dates: bool = True) -> dict[str, Any]:
        """The exact JSON document an agent is shown.

        `vote_count` and `participating_count` are derived here rather than left to be counted
        by the reader, because a count the agent can legitimately state must be a number it was
        given — see the module docstring on why the input grows instead of the check loosening.
        """
        document: dict[str, Any] = {
            "symbol": self.symbol,
            "regime": {
                "session": self.session,
                "sessions": list(self.sessions),
                "market_open": self.market_open,
                "minutes_until_weekly_close": self.minutes_until_weekly_close,
                "trend_strength": _round(self.trend_strength),
                "volatility_percentile": _round(self.volatility_percentile),
                "is_trending": self.is_trending,
                "is_ranging": self.is_ranging,
            },
            "votes": [
                {
                    "strategy": vote.strategy,
                    "weight": _round(vote.weight),
                    "direction": vote.direction,
                    "confidence": _round(vote.confidence),
                    "participated": vote.participated,
                    "reason": vote.reason,
                }
                for vote in self.votes
            ],
            "vote_count": len(self.votes),
            "participating_count": len(self.participating),
            "decision": {
                "fired": self.fired,
                "reason": self.reason,
                "winning_direction": self.winning_direction,
                "long_weight": _round(self.long_weight),
                "long_votes": self.long_votes,
                "short_weight": _round(self.short_weight),
                "short_votes": self.short_votes,
                "min_total_weight": _round(self.min_total_weight),
                "min_agreeing": self.min_agreeing,
            },
            "plan": (
                None
                if self.plan is None
                else {
                    "direction": self.plan.direction,
                    "entry_price": _round(self.plan.entry_price),
                    "stop_loss": _round(self.plan.stop_loss),
                    "take_profit": _round(self.plan.take_profit),
                    "reward_risk": _round(self.plan.reward_risk),
                    "confidence": _round(self.plan.confidence),
                    "primary_strategy": self.plan.primary_strategy,
                }
            ),
            "indicators": {name: _round(value) for name, value in sorted(self.indicators.items())},
            "patterns": [
                {
                    "name": pattern.name,
                    "definition": pattern.definition,
                    "bar_index": pattern.bar_index,
                    "criteria": {
                        key: _round(value) for key, value in sorted(pattern.criteria.items())
                    },
                    "label": pattern.label,
                }
                for pattern in self.patterns
            ],
            "execution": (
                None
                if self.execution is None
                else {
                    "volume": _round(self.execution.volume),
                    "risk_fraction": _round(self.execution.risk_fraction),
                    "risk_amount": _round(self.execution.risk_amount),
                    "stop_distance": _round(self.execution.stop_distance),
                    "total_open_risk": _round(self.execution.total_open_risk),
                    "max_risk_per_trade": _round(self.execution.max_risk_per_trade),
                    "max_total_risk": _round(self.execution.max_total_risk),
                    # The same four figures as percentages. A fraction of equity is how the
                    # sizer thinks and "0.40%" is how a person reads it, so both are shown.
                    # Otherwise a correct narration is discarded for expressing a cap the way
                    # the cap is written down: the input grows, the check does not loosen.
                    "risk_percent": _round(self.execution.risk_fraction * 100),
                    "total_open_risk_percent": _round(self.execution.total_open_risk * 100),
                    "max_risk_per_trade_percent": _round(self.execution.max_risk_per_trade * 100),
                    "max_total_risk_percent": _round(self.execution.max_total_risk * 100),
                }
            ),
            "spread": (
                None
                if self.spread is None
                else {
                    "spread": _round(self.spread.spread),
                    "median_spread": _round(self.spread.median_spread),
                    "over_median": _round(self.spread.over_median),
                }
            ),
            "events": [
                {
                    "title": event.title,
                    "currency": event.currency,
                    "importance": event.importance,
                    "minutes_until": event.minutes_until,
                    "forecast": _round(event.forecast),
                    "previous": _round(event.previous),
                    "actual": _round(event.actual),
                }
                for event in self.events
            ],
            "agent_notes": [
                {"agent": echo.agent, "text": echo.text, "provider": echo.provider}
                for echo in self.agent_notes
            ],
            "risk_flags": [
                {"flag": flag.flag, "severity": flag.severity} for flag in self.risk_flags
            ],
            "risk_flag_count": len(self.risk_flags),
            "analogues": [
                {
                    "id": analogue.id,
                    "symbol": analogue.symbol,
                    "similarity": _round(analogue.similarity),
                    "outcome": analogue.outcome,
                    "outcome_r": _round(analogue.outcome_r),
                    **(
                        {
                            "timestamp": analogue.timestamp.isoformat(),
                            "resolved_at": (
                                analogue.resolved_at.isoformat()
                                if analogue.resolved_at is not None
                                else None
                            ),
                        }
                        if include_dates
                        else {}
                    ),
                }
                for analogue in self.analogues
            ],
        }
        if include_dates:
            document["timestamp"] = self.timestamp.isoformat()
        return document

    def rendered(self, *, include_dates: bool = True) -> str:
        """The payload as the string that actually goes into the prompt."""
        return json.dumps(self.payload(include_dates=include_dates), sort_keys=True, indent=2)

    def grounding(self, *, include_dates: bool = True) -> Grounding:
        """The numbers a response may use, taken from the payload it will be shown."""
        return Grounding.from_payload(self.payload(include_dates=include_dates))


# -- the output ---------------------------------------------------------------------------


class AgentNote(_Document):
    """Base of every agent response: the number check, and nothing else.

    Each agent declares its own narration field — the chartist's is `read`, the historian's is
    `text` — because the two are asked different questions and a shared field name would be the
    only thing they had in common. Whatever it is called, it is **serialised as `text`**, which
    is the single field `fxagent.dashboard.contract` renders. `tests/agents/test_narrate.py`
    asserts every registered agent dumps one.

    `extra="forbid"`, in deliberate contrast to `fxagent.dashboard.contract`, which ignores
    unknown keys. The dashboard is reading documents this system wrote and must tolerate the
    analyst growing a field; this is reading a model's answer to a stated schema, and a key
    that was not asked for means the schema was not followed. Under hard rule 5 that is a
    discard, and the cost of discarding is a template paragraph.
    """

    @model_validator(mode="after")
    def _numbers_must_be_grounded(self, info: ValidationInfo) -> Self:
        grounding = (info.context or {}).get(GROUNDING_KEY)
        if not isinstance(grounding, Grounding):
            raise ValueError(
                "an agent note must be validated with a Grounding in the validation context; "
                "without it the number check would silently pass and hard rule 5's "
                "'never a number that was not in its input' would be a comment, not a control"
            )

        # Re-read the note's own JSON rather than only the narration field: a subclass may
        # score into a numeric field, and a level invented there is the same defect as one
        # invented in a sentence. Checking the serialised form covers every field a subclass
        # can add without each subclass having to remember to opt in.
        invented = grounding.ungrounded(self.model_dump_json())
        if invented:
            raise ValueError(
                f"response used {len(invented)} number(s) absent from its input: "
                f"{', '.join(invented[:5])}"
            )
        return self


class PatternSeen(_Document):
    """One formation the chartist chose to mention, and what it says about it.

    `label` is a defaulted field rather than something the model supplies, so a formation
    cannot reach a screen without its CONTEXT stamp — the same arrangement
    `fxagent.dashboard.models.PatternNote` uses, and for the same reason: a caption can be
    styled away, a value being printed cannot.

    `significance` is where a model would most naturally write "suggests a reversal". It is
    displayed next to the label saying the evidence does not support that, and it enters
    nothing. There is no field on this class that a number could be read out of.
    """

    #: Must be one of the formations actually detected. Checked, not requested.
    name: str = Field(min_length=1)
    meaning: str = Field(min_length=1, max_length=300)
    significance: str = Field(min_length=1, max_length=300)
    label: str = CONTEXT_ONLY


class TermExplained(_Document):
    """One piece of jargon the note used, defined.

    The panel is read by one person at 7am deciding whether to trust a machine. A narration
    that says "ADX is above the threshold" and leaves it there is not an explanation, it is a
    restatement — so any term the chartist uses, it defines.
    """

    term: str = Field(min_length=1, max_length=60)
    definition: str = Field(min_length=1, max_length=300)


class ChartistNote(AgentNote):
    """The chartist's reading of the structured core output. Never pixels, never raw bars.

    `read` is the narration — short on purpose. The panel gives this a card, and a chartist
    that needs four hundred characters to say what the regime is has not read it, it has
    restated the input.

    `disagreement` is the one place the chartist may push back, and it is `None` on most bars.
    It is prose and it is advisory: it is displayed and logged, it is not scored, nothing
    compares it to anything, and no code path branches on it. That is the same status
    CLAUDE.md gives the risk officer's `proceed_recommendation`, for the same reason — an
    agent whose objection could gate something is an agent that decides.
    """

    read: str = Field(
        min_length=1,
        max_length=MAX_READ_CHARS,
        serialization_alias="text",
        description="The reading itself. Rendered as the card's text.",
    )
    patterns_seen: tuple[PatternSeen, ...] = ()
    terms_explained: tuple[TermExplained, ...] = ()
    disagreement: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)

    @model_validator(mode="after")
    def _patterns_were_actually_detected(self, info: ValidationInfo) -> Self:
        """A formation the detector did not find cannot be narrated into existence.

        The numeric check does not cover this — a formation name has no digits in it — so
        without this a model could report a textbook morning star nobody detected, on a bar
        where the deterministic scan found a doji.
        """
        grounding = (info.context or {}).get(GROUNDING_KEY)
        if isinstance(grounding, Grounding):
            unknown = sorted({item.name for item in self.patterns_seen} - grounding.pattern_names)
            if unknown:
                raise ValueError(
                    f"named formation(s) that were not detected on this bar: {unknown}; "
                    "the detector decides what is present, not the narration"
                )
        return self


class HistorianNote(AgentNote):
    """The historian's reading of the retrieved analogues.

    `analogue_ids` names which retrieved windows the prose is about. It cannot introduce one:
    every id is checked against the briefing, so an analogue that failed the point-in-time
    filter cannot re-enter the narration by being cited. The analogues the panel displays are
    the retrieval's own records, attached by the caller — this field selects, it does not
    supply.
    """

    text: str = Field(min_length=1, max_length=MAX_NOTE_CHARS)
    analogue_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _analogues_were_retrieved(self, info: ValidationInfo) -> Self:
        grounding = (info.context or {}).get(GROUNDING_KEY)
        if isinstance(grounding, Grounding):
            unknown = [item for item in self.analogue_ids if item not in grounding.analogue_ids]
            if unknown:
                raise ValueError(
                    f"cited analogue id(s) that were not retrieved: {unknown}; a window that "
                    "did not pass the point-in-time filter cannot be cited back into the record"
                )
        return self


class RiskOfficerNote(AgentNote):
    """The risk officer's reading of a plan it did not choose and cannot change.

    (Distinct from `fxagent.dashboard.models.RiskOfficerNote`, which is the panel's *view* of
    this. Same name because it is the same thing at two ends of a JSONB document; this one is
    what an LLM must produce, that one is what a browser receives.)

    **`proceed_recommendation` is advisory only.** It is displayed, it is logged, and it gates
    nothing. The deterministic permission layer decides whether an order is placed, and
    `tests/agents/test_advisory_only.py` asserts both halves of that: behaviourally, that a WAIT
    leaves the trade plan and the consensus decision bit-identical; and structurally, that no
    module outside the narration and display paths so much as mentions the field.

    That is not defensive paranoia about a string. It is CLAUDE.md hard rule 4 at its most
    load-bearing point: this is the one agent whose output *reads* like a control, sitting next
    to the one decision where being wrong costs money. An agent that can stop a trade is an
    agent that can stop the right trade, and the failure mode is invisible — a suppressed
    winner leaves no trace in the equity curve to explain.
    """

    plan_summary: str = Field(
        min_length=1,
        max_length=MAX_PLAN_SUMMARY_CHARS,
        serialization_alias="text",
        description="What the plan is, in one or two sentences. Rendered as the card's text.",
    )
    risk_flags: tuple[RiskFlag, ...] = ()
    size_rationale: str = Field(default="", max_length=MAX_NOTE_CHARS)
    proceed_recommendation: str = Field(default="CAUTION")
    reasoning: str = Field(default="", max_length=MAX_NOTE_CHARS)

    @model_validator(mode="after")
    def _recommendation_is_one_of_three(self) -> Self:
        if self.proceed_recommendation not in PROCEED_RECOMMENDATIONS:
            raise ValueError(
                f"proceed_recommendation must be one of {PROCEED_RECOMMENDATIONS}, got "
                f"{self.proceed_recommendation!r}"
            )
        return self


def validate_note[T: AgentNote](
    note: type[T],
    raw: Any,
    briefing: Briefing,
    *,
    include_dates: bool = True,
) -> T | None:
    """Validate one response against `note`, or discard it whole and return `None`.

    `raw` is a JSON string or an already-decoded object. A string that is not valid JSON is a
    discard: there is no repair step, because every repair is a guess about what a model meant
    and a guess that lands in a narration next to a real decision is worse than no narration.

    The content is never logged. A briefing quotes plan levels and a narration quotes them
    back, and a log line is the one place that outlives the request.
    """
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            # The exception type and position are logged; the text that failed to parse is not.
            logger.warning(
                "discarded %s response: not valid JSON (%s)", note.__name__, type(error).__name__
            )
            return None

    try:
        return note.model_validate(
            raw,
            context={GROUNDING_KEY: briefing.grounding(include_dates=include_dates)},
        )
    except ValidationError as error:
        logger.warning(
            "discarded %s response: %d validation error(s)", note.__name__, error.error_count()
        )
        return None


def sentences(parts: Iterable[str]) -> str:
    """Join non-empty fragments into one paragraph. Shared by the templates and the prompts."""
    return " ".join(part.strip() for part in parts if part and part.strip())


def direction_word(direction: str | None) -> str:
    """A direction as prose, tolerating the enum, the string and the absence of both."""
    if direction is None:
        return "no direction"
    text = str(direction).upper()
    if text == str(SignalDirection.FLAT):
        return "flat"
    return text.lower()
