"""The narration pass: briefing in, dashboard-shaped agent blocks out. Never fails.

The loop, and nothing else. Each agent's prompt, schema and fallback live in its own module —
`chartist`, `historian` — and arrive here as an `AgentSpec`. This file asks the gateway,
validates whatever comes back, and on *any* miss writes that agent's deterministic block
instead: no key, no provider, a rate limit that outlasted the backoff, malformed JSON, a schema
violation, an invented number, a formation nobody detected.

**There is no failure path.** `narrate` has no error return and raises nothing a caller is
expected to handle. Every branch ends in a block of text, because the narration is commentary
on a decision that was already made without it: the regime was measured, the strategies voted
and the plan was chosen by deterministic Python before this function was called, and a pass
that fell over because a free-tier API was down would be reporting an outage in the part of the
system that does not matter as though it were an outage in the part that does.

**Agents are asked in order, and only the risk officer is shown what came before.** Its spec
sets `reads_agents`; the chartist's and the historian's do not. That is not caution about
prompt size — the two of them are asked independent questions, and showing one the other's
answer turns two readings into one with a preamble. The risk officer is reviewing the whole
record, so the record includes what was said about it.

**Nothing an agent returns is ever read as a decision here.** The risk officer's
`proceed_recommendation` travels into its block and no further: this function has no branch on
it, and `tests/agents/test_advisory_only.py` asserts that a WAIT leaves the trade plan and the
consensus result bit-identical. A fourth agent is a module beside the other three and one entry
in `AGENTS`; it will have the same standing.

**Analogues come from retrieval, not from the historian.** The `analogues` list attached to the
historian's block is copied off the briefing whatever the agent said, or whether it said
anything. The retrieval is point-in-time filtered in SQL; an agent that narrates it cannot add
a window to it, and the panel shows what was actually retrieved either way.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from fxagent.agents import chartist, historian, risk_officer
from fxagent.agents.gateway import Gateway, Prompt
from fxagent.agents.schemas import AgentEcho, Briefing, validate_note
from fxagent.agents.spec import AgentSpec

# The dashboard owns these names because it reads them; `contract.py` says in as many words
# that the analyst is expected to import them to write the same keys. One definition of the
# JSONB layout, imported by both ends, beats two that agree until one of them is edited.
from fxagent.dashboard.contract import AGENTS_KEY, HISTORIAN, PATTERNS_KEY

__all__ = [
    "AGENTS",
    "LEGACY_AGENTS",
    "TEMPLATE_PROVIDER",
    "AgentSpec",
    "attach_narration",
    "attach_patterns",
    "narrate",
]

logger = logging.getLogger(__name__)

#: What `provider` reads as when the template wrote the block. A real name rather than `None`,
#: so the panel can say which paragraphs came from a model and a stored evaluation still says
#: so a month later. `None` would be indistinguishable from a field nobody filled in.
TEMPLATE_PROVIDER = "template"

#: The three agents, in the order the panel shows them — and the order they are asked in,
#: which matters: the risk officer's spec opts into reading what the other two said, and it can
#: only read what has already been produced. A fourth agent is a module beside theirs and a
#: fourth entry here.
## The active registry contains the chartist only. The retired specs remain available in
## LEGACY_AGENTS for compatibility with historical evaluations.
AGENTS: tuple[AgentSpec, ...] = (chartist.SPEC,)

# Compatibility registry for re-rendering old three-agent evaluations. This is deliberately not
# the default and must never be passed by the trader.
LEGACY_AGENTS: tuple[AgentSpec, ...] = (chartist.SPEC, historian.SPEC, risk_officer.SPEC)


def _analogue_blocks(briefing: Briefing) -> list[dict[str, Any]]:
    """The retrieved analogues in the shape `dashboard.contract` reads."""
    return [
        {
            "timestamp": analogue.timestamp.isoformat(),
            "symbol": analogue.symbol,
            "similarity": analogue.similarity,
            "outcome": analogue.outcome,
            "outcome_r": analogue.outcome_r,
            "resolved_at": (
                analogue.resolved_at.isoformat() if analogue.resolved_at is not None else None
            ),
        }
        for analogue in briefing.analogues
    ]


def _stamp(block: dict[str, Any], briefing: Briefing) -> dict[str, Any]:
    return {
        **block,
        "provider": TEMPLATE_PROVIDER,
        "model": None,
        "generated_at": briefing.timestamp.isoformat(),
    }


async def narrate(
    briefing: Briefing,
    *,
    gateway: Gateway | None = None,
    include_dates: bool = True,
    agents: Sequence[AgentSpec] = AGENTS,
) -> dict[str, Any]:
    """One narration block per agent, LLM-written where possible and templated where not.

    `gateway=None` skips the LLM path entirely and templates everything — the honest way to run
    a backtest, where the point is to replay the deterministic core and a model's commentary
    would be both a cost and a source of run-to-run variation.

    `include_dates=False` drops every timestamp from the prompt, for backtest-mode narration
    where the current date is look-ahead. It is passed to the grounding as well as to the
    payload, so a response cannot quote a date it was not shown either.

    Agents are awaited one after another, never gathered. The gateway serialises anyway, but
    fanning out here would queue every symbol's prompts behind one lock and turn a rate-limit
    wait into a wait for all of them.
    """
    blocks: dict[str, Any] = {}

    echoes: list[AgentEcho] = []

    for spec in agents:
        block: dict[str, Any] | None = None
        # Only the agents that opted in see the ones before them; everyone else is handed the
        # briefing exactly as it arrived, so a reordering of `AGENTS` cannot change what an
        # independent agent was asked.
        shown = spec.augment(
            briefing.model_copy(update={"agent_notes": tuple(echoes)})
            if spec.reads_agents
            else briefing
        )

        # Evaluated on every path, including the ones with no gateway and no key. "Is there
        # anything here worth a call" is a question about the briefing, and answering it only
        # when a provider happens to be configured would leave the predicate unexercised until
        # somebody set a key — so the first day it ran would be the first day it was tested.
        # It also keeps the two reasons for silence apart in the log: nothing to review is a
        # property of the bar, no provider is a property of the deployment, and a card that
        # reads the same either way is how a misconfiguration hides for a week.
        worth_asking = spec.should_ask(shown)
        if not worth_asking:
            logger.debug("%s not asked: this briefing has nothing for it to review", spec.name)

        if gateway is not None and worth_asking:
            prompt = Prompt(
                agent=spec.name,
                system=spec.system,
                user=shown.rendered(include_dates=include_dates),
                prefer=spec.prefer,
                daily_call_limit=spec.daily_call_limit,
            )
            completion = await gateway.complete(prompt)
            if completion is not None:
                note = validate_note(spec.note, completion.text, shown, include_dates=include_dates)
                if note is None:
                    logger.warning(
                        "%s response from %s was discarded; using the deterministic template",
                        spec.name,
                        completion.provider,
                    )
                else:
                    block = {
                        # `mode="json"` because this document is written to JSONB: it turns the
                        # tuples into lists and anything datetime-shaped into a string, so what
                        # is stored is what the dashboard reads back rather than whatever the
                        # driver's encoder happened to make of a Python object.
                        # `by_alias=True` because each agent names its own narration field and
                        # they all serialise as `text`, which is the one field the panel renders.
                        **note.model_dump(mode="json", by_alias=True),
                        "provider": completion.provider,
                        "model": completion.model,
                        "generated_at": completion.generated_at.isoformat(),
                    }

        if block is None:
            block = _stamp(spec.fallback(shown), briefing)

        echoes.append(
            AgentEcho(agent=spec.name, text=str(block.get("text", "")), provider=block["provider"])
        )

        if spec.name == HISTORIAN:
            # Attached whichever path wrote the text. See the module docstring: the panel shows
            # what retrieval actually returned, not what an agent chose to mention.
            block["analogues"] = _analogue_blocks(briefing)

        blocks[spec.name] = block

    return blocks


def attach_narration(diagnostics: dict[str, Any], blocks: dict[str, Any]) -> dict[str, Any]:
    """Put the blocks under the reserved key, returning a new document.

    A copy rather than a mutation: `diagnostics` is the object `Consensus.evaluate` returned,
    it is also what gets written to the journal, and a function that quietly edited it would
    make "what did consensus decide" depend on whether narration had run yet.
    """
    return {**diagnostics, AGENTS_KEY: blocks}


def attach_patterns(diagnostics: dict[str, Any], briefing: Briefing) -> dict[str, Any]:
    """Write the detected formations under the panel's reserved `patterns` key.

    Separate from `attach_narration` because they are separate facts with separate producers: a
    formation is a deterministic detection that exists whether or not any agent ran, and folding
    it into the narration writer would make it disappear on the path where narration is skipped.

    The stamp is carried through from `PatternHit` rather than applied here — the label is part
    of the data, and this is only choosing where to put it.
    """
    return {
        **diagnostics,
        PATTERNS_KEY: [
            {
                "name": pattern.name,
                "definition": pattern.definition,
                "bar_time": (
                    pattern.timestamp.isoformat() if pattern.timestamp is not None else None
                ),
                "label": pattern.label,
            }
            for pattern in briefing.patterns
        ],
    }
