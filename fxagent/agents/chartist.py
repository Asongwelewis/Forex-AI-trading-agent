"""The chartist: reads the computed record and says what it sees. Decides nothing.

**Structured data only.** The prompt is the `Briefing` JSON — regime measurement, each
strategy's vote and the weight the router gave it, named indicator readings, detected candle
formations, session state. No chart image, no raw price series, not one OHLC tuple. That is
CLAUDE.md's row for this agent ("Structured core output / Never: pixels, raw price series")
and it is a property of `Briefing`, which has no field a bar could travel in — so the rule
holds by construction rather than by the prompt asking nicely.

The reason is not squeamishness about images. A model handed a price series will pattern-match
on it and produce a directional opinion that sounds like analysis and is worth nothing, and it
will do so in the one part of the output a human is most likely to read as a second opinion.
Handing it only what the deterministic core already computed means the most it can do is
describe that computation — which is the entire job.

**The formations are context and the output says so on every one of them.** Two studies found
candlestick formations produce no net positive return on EUR/USD after costs. `PatternSeen`
carries the CONTEXT label as a defaulted field, the detected set is checked so the chartist
cannot narrate a formation nobody found, and nothing downstream reads any of it. They reach
this agent and the panel; `tests/patterns/test_patterns_never_reach_consensus.py` holds the
other end.

**`disagreement` is prose, and prose is all it is.** The chartist may say the regime does not
support the direction. Nothing scores that, nothing counts it, and no branch anywhere tests it
— it is displayed and logged, the same status CLAUDE.md gives the risk officer's
`proceed_recommendation`. An agent whose objection could gate something is an agent that
decides, and hard rule 4 exists because an agent that can move a stop can empty an account.

**Jargon is defined by the agent that used it.** `terms_explained` is not decoration: the panel
exists so one person at 7am can understand every decision without opening a log file, and a
narration that says "ADX is above the threshold" has restated the input rather than explained
it. The deterministic fallback fills the same field from a fixed glossary, so the card does not
lose its explanations when a provider is down.
"""

from __future__ import annotations

from typing import Any

from fxagent.agents import templates
from fxagent.agents.schemas import (
    MAX_READ_CHARS,
    Briefing,
    ChartistNote,
)
from fxagent.agents.spec import AgentSpec
from fxagent.dashboard.contract import CHARTIST
from fxagent.patterns import CONTEXT_ONLY

__all__ = ["GLOSSARY", "NAME", "SPEC", "SYSTEM", "fallback"]

NAME = CHARTIST

#: What the fallback defines when the prose uses it. Deliberately a fixed table rather than
#: anything generated: these definitions appear on a panel a person makes decisions from, and a
#: definition that varies run to run is a definition nobody can check. The LLM path writes its
#: own, which is why this only has to cover the words the *template* uses.
GLOSSARY: dict[str, str] = {
    "ADX": (
        "Average Directional Index: how strongly the market is trending, regardless of "
        "direction. Higher means a stronger trend, not a rising price."
    ),
    "percentile": (
        "Where the current reading sits in its own recent history. The 62nd percentile means "
        "it was lower than this on most recent bars."
    ),
    "R": (
        "One R is the distance from entry to the stop. A 2R target is twice as far away as the "
        "stop, so the trade can be wrong more often than right and still make money."
    ),
    "summed weight": (
        "The router gives each strategy a weight for the current regime. Consensus adds the "
        "weights of the strategies that agree, and the total has to clear a fixed bar."
    ),
    "session": (
        "Which of the major trading centres — Tokyo, London, New York — is open at this bar. "
        "Boundaries are computed in local time, so they shift with daylight saving."
    ),
    "consensus": (
        "The rule that decides whether anything happens: enough summed weight and enough "
        "separate strategies agreeing on the same direction."
    ),
}

SYSTEM = (
    "You are the chartist. You are given a JSON record of an analysis that deterministic code "
    "has already completed: the regime it measured, every strategy's vote and the weight the "
    "router gave it, indicator readings, any candle formations a detector found, and the trade "
    "plan if one qualified.\n"
    "You never see a chart image and never see raw price bars, and you must not describe price "
    "action you were not given.\n\n"
    "Rules, all absolute:\n"
    "1. Never recommend a trade, a direction, a size, a stop or a target. Whether to trade has "
    "already been settled and nothing you write can change it.\n"
    "2. Never state a number that does not appear in the input. Not a rounded estimate, not a "
    "derived figure, not a percentage you worked out. Any response containing one is discarded "
    "in full.\n"
    "3. Only name a formation that appears under `patterns`. Candle formations are CONTEXT: two "
    "studies found they produce no net positive return on EUR/USD after costs. Say what the "
    "shape is; do not say what it predicts.\n"
    "4. Describe only what the input says. Where it does not answer something, say so.\n\n"
    "Return one JSON object and nothing else — no prose outside it, no code fence:\n"
    '{"read": str, "patterns_seen": [{"name": str, "meaning": str, "significance": str}], '
    '"terms_explained": [{"term": str, "definition": str}], "disagreement": str|null}\n\n'
    f"`read` is the whole reading, under {MAX_READ_CHARS} characters — the regime, what the "
    "votes did, and what the plan is.\n"
    "`patterns_seen` explains any formation you mention: `meaning` is the shape, `significance` "
    "is what it would be taken to mean, stated as the claim it is rather than as a fact.\n"
    "`terms_explained` defines every piece of jargon your `read` uses. A reader who does not "
    "know what ADX is must be able to follow the whole card.\n"
    "`disagreement` is null unless the measurements genuinely do not support the direction "
    "chosen, in which case say why in one sentence. It is displayed and recorded and it changes "
    "nothing; do not use it to argue for a different trade."
)


def _patterns_seen(briefing: Briefing) -> list[dict[str, Any]]:
    """The detected formations with their own definitions, and no claim about what they do.

    `significance` is the same sentence on every formation on purpose. The honest thing to say
    about a candle formation is what the evidence says, and the evidence says it does not pay —
    so the deterministic path says exactly that rather than inventing a per-formation story the
    LLM path would at least be visibly responsible for.
    """
    seen: dict[str, dict[str, Any]] = {}
    for pattern in briefing.patterns:
        seen.setdefault(
            pattern.name,
            {
                "name": pattern.name,
                "meaning": pattern.definition or f"A {pattern.name.replace('_', ' ')} formation.",
                "significance": (
                    "Displayed for context. Two studies found candle formations produce no net "
                    "positive return on EUR/USD after costs, and nothing in this system reads "
                    "them."
                ),
                "label": CONTEXT_ONLY,
            },
        )
    return list(seen.values())


def _terms_explained(read: str) -> list[dict[str, str]]:
    """Every glossary term the read actually used, in the order the glossary lists them.

    Matched against the produced text rather than emitted wholesale, so the card explains the
    words on it and not a dictionary the reader has to search.
    """
    lowered = read.lower()
    return [
        {"term": term, "definition": definition}
        for term, definition in GLOSSARY.items()
        if term.lower() in lowered
    ]


def fallback(briefing: Briefing) -> dict[str, Any]:
    """The whole card, deterministically. No provider, no key, no network.

    Returns the same shape the LLM path does — `patterns_seen` and `terms_explained` included —
    so a card does not lose half its structure the moment a free-tier API is unreachable. The
    formations are a deterministic fact either way; there is no reason for them to vanish.

    `disagreement` is always `None` here. The template has no basis for one: it can restate what
    the core measured, and "these measurements do not support this direction" is a judgement,
    which is the one thing this file exists not to make up.
    """
    read = templates.chartist_read(briefing)
    return {
        "text": read,
        "patterns_seen": _patterns_seen(briefing),
        "terms_explained": _terms_explained(read),
        "disagreement": None,
    }


SPEC = AgentSpec(
    name=NAME,
    note=ChartistNote,
    system=SYSTEM,
    fallback=fallback,
    # CLAUDE.md pins the chartist to Groq. A preference, not a requirement — see `AgentSpec`.
    prefer=("groq",),
)
