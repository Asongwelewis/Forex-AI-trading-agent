"""The historian: reads retrieved analogues and how each one actually resolved.

Beside `chartist` so `narrate.AGENTS` reads as one kind of thing — a tuple of agent modules'
specs — rather than as one agent with a module and another with a prompt string inlined in the
orchestrator. The risk officer joins as a third file.

**Every analogue it sees resolved before the bar under analysis.** That is enforced in SQL by
`WindowRepository.search`, which requires `as_of` and filters on `outcome_resolved_at`; there
is no unfiltered search to reach for. `HistorianNote.analogue_ids` is checked against the
briefing on top of that, so a window that failed the point-in-time filter cannot be cited back
into the record by an agent that heard about it some other way.

**An empty retrieval is a normal answer.** Early in the record nothing has resolved yet, and
"no analogue was retrievable" is informative. Saying so is not the same as a lookup failing,
and the prompt asks for the distinction explicitly.
"""

from __future__ import annotations

from typing import Any

from fxagent.agents import templates
from fxagent.agents.schemas import MAX_NOTE_CHARS, Briefing, HistorianNote
from fxagent.agents.spec import AgentSpec
from fxagent.dashboard.contract import HISTORIAN

__all__ = ["NAME", "SPEC", "SYSTEM", "fallback"]

NAME = HISTORIAN

SYSTEM = (
    "You are the historian. You are given a JSON record of an analysis that deterministic code "
    "has already completed, together with historical windows retrieved as analogues and how "
    "each one actually resolved.\n"
    "Every analogue you were given resolved before the bar under analysis. You must not reason "
    "about anything later than that, and you must not cite an analogue that is not in the "
    "input.\n\n"
    "Rules, all absolute:\n"
    "1. Never recommend a trade, a direction, a size, a stop or a target. Whether to trade has "
    "already been settled and nothing you write can change it.\n"
    "2. Never state a number that does not appear in the input. Not a rounded estimate, not a "
    "derived figure, not a rate you worked out from the outcomes. Any response containing one "
    "is discarded in full.\n"
    "3. Describe only what the input says. Where it does not answer something, say so.\n\n"
    "Return one JSON object and nothing else — no prose outside it, no code fence:\n"
    '{"text": str, "analogue_ids": [int]}\n\n'
    f"`text` is under {MAX_NOTE_CHARS} characters. `analogue_ids` lists the ids your text is "
    "about, drawn only from the input.\n"
    "An empty analogue list is a normal result, not an error: say the record does not yet reach "
    "back far enough for a resolved comparison."
)


def fallback(briefing: Briefing) -> dict[str, Any]:
    """The retrieved analogues summarised from their own numbers, with no provider involved."""
    return {"text": templates.historian_fallback(briefing)}


SPEC = AgentSpec(
    name=NAME,
    note=HistorianNote,
    system=SYSTEM,
    fallback=fallback,
    # CLAUDE.md pins the historian to Gemini. A preference, not a requirement.
    prefer=("gemini",),
)
