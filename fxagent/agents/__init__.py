"""The narration layer. Explains decisions; never makes one.

Seven modules, layered so that the bottom one can never depend on the top:

| Module | Job |
|---|---|
| `schemas` | The `Briefing` an agent is shown, the shapes it may return, and the checks. |
| `templates` | Deterministic explanation from the same briefing. No LLM, no network. |
| `gateway` | N providers, sequential, backed off, budgeted, cached. Returns `None` on failure. |
| `spec` | What an agent is, as a record — so an agent needs no import of the loop. |
| `chartist`, `historian`, `risk_officer` | One module per agent: prompt, schema, fallback. |
| `narrate` | Briefing → dashboard-shaped blocks. LLM where possible, template where not. |

`templates` is the floor and imports only `schemas`. With every key unset, every provider down
and the daily call budget spent, `narrate` still returns a full set of blocks — which is the
property `tests/agents/test_narration_survives_total_outage.py` exists to hold.

Nothing here can move a number. Hard rule 4: indicators, signals, sizes, stops and order types
are deterministic Python, and an agent that could touch one of them is an agent that could
empty an account. Hard rule 5: any response failing validation is discarded entirely and the
template is used, with no partial parsing and no retry-until-it-parses.

The risk officer's `proceed_recommendation` is **advisory only** — displayed, logged, and read
by nothing. `tests/agents/test_advisory_only.py` holds that behaviourally (a WAIT leaves the
trade plan and the consensus result bit-identical) and structurally (no module outside the
producer and the panel names the field in code, and no branch anywhere decides on it).

LiteLLM is an optional extra (`uv sync --extra llm`) imported inside the call that needs it, so
this package imports and this package's tests run without it.
"""

from __future__ import annotations

from fxagent.agents.gateway import (
    DEFAULT_DAILY_CALL_LIMIT,
    DEFAULT_PROVIDERS,
    CallBudget,
    Completion,
    Gateway,
    Prompt,
    ProviderConfig,
    ProviderError,
    RateLimited,
)
from fxagent.agents.narrate import (
    AGENTS,
    TEMPLATE_PROVIDER,
    attach_narration,
    attach_patterns,
    narrate,
)
from fxagent.agents.schemas import (
    AgentEcho,
    AgentNote,
    AnalogueBrief,
    Briefing,
    CalendarEventBrief,
    ChartistNote,
    ExecutionPlan,
    Grounding,
    HistorianNote,
    PatternBrief,
    PatternSeen,
    RiskFlag,
    RiskOfficerNote,
    SpreadState,
    TermExplained,
    TradePlan,
    VoteLine,
    validate_note,
)
from fxagent.agents.spec import AgentSpec
from fxagent.agents.templates import explain

__all__ = [
    "AGENTS",
    "AgentEcho",
    "AgentNote",
    "AgentSpec",
    "AnalogueBrief",
    "Briefing",
    "CalendarEventBrief",
    "CallBudget",
    "ChartistNote",
    "Completion",
    "DEFAULT_DAILY_CALL_LIMIT",
    "DEFAULT_PROVIDERS",
    "ExecutionPlan",
    "Gateway",
    "Grounding",
    "HistorianNote",
    "PatternBrief",
    "PatternSeen",
    "Prompt",
    "ProviderConfig",
    "ProviderError",
    "RateLimited",
    "RiskFlag",
    "RiskOfficerNote",
    "SpreadState",
    "TEMPLATE_PROVIDER",
    "TermExplained",
    "TradePlan",
    "VoteLine",
    "attach_narration",
    "attach_patterns",
    "explain",
    "narrate",
    "validate_note",
]
