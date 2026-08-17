"""What an agent is, as a record the orchestrator can iterate.

Its own module so each agent can declare its spec without importing the loop that runs it, and
the loop can import every agent without a cycle. The dependency graph is a line:

    spec -> schemas
    chartist, historian -> spec, schemas, templates
    narrate -> spec, the agents, gateway

Adding the risk officer is a module beside `chartist` and one entry in `narrate.AGENTS`. That
is what "a config entry, not a refactor" has to mean to be worth claiming.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fxagent.agents.schemas import AgentNote, Briefing

__all__ = ["AgentSpec"]


@dataclass(frozen=True)
class AgentSpec:
    """One agent: its name, its schema, its prompt, and what to write when it cannot answer.

    `fallback` returns a whole block rather than a string. Both paths through `narrate` then
    produce the same shape, so a card does not lose its structured half the moment a provider
    goes down — the chartist's detected formations are deterministic facts, and there is no
    reason for them to disappear because a model was unreachable.
    """

    name: str
    note: type[AgentNote]
    system: str
    fallback: Callable[[Briefing], dict[str, Any]]
    #: The provider CLAUDE.md pins this agent to, tried first. A preference and not a
    #: requirement: with that provider unconfigured or failing, the ordinary chain serves the
    #: call, because a narration from the second-choice model beats no narration at all.
    #:
    #: More than one name makes the preference a ladder, which is how the risk officer's three
    #: NVIDIA rungs are expressed — same key, same endpoint, three models.
    prefer: tuple[str, ...] = field(default=())
    #: Outbound requests this agent may make in a day, on top of the gateway's shared cap. The
    #: number lives here, beside the agent it governs, and travels to the gateway on the prompt
    #: — so the enforcement stays in one place without the gateway learning what an agent is.
    daily_call_limit: int | None = None
    #: Whether this agent is shown what the agents before it said. `False` everywhere except
    #: the risk officer: the chartist and the historian are asked independent questions, and
    #: showing one the other's answer would turn two readings into one with a preamble.
    reads_agents: bool = False
    #: Derives whatever this agent needs that the briefing does not already carry, returning
    #: the briefing it will actually be shown. The risk officer uses it to compute its
    #: deterministic flags: they belong to the agent that reasons about them, not to a model
    #: every other agent also reads — and because grounding is taken from the shown briefing,
    #: deriving them here is also what makes them quotable.
    augment: Callable[[Briefing], Briefing] = field(default=lambda briefing: briefing)
    #: Whether it is worth asking a provider at all on this briefing. Returning `False` skips
    #: straight to the fallback without spending a call — which is how a tight per-agent budget
    #: survives the eight hours a day on which nothing qualifies.
    should_ask: Callable[[Briefing], bool] = field(default=lambda _briefing: True)
