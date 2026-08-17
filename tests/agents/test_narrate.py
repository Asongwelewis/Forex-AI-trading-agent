"""The LLM path, when a provider does answer — and the discards that send it back to the floor.

The outage test proves the fallback. This one proves there is something to fall back *from*,
and that everything hard rule 5 calls a discard actually reaches the template rather than the
panel.
"""

from __future__ import annotations

import json

from fxagent.agents.gateway import Gateway, Prompt, ProviderConfig
from fxagent.agents.narrate import AGENTS, TEMPLATE_PROVIDER, narrate
from fxagent.dashboard.contract import CHARTIST, HISTORIAN, RISK_OFFICER, read_agents
from tests.agents.builders import analogue, fired_briefing

PROVIDER = ProviderConfig(
    name="alpha", model="alpha/m", api_key_env="ALPHA_KEY", min_interval_seconds=0
)
ENV = {"ALPHA_KEY": "k"}

GOOD_CHARTIST = json.dumps(
    {
        "read": "The measured trend and the chosen direction agree.",
        "patterns_seen": [],
        "terms_explained": [{"term": "ADX", "definition": "how strongly it is trending"}],
        "disagreement": None,
    }
)
GOOD_HISTORIAN = json.dumps(
    {"text": "The nearest retrieved window resolved at its target.", "analogue_ids": [1]}
)
GOOD_RISK = json.dumps(
    {
        "plan_summary": "A long, already sized and stopped.",
        "risk_flags": [{"flag": "Nothing unusual in the record.", "severity": "INFO"}],
        "size_rationale": "Risk over stop distance, rounded down.",
        "proceed_recommendation": "PROCEED",
        "reasoning": "The record supports the plan.",
    }
)


class ByAgentTransport:
    """Returns a canned response per agent name, so both agents can be scripted separately."""

    def __init__(self, **responses: str) -> None:
        self._responses = responses
        self.seen: list[tuple[str, str]] = []

    async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
        self.seen.append((prompt.agent, prompt.user))
        return self._responses.get(prompt.agent, "{}")


async def _nap(_seconds: float) -> None: ...


def _gateway(transport: object) -> Gateway:
    return Gateway((PROVIDER,), transport=transport, env=ENV, sleep=_nap)  # type: ignore[arg-type]


async def test_a_valid_response_is_used_and_stamped_with_who_wrote_it() -> None:
    transport = ByAgentTransport(chartist=GOOD_CHARTIST, historian=GOOD_HISTORIAN)
    briefing = fired_briefing(analogues=(analogue(1),))

    blocks = await narrate(briefing, gateway=_gateway(transport))

    assert blocks[CHARTIST]["provider"] == "alpha"
    assert blocks[CHARTIST]["model"] == "alpha/m"
    assert blocks[CHARTIST]["text"].startswith("The measured trend")
    assert blocks[CHARTIST]["terms_explained"][0]["term"] == "ADX"
    assert blocks[CHARTIST]["disagreement"] is None
    # The agent's own field name never reaches the stored document; `text` is what the panel
    # renders and the only narration key either path writes.
    assert "read" not in blocks[CHARTIST]
    assert blocks[HISTORIAN]["provider"] == "alpha"
    assert blocks[HISTORIAN]["analogue_ids"] == [1]


async def test_the_agents_are_asked_one_at_a_time_in_the_order_the_registry_lists() -> None:
    transport = ByAgentTransport(
        chartist=GOOD_CHARTIST, historian=GOOD_HISTORIAN, risk_officer=GOOD_RISK
    )
    briefing = fired_briefing()

    await narrate(briefing, gateway=_gateway(transport))

    assert [agent for agent, _ in transport.seen] == [spec.name for spec in AGENTS]


async def test_only_the_agent_that_opted_in_is_shown_what_the_others_said() -> None:
    """Two independent readings, not one with a preamble — and a reviewer that sees both."""
    transport = ByAgentTransport(
        chartist=GOOD_CHARTIST, historian=GOOD_HISTORIAN, risk_officer=GOOD_RISK
    )
    briefing = fired_briefing()

    await narrate(briefing, gateway=_gateway(transport))

    shown = dict(transport.seen)
    assert shown[CHARTIST] == briefing.rendered()
    assert shown[HISTORIAN] == briefing.rendered()

    reviewed = json.loads(shown[RISK_OFFICER])
    assert [note["agent"] for note in reviewed["agent_notes"]] == [CHARTIST, HISTORIAN]
    assert reviewed["agent_notes"][0]["text"].startswith("The measured trend")


async def test_an_invented_number_sends_that_agent_back_to_the_template() -> None:
    """And only that agent. One bad narration must not cost the other one its LLM answer."""
    invented = json.dumps({"read": "Watch 1.2345 closely."})
    transport = ByAgentTransport(chartist=invented, historian=GOOD_HISTORIAN)

    blocks = await narrate(fired_briefing(analogues=(analogue(1),)), gateway=_gateway(transport))

    assert blocks[CHARTIST]["provider"] == TEMPLATE_PROVIDER
    assert blocks[HISTORIAN]["provider"] == "alpha"


async def test_a_response_that_is_not_json_falls_back() -> None:
    transport = ByAgentTransport(chartist="I think you should buy EURUSD here.")

    blocks = await narrate(fired_briefing(), gateway=_gateway(transport))

    assert blocks[CHARTIST]["provider"] == TEMPLATE_PROVIDER


async def test_a_recommendation_field_the_schema_never_asked_for_is_discarded() -> None:
    """Hard rule 4 in the shape it would actually arrive in: an agent volunteering a decision."""
    overreaching = json.dumps({"read": "The trend supports it.", "proceed": "YES"})
    transport = ByAgentTransport(chartist=overreaching)

    blocks = await narrate(fired_briefing(), gateway=_gateway(transport))

    assert blocks[CHARTIST]["provider"] == TEMPLATE_PROVIDER
    assert "proceed" not in blocks[CHARTIST]


async def test_an_llm_written_block_is_readable_by_the_dashboard() -> None:
    transport = ByAgentTransport(chartist=GOOD_CHARTIST, historian=GOOD_HISTORIAN)

    blocks = await narrate(fired_briefing(analogues=(analogue(1),)), gateway=_gateway(transport))
    parsed = read_agents({"agents": blocks})

    assert parsed.discarded == ()
    assert parsed.chartist is not None and parsed.chartist.provider == "alpha"
    assert parsed.historian is not None
    assert len(parsed.analogues) == 1


async def test_every_prompt_states_the_two_rules_that_matter() -> None:
    """Enforced downstream regardless — this only keeps the discard rate down."""
    for spec in AGENTS:
        assert "Never state a number that does not appear in the input" in spec.system
        # Phrased for the job: the first two are told not to recommend a trade, and the risk
        # officer — which is reviewing one — is told not to propose a different one.
        assert (
            "Never recommend a trade" in spec.system or "Never propose a different" in spec.system
        )


async def test_backtest_mode_prompts_carry_no_date() -> None:
    transport = ByAgentTransport(chartist=GOOD_CHARTIST, historian=GOOD_HISTORIAN)
    briefing = fired_briefing(analogues=(analogue(1),))

    await narrate(briefing, gateway=_gateway(transport), include_dates=False)

    for _, user in transport.seen:
        assert "2026" not in user
        assert "2025" not in user


async def test_each_agent_asks_the_provider_claude_md_pins_it_to() -> None:
    by_name = {spec.name: spec for spec in AGENTS}

    assert by_name[CHARTIST].prefer == ("groq",)
    assert by_name[HISTORIAN].prefer == ("gemini",)
    # The risk officer's pin is a ladder: three models on the one NVIDIA endpoint, because a
    # developer reported it returning 500s and a single rung would make that a total outage.
    assert by_name[RISK_OFFICER].prefer == (
        "nvidia_deepseek",
        "nvidia_nemotron",
        "nvidia_glm",
    )


async def test_the_pin_reaches_the_gateway_as_a_preference_on_the_prompt() -> None:
    seen: list[tuple[str, tuple[str, ...]]] = []

    class PreferenceRecordingTransport:
        async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
            seen.append((prompt.agent, prompt.prefer))
            return {CHARTIST: GOOD_CHARTIST, HISTORIAN: GOOD_HISTORIAN}.get(prompt.agent, GOOD_RISK)

    await narrate(
        fired_briefing(analogues=(analogue(1),)), gateway=_gateway(PreferenceRecordingTransport())
    )

    assert seen == [
        (CHARTIST, ("groq",)),
        (HISTORIAN, ("gemini",)),
        (RISK_OFFICER, ("nvidia_deepseek", "nvidia_nemotron", "nvidia_glm")),
    ]


async def test_the_detected_formations_are_written_under_the_panels_reserved_key() -> None:
    """Separate from the narration writer: a formation is a fact whether or not an agent ran."""
    from fxagent.agents.narrate import attach_patterns
    from fxagent.dashboard.contract import PATTERNS_KEY, read_patterns
    from fxagent.patterns import CONTEXT_ONLY
    from tests.agents.builders import pattern

    briefing = fired_briefing(patterns=(pattern("doji"), pattern("pin_bar")))
    document = attach_patterns({"fired": True}, briefing)

    notes, discarded = read_patterns(document)

    assert document["fired"] is True
    assert len(document[PATTERNS_KEY]) == 2
    assert discarded == ()
    assert [note.name for note in notes] == ["doji", "pin_bar"]
    assert all(note.definition for note in notes)
    assert all(note.label == CONTEXT_ONLY for note in notes)


async def test_attaching_patterns_leaves_the_diagnostics_it_was_given_alone() -> None:
    from fxagent.agents.narrate import attach_patterns

    diagnostics = {"fired": False}
    attach_patterns(diagnostics, fired_briefing())

    assert diagnostics == {"fired": False}
