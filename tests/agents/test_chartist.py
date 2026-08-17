"""The chartist: what it is shown, what it may say back, and what it can never do.

Three claims are worth testing here and each has its own section: the input is structured data
and nothing else, the formations it reports are the ones a detector found and are labelled
context, and the deterministic fallback produces the same card the LLM path does.
"""

from __future__ import annotations

import json

from fxagent.agents import chartist
from fxagent.agents.gateway import Gateway, Prompt, ProviderConfig
from fxagent.agents.narrate import TEMPLATE_PROVIDER, narrate
from fxagent.agents.schemas import MAX_READ_CHARS, validate_note
from fxagent.dashboard.contract import CHARTIST, read_agents
from fxagent.patterns import CONTEXT_ONLY
from tests.agents.builders import INDICATORS, declined_briefing, fired_briefing, pattern

PROVIDER = ProviderConfig(
    name="alpha", model="alpha/m", api_key_env="ALPHA_KEY", min_interval_seconds=0
)


class Canned:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[Prompt] = []

    async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
        self.prompts.append(prompt)
        return self._response


async def _nap(_seconds: float) -> None: ...


def _gateway(transport: object) -> Gateway:
    return Gateway(
        (PROVIDER,),
        transport=transport,  # type: ignore[arg-type]
        env={"ALPHA_KEY": "k"},
        sleep=_nap,
    )


# -- structured data only --------------------------------------------------------------------


async def test_the_prompt_carries_the_computed_record_and_no_price_series() -> None:
    """CLAUDE.md's row for this agent: structured core output, never pixels or raw bars."""
    transport = Canned(json.dumps({"read": "The regime is trending."}))
    briefing = fired_briefing(patterns=(pattern("doji"),), indicators=INDICATORS)

    await narrate(briefing, gateway=_gateway(transport), agents=(chartist.SPEC,))

    sent = json.loads(transport.prompts[0].user)
    assert set(sent) >= {"regime", "votes", "decision", "indicators", "patterns", "plan"}
    # Session state and each vote's weight are there; a bar is not, and cannot be.
    assert sent["regime"]["sessions"] == ["LONDON"]
    assert {vote["strategy"]: vote["weight"] for vote in sent["votes"]}["range_reversion"] == 0.0
    assert {"bars", "ohlc", "open", "high", "low", "close"}.isdisjoint(sent)


async def test_the_system_prompt_states_the_rules_that_are_also_enforced() -> None:
    """The prompt only lowers the discard rate. Every rule below is checked downstream too."""
    system = chartist.SYSTEM

    assert "never see a chart image" in system
    assert "Never recommend a trade" in system
    assert "Never state a number that does not appear in the input" in system
    assert "Only name a formation that appears under `patterns`" in system
    assert "no net positive return on EUR/USD after costs" in system


# -- formations are context ------------------------------------------------------------------


async def test_a_narrated_formation_must_be_one_the_detector_found() -> None:
    invented = json.dumps(
        {
            "read": "A morning star printed here.",
            "patterns_seen": [
                {"name": "morning_star", "meaning": "three bars", "significance": "context"}
            ],
        }
    )
    transport = Canned(invented)
    briefing = fired_briefing(patterns=(pattern("doji"),))

    blocks = await narrate(briefing, gateway=_gateway(transport), agents=(chartist.SPEC,))

    assert blocks[CHARTIST]["provider"] == TEMPLATE_PROVIDER


async def test_an_llm_written_card_keeps_the_context_label_on_every_formation() -> None:
    answered = json.dumps(
        {
            "read": "A doji printed on this bar.",
            "patterns_seen": [
                {"name": "doji", "meaning": "almost no body", "significance": "often quoted"}
            ],
        }
    )
    briefing = fired_briefing(patterns=(pattern("doji"),))

    blocks = await narrate(briefing, gateway=_gateway(Canned(answered)), agents=(chartist.SPEC,))

    assert blocks[CHARTIST]["provider"] == "alpha"
    assert blocks[CHARTIST]["patterns_seen"][0]["label"] == CONTEXT_ONLY


def test_the_note_type_has_no_field_a_score_could_be_read_out_of() -> None:
    """`disagreement` is prose and everything else is prose or a name. Nothing to total up."""
    from fxagent.agents.schemas import ChartistNote

    numeric = {
        name
        for name, field in ChartistNote.model_fields.items()
        if field.annotation in (int, float, "int", "float")
    }

    assert numeric == set()


# -- the fallback ----------------------------------------------------------------------------


def test_the_fallback_produces_the_same_shape_the_llm_path_does() -> None:
    """A card must not lose its structured half because a free-tier API was unreachable."""
    block = chartist.fallback(fired_briefing(patterns=(pattern("doji"),)))

    assert set(block) == {"text", "patterns_seen", "terms_explained", "disagreement"}
    assert block["patterns_seen"][0]["name"] == "doji"
    assert block["patterns_seen"][0]["label"] == CONTEXT_ONLY
    assert block["disagreement"] is None


def test_the_fallback_read_fits_the_length_its_schema_allows() -> None:
    """Trimmed by whole sentences. A card clipped mid-number reads as a real number."""
    for briefing in (
        fired_briefing(),
        declined_briefing(),
        fired_briefing(patterns=(pattern("doji"), pattern("pin_bar"))),
    ):
        read = chartist.fallback(briefing)["text"]

        assert len(read) <= MAX_READ_CHARS
        assert read.endswith("Written from the computed record, not by a model.")


def test_the_fallback_defines_the_jargon_it_used_and_nothing_it_did_not() -> None:
    terms = {item["term"] for item in chartist.fallback(fired_briefing())["terms_explained"]}

    assert "ADX" in terms
    assert terms <= set(chartist.GLOSSARY)
    read = chartist.fallback(fired_briefing())["text"]
    for term in terms:
        assert term.lower() in read.lower()


def test_the_fallback_never_claims_a_formation_predicts_anything() -> None:
    """The evidence says they do not, and the deterministic path is where that must not slip."""
    seen = chartist.fallback(fired_briefing(patterns=(pattern("doji"),)))["patterns_seen"]

    assert "no net positive return" in seen[0]["significance"]
    assert "reversal" not in seen[0]["significance"].lower()


def test_the_fallback_card_is_readable_by_the_dashboard() -> None:
    blocks = {CHARTIST: {**chartist.fallback(fired_briefing()), "provider": TEMPLATE_PROVIDER}}

    parsed = read_agents({"agents": blocks})

    assert parsed.discarded == ()
    assert parsed.chartist is not None and parsed.chartist.text


def test_a_fallback_card_satisfies_the_schema_the_llm_path_is_held_to() -> None:
    """Deliberately not enforced at runtime — the floor must not be able to fail — but true.

    If the template drifted out of the shape the LLM path produces, the panel would render two
    different cards depending on whether a provider answered, and only one of them would have
    been designed. Checking it here keeps them one card without putting a validator on the
    path whose whole job is to have no way of failing.
    """
    briefing = fired_briefing(patterns=(pattern("doji"),))
    block = chartist.fallback(briefing)

    # The only difference is the narration key: the fallback writes the serialised name, the
    # schema validates the field name. Everything else must pass unchanged, grounding included.
    as_response = {key: value for key, value in block.items() if key != "text"}
    as_response["read"] = block["text"]

    assert validate_note(chartist.SPEC.note, as_response, briefing) is not None
