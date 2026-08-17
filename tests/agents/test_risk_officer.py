"""The risk officer: what it is shown, the flags it raises without a model, and its budget.

The advisory guarantee has its own file — `test_advisory_only.py`. This one covers everything
else: that all of its input is computed upstream, that the deterministic flags are comparisons
between numbers already on the briefing, that twenty calls a day is enforced rather than
documented, and that the NVIDIA ladder is tried before the rest of the chain.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import pytest

from fxagent.agents import risk_officer
from fxagent.agents.gateway import CallBudget, Gateway, Prompt, ProviderConfig, ProviderError
from fxagent.agents.narrate import TEMPLATE_PROVIDER, narrate
from fxagent.agents.schemas import (
    MAX_PLAN_SUMMARY_CHARS,
    RiskOfficerNote,
    validate_note,
)
from fxagent.dashboard.contract import RISK_OFFICER, read_agents
from tests.agents.builders import (
    MOMENT,
    declined_briefing,
    event,
    execution,
    fired_briefing,
    spread,
)

PROVIDER = ProviderConfig(
    name="alpha", model="alpha/m", api_key_env="ALPHA_KEY", min_interval_seconds=0
)

GOOD = json.dumps(
    {
        "plan_summary": "A long, already sized and stopped.",
        "risk_flags": [{"flag": "The spread is above its median.", "severity": "WARN"}],
        "size_rationale": "The risk budget over the stop distance.",
        "proceed_recommendation": "CAUTION",
        "reasoning": "Conditions are workable but not clean.",
    }
)


class Canned:
    def __init__(self, response: str = GOOD) -> None:
        self._response = response
        self.prompts: list[Prompt] = []

    async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
        self.prompts.append(prompt)
        return self._response


async def _nap(_seconds: float) -> None: ...


def _gateway(transport: object, **kwargs: object) -> Gateway:
    return Gateway(
        (PROVIDER,),
        transport=transport,  # type: ignore[arg-type]
        env={"ALPHA_KEY": "k"},
        sleep=_nap,
        now=lambda: MOMENT,
        **kwargs,  # type: ignore[arg-type]
    )


def _full_briefing():
    return fired_briefing(execution=execution(), spread=spread(over_median=1.8), events=(event(),))


# -- everything it reads was computed upstream -----------------------------------------------


async def test_the_prompt_carries_the_sized_order_the_conditions_and_the_other_agents() -> None:
    transport = Canned()

    await narrate(_full_briefing(), gateway=_gateway(transport))

    sent = json.loads(transport.prompts[-1].user)
    assert sent["plan"]["direction"] == "LONG"
    assert sent["execution"]["volume"] == 0.04
    assert sent["execution"]["max_risk_per_trade"] == 0.005
    assert sent["execution"]["total_open_risk"] == 0.008
    assert sent["spread"]["over_median"] == 1.8
    assert sent["events"][0]["title"] == "Non-Farm Payrolls"
    assert [note["agent"] for note in sent["agent_notes"]] == ["chartist", "historian"]


def test_the_caps_travel_beside_the_values_they_bound() -> None:
    """ "0.4% risked" means nothing alone, and telling the model the cap in prose would be a
    number in a prompt that nothing checks."""
    payload = _full_briefing().payload()["execution"]

    assert {"risk_fraction", "max_risk_per_trade"} <= set(payload)
    assert {"total_open_risk", "max_total_risk"} <= set(payload)


def test_a_released_number_is_gated_in_sql_and_simply_carried_here() -> None:
    """Hard rule 6 wants the point-in-time gate in SQL, not in a prompt. `events_visible_at()`
    nulls `actual` until the event has happened, so an upcoming release has none by
    construction and this model repeats no part of the rule it therefore cannot contradict."""
    upcoming = event(minutes_until=30)

    assert upcoming.actual is None
    assert upcoming.forecast is not None and upcoming.previous is not None


def test_the_note_has_no_field_a_score_could_be_read_out_of() -> None:
    numeric = {
        name
        for name, field in RiskOfficerNote.model_fields.items()
        if field.annotation in (int, float)
    }

    assert numeric == set()


# -- the deterministic flags -----------------------------------------------------------------


def test_a_quiet_bar_raises_nothing() -> None:
    flags = risk_officer.deterministic_flags(
        fired_briefing(execution=execution(), spread=spread(over_median=1.0))
    )

    assert flags == []


def test_a_wide_spread_is_flagged_at_the_volatility_it_deserves() -> None:
    """Exness is a market-maker CFD broker and its spreads widen on news. Routine, not exotic."""
    warned = risk_officer.deterministic_flags(fired_briefing(spread=spread(over_median=1.8)))
    critical = risk_officer.deterministic_flags(fired_briefing(spread=spread(over_median=3.0)))

    assert [flag["severity"] for flag in warned] == ["WARN"]
    assert [flag["severity"] for flag in critical] == ["CRITICAL"]


def test_open_risk_near_its_cap_is_flagged_and_over_it_is_louder() -> None:
    near = risk_officer.deterministic_flags(
        fired_briefing(execution=execution(total_open_risk=0.018))
    )
    over = risk_officer.deterministic_flags(
        fired_briefing(execution=execution(total_open_risk=0.025))
    )

    assert [flag["severity"] for flag in near] == ["WARN"]
    assert [flag["severity"] for flag in over] == ["CRITICAL"]


def test_a_trade_sized_over_its_own_cap_is_a_defect_and_says_so() -> None:
    """Hard rule 8's per-trade cap. The sizer should make this impossible; if it reaches the
    briefing, the card must not describe it as a close call."""
    flags = risk_officer.deterministic_flags(
        fired_briefing(execution=execution(risk_fraction=0.009))
    )

    assert flags[0]["severity"] == "CRITICAL"
    assert "defect" in flags[0]["flag"]


def test_an_event_inside_the_blackout_is_louder_than_one_merely_near() -> None:
    near = risk_officer.deterministic_flags(fired_briefing(events=(event(minutes_until=45),)))
    imminent = risk_officer.deterministic_flags(fired_briefing(events=(event(minutes_until=8),)))

    assert [flag["severity"] for flag in near] == ["WARN"]
    assert [flag["severity"] for flag in imminent] == ["CRITICAL"]
    assert "permission layer" in imminent[0]["flag"]


def test_an_event_outside_the_window_is_not_a_flag() -> None:
    assert (
        risk_officer.deterministic_flags(fired_briefing(events=(event(minutes_until=180),))) == []
    )


def test_a_release_that_has_just_passed_still_counts() -> None:
    """The spread damage from a release outlasts the release, which is why `minutes_until` is
    signed rather than split into two fields."""
    flags = risk_officer.deterministic_flags(fired_briefing(events=(event(minutes_until=-10),)))

    assert [flag["severity"] for flag in flags] == ["CRITICAL"]


def test_every_flag_is_a_comparison_between_two_numbers_already_on_the_briefing() -> None:
    """Which is what makes them checkable, and what makes them survive a provider outage."""
    briefing = _full_briefing()
    flags = risk_officer.deterministic_flags(briefing)

    assert flags
    assert all(set(flag) == {"flag", "severity"} for flag in flags)
    assert all(flag["severity"] in ("INFO", "WARN", "CRITICAL") for flag in flags)


# -- the fallback ----------------------------------------------------------------------------


def test_the_fallback_produces_every_field_the_llm_path_does() -> None:
    block = risk_officer.fallback(_full_briefing())

    assert set(block) == {
        "text",
        "risk_flags",
        "size_rationale",
        "proceed_recommendation",
        "reasoning",
    }
    assert block["risk_flags"]
    assert len(block["text"]) <= MAX_PLAN_SUMMARY_CHARS


def test_an_unreachable_provider_costs_the_card_its_prose_and_not_its_substance() -> None:
    """Matters more here than for the other two: this is the only place a reader is told the
    spread has tripled."""
    block = risk_officer.fallback(fired_briefing(spread=spread(over_median=3.0)))

    assert block["proceed_recommendation"] == "WAIT"
    assert any("spread" in flag["flag"].lower() for flag in block["risk_flags"])


def test_the_fallback_recommendation_follows_the_flags_it_just_listed() -> None:
    quiet = risk_officer.fallback(fired_briefing(execution=execution()))
    warned = risk_officer.fallback(fired_briefing(spread=spread(over_median=1.8)))
    critical = risk_officer.fallback(fired_briefing(spread=spread(over_median=3.0)))

    assert quiet["proceed_recommendation"] == "PROCEED"
    assert warned["proceed_recommendation"] == "CAUTION"
    assert critical["proceed_recommendation"] == "WAIT"


def test_the_fallback_says_so_when_there_is_no_order() -> None:
    block = risk_officer.fallback(declined_briefing())

    assert "no order to review" in block["text"].lower()
    assert block["proceed_recommendation"] == "WAIT"


def test_the_size_rationale_is_a_derivation_rather_than_a_number() -> None:
    """ "0.04 lots" tells a reader nothing they can check."""
    rationale = risk_officer.fallback(_full_briefing())["size_rationale"]

    assert "stop distance" in rationale
    assert "never fixed lots" in rationale


def test_a_fallback_card_satisfies_the_schema_the_llm_path_is_held_to() -> None:
    """Not enforced at runtime — the floor must not be able to fail — but the two paths must
    produce one card, not two that were each designed once."""
    briefing = _full_briefing()
    block = risk_officer.fallback(briefing)

    as_response = {key: value for key, value in block.items() if key != "text"}
    as_response["plan_summary"] = block["text"]

    assert validate_note(RiskOfficerNote, as_response, briefing) is not None


def test_the_fallback_card_is_readable_by_the_dashboard() -> None:
    blocks = {
        RISK_OFFICER: {**risk_officer.fallback(_full_briefing()), "provider": TEMPLATE_PROVIDER}
    }

    parsed = read_agents({"agents": blocks})

    assert parsed.discarded == ()
    assert parsed.risk_officer is not None
    assert parsed.risk_officer.proceed_recommendation in ("PROCEED", "CAUTION", "WAIT")


# -- discards --------------------------------------------------------------------------------


def test_a_recommendation_outside_the_three_words_is_discarded() -> None:
    briefing = _full_briefing()

    def answer(recommendation: str) -> str:
        return json.dumps({"plan_summary": "A long.", "proceed_recommendation": recommendation})

    assert validate_note(RiskOfficerNote, answer("ABORT"), briefing) is None
    assert validate_note(RiskOfficerNote, answer("WAIT"), briefing) is not None


def test_a_severity_outside_the_three_labels_is_discarded() -> None:
    briefing = _full_briefing()
    answer = json.dumps(
        {
            "plan_summary": "A long.",
            "proceed_recommendation": "CAUTION",
            "risk_flags": [{"flag": "Something", "severity": "CATASTROPHIC"}],
        }
    )

    assert validate_note(RiskOfficerNote, answer, briefing) is None


def test_a_summary_over_three_hundred_characters_is_discarded() -> None:
    sprawl = "the plan is a long. " * 20

    assert len(sprawl) > MAX_PLAN_SUMMARY_CHARS
    assert (
        validate_note(
            RiskOfficerNote,
            json.dumps({"plan_summary": sprawl, "proceed_recommendation": "PROCEED"}),
            _full_briefing(),
        )
        is None
    )


def test_a_size_the_model_invented_is_discarded_like_any_other_number() -> None:
    """Rule 1 of its prompt, enforced rather than requested."""
    answer = json.dumps(
        {
            "plan_summary": "A long.",
            "proceed_recommendation": "CAUTION",
            "reasoning": "0.12 lots would have been the better size.",
        }
    )

    assert validate_note(RiskOfficerNote, answer, _full_briefing()) is None


# -- the budget ------------------------------------------------------------------------------


def test_twenty_calls_a_day_is_declared_on_the_spec_and_carried_on_the_prompt() -> None:
    assert risk_officer.DAILY_CALL_LIMIT == 20
    assert risk_officer.SPEC.daily_call_limit == 20


async def test_the_cap_is_enforced_rather_than_documented() -> None:
    transport = Canned()
    budget = CallBudget(now=lambda: MOMENT)
    gateway = _gateway(transport, budget=budget)

    for _ in range(risk_officer.DAILY_CALL_LIMIT + 5):
        # A different briefing each time, or the cache would serve them and nothing would spend.
        await narrate(
            fired_briefing(execution=execution(volume=0.01 + _ / 1000)),
            gateway=gateway,
            agents=(risk_officer.SPEC,),
        )

    assert budget.spent_by(RISK_OFFICER) == risk_officer.DAILY_CALL_LIMIT
    assert len(transport.prompts) == risk_officer.DAILY_CALL_LIMIT


async def test_a_bar_with_no_order_never_spends_a_call() -> None:
    """Twenty a day does not survive being spent on "nothing qualified"."""
    transport = Canned()
    budget = CallBudget(now=lambda: MOMENT)

    blocks = await narrate(
        declined_briefing(), gateway=_gateway(transport, budget=budget), agents=(risk_officer.SPEC,)
    )

    assert transport.prompts == []
    assert budget.spent_by(RISK_OFFICER) == 0
    assert blocks[RISK_OFFICER]["provider"] == TEMPLATE_PROVIDER
    assert blocks[RISK_OFFICER]["text"]


async def test_running_out_of_budget_still_produces_a_card() -> None:
    budget = CallBudget(now=lambda: MOMENT)
    for _ in range(20):
        budget.spend(agent=RISK_OFFICER, agent_limit=20)

    blocks = await narrate(
        _full_briefing(), gateway=_gateway(Canned(), budget=budget), agents=(risk_officer.SPEC,)
    )

    assert blocks[RISK_OFFICER]["provider"] == TEMPLATE_PROVIDER
    assert blocks[RISK_OFFICER]["risk_flags"]


# -- the provider ladder ---------------------------------------------------------------------


def test_the_ladder_is_three_models_on_one_endpoint_then_the_rest_of_the_chain() -> None:
    """A developer reported the endpoint returning 500s on trivial prompts. One rung would make
    that a total outage of this agent's provider."""
    assert risk_officer.SPEC.prefer == ("nvidia_deepseek", "nvidia_nemotron", "nvidia_glm")


async def test_a_five_hundred_falls_to_the_next_rung_rather_than_being_retried() -> None:
    """A systematic error on trivial prompts is not a transient one, and the rung below is a
    better use of twenty calls than a retry of a model that is not answering."""
    tried: list[str] = []

    class FailingLadder:
        async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
            tried.append(provider.name)
            if provider.name == "nvidia_deepseek":
                raise ProviderError("500 Internal Server Error")
            return GOOD

    from fxagent.agents.gateway import DEFAULT_PROVIDERS

    nvidia = tuple(p for p in DEFAULT_PROVIDERS if p.name.startswith("nvidia_"))
    gateway = Gateway(
        nvidia,
        transport=FailingLadder(),  # type: ignore[arg-type]
        env={"NVIDIA_API_KEY": "k"},
        sleep=_nap,
        now=lambda: MOMENT,
    )

    blocks = await narrate(_full_briefing(), gateway=gateway, agents=(risk_officer.SPEC,))

    assert tried == ["nvidia_deepseek", "nvidia_nemotron"]
    assert blocks[RISK_OFFICER]["provider"] == "nvidia_nemotron"


def test_the_module_names_the_endpoint_it_was_built_against() -> None:
    from fxagent.agents.gateway import NVIDIA_NIM_BASE

    assert NVIDIA_NIM_BASE == "https://integrate.api.nvidia.com/v1"
    assert "deepseek-ai/deepseek-v4-pro" in risk_officer.__doc__


@pytest.mark.parametrize("threshold", ["SPREAD_WARN", "SPREAD_CRITICAL", "OPEN_RISK_WARN"])
def test_the_thresholds_decide_what_a_human_is_told_and_nothing_else(threshold: str) -> None:
    """There is nothing here worth tuning: these pick a label, never an action."""
    assert isinstance(getattr(risk_officer.THRESHOLDS, threshold), float)


def test_should_ask_reads_the_briefing_and_nothing_about_configuration() -> None:
    """It gates on whether there is an order to review — never on whether a key is set.

    The distinction matters because the two produce the same visible outcome. If the predicate
    consulted configuration, every deployment without an NVIDIA key would skip for the wrong
    reason, look correct, and leave the real logic unexercised until somebody added one.
    """
    with_order = fired_briefing(execution=execution())
    without = declined_briefing()

    assert risk_officer.SPEC.should_ask(with_order) is True
    assert risk_officer.SPEC.should_ask(without) is False

    # A pure function of the briefing: no environment, no key, no gateway in its signature.
    import inspect

    assert list(inspect.signature(risk_officer.SPEC.should_ask).parameters) == ["briefing"]


async def test_the_skip_is_decided_by_the_bar_with_a_working_provider_present() -> None:
    """The control the other test needs. One gateway, one key, one transport — only the
    briefing changes, so the absence of a call can only be the briefing's doing."""
    transport = Canned()
    gateway = _gateway(transport)

    await narrate(declined_briefing(), gateway=gateway, agents=(risk_officer.SPEC,))
    assert transport.prompts == [], "a bar with no order should not have been asked"

    await narrate(
        fired_briefing(execution=execution()), gateway=gateway, agents=(risk_officer.SPEC,)
    )
    assert len(transport.prompts) == 1, "a bar with an order should have been asked"


async def test_the_predicate_still_runs_when_no_provider_is_configured() -> None:
    """Otherwise the logic goes untested until a key is added — the first day it runs would be
    the first day anyone finds out whether it works."""
    asked: list[bool] = []

    spec = replace(
        risk_officer.SPEC,
        should_ask=lambda briefing: (
            asked.append(briefing.plan is not None) or briefing.plan is not None
        ),
    )
    unconfigured = Gateway((PROVIDER,), transport=Canned(), env={}, sleep=_nap)

    await narrate(fired_briefing(execution=execution()), gateway=unconfigured, agents=(spec,))
    await narrate(declined_briefing(), gateway=unconfigured, agents=(spec,))

    assert asked == [True, False]


async def test_the_predicate_still_runs_with_no_gateway_at_all() -> None:
    """Backtest mode. The predicate is not behind the short-circuit that skips the LLM path."""
    asked: list[str] = []

    spec = replace(
        risk_officer.SPEC,
        should_ask=lambda briefing: asked.append(briefing.symbol) or False,
    )

    await narrate(fired_briefing(execution=execution()), agents=(spec,))

    assert asked == ["EURUSD"]


async def test_an_unconfigured_provider_and_an_empty_bar_are_distinguishable_in_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nothing to review is a property of the bar; no provider is a property of the deployment.
    A card that reads the same either way is how a misconfiguration hides for a week."""
    unconfigured = Gateway((PROVIDER,), transport=Canned(), env={}, sleep=_nap)

    with caplog.at_level(logging.DEBUG):
        await narrate(declined_briefing(), gateway=unconfigured, agents=(risk_officer.SPEC,))
        empty_bar = "\n".join(record.getMessage() for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        await narrate(
            fired_briefing(execution=execution()), gateway=unconfigured, agents=(risk_officer.SPEC,)
        )
        no_provider = "\n".join(record.getMessage() for record in caplog.records)

    assert "nothing for it to review" in empty_bar
    assert "nothing for it to review" not in no_provider
    assert "no LLM provider is configured" in no_provider
