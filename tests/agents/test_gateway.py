"""The provider chain: order, backoff, the daily ceiling, the cache, and what never gets logged.

Every provider here is fictional and every response is a stub. The gateway's job is routing and
restraint, and neither needs a real model to be wrong.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest

from fxagent.agents.gateway import (
    DEFAULT_DAILY_CALL_LIMIT,
    DEFAULT_PROVIDERS,
    NVIDIA_NIM_BASE,
    CallBudget,
    Completion,
    DailyCallLimitReached,
    Gateway,
    Prompt,
    PromptCache,
    ProviderConfig,
    ProviderError,
    RateLimited,
    RetryPolicy,
    providers_from_env,
)

ALPHA = ProviderConfig(
    name="alpha", model="alpha/m", api_key_env="ALPHA_KEY", min_interval_seconds=0
)
BETA = ProviderConfig(name="beta", model="beta/m", api_key_env="BETA_KEY", min_interval_seconds=0)
GAMMA = ProviderConfig(
    name="gamma", model="gamma/m", api_key_env="GAMMA_KEY", min_interval_seconds=0
)
CHAIN = (ALPHA, BETA, GAMMA)
ENV = {"ALPHA_KEY": "k", "BETA_KEY": "k", "GAMMA_KEY": "k"}

MOMENT = datetime(2026, 1, 12, 9, 0, tzinfo=UTC)


class ScriptedTransport:
    """Answers each provider according to `script`; anything unlisted returns its own name."""

    def __init__(self, script: dict[str, object] | None = None) -> None:
        self._script = script or {}
        self.calls: list[str] = []

    async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
        self.calls.append(provider.name)
        outcome = self._script.get(provider.name)
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if outcome else None
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome) if outcome is not None else f"answer from {provider.name}"


def _gateway(transport: object, **kwargs: object) -> Gateway:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    gateway = Gateway(
        CHAIN,
        transport=transport,  # type: ignore[arg-type]
        env=ENV,
        sleep=kwargs.pop("sleep", sleep),  # type: ignore[arg-type]
        jitter=kwargs.pop("jitter", lambda _low, high: high),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )
    gateway.slept = slept  # type: ignore[attr-defined]
    return gateway


def prompt(user: str = "briefing") -> Prompt:
    return Prompt(agent="chartist", system="rules", user=user)


# -- the chain -----------------------------------------------------------------------------


async def test_the_first_configured_provider_serves_the_call() -> None:
    transport = ScriptedTransport()
    completion = await _gateway(transport).complete(prompt())

    assert completion is not None
    assert completion.provider == "alpha"
    assert completion.model == "alpha/m"
    assert transport.calls == ["alpha"]


async def test_a_failing_provider_falls_through_to_the_next_immediately() -> None:
    """Not a rate limit: there is nothing to wait for, so the next provider is tried at once."""
    transport = ScriptedTransport({"alpha": ProviderError("502")})
    gateway = _gateway(transport)

    completion = await gateway.complete(prompt())

    assert completion is not None and completion.provider == "beta"
    assert transport.calls == ["alpha", "beta"]
    assert gateway.slept == []  # type: ignore[attr-defined]


async def test_the_provider_that_served_the_call_is_logged_by_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = ScriptedTransport({"alpha": ProviderError("502"), "beta": ProviderError("502")})
    with caplog.at_level(logging.INFO):
        await _gateway(transport).complete(prompt())

    assert any("served by gamma" in record.getMessage() for record in caplog.records)


async def test_a_provider_with_no_key_is_skipped_without_being_called() -> None:
    transport = ScriptedTransport()
    gateway = Gateway(CHAIN, transport=transport, env={"BETA_KEY": "k"})

    completion = await gateway.complete(prompt())

    assert gateway.configured() == (BETA,)
    assert completion is not None and completion.provider == "beta"
    assert transport.calls == ["beta"]


async def test_no_provider_configured_returns_none_rather_than_raising() -> None:
    """The caller always has a template. An exception here would fail a pass over commentary."""
    gateway = Gateway(CHAIN, transport=ScriptedTransport(), env={})

    assert await gateway.complete(prompt()) is None


async def test_every_provider_failing_returns_none() -> None:
    transport = ScriptedTransport(dict.fromkeys(("alpha", "beta", "gamma"), ProviderError("down")))

    assert await _gateway(transport).complete(prompt()) is None
    assert transport.calls == ["alpha", "beta", "gamma"]


# -- rate limits ---------------------------------------------------------------------------


async def test_a_rate_limit_is_retried_on_the_same_provider_before_moving_on() -> None:
    transport = ScriptedTransport({"alpha": [RateLimited("429"), "recovered"]})
    gateway = _gateway(transport)

    completion = await gateway.complete(prompt())

    assert completion is not None
    assert completion.provider == "alpha"
    assert completion.text == "recovered"
    assert transport.calls == ["alpha", "alpha"]
    assert gateway.slept == [1.0]  # type: ignore[attr-defined]


async def test_the_backoff_grows_and_is_capped() -> None:
    transport = ScriptedTransport({"alpha": [RateLimited("429")] * 4})
    gateway = _gateway(
        transport, policy=RetryPolicy(attempts=4, base_delay_seconds=1.0, max_delay_seconds=5.0)
    )

    await gateway.complete(prompt())

    assert gateway.slept[:3] == [1.0, 3.0, 5.0]  # type: ignore[attr-defined]


async def test_a_rate_limit_that_outlasts_the_retries_moves_to_the_next_provider() -> None:
    transport = ScriptedTransport({"alpha": [RateLimited("429")] * 3})

    completion = await _gateway(transport).complete(prompt())

    assert completion is not None and completion.provider == "beta"
    assert transport.calls == ["alpha", "alpha", "alpha", "beta"]


async def test_the_providers_own_retry_after_is_never_undercut() -> None:
    """Full jitter can draw a short delay; a provider that named a wait is still obeyed."""
    transport = ScriptedTransport({"alpha": [RateLimited("429", retry_after=12.0), "ok"]})
    gateway = _gateway(transport, jitter=lambda _low, _high: 0.0)

    await gateway.complete(prompt())

    assert gateway.slept == [12.0]  # type: ignore[attr-defined]


# -- the daily ceiling ---------------------------------------------------------------------


def test_the_budget_refuses_the_call_past_the_limit() -> None:
    budget = CallBudget(daily_limit=2, now=lambda: MOMENT)
    budget.spend()
    budget.spend()

    assert budget.remaining_today == 0
    with pytest.raises(DailyCallLimitReached, match="hard daily cap of 2"):
        budget.spend()


def test_the_budget_rolls_over_at_utc_midnight() -> None:
    moment = MOMENT
    budget = CallBudget(daily_limit=1, now=lambda: moment)
    budget.spend()

    moment = MOMENT + timedelta(days=1)
    assert budget.spent_today == 0
    budget.spend()


def test_the_default_limit_is_the_one_claude_md_states() -> None:
    assert DEFAULT_DAILY_CALL_LIMIT == 50
    assert CallBudget().daily_limit == 50


async def test_an_exhausted_budget_refuses_and_logs_at_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ERROR because reaching fifty calls at this volume means a loop, and a loop must be found."""
    transport = ScriptedTransport()
    budget = CallBudget(daily_limit=1, now=lambda: MOMENT)
    budget.spend()

    with caplog.at_level(logging.ERROR):
        assert await _gateway(transport, budget=budget).complete(prompt()) is None

    assert transport.calls == []
    assert any(record.levelno == logging.ERROR for record in caplog.records)


async def test_falling_through_the_chain_spends_one_call_per_provider_tried() -> None:
    """It is outbound requests that burn a free tier, not logical calls."""
    transport = ScriptedTransport({"alpha": ProviderError("down"), "beta": ProviderError("down")})
    budget = CallBudget(daily_limit=10, now=lambda: MOMENT)

    await _gateway(transport, budget=budget).complete(prompt())

    assert budget.spent_today == 3


# -- cache ---------------------------------------------------------------------------------


async def test_an_identical_prompt_is_served_from_the_cache_without_a_second_call() -> None:
    transport = ScriptedTransport()
    budget = CallBudget(daily_limit=10, now=lambda: MOMENT)
    gateway = _gateway(transport, budget=budget)

    first = await gateway.complete(prompt())
    second = await gateway.complete(prompt())

    assert first is not None and second is not None
    assert second.text == first.text
    assert second.cached is True and first.cached is False
    assert transport.calls == ["alpha"]
    assert budget.spent_today == 1


async def test_a_different_briefing_is_a_different_key() -> None:
    transport = ScriptedTransport()
    gateway = _gateway(transport)

    await gateway.complete(prompt("one"))
    await gateway.complete(prompt("two"))

    assert transport.calls == ["alpha", "alpha"]


def test_the_cache_is_bounded() -> None:
    cache = PromptCache(max_entries=2)
    for index in range(3):
        cache.put(
            str(index),
            Completion(text="t", provider="alpha", model="alpha/m", generated_at=MOMENT),
        )

    assert len(cache) == 2
    assert cache.get("0") is None
    assert cache.get("2") is not None


# -- sequencing ----------------------------------------------------------------------------


async def test_two_callers_never_have_requests_in_flight_at_once() -> None:
    """RPM is the limit that bites. Fanning out across symbols is what trips it."""
    in_flight = 0
    peak = 0

    class OverlapDetectingTransport:
        async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return "ok"

    gateway = _gateway(OverlapDetectingTransport())
    await asyncio.gather(*(gateway.complete(prompt(f"symbol-{index}")) for index in range(5)))

    assert peak == 1


async def test_a_second_call_to_one_provider_waits_out_its_minimum_interval() -> None:
    clock = MOMENT
    spaced = ProviderConfig(
        name="alpha", model="alpha/m", api_key_env="ALPHA_KEY", min_interval_seconds=6.0
    )
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    transport = ScriptedTransport()
    gateway = Gateway(
        (spaced,), transport=transport, env=ENV, now=lambda: clock, sleep=sleep, cache=None
    )

    await gateway.complete(prompt("one"))
    clock = MOMENT + timedelta(seconds=2)
    await gateway.complete(prompt("two"))

    assert slept == [4.0]


# -- what never gets logged ----------------------------------------------------------------


async def test_prompt_content_never_reaches_a_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A briefing quotes plan levels. A log line outlives the request and gets pasted around."""
    secret = "STOP-AT-1.09800-ON-ACCOUNT-476187411"
    transport = ScriptedTransport(dict.fromkeys(("alpha", "beta", "gamma"), ProviderError("down")))

    with caplog.at_level(logging.DEBUG):
        await _gateway(transport).complete(Prompt(agent="chartist", system=secret, user=secret))

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert rendered  # the failures were logged at all
    assert secret not in rendered
    assert "476187411" not in rendered


# -- configuration -------------------------------------------------------------------------


def test_the_env_names_the_first_two_and_keeps_the_rest_behind_them() -> None:
    ordered = providers_from_env(
        {"LLM_PRIMARY_PROVIDER": "groq", "LLM_FALLBACK_PROVIDER": "nvidia_deepseek"}
    )

    assert [config.name for config in ordered][:3] == ["groq", "nvidia_deepseek", "gemini"]
    # Nothing is dropped by being left out of the two variables.
    assert len(ordered) == len(DEFAULT_PROVIDERS)


def test_an_unknown_provider_name_is_ignored_rather_than_emptying_the_chain() -> None:
    ordered = providers_from_env({"LLM_PRIMARY_PROVIDER": "typo"})

    assert [config.name for config in ordered] == [config.name for config in DEFAULT_PROVIDERS]


def test_the_default_chain_has_more_than_two_providers() -> None:
    """Built for N, and the names are distinct because the chain is ordered by them."""
    names = [config.name for config in DEFAULT_PROVIDERS]

    assert len(DEFAULT_PROVIDERS) >= 3
    assert set(names) >= {"gemini", "groq", "nvidia_deepseek"}
    assert len(set(names)) == len(names)
    assert len({config.model for config in DEFAULT_PROVIDERS}) == len(DEFAULT_PROVIDERS)


def test_the_nvidia_rungs_share_one_key_and_one_endpoint() -> None:
    """The reported 500s make a single NIM entry a single point of failure. Three rungs, one
    account: the fall-through the chain already implements, at no extra configuration."""
    nvidia = [config for config in DEFAULT_PROVIDERS if config.name.startswith("nvidia_")]

    assert len(nvidia) == 3
    assert {config.api_key_env for config in nvidia} == {"NVIDIA_API_KEY"}
    assert {config.api_base for config in nvidia} == {NVIDIA_NIM_BASE}
    assert NVIDIA_NIM_BASE == "https://integrate.api.nvidia.com/v1"


def test_one_nvidia_key_configures_the_whole_rung_ladder() -> None:
    gateway = Gateway(DEFAULT_PROVIDERS, env={"NVIDIA_API_KEY": "k"})

    assert [config.name for config in gateway.configured()] == [
        "nvidia_deepseek",
        "nvidia_nemotron",
        "nvidia_glm",
    ]


async def test_the_base_url_reaches_the_transport_with_the_request() -> None:
    """LiteLLM knows the well-known hosts; NIM is pinned so the endpoint is in the repository
    rather than in a default that can move under it."""
    seen: list[str | None] = []

    class BaseRecordingTransport:
        async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
            seen.append(provider.api_base)
            return "ok"

    nim = next(config for config in DEFAULT_PROVIDERS if config.name == "nvidia_deepseek")
    gateway = Gateway((nim,), transport=BaseRecordingTransport(), env={"NVIDIA_API_KEY": "k"})

    await gateway.complete(prompt())

    assert seen == [NVIDIA_NIM_BASE]


def test_adding_a_provider_needs_no_change_to_the_gateway() -> None:
    extra = ProviderConfig(name="delta", model="delta/m", api_key_env="DELTA_KEY")
    ordered = providers_from_env(
        {"LLM_PRIMARY_PROVIDER": "delta"}, providers=(*DEFAULT_PROVIDERS, extra)
    )

    assert ordered[0] is extra
    assert Gateway(ordered, env={"DELTA_KEY": "k"}).configured() == (extra,)


# -- per-agent pinning -----------------------------------------------------------------------


async def test_a_prompts_preferred_provider_goes_first() -> None:
    """CLAUDE.md pins the chartist to Groq. The gateway honours that without knowing why."""
    transport = ScriptedTransport()
    gateway = _gateway(transport)

    completion = await gateway.complete(
        Prompt(agent="chartist", system="rules", user="briefing", prefer=("gamma",))
    )

    assert completion is not None and completion.provider == "gamma"
    assert transport.calls == ["gamma"]


async def test_a_preferred_provider_that_fails_falls_through_to_the_rest() -> None:
    """A pin is a preference. The second-choice model beats no narration at all."""
    transport = ScriptedTransport({"gamma": ProviderError("down")})
    gateway = _gateway(transport)

    completion = await gateway.complete(
        Prompt(agent="chartist", system="rules", user="briefing", prefer=("gamma",))
    )

    assert completion is not None and completion.provider == "alpha"
    assert transport.calls == ["gamma", "alpha"]


async def test_an_unconfigured_preference_is_skipped_silently() -> None:
    transport = ScriptedTransport()
    gateway = Gateway(CHAIN, transport=transport, env={"BETA_KEY": "k"})

    completion = await gateway.complete(
        Prompt(agent="chartist", system="rules", user="briefing", prefer=("gamma",))
    )

    assert completion is not None and completion.provider == "beta"


async def test_a_routing_preference_does_not_split_the_cache() -> None:
    """Same question, same answer. A preference must not make a re-run pay twice."""
    transport = ScriptedTransport()
    gateway = _gateway(transport)

    await gateway.complete(Prompt(agent="chartist", system="r", user="b", prefer=("gamma",)))
    second = await gateway.complete(Prompt(agent="chartist", system="r", user="b"))

    assert second is not None and second.cached is True
    assert transport.calls == ["gamma"]


# -- the per-agent ceiling -------------------------------------------------------------------


def test_an_agents_own_cap_binds_before_the_shared_one() -> None:
    """The risk officer runs at most 20 times a day; the shared cap is 50. The tighter wins."""
    budget = CallBudget(daily_limit=50, now=lambda: MOMENT)
    for _ in range(3):
        budget.spend(agent="risk_officer", agent_limit=3)

    assert budget.spent_by("risk_officer") == 3
    assert budget.spent_today == 3
    with pytest.raises(DailyCallLimitReached, match="risk_officer has spent its cap of 3"):
        budget.spend(agent="risk_officer", agent_limit=3)


def test_one_agents_cap_does_not_constrain_another() -> None:
    budget = CallBudget(daily_limit=50, now=lambda: MOMENT)
    budget.spend(agent="risk_officer", agent_limit=1)

    budget.spend(agent="chartist")
    budget.spend(agent="chartist")

    assert budget.spent_by("chartist") == 2
    assert budget.spent_by("risk_officer") == 1


def test_the_shared_cap_still_binds_when_no_agent_cap_is_reached() -> None:
    budget = CallBudget(daily_limit=2, now=lambda: MOMENT)
    budget.spend(agent="chartist", agent_limit=99)
    budget.spend(agent="chartist", agent_limit=99)

    with pytest.raises(DailyCallLimitReached, match="hard daily cap of 2"):
        budget.spend(agent="chartist", agent_limit=99)


def test_two_different_limits_for_one_agent_is_a_wiring_bug_not_a_preference() -> None:
    """A cap the next caller can widen is not a cap."""
    budget = CallBudget(now=lambda: MOMENT)
    budget.spend(agent="risk_officer", agent_limit=20)

    with pytest.raises(ValueError, match="already declared a daily limit of 20"):
        budget.spend(agent="risk_officer", agent_limit=200)


def test_the_per_agent_count_rolls_over_with_the_day() -> None:
    moment = MOMENT
    budget = CallBudget(now=lambda: moment)
    budget.spend(agent="risk_officer", agent_limit=1)

    moment = MOMENT + timedelta(days=1)
    assert budget.spent_by("risk_officer") == 0
    budget.spend(agent="risk_officer", agent_limit=1)


async def test_the_gateway_enforces_the_limit_the_prompt_declares() -> None:
    """Declared by the agent's spec and enforced here, without the gateway knowing whose it is."""
    transport = ScriptedTransport()
    budget = CallBudget(now=lambda: MOMENT)
    gateway = _gateway(transport, budget=budget)

    for index in range(3):
        await gateway.complete(
            Prompt(agent="risk_officer", system="r", user=f"b{index}", daily_call_limit=2)
        )

    assert transport.calls == ["alpha", "alpha"]
    assert budget.spent_by("risk_officer") == 2


async def test_a_declared_limit_does_not_split_the_cache() -> None:
    transport = ScriptedTransport()
    gateway = _gateway(transport)

    await gateway.complete(Prompt(agent="risk_officer", system="r", user="b", daily_call_limit=20))
    second = await gateway.complete(Prompt(agent="risk_officer", system="r", user="b"))

    assert second is not None and second.cached is True
    assert transport.calls == ["alpha"]
