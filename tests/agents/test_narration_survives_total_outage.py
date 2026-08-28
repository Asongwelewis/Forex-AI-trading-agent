"""The point of this commit: with every provider unreachable, the pass still explains itself.

The narration layer sits next to a decision that was already made without it. So the failure
this file rules out is not "the agents were wrong" but "the agents being unavailable stopped
the system reporting what it decided" — a free-tier outage taking down the part of the pipeline
that never needed a model in the first place.

Every test here drives the real `narrate`, the real `Gateway` and the real `Briefing`
assembly. Only the transport is substituted, and it is substituted with failure.
"""

from __future__ import annotations

import importlib.util
import logging

import pytest

from fxagent.agents.gateway import (
    CallBudget,
    Gateway,
    LiteLLMTransport,
    Prompt,
    ProviderConfig,
    ProviderError,
    RateLimited,
)
from fxagent.agents.narrate import (
    AGENTS,
    LEGACY_AGENTS,
    TEMPLATE_PROVIDER,
    attach_narration,
    narrate,
)
from fxagent.dashboard.contract import AGENTS_KEY, CHARTIST, HISTORIAN, read_agents
from tests.agents.builders import MOMENT, analogue, declined_briefing, fired_briefing

#: Two providers with keys present, so every test below reaches the transport rather than
#: being skipped for want of configuration. Short intervals keep the injected sleep trivial.
PROVIDERS = (
    ProviderConfig(name="alpha", model="alpha/m", api_key_env="ALPHA_KEY", min_interval_seconds=0),
    ProviderConfig(name="beta", model="beta/m", api_key_env="BETA_KEY", min_interval_seconds=0),
)
ENV = {"ALPHA_KEY": "present", "BETA_KEY": "present"}


class DeadTransport:
    """Every provider unreachable, in whichever way is being modelled."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or ProviderError("connection refused")
        self.attempts: list[str] = []

    async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
        self.attempts.append(provider.name)
        raise self._error


async def _nap(_seconds: float) -> None:
    """Injected sleep. The backoff schedule is asserted elsewhere; here it just must not wait."""


def _gateway(transport: DeadTransport, **kwargs: object) -> Gateway:
    return Gateway(
        PROVIDERS,
        transport=transport,
        env=ENV,
        sleep=_nap,
        jitter=lambda _low, high: high,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_every_provider_unreachable_still_produces_a_block_per_agent() -> None:
    transport = DeadTransport()
    briefing = fired_briefing(analogues=(analogue(),))

    blocks = await narrate(briefing, gateway=_gateway(transport))

    assert set(blocks) == {spec.name for spec in AGENTS}
    for spec in AGENTS:
        block = blocks[spec.name]
        assert block["provider"] == TEMPLATE_PROVIDER
        assert block["text"].strip()
        assert block["model"] is None

    # The fallback was reached *through* the chain, not by skipping it: both providers were
    # tried for every agent. A test that passed because nothing was ever attempted would prove
    # nothing about an outage.
    assert transport.attempts == ["alpha", "beta"] * len(AGENTS)


async def test_the_template_blocks_state_what_the_core_actually_decided() -> None:
    briefing = fired_briefing()
    blocks = await narrate(briefing, gateway=_gateway(DeadTransport()))

    chartist = blocks[CHARTIST]["text"]
    assert briefing.symbol in chartist
    assert briefing.plan is not None
    assert str(briefing.plan.entry_price) in chartist
    assert str(briefing.plan.stop_loss) in chartist
    assert str(briefing.plan.take_profit) in chartist


async def test_the_declined_path_explains_itself_too() -> None:
    """On most bars nothing fires, and that is the bar whose explanation matters most."""
    briefing = declined_briefing()
    blocks = await narrate(briefing, gateway=_gateway(DeadTransport()))

    text = blocks[CHARTIST]["text"]
    assert "no trade plan" in text.lower()
    assert briefing.reason in text or "declined" in text.lower()


async def test_the_blocks_are_readable_by_the_dashboard_without_a_single_discard() -> None:
    """The writer and the reader are checked against each other, not against a fixture.

    `read_agents` is the dashboard's own parser and it discards any block it cannot validate.
    Running it over what `narrate` produced is the only assertion that catches the two ends
    drifting apart, which is the failure a shape test written twice cannot see.
    """
    briefing = fired_briefing(analogues=(analogue(1), analogue(2, similarity=0.81)))
    blocks = await narrate(briefing, gateway=_gateway(DeadTransport()), agents=LEGACY_AGENTS)

    document = attach_narration({"fired": True}, blocks)
    assert document[AGENTS_KEY] is blocks
    assert document["fired"] is True

    parsed = read_agents(document)
    assert parsed.discarded == ()
    assert parsed.chartist is not None and parsed.chartist.agent == CHARTIST
    assert parsed.historian is not None and parsed.historian.agent == HISTORIAN
    assert [item.similarity for item in parsed.analogues] == [0.93, 0.81]


async def test_retrieved_analogues_survive_the_outage() -> None:
    """The analogues are retrieval's output. No agent had to be reachable for them to appear."""
    briefing = fired_briefing(analogues=(analogue(7),))
    blocks = await narrate(briefing, gateway=_gateway(DeadTransport()), agents=LEGACY_AGENTS)

    assert blocks[HISTORIAN]["provider"] == TEMPLATE_PROVIDER
    assert blocks[HISTORIAN]["analogues"] == [
        {
            "timestamp": "2025-03-04T09:00:00+00:00",
            "symbol": "EURUSD",
            "similarity": 0.93,
            "outcome": "TAKE_PROFIT",
            "outcome_r": 2.0,
            "resolved_at": "2025-03-06T09:00:00+00:00",
        }
    ]


async def test_rate_limits_that_outlast_the_backoff_end_in_the_template() -> None:
    transport = DeadTransport(RateLimited("429", retry_after=None))
    blocks = await narrate(fired_briefing(), gateway=_gateway(transport))

    assert all(block["provider"] == TEMPLATE_PROVIDER for block in blocks.values())
    # Three attempts per provider, two providers, every agent.
    assert len(transport.attempts) == 3 * 2 * len(AGENTS)


async def test_no_key_anywhere_never_reaches_a_provider() -> None:
    """The README's promise: with no keys set at all, the pass completes normally."""
    transport = DeadTransport()
    gateway = Gateway(PROVIDERS, transport=transport, env={}, sleep=_nap)

    blocks = await narrate(fired_briefing(), gateway=gateway)

    assert gateway.configured() == ()
    assert transport.attempts == []
    assert all(block["provider"] == TEMPLATE_PROVIDER for block in blocks.values())


async def test_an_exhausted_daily_budget_falls_back_rather_than_raising() -> None:
    transport = DeadTransport()
    budget = CallBudget(daily_limit=1, now=lambda: MOMENT)
    budget.spend()

    blocks = await narrate(fired_briefing(), gateway=_gateway(transport, budget=budget))

    assert transport.attempts == []
    assert all(block["provider"] == TEMPLATE_PROVIDER for block in blocks.values())


async def test_no_gateway_at_all_templates_everything() -> None:
    """How a backtest runs: no model, no cost, and no run-to-run variation in the commentary."""
    blocks = await narrate(fired_briefing(analogues=(analogue(),)), agents=LEGACY_AGENTS)

    assert all(block["provider"] == TEMPLATE_PROVIDER for block in blocks.values())
    assert blocks[HISTORIAN]["analogues"]


async def test_backtest_mode_narration_carries_no_date(caplog: pytest.LogCaptureFixture) -> None:
    """`include_dates=False` is about the prompt; the blocks still complete."""
    briefing = fired_briefing()
    with caplog.at_level(logging.DEBUG):
        blocks = await narrate(briefing, gateway=_gateway(DeadTransport()), include_dates=False)

    assert "timestamp" not in briefing.payload(include_dates=False)
    assert all(block["text"].strip() for block in blocks.values())


@pytest.mark.skipif(
    importlib.util.find_spec("litellm") is not None,
    reason="litellm is installed, so the real transport would attempt a network call",
)
async def test_the_real_transport_degrades_to_the_template_when_litellm_is_absent() -> None:
    """The optional extra being uninstalled is an outage like any other, not an ImportError.

    This is the configuration the base install ships in, so it is the one most likely to run
    in production before anyone sets a key.
    """
    gateway = Gateway(PROVIDERS, transport=LiteLLMTransport(), env=ENV, sleep=_nap)

    blocks = await narrate(fired_briefing(), gateway=gateway)

    assert all(block["provider"] == TEMPLATE_PROVIDER for block in blocks.values())
