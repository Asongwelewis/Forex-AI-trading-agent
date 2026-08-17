"""The deterministic floor. No provider, no key, no network — and still a full explanation.

These read like prose assertions because the thing under test is prose. What they are actually
pinning is that the template never invents, never rounds a warm-up to zero, and never renders a
missing section as an empty one.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fxagent.agents import templates
from fxagent.agents.schemas import Briefing
from tests.agents.builders import analogue, declined_briefing, fired_briefing

MOMENT = datetime(2026, 1, 12, 9, 0, tzinfo=UTC)


def test_the_explanation_names_the_symbol_the_session_and_the_state() -> None:
    text = templates.explain(fired_briefing())

    assert "EURUSD" in text
    assert "LONDON" in text
    assert "ADX 30.0" in text
    assert "62nd percentile" in text


def test_warm_up_reports_as_unknown_and_never_as_a_flat_market() -> None:
    """`None` trend strength is the classifier saying "not yet", which is not the same as zero."""
    briefing = Briefing(symbol="EURUSD", timestamp=MOMENT, trend_strength=None)

    text = templates.regime_sentence(briefing)

    assert "warming up" in text
    assert "0.0" not in text


def test_every_strategy_on_the_slate_is_named_including_the_ones_that_did_not_vote() -> None:
    """A list of only the strategies that spoke reads as unanimity."""
    text = templates.votes_sentence(fired_briefing())

    assert "session_breakout" in text
    assert "carry_divergence" in text
    assert "range_reversion" in text
    assert "gated" in text


def test_a_silent_strategy_is_reported_with_the_cores_own_reason() -> None:
    text = templates.votes_sentence(declined_briefing())

    assert "silent: the setup is not present" in text


def test_the_verdict_carries_the_thresholds_it_was_measured_against() -> None:
    text = templates.verdict_sentence(fired_briefing())

    assert "Consensus fired LONG" in text
    assert "1.60" in text  # the summed long weight
    assert "2 agreeing strategies" in text


def test_a_declined_verdict_quotes_why() -> None:
    briefing = declined_briefing()

    assert briefing.reason in templates.verdict_sentence(briefing)
    assert "declined" in templates.verdict_sentence(briefing).lower()


def test_the_plan_sentence_carries_the_levels_unrounded() -> None:
    briefing = fired_briefing()
    assert briefing.plan is not None

    text = templates.plan_sentence(briefing)

    assert str(briefing.plan.entry_price) in text
    assert str(briefing.plan.stop_loss) in text
    assert str(briefing.plan.take_profit) in text
    assert "session_breakout" in text


def test_no_plan_is_a_sentence_rather_than_an_omission() -> None:
    """An empty gap where the levels go is indistinguishable from a rendering failure."""
    assert "no trade plan" in templates.plan_sentence(declined_briefing()).lower()


def test_an_empty_retrieval_says_why_it_is_empty() -> None:
    text = templates.historian_fallback(fired_briefing())

    assert "No resolved analogue" in text
    assert "point" not in text.lower() or "resolved" in text.lower()
    assert "not that the lookup failed" in text


def test_retrieved_analogues_are_summarised_from_their_own_numbers() -> None:
    briefing = fired_briefing(
        analogues=(
            analogue(1, similarity=0.93, outcome_r=2.0),
            analogue(2, similarity=0.81, outcome_r=-1.0),
        )
    )

    text = templates.historian_fallback(briefing)

    assert "2 resolved analogue(s)" in text
    assert "0.93 similarity" in text
    assert "-1.0R to 2.0R" in text


def test_every_template_paragraph_says_a_model_did_not_write_it() -> None:
    briefing = fired_briefing(analogues=(analogue(),))

    for text in (
        templates.explain(briefing),
        templates.chartist_fallback(briefing),
        templates.historian_fallback(briefing),
    ):
        assert text.endswith(templates.PROVENANCE)


def test_the_templates_need_nothing_but_the_briefing() -> None:
    """The floor cannot import the ceiling: a template that reached the gateway is not a floor."""
    import inspect

    source = inspect.getsource(templates)

    assert "gateway" not in source.replace("fxagent.agents.gateway", "")
    assert "import litellm" not in source
    assert "httpx" not in source
