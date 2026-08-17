"""The briefing an agent is shown, and what it is allowed to say back.

The interesting tests here are the discards. Hard rule 5 says a response failing validation is
thrown away entirely, and the rule that makes that enforceable rather than aspirational is the
number check: an agent may not state a figure that was not in its input.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fxagent.agents.schemas import (
    MAX_NOTE_CHARS,
    MAX_READ_CHARS,
    Briefing,
    ChartistNote,
    Grounding,
    HistorianNote,
    validate_note,
)
from fxagent.patterns import CONTEXT_ONLY
from tests.agents.builders import (
    INDICATORS,
    analogue,
    declined_briefing,
    fired_briefing,
    pattern,
)


def _note(**overrides: object) -> dict[str, object]:
    return {"read": "The regime is trending and the plan is long.", **overrides}


# -- assembly ------------------------------------------------------------------------------


def test_the_briefing_carries_the_whole_slate_including_the_votes_that_did_not_count() -> None:
    briefing = fired_briefing()

    by_name = {vote.strategy: vote for vote in briefing.votes}
    assert set(by_name) == {"session_breakout", "carry_divergence", "range_reversion"}
    assert by_name["range_reversion"].participated is False
    assert "gated" in by_name["range_reversion"].reason
    assert by_name["carry_divergence"].participated is True


def test_the_plan_is_the_primary_strategys_levels_taken_whole() -> None:
    briefing = fired_briefing()

    assert briefing.plan is not None
    assert briefing.fired is True
    assert briefing.plan.direction == "LONG"
    assert briefing.plan.primary_strategy == "session_breakout"
    assert briefing.plan.stop_loss < briefing.plan.entry_price < briefing.plan.take_profit


def test_a_declined_evaluation_has_no_plan_and_still_has_its_reason() -> None:
    briefing = declined_briefing()

    assert briefing.plan is None
    assert briefing.fired is False
    assert briefing.reason


def test_counts_the_agent_may_legitimately_state_are_in_the_payload() -> None:
    """Rather than being special-cased in the check. See the module docstring in `schemas`."""
    payload = fired_briefing().payload()

    assert payload["vote_count"] == 3
    assert payload["participating_count"] == 2


# -- grounding -----------------------------------------------------------------------------


def test_a_number_quoted_from_the_input_is_accepted_at_the_precision_it_was_written() -> None:
    briefing = fired_briefing()
    assert briefing.plan is not None

    note = validate_note(
        ChartistNote,
        json.dumps(_note(read="Entry is 1.1000 with the stop at 1.0980.")),
        briefing,
    )

    assert note is not None
    assert "1.1000" in note.read
    # Whatever an agent calls its narration field, it serialises as the one the panel reads.
    assert "text" in note.model_dump(by_alias=True)


def test_a_number_that_was_not_in_the_input_discards_the_whole_response() -> None:
    briefing = fired_briefing()

    assert validate_note(ChartistNote, json.dumps(_note(read="Watch 1.2345.")), briefing) is None


def test_an_invented_number_in_a_nested_field_is_caught_too() -> None:
    """A nested field is not a side channel: the check reads the whole response's own JSON."""
    briefing = fired_briefing(patterns=(pattern("doji"),))

    def seen(meaning: str) -> str:
        return json.dumps(
            _note(
                patterns_seen=[{"name": "doji", "meaning": meaning, "significance": "context only"}]
            )
        )

    assert validate_note(ChartistNote, seen("a body under 1.2345"), briefing) is None
    assert validate_note(ChartistNote, seen("a body under 0.002"), briefing) is not None


def test_grounding_permits_a_rounded_quotation_but_not_a_rounded_invention() -> None:
    grounding = Grounding(values=frozenset({27.34567}))

    assert grounding.permits(27.3, 1) is True
    assert grounding.permits(27.0, 0) is True
    assert grounding.permits(27.9, 1) is False
    assert grounding.ungrounded("ADX read 27.3, up from 19.0") == ("19.0",)


def test_a_date_is_read_as_three_positive_numbers_not_two_negative_ones() -> None:
    """The lookbehind in the literal pattern. `2026-08-16` is 2026, 8 and 16, never -8."""
    grounding = Grounding(values=frozenset({2026.0, 8.0, 16.0}))

    assert grounding.ungrounded("the window opened 2026-08-16") == ()


# -- discards ------------------------------------------------------------------------------


def test_malformed_json_is_discarded_rather_than_repaired() -> None:
    briefing = fired_briefing()

    assert validate_note(ChartistNote, "{'text': 'nearly json'", briefing) is None
    assert validate_note(ChartistNote, "here is my answer: {}", briefing) is None


def test_an_unrequested_field_discards_the_response() -> None:
    briefing = fired_briefing()

    assert validate_note(ChartistNote, json.dumps(_note(recommendation="BUY")), briefing) is None


def test_the_chartist_cannot_narrate_a_formation_nobody_detected() -> None:
    """A formation name carries no digits, so the numeric check cannot police it on its own."""
    briefing = fired_briefing(patterns=(pattern("doji"),))

    def seen(name: str) -> str:
        return json.dumps(
            _note(patterns_seen=[{"name": name, "meaning": "a shape", "significance": "context"}])
        )

    assert validate_note(ChartistNote, seen("morning_star"), briefing) is None
    assert validate_note(ChartistNote, seen("doji"), briefing) is not None


def test_a_formation_the_chartist_mentions_is_stamped_context_by_default() -> None:
    """A defaulted field, so it cannot be omitted by a model that would rather not say it."""
    briefing = fired_briefing(patterns=(pattern("doji"),))

    note = validate_note(
        ChartistNote,
        json.dumps(
            _note(patterns_seen=[{"name": "doji", "meaning": "no body", "significance": "ctx"}])
        ),
        briefing,
    )

    assert note is not None
    assert note.patterns_seen[0].label == CONTEXT_ONLY
    assert "NOT A SIGNAL" in note.patterns_seen[0].label


def test_a_read_over_four_hundred_characters_is_discarded() -> None:
    """The chartist's card is short on purpose — see `MAX_READ_CHARS`."""
    sprawl = "the market is trending. " * 20

    assert len(sprawl) > MAX_READ_CHARS
    assert validate_note(ChartistNote, json.dumps(_note(read=sprawl)), fired_briefing()) is None


def test_disagreement_defaults_to_absent_and_is_only_ever_prose() -> None:
    """It is displayed and logged. Nothing scores it and no branch tests it — hard rule 4."""
    briefing = fired_briefing()

    quiet = validate_note(ChartistNote, json.dumps(_note()), briefing)
    speaking = validate_note(
        ChartistNote,
        json.dumps(_note(disagreement="The measured trend does not support this direction.")),
        briefing,
    )

    assert quiet is not None and quiet.disagreement is None
    assert speaking is not None and isinstance(speaking.disagreement, str)


def test_terms_explained_travels_as_term_and_definition() -> None:
    briefing = fired_briefing()

    note = validate_note(
        ChartistNote,
        json.dumps(_note(terms_explained=[{"term": "ADX", "definition": "trend strength"}])),
        briefing,
    )

    assert note is not None
    assert note.terms_explained[0].term == "ADX"


def test_an_overlong_historian_narration_is_discarded() -> None:
    sprawl = "the record does not reach back. " * (MAX_NOTE_CHARS // 10)

    assert validate_note(HistorianNote, json.dumps({"text": sprawl}), fired_briefing()) is None


def test_validating_without_a_grounding_raises_rather_than_passing() -> None:
    """There must be no path on which the number check is silently skipped."""
    with pytest.raises(ValidationError, match="validation context"):
        ChartistNote.model_validate(_note())


# -- the historian's analogues -------------------------------------------------------------


def test_the_historian_may_cite_a_retrieved_analogue() -> None:
    briefing = fired_briefing(analogues=(analogue(1),))

    note = validate_note(
        HistorianNote,
        json.dumps({"text": "The nearest window resolved at target.", "analogue_ids": [1]}),
        briefing,
    )

    assert note is not None and note.analogue_ids == (1,)


def test_the_historian_cannot_cite_a_window_that_was_not_retrieved() -> None:
    """Id 2 is a grounded *number* — it is the vote count — and still not a retrieved window.

    Which is the point: the numeric check alone would let a filtered-out analogue back into the
    record on the strength of its id appearing somewhere else in the payload.
    """
    briefing = fired_briefing(analogues=(analogue(1),))
    assert 2.0 in briefing.grounding().values

    assert (
        validate_note(
            HistorianNote,
            json.dumps({"text": "The second window resolved well.", "analogue_ids": [2]}),
            briefing,
        )
        is None
    )


# -- backtest mode -------------------------------------------------------------------------


def test_backtest_mode_removes_every_date_from_the_payload() -> None:
    briefing = fired_briefing(analogues=(analogue(),))

    dated = briefing.payload(include_dates=True)
    undated = briefing.payload(include_dates=False)

    assert "timestamp" in dated
    assert "timestamp" not in undated
    assert "timestamp" in dated["analogues"][0]
    assert "timestamp" not in undated["analogues"][0]
    assert "resolved_at" not in undated["analogues"][0]


def test_a_date_dropped_from_the_prompt_is_dropped_from_the_grounding_too() -> None:
    """Otherwise backtest mode would hide the year from the model and still let it quote one."""
    briefing = fired_briefing()
    quoting_the_year = json.dumps(_note(read="The comparable move was in 2026."))

    assert validate_note(ChartistNote, quoting_the_year, briefing, include_dates=True) is not None
    assert validate_note(ChartistNote, quoting_the_year, briefing, include_dates=False) is None


def test_the_rendered_prompt_is_stable_across_calls() -> None:
    """Two identical briefings must hash to one cache key, so a re-run costs one call."""
    assert fired_briefing().rendered() == fired_briefing().rendered()


def test_float_noise_never_reaches_the_prompt() -> None:
    """A price of 1.0999999999999999 in a prompt is also 1.1 being ungrounded when quoted."""
    briefing = Briefing(
        symbol="EURUSD", timestamp=fired_briefing().timestamp, long_weight=0.1 + 0.2
    )

    assert briefing.payload()["decision"]["long_weight"] == 0.3


# -- the chartist's structured input -------------------------------------------------------


def test_indicator_readings_reach_the_payload_with_warm_up_left_as_unknown() -> None:
    """`None` is the classifier's "not yet". Rendering it as 0.0 would be a measurement."""
    payload = fired_briefing(indicators=INDICATORS).payload()

    assert payload["indicators"] == {"adx_14": 30.0, "atr_14": 0.002, "ema_200": None}


def test_detected_formations_reach_the_payload_carrying_their_context_label() -> None:
    """The model reads the stamp before it writes about the formation, not after."""
    payload = fired_briefing(patterns=(pattern("doji"),)).payload()

    assert payload["patterns"][0]["name"] == "doji"
    assert payload["patterns"][0]["label"] == CONTEXT_ONLY
    assert payload["patterns"][0]["definition"]
    assert payload["patterns"][0]["criteria"]["atr"] == 0.002


def test_the_briefing_has_no_field_a_price_bar_could_travel_in() -> None:
    """CLAUDE.md: the chartist never sees pixels or a raw price series. Held by construction.

    An assertion about the shape rather than about the prompt, because a prompt that asks a
    model not to look at bars is only as good as the caller that assembled it.
    """
    forbidden = {"bars", "ohlc", "series", "candles", "prices", "image", "chart"}

    assert forbidden.isdisjoint(Briefing.model_fields)
    assert forbidden.isdisjoint(fired_briefing().payload())
