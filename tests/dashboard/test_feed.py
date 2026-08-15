"""The right pane: the votes, the narrations, and what happens when a narration is malformed.

Two properties get most of the attention here, because both are ones a renderer tends to lose:

* the three vote states stay three states, so "never asked" and "asked and declined" do not
  collapse into one grey row;
* a block that fails validation is dropped whole and *announced*, rather than half-drawn.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fxagent.dashboard.contract import read_agents, read_patterns, read_regime, read_votes
from fxagent.dashboard.feed import AGENT_LAYER_ABSENT, build_feed
from fxagent.dashboard.models import GrantSnapshot, GrantState
from fxagent.regime.router import CARRY_DIVERGENCE, RANGE_REVERSION, SESSION_BREAKOUT
from tests.dashboard.builders import evaluation, trade, vote

ADVISORY = GrantSnapshot(state=GrantState.ADVISORY, reason="not built")

NARRATION = {
    "text": "Price closed above the Asian range on rising ADX.",
    "provider": "groq",
    "model": "llama-3.3-70b",
    "generated_at": "2026-01-12T09:00:00+00:00",
}


def feed_for(*evaluations, trades=()):
    return build_feed("EURUSD", list(evaluations), list(trades), ADVISORY)


# --- the shape of the feed ----------------------------------------------------


def test_the_feed_is_newest_first() -> None:
    older = evaluation(identifier=1, ts_utc=datetime(2026, 1, 12, 8, tzinfo=UTC))
    newer = evaluation(identifier=2, ts_utc=datetime(2026, 1, 12, 9, tzinfo=UTC))

    payload = feed_for(older, newer)

    assert [entry.evaluation_id for entry in payload.entries] == [2, 1]


def test_evaluations_that_fired_nothing_are_still_entries() -> None:
    """On most bars a refusal is the only thing that happened, and it is the thing to read."""
    payload = feed_for(
        evaluation(fired=False, reason="best was LONG with 1 agreeing and summed weight 1.00")
    )

    assert len(payload.entries) == 1
    assert payload.entries[0].fired is False
    assert "1 agreeing" in payload.entries[0].reason


def test_an_empty_feed_says_the_analyst_has_not_run() -> None:
    payload = feed_for()

    assert payload.entries == ()
    assert any("analyst has not run" in note for note in payload.notes)


def test_a_trade_is_attached_to_the_evaluation_that_produced_it() -> None:
    payload = feed_for(
        evaluation(identifier=7, fired=True),
        trades=[trade(identifier=3, evaluation_id=7, r_multiple=1.8)],
    )

    entry = payload.entries[0]
    assert [view.trade_id for view in entry.trades] == [3]
    assert entry.trades[0].r_multiple == 1.8


def test_a_trade_from_another_evaluation_is_not_attached_to_this_one() -> None:
    payload = feed_for(evaluation(identifier=7), trades=[trade(evaluation_id=99)])

    assert payload.entries[0].trades == ()


# --- votes --------------------------------------------------------------------


def test_silent_gated_and_flat_stay_three_different_facts() -> None:
    payload = feed_for(
        evaluation(
            votes=[
                vote(SESSION_BREAKOUT, weight=0.0, reason="silent: the setup is not present"),
                vote(
                    RANGE_REVERSION,
                    weight=0.0,
                    direction="LONG",
                    participated=False,
                    reason="gated: the router does not permit this strategy in this regime",
                ),
                vote(
                    CARRY_DIVERGENCE,
                    weight=0.5,
                    direction="FLAT",
                    confidence=0.3,
                    participated=True,
                    reason="flat: voted, but an abstention carries no direction",
                ),
            ]
        )
    )

    votes = {item.strategy: item for item in payload.entries[0].votes}

    assert votes[SESSION_BREAKOUT].direction is None
    assert votes[SESSION_BREAKOUT].participated is False

    assert votes[RANGE_REVERSION].direction == "LONG"
    assert votes[RANGE_REVERSION].participated is False

    assert votes[CARRY_DIVERGENCE].direction == "FLAT"
    assert votes[CARRY_DIVERGENCE].participated is True


def test_the_vote_list_survives_being_handed_the_bare_list() -> None:
    """A caller holding `diagnostics["votes"]` already has the list; re-wrapping it is one bug."""
    rows = [vote(SESSION_BREAKOUT, direction="LONG", participated=True)]

    assert read_votes({"votes": rows}) == read_votes(rows)


def test_a_vote_with_no_strategy_names_nothing_and_is_skipped() -> None:
    assert read_votes([{"weight": 1.0, "direction": "LONG"}]) == ()


# --- regime -------------------------------------------------------------------


def test_the_dominant_session_is_derived_when_a_stored_row_lacks_it() -> None:
    view = read_regime({"sessions": ["LONDON", "NEW_YORK", "OVERLAP"], "market_open": True})

    assert view.session == "OVERLAP"


def test_a_stored_session_label_is_used_as_written() -> None:
    view = read_regime({"sessions": ["TOKYO"], "session": "TOKYO", "market_open": True})

    assert view.session == "TOKYO"


def test_a_regime_that_never_warmed_up_reports_unknown_rather_than_zero() -> None:
    view = read_regime({"sessions": [], "market_open": False, "trend_strength": None})

    assert view.trend_strength is None
    assert view.is_trending is False
    assert view.is_ranging is False


# --- agent blocks --------------------------------------------------------------


def test_the_panel_says_when_the_agent_layer_has_produced_nothing() -> None:
    """An empty section for an absent producer must not look like a quiet one."""
    payload = feed_for(evaluation())

    assert AGENT_LAYER_ABSENT in payload.notes


def test_a_narration_is_passed_through_as_text() -> None:
    payload = feed_for(evaluation(extra={"agents": {"chartist": NARRATION}}))

    entry = payload.entries[0]
    assert entry.chartist is not None
    assert entry.chartist.text == NARRATION["text"]
    assert entry.chartist.provider == "groq"
    assert AGENT_LAYER_ABSENT not in payload.notes


def test_analogues_carry_the_similarity_and_how_they_resolved() -> None:
    payload = feed_for(
        evaluation(
            extra={
                "agents": {
                    "historian": {
                        **NARRATION,
                        "analogues": [
                            {
                                "timestamp": "2025-03-04T08:00:00+00:00",
                                "symbol": "EURUSD",
                                "similarity": 0.91,
                                "outcome": "TARGET",
                                "outcome_r": 2.0,
                                "resolved_at": "2025-03-06T08:00:00+00:00",
                            }
                        ],
                    }
                }
            }
        )
    )

    analogue = payload.entries[0].analogues[0]
    assert analogue.similarity == 0.91
    assert analogue.outcome_r == 2.0
    assert analogue.resolved_at == "2025-03-06T08:00:00+00:00"


def test_the_risk_officers_recommendation_travels_as_advisory_text() -> None:
    payload = feed_for(
        evaluation(
            extra={
                "agents": {
                    "risk_officer": {**NARRATION, "proceed_recommendation": "PROCEED"},
                }
            }
        )
    )

    assert payload.entries[0].risk_officer.proceed_recommendation == "PROCEED"


def test_a_malformed_narration_is_discarded_whole_and_named() -> None:
    """Hard rule 5 on the read side: no partial parsing, no repair, and no silent blank."""
    payload = feed_for(
        evaluation(
            extra={
                "agents": {
                    "chartist": {"provider": "groq"},  # no text
                    "historian": NARRATION,
                }
            }
        )
    )

    entry = payload.entries[0]
    assert entry.chartist is None
    assert "chartist" in entry.discarded
    assert entry.historian is not None  # one bad block does not take the others with it


def test_an_unknown_field_does_not_discard_a_narration() -> None:
    """The analyst will grow fields. A reader that treated that as corruption would be an
    outage every time it did."""
    payload = feed_for(
        evaluation(extra={"agents": {"chartist": {**NARRATION, "token_count": 812}}})
    )

    assert payload.entries[0].chartist is not None
    assert payload.entries[0].discarded == ()


def test_agents_that_have_not_run_are_absent_rather_than_discarded() -> None:
    blocks = read_agents({"agents": {"chartist": NARRATION}})

    assert blocks.chartist is not None
    assert blocks.historian is None
    assert blocks.discarded == ()


# --- candle formations ----------------------------------------------------------


def test_a_formation_carries_its_context_only_label_in_the_data() -> None:
    """Not applied by the template — the label must not be style-able away."""
    payload = feed_for(
        evaluation(
            extra={
                "patterns": [
                    {
                        "name": "bullish engulfing",
                        "definition": "A green body that wholly contains the prior red body.",
                        "bar_time": "2026-01-12T09:00:00+00:00",
                    }
                ]
            }
        )
    )

    pattern = payload.entries[0].patterns[0]
    assert pattern.name == "bullish engulfing"
    assert pattern.label == "CONTEXT ONLY — NOT A SIGNAL"


def test_a_formation_without_a_definition_is_discarded() -> None:
    """A pattern name with no definition is a label, and a label next to a chart reads as a
    recommendation."""
    notes, discarded = read_patterns({"patterns": [{"name": "doji"}]})

    assert notes == ()
    assert discarded == ("patterns[0]",)


def test_patterns_are_ignored_when_the_key_is_not_a_list() -> None:
    assert read_patterns({"patterns": "bullish engulfing"}) == ((), ())


# --- permission ------------------------------------------------------------------


def test_the_grant_travels_with_the_feed() -> None:
    payload = feed_for(evaluation())

    assert payload.grant.state is GrantState.ADVISORY
