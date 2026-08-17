"""Deterministic explanation, assembled from the same `Briefing` the agents receive.

**No LLM here, and no import that could reach one.** This module is the floor: with every
provider unreachable, every key unset and the daily call budget spent, these functions still
produce a usable explanation of what the system decided and why. The SRS requires that, and
`tests/agents/test_narration_survives_total_outage.py` is the test that holds it.

That makes this the most important file in the package and the least interesting one. It is
plain sentence assembly over fields that are already there — no scoring, no interpretation, no
judgement the deterministic core did not already make. Where the core wrote a reason, the
template quotes it; where the core stayed silent, the template says so rather than filling in.

**The numbers are the core's own.** Nothing here computes a level, a ratio or a threshold that
is not already on the briefing, which is the same constraint the agents work under — so a
template paragraph and a model paragraph make claims of exactly the same kind, and a reader
cannot be misled about which one they are looking at by how confident it sounds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fxagent.agents.schemas import (
    MAX_PLAN_SUMMARY_CHARS,
    MAX_READ_CHARS,
    Briefing,
    VoteLine,
    direction_word,
    sentences,
)

__all__ = [
    "ADVISORY_ONLY",
    "PROVENANCE",
    "execution_sentence",
    "risk_officer_summary",
    "risk_reasoning",
    "size_rationale",
    "spread_sentence",
    "chartist_fallback",
    "chartist_read",
    "explain",
    "fit",
    "historian_fallback",
    "plan_sentence",
    "regime_sentence",
    "verdict_headline",
    "verdict_sentence",
    "votes_sentence",
]

#: Appended to every template paragraph. A reader looking at a card in the panel needs to know
#: whether a model wrote it, and the alternative — relying on the provider label being rendered
#: — puts that distinction in the hands of a stylesheet.
PROVENANCE = "Written from the computed record, not by a model."

#: Carried in the risk officer's own `reasoning` text, not applied by whatever renders it. This
#: is the one field in the system most likely to be read as an instruction, and a disclaimer
#: that lives in a stylesheet is a disclaimer one CSS change removes.
ADVISORY_ONLY = (
    "This recommendation is advisory. It is displayed and recorded, and it gates nothing — the "
    "deterministic permission layer decides whether an order is placed."
)


def _pct(value: float) -> str:
    """A fraction of equity as a percentage. `0.005` is unreadable; `0.50%` is the cap."""
    return f"{value * 100:.2f}%"


def _fmt(value: float | None, places: int = 2) -> str:
    return "unknown" if value is None else f"{value:.{places}f}"


def _ordinal(value: float) -> str:
    """`62` -> `62nd`. The teens are the exception every naive version gets wrong."""
    number = int(round(value))
    suffix = (
        "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    )
    return f"{number}{suffix}"


def regime_sentence(briefing: Briefing) -> str:
    """Where the market is and what it is doing. Warm-up reports as unknown, never as zero."""
    if not briefing.market_open:
        where = f"{briefing.symbol} with the market closed"
    elif briefing.sessions:
        where = f"{briefing.symbol} in the {', '.join(briefing.sessions)} session"
        if len(briefing.sessions) > 1:
            where += "s"
    else:
        where = f"{briefing.symbol} outside any named session"

    if briefing.trend_strength is None:
        state = (
            "trend strength is still warming up, so the market is classified as neither "
            "trending nor ranging"
        )
    elif briefing.is_trending:
        state = f"ADX {_fmt(briefing.trend_strength, 1)} puts it in a trend"
    elif briefing.is_ranging:
        state = f"ADX {_fmt(briefing.trend_strength, 1)} puts it in a range"
    else:
        state = (
            f"ADX {_fmt(briefing.trend_strength, 1)} sits between the two thresholds, "
            "so it is neither"
        )

    if briefing.volatility_percentile is not None:
        state += (
            f", with the current range at the {_ordinal(briefing.volatility_percentile)} "
            "percentile of its trailing window"
        )

    return f"{where}: {state}."


def _vote_phrase(vote: VoteLine) -> str:
    """One vote as a clause, quoting the reason `Consensus` itself recorded."""
    if vote.direction is None:
        return f"{vote.strategy} was {vote.reason or 'silent'}"
    stated = f"{vote.strategy} said {direction_word(vote.direction)}"
    if vote.confidence is not None:
        stated += f" at {_fmt(vote.confidence)} confidence"
    stated += f" on weight {_fmt(vote.weight)}"
    if vote.reason:
        stated += f" ({vote.reason})"
    return stated


def votes_sentence(briefing: Briefing) -> str:
    """Every strategy on the router's slate, including the ones that did not vote.

    The silent and gated ones are named rather than omitted. A list of only the strategies
    that spoke reads as unanimity, and on most bars the interesting fact is which strategy was
    not allowed to speak.
    """
    if not briefing.votes:
        return "No strategy was on the router's slate for this bar."
    return "; ".join(_vote_phrase(vote) for vote in briefing.votes) + "."


def verdict_sentence(briefing: Briefing) -> str:
    """What consensus concluded, with the thresholds it was measured against."""
    tally = (
        f"LONG carried {_fmt(briefing.long_weight)} across {briefing.long_votes} vote(s) and "
        f"SHORT carried {_fmt(briefing.short_weight)} across {briefing.short_votes}, against a "
        f"bar of {_fmt(briefing.min_total_weight)} summed weight and {briefing.min_agreeing} "
        f"agreeing strategies"
    )
    if briefing.fired:
        headline = f"Consensus fired {direction_word(briefing.winning_direction).upper()}"
    else:
        headline = "Consensus declined"
    reason = f" — {briefing.reason}" if briefing.reason else ""
    return f"{headline}{reason}. {tally}."


def plan_sentence(briefing: Briefing) -> str:
    """The levels, or an explicit statement that there are none.

    "No plan" is a sentence rather than an omission: a card with a regime and votes and then
    nothing where the levels go is indistinguishable from a card whose levels failed to render.
    """
    plan = briefing.plan
    if plan is None:
        return "There is no trade plan, because nothing qualified."
    return (
        f"The plan is {direction_word(plan.direction).upper()} from {plan.entry_price}, "
        f"stop {plan.stop_loss}, target {plan.take_profit} — {_fmt(plan.reward_risk, 1)}R at "
        f"{_fmt(plan.confidence)} confidence, on {plan.primary_strategy}'s levels taken whole."
    )


def explain(briefing: Briefing) -> str:
    """The full deterministic explanation of one evaluation. Always available."""
    return sentences(
        [
            regime_sentence(briefing),
            votes_sentence(briefing),
            verdict_sentence(briefing),
            plan_sentence(briefing),
            PROVENANCE,
        ]
    )


def verdict_headline(briefing: Briefing) -> str:
    """The verdict without the threshold tally — what fits on a card beside the vote list."""
    if briefing.fired:
        headline = f"Consensus fired {direction_word(briefing.winning_direction).upper()}"
    else:
        headline = "Consensus declined"
    return f"{headline}{f' — {briefing.reason}' if briefing.reason else ''}."


def fit(parts: Sequence[str], limit: int, *, always: str = "") -> str:
    """Assemble `parts` into at most `limit` characters, dropping whole sentences from the end.

    Never truncates mid-sentence. A card cut off at "the plan is LONG from 1.1, sto" is worse
    than one that stopped a sentence earlier, because a clipped number reads as a real one.
    `always` is appended whatever else is dropped, which is how the provenance line survives.
    """
    kept = list(parts)
    while kept:
        candidate = sentences([*kept, always])
        if len(candidate) <= limit:
            return candidate
        kept.pop()
    return sentences([always])[:limit]


def chartist_read(briefing: Briefing) -> str:
    """The chartist's card, deterministically, inside the same length its schema allows.

    Ordered most-important-first and trimmed from the end, so on a bar with a lot to say the
    formations drop out before the plan does — the formations are context by definition, and
    they travel structurally in `patterns_seen` regardless of whether the prose mentions them.
    """
    parts = [regime_sentence(briefing), verdict_headline(briefing), plan_sentence(briefing)]
    if briefing.patterns:
        names = ", ".join(sorted({pattern.name for pattern in briefing.patterns}))
        parts.append(f"Formations present, as context only: {names}.")
    return fit(parts, MAX_READ_CHARS, always=PROVENANCE)


def chartist_fallback(briefing: Briefing) -> str:
    """The long-form structural narration, for callers that are not filling a card."""
    if briefing.plan is None:
        closing = "There is no direction to check the regime against."
    elif briefing.is_trending and briefing.plan.direction in {"LONG", "SHORT"}:
        closing = (
            f"The measured trend and the chosen {direction_word(briefing.plan.direction).upper()} "
            "are consistent, which is a statement about the two measurements and not a forecast."
        )
    elif briefing.is_ranging:
        closing = (
            "The market measures as ranging, so a directional plan here rests on the strategies' "
            "own gates rather than on trend strength."
        )
    else:
        closing = "Trend strength does not classify this bar either way."

    # The verdict is carried even though this card is nominally about structure. On a bar that
    # declined — which is most of them — "why did nothing happen" is the only question a reader
    # has, and a card that described the regime and then went quiet would answer the one
    # question nobody asked.
    return sentences(
        [
            regime_sentence(briefing),
            verdict_sentence(briefing),
            plan_sentence(briefing),
            closing,
            PROVENANCE,
        ]
    )


def historian_fallback(briefing: Briefing) -> str:
    """What the historian would have narrated: the retrieved analogues and how they resolved.

    An empty retrieval is stated as an empty retrieval. Point-in-time filtering removes windows
    whose outcome had not resolved by this bar, so "no analogue" is a routine and informative
    answer early in the record — and one that must never be confused with a failed lookup.
    """
    if not briefing.analogues:
        return sentences(
            [
                f"No resolved analogue was retrievable for {briefing.symbol} at this bar.",
                "Retrieval only returns windows whose outcome had already resolved, so an empty "
                "result means the record does not yet reach back far enough, not that the "
                "lookup failed.",
                PROVENANCE,
            ]
        )

    resolved = [item for item in briefing.analogues if item.outcome is not None]
    lines = [
        f"{len(briefing.analogues)} resolved analogue(s) were retrieved for {briefing.symbol}, "
        f"the nearest at {_fmt(max(item.similarity for item in briefing.analogues))} similarity."
    ]
    if resolved:
        scored = [item.outcome_r for item in resolved if item.outcome_r is not None]
        if scored:
            lines.append(
                f"Their outcomes range from {_fmt(min(scored), 1)}R to {_fmt(max(scored), 1)}R."
            )
        lines.append(
            "Outcomes are reported as they resolved, and every window shown resolved before "
            "this bar."
        )
    lines.append(PROVENANCE)
    return sentences(lines)


# -- the risk officer's card -----------------------------------------------------------------


def execution_sentence(briefing: Briefing) -> str:
    """The sized order against the caps it was sized under, or a statement that there is none."""
    execution = briefing.execution
    if execution is None:
        return "No order was sized, so there is nothing to review."
    return (
        f"{execution.volume} lots risking {_fmt(execution.risk_amount)} over a "
        f"{execution.stop_distance} stop — {_pct(execution.risk_fraction)} of equity against a "
        f"{_pct(execution.max_risk_per_trade)} cap, with {_pct(execution.total_open_risk)} open "
        f"in total against {_pct(execution.max_total_risk)}."
    )


def spread_sentence(briefing: Briefing) -> str:
    """Dealing conditions, as a multiple of their own median."""
    if briefing.spread is None:
        return "Spread was not measured for this bar."
    return (
        f"The spread is {briefing.spread.spread} against a median of "
        f"{briefing.spread.median_spread} — {_fmt(briefing.spread.over_median, 1)}x."
    )


def risk_officer_summary(briefing: Briefing) -> str:
    """The card's text, inside the length the schema allows. Trimmed by whole sentences."""
    plan = briefing.plan
    opening = (
        "There is no order to review."
        if plan is None
        else (
            f"{direction_word(plan.direction).upper()} {briefing.symbol} from {plan.entry_price}, "
            f"stop {plan.stop_loss}, target {plan.take_profit}."
        )
    )
    return fit(
        [opening, execution_sentence(briefing), spread_sentence(briefing)],
        MAX_PLAN_SUMMARY_CHARS,
        always=PROVENANCE,
    )


def size_rationale(briefing: Briefing) -> str:
    """Where the size came from. Stop distance and risk fraction, which is the whole derivation.

    Stated as a derivation rather than as a number, because "0.04 lots" tells a reader nothing
    they can check and "risk divided by stop distance, rounded down to the lot step" tells them
    exactly what to check it against.
    """
    execution = briefing.execution
    if execution is None:
        return "No size was computed, because nothing qualified."
    return sentences(
        [
            f"Size came from the risk budget divided by the stop distance: "
            f"{_pct(execution.risk_fraction)} of equity ({_fmt(execution.risk_amount)}) over "
            f"{execution.stop_distance}, rounded down to the broker's lot step at "
            f"{execution.volume}.",
            "Risk units, never fixed lots — so a wider stop buys a smaller position rather than "
            "a larger loss.",
        ]
    )


def risk_reasoning(briefing: Briefing, flags: Sequence[Mapping[str, str]]) -> str:
    """Why the card says what it says, and what the recommendation is worth.

    The advisory disclaimer is part of the text rather than something the panel adds, for the
    same reason a pattern carries its own CONTEXT label: a caption can be styled away, and this
    is the one field in the system most likely to be read as an instruction.
    """
    if not flags:
        counted = "No flagged condition was found in the computed record."
    else:
        loudest = "CRITICAL" if any(f["severity"] == "CRITICAL" for f in flags) else "WARN"
        counted = f"{len(flags)} condition(s) flagged, the loudest at {loudest}: " + " ".join(
            flag["flag"] for flag in flags
        )
    return sentences([counted, ADVISORY_ONLY, PROVENANCE])
