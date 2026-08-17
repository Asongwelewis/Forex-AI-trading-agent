"""The risk officer: reads a plan it did not choose, and cannot change.

**`proceed_recommendation` is advisory. That is the whole design constraint of this file.**

It is displayed on the panel, written to the journal, and read by nothing. The deterministic
permission layer decides whether an order is placed; this agent's opinion is not an input to
that decision, not a tie-breaker, and not a veto. `tests/agents/test_advisory_only.py` holds
both halves of the claim — behaviourally, that a WAIT leaves the trade plan and the consensus
result bit-identical to a PROCEED; and structurally, that no module outside this package and
the panel so much as names the field.

The temptation here is stronger than anywhere else in the system, and worth naming. Of the
three agents this is the only one whose output *reads* like a control, and it sits next to the
one decision where being wrong costs money — so "surely we should at least pause on a WAIT"
is the obvious next thought, and it is the thought hard rule 4 exists to refuse. An agent that
can stop a trade is an agent that can stop the right trade, and that failure is invisible: a
suppressed winner leaves nothing in the equity curve to explain itself, so the harm accrues
silently and the model looks prudent the whole time. The deterministic gate can be backtested.
This cannot.

**Everything it reads was computed upstream.** The consensus signal, the sized volume, the stop
and target, the spread against its own median, the events near this bar, the total open risk
against its cap, and what the other two agents said. It computes nothing, and there is no field
on `RiskOfficerNote` that a number could be read out of — `risk_flags` carries a severity label
with no weight, on purpose, because a severity that could be summed is a score.

**Provider.** NVIDIA NIM, `deepseek-ai/deepseek-v4-pro`, via `integrate.api.nvidia.com/v1`.
A developer reported that endpoint returning internal server errors on trivial prompts, so the
fallback is here from the first commit rather than after the first outage: `prefer` names three
models on that one endpoint — deepseek, then Nemotron, then GLM — and the ordinary chain
continues past them to Gemini and Groq. A 500 is classified as "this model will not serve this
call" and falls through immediately rather than being retried, because a systematic error on
trivial prompts is not a transient one and the rung below is a better use of the budget.

**Twenty calls a day, enforced in code.** `DAILY_CALL_LIMIT` travels to the gateway on the
prompt and is refused there by the same counter that holds the shared cap. It is also not asked
at all on a bar with no plan: there is nothing to review, and spending one of twenty on
"nothing qualified" is how a budget is gone by lunchtime.
"""

from __future__ import annotations

from typing import Any

from fxagent.agents import templates
from fxagent.agents.schemas import (
    MAX_NOTE_CHARS,
    MAX_PLAN_SUMMARY_CHARS,
    Briefing,
    RiskFlag,
    RiskOfficerNote,
)
from fxagent.agents.spec import AgentSpec
from fxagent.dashboard.contract import RISK_OFFICER

__all__ = [
    "DAILY_CALL_LIMIT",
    "NAME",
    "SPEC",
    "SYSTEM",
    "THRESHOLDS",
    "RiskThresholds",
    "deterministic_flags",
    "fallback",
    "with_flags",
]

NAME = RISK_OFFICER

#: Outbound requests a day. Low because this is the most expensive prompt in the system — it
#: carries the execution plan, the calendar and both other narrations — and because it is only
#: worth asking on a bar that produced an order. Enforced by `CallBudget`, not by convention.
DAILY_CALL_LIMIT = 20


class RiskThresholds:
    """When a condition becomes worth a flag, and how loud.

    A plain class of constants rather than anything configurable: these decide what a human is
    *told*, never what the system does, so there is nothing here to tune and nothing that could
    be tuned into a decision. The two caps that do matter — per-trade and total risk — are hard
    rule 8's, live in `fxagent.risk`, and arrive on the briefing already applied.
    """

    #: Spread at this multiple of its own median is worth mentioning...
    SPREAD_WARN = 1.5
    #: ...and at this multiple it is the dominant fact about the trade. Exness is a market-maker
    #: CFD broker and its spreads widen on news, so this is routine rather than exotic.
    SPREAD_CRITICAL = 2.5

    #: The window the brief covers. The permission layer blacks out fifteen minutes around a
    #: high-impact release; an hour is the wider view a human wants before that bites.
    EVENT_WINDOW_MINUTES = 60
    #: Inside the blackout the permission layer is about to act on its own.
    EVENT_BLACKOUT_MINUTES = 15

    #: Open risk at this fraction of the total cap leaves little room for the next setup.
    OPEN_RISK_WARN = 0.75

    #: Minutes to the weekly close under which a multi-day plan is a weekend gap waiting to
    #: happen. Grants expire and force flat before Friday close (CLAUDE.md, known traps).
    WEEKEND_WARN_MINUTES = 240


THRESHOLDS = RiskThresholds()


SYSTEM = (
    "You are the risk officer. You are given a JSON record of an analysis and an order that "
    "deterministic code has already completed and sized: the agreed direction, the volume, the "
    "stop and target, the risk taken against its cap, the total open risk against its cap, the "
    "spread against its own median, scheduled releases near this bar, and what the other agents "
    "said.\n"
    "You review it for a human reader. You did not choose any of it and you cannot change any "
    "of it.\n\n"
    "Rules, all absolute:\n"
    "1. Never propose a different size, stop, target or direction. They are already set. If you "
    "think one is wrong, say so in `reasoning` as an observation — not as an instruction.\n"
    "2. Never state a number that does not appear in the input. Not a rounded estimate, not a "
    "derived figure, not a ratio you worked out. Any response containing one is discarded in "
    "full.\n"
    "3. `proceed_recommendation` is ADVISORY. It is shown to a person and recorded. It does not "
    "gate execution — a deterministic permission layer does that, and it will not read your "
    "answer. Recommend WAIT freely where the conditions warrant it; understand that doing so "
    "stops nothing.\n"
    "4. Describe only what the input says. Where it does not answer something, say so.\n\n"
    "Return one JSON object and nothing else — no prose outside it, no code fence:\n"
    '{"plan_summary": str, "risk_flags": [{"flag": str, "severity": '
    '"INFO"|"WARN"|"CRITICAL"}], "size_rationale": str, "proceed_recommendation": '
    '"PROCEED"|"CAUTION"|"WAIT", "reasoning": str}\n\n'
    f"`plan_summary` is under {MAX_PLAN_SUMMARY_CHARS} characters: what the order is, in a "
    "sentence or two. It sits directly above the levels, so do not restate them at length.\n"
    "`risk_flags` are the conditions a reader should notice. Some are already listed in the "
    "input under `risk_flags`; repeat the ones that matter and add any the input supports.\n"
    "`size_rationale` explains what the size was derived from — the stop distance and the risk "
    "fraction are both in the input.\n"
    f"`reasoning` is under {MAX_NOTE_CHARS} characters and is where a disagreement belongs."
)


def _flag(flag: str, severity: str) -> dict[str, str]:
    return {"flag": flag, "severity": severity}


def deterministic_flags(briefing: Briefing) -> list[dict[str, str]]:
    """The conditions the computed record already establishes, without a model involved.

    These are facts, not judgements: each one is a comparison between two numbers that are both
    on the briefing. They go *into* the prompt as well as into the fallback, so the agent is
    reviewing the same conditions a reader can check rather than hunting for them — and so a
    provider outage costs the card its prose and not its content.
    """
    flags: list[dict[str, str]] = []
    execution = briefing.execution

    if execution is not None:
        if execution.over_trade_cap:
            flags.append(
                _flag(
                    "The risk on this trade is above the per-trade cap. The sizer should have "
                    "made that impossible, so this is a defect and not a close call.",
                    "CRITICAL",
                )
            )
        if execution.over_total_cap:
            flags.append(
                _flag("Total open risk is above its cap with this order included.", "CRITICAL")
            )
        elif execution.total_risk_used >= THRESHOLDS.OPEN_RISK_WARN:
            flags.append(
                _flag(
                    "Total open risk is most of the way to its cap, leaving little room for the "
                    "next setup.",
                    "WARN",
                )
            )

    if briefing.spread is not None:
        over = briefing.spread.over_median
        if over >= THRESHOLDS.SPREAD_CRITICAL:
            flags.append(
                _flag(
                    "The spread is far above its own median. Entry and exit both cost more than "
                    "the plan assumed.",
                    "CRITICAL",
                )
            )
        elif over >= THRESHOLDS.SPREAD_WARN:
            flags.append(_flag("The spread is above its own median.", "WARN"))

    for event in briefing.events:
        if abs(event.minutes_until) > THRESHOLDS.EVENT_WINDOW_MINUTES:
            continue
        inside_blackout = abs(event.minutes_until) <= THRESHOLDS.EVENT_BLACKOUT_MINUTES
        flags.append(
            _flag(
                f"{event.importance} release near this bar: {event.title} ({event.currency})."
                + (
                    " Inside the window the permission layer blacks out on its own."
                    if inside_blackout
                    else ""
                ),
                "CRITICAL" if inside_blackout else "WARN",
            )
        )

    if (
        briefing.plan is not None
        and briefing.minutes_until_weekly_close <= THRESHOLDS.WEEKEND_WARN_MINUTES
    ):
        flags.append(
            _flag(
                "The weekly close is near. A position carried through it is exposed to the "
                "weekend gap, and grants expire before Friday close.",
                "WARN",
            )
        )

    return flags


def _recommendation(flags: list[dict[str, str]], briefing: Briefing) -> str:
    """The deterministic card's recommendation, derived from the flags it just listed.

    Derived rather than hard-coded so the fallback card is worth reading, and — to be explicit
    about what this is not — derived on a path that gates nothing. It produces the same
    advisory string the LLM produces, into the same field nothing reads. If that ever stops
    being true, `tests/agents/test_advisory_only.py` fails before anything reaches a broker.
    """
    if briefing.plan is None:
        return "WAIT"
    severities = {flag["severity"] for flag in flags}
    if "CRITICAL" in severities:
        return "WAIT"
    if "WARN" in severities:
        return "CAUTION"
    return "PROCEED"


def fallback(briefing: Briefing) -> dict[str, Any]:
    """The whole card, deterministically — every field the LLM path fills, from the record.

    The flags are the interesting half and they need no model at all: each is a comparison
    between two numbers already on the briefing. So an unreachable provider costs this card its
    prose and none of its substance, which matters more here than for the other two agents —
    the chartist going quiet loses commentary, and this going quiet would lose the only place a
    reader is told the spread has tripled.
    """
    # Already derived when `narrate` applied `with_flags`; recomputed identically when a
    # caller invokes this directly. The same comparison over the same numbers either way.
    flags = [
        {"flag": flag.flag, "severity": flag.severity} for flag in briefing.risk_flags
    ] or deterministic_flags(briefing)
    return {
        "text": templates.risk_officer_summary(briefing),
        "risk_flags": flags,
        "size_rationale": templates.size_rationale(briefing),
        "proceed_recommendation": _recommendation(flags, briefing),
        "reasoning": templates.risk_reasoning(briefing, flags),
    }


def with_flags(briefing: Briefing) -> Briefing:
    """The briefing this agent is shown: the record, plus the flags the record already implies.

    The prompt tells the model some conditions are already listed under `risk_flags`, and this
    is what makes that true. It also makes them *quotable*: grounding is taken from the payload
    actually rendered, so a flag the model repeats — and the count of them — is a phrase and a
    number it was given rather than ones it produced.
    """
    return briefing.model_copy(
        update={"risk_flags": tuple(RiskFlag(**flag) for flag in deterministic_flags(briefing))}
    )


def _worth_asking(briefing: Briefing) -> bool:
    """Only ask a provider when there is an order to review.

    On most bars nothing qualifies, and a briefing with no execution plan has nothing for this
    agent to say that the template cannot say better. Twenty calls a day does not survive being
    spent on "there is no order".
    """
    return briefing.execution is not None or briefing.plan is not None


SPEC = AgentSpec(
    name=NAME,
    note=RiskOfficerNote,
    system=SYSTEM,
    fallback=fallback,
    # The ladder is the fallback. Three models on one NVIDIA endpoint, then the ordinary chain.
    prefer=("nvidia_deepseek", "nvidia_nemotron", "nvidia_glm"),
    daily_call_limit=DAILY_CALL_LIMIT,
    # The only agent shown the others' output — CLAUDE.md gives it the computed execution plan,
    # and a reviewer of that plan should see what was said about the analysis behind it.
    reads_agents=True,
    augment=with_flags,
    should_ask=_worth_asking,
)
