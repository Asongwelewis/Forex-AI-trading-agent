"""A WAIT stops nothing. The single most important assertion in the agent layer.

CLAUDE.md: `proceed_recommendation` is advisory only. It is displayed and logged. It does not
gate execution — the deterministic permission layer does, and nothing else.

Two halves, and both are needed.

**Behavioural.** Narrate the same briefing three times: with the risk officer recommending
PROCEED, recommending WAIT, and unreachable. Everything except the risk officer's own block
must come out identical, and the trade plan must survive all three unchanged. A gate wired in
anywhere would show up as a difference here.

**Structural.** The behavioural test proves today's code does not read the field. It cannot
prove tomorrow's will not, because the wiring that would break it does not exist yet — the risk
and permission layers are unbuilt, and a gate added there would sail past a test that only
narrates. So the second half greps the package: outside the agent that produces the field and
the panel that displays it, nothing may so much as name it.

That is the same shape as `tests/patterns/test_patterns_never_reach_consensus.py`, and for the
same reason. The failure being prevented is invisible from the outside: a suppressed winner
leaves nothing in an equity curve to explain itself, so it accrues silently while the system
looks prudent the whole time.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import fxagent
from fxagent.agents import risk_officer
from fxagent.agents.gateway import Gateway, Prompt, ProviderConfig, ProviderError
from fxagent.agents.narrate import LEGACY_AGENTS, TEMPLATE_PROVIDER, attach_narration, narrate
from fxagent.agents.schemas import PROCEED_RECOMMENDATIONS
from fxagent.dashboard.contract import CHARTIST, HISTORIAN, RISK_OFFICER, read_agents
from tests.agents.builders import MOMENT, execution, fired_briefing

PACKAGE_ROOT = Path(fxagent.__file__).parent
FIELD = "proceed_recommendation"

#: The two places the field is allowed to exist: the agent that produces it, and the panel that
#: renders it. `schemas` defines it, `narrate` copies the block through, `contract` and `models`
#: read it back out for display.
PERMITTED = {
    Path("agents/risk_officer.py"),
    Path("agents/schemas.py"),
    Path("agents/narrate.py"),
    Path("dashboard/contract.py"),
    Path("dashboard/models.py"),
}

PROVIDER = ProviderConfig(
    name="alpha", model="alpha/m", api_key_env="ALPHA_KEY", min_interval_seconds=0
)


def _risk_response(recommendation: str) -> str:
    return json.dumps(
        {
            "plan_summary": "A long, already sized and stopped.",
            "risk_flags": [{"flag": "The spread is wide.", "severity": "CRITICAL"}],
            "size_rationale": "Risk over stop distance, rounded down to the lot step.",
            "proceed_recommendation": recommendation,
            "reasoning": "Conditions are poor.",
        }
    )


CHARTIST_RESPONSE = json.dumps({"read": "The regime is trending."})
HISTORIAN_RESPONSE = json.dumps({"text": "No resolved analogue was retrievable."})


class Scripted:
    def __init__(self, risk: str | None) -> None:
        self._risk = risk

    async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
        if prompt.agent == CHARTIST:
            return CHARTIST_RESPONSE
        if prompt.agent == HISTORIAN:
            return HISTORIAN_RESPONSE
        if self._risk is None:
            raise ProviderError("the endpoint returned 500")
        return self._risk


async def _nap(_seconds: float) -> None: ...


def _gateway(risk: str | None) -> Gateway:
    # Fixed clock: `generated_at` is stamped from it, and three runs compared block-for-block
    # would otherwise differ on the one field that has nothing to do with the recommendation.
    return Gateway(
        (PROVIDER,),
        transport=Scripted(risk),  # type: ignore[arg-type]
        env={"ALPHA_KEY": "k"},
        sleep=_nap,
        now=lambda: MOMENT,
    )


# -- behavioural -----------------------------------------------------------------------------


async def test_a_wait_leaves_the_trade_plan_exactly_as_the_core_computed_it() -> None:
    """The requirement, stated directly. The plan is the deterministic core's and stays so."""
    briefing = fired_briefing(execution=execution())
    before = briefing.plan.model_dump() if briefing.plan is not None else None

    blocks = await narrate(briefing, gateway=_gateway(_risk_response("WAIT")), agents=LEGACY_AGENTS)

    assert blocks[RISK_OFFICER][FIELD] == "WAIT"
    assert briefing.plan is not None
    assert briefing.plan.model_dump() == before
    assert briefing.plan.direction == "LONG"
    assert briefing.fired is True


async def test_a_wait_changes_nothing_a_proceed_would_not() -> None:
    """Three runs, one briefing. Only the risk officer's own block may differ.

    This is the assertion a gate would break, wherever it were wired in: if narration could
    suppress anything, the WAIT run and the PROCEED run would stop being the same document.
    """
    briefing = fired_briefing(execution=execution())

    proceeding = await narrate(
        briefing, gateway=_gateway(_risk_response("PROCEED")), agents=LEGACY_AGENTS
    )
    waiting = await narrate(
        briefing, gateway=_gateway(_risk_response("WAIT")), agents=LEGACY_AGENTS
    )
    silent = await narrate(briefing, gateway=_gateway(None), agents=LEGACY_AGENTS)

    for other in (waiting, silent):
        assert set(other) == set(proceeding)
        assert other[CHARTIST] == proceeding[CHARTIST]
        assert other[HISTORIAN] == proceeding[HISTORIAN]

    assert proceeding[RISK_OFFICER][FIELD] == "PROCEED"
    assert waiting[RISK_OFFICER][FIELD] == "WAIT"
    # Unreachable falls to the template, which fills the same field. Also advisory, also read
    # by nothing — the derivation lives in `risk_officer._recommendation`.
    assert silent[RISK_OFFICER][FIELD] in PROCEED_RECOMMENDATIONS
    assert silent[RISK_OFFICER]["provider"] == TEMPLATE_PROVIDER


async def test_the_stored_evaluation_still_says_the_signal_fired() -> None:
    """A WAIT must not be able to turn a fired evaluation into a declined one in the record."""
    briefing = fired_briefing(execution=execution())
    diagnostics = {"fired": True, "winning_direction": "LONG", "reason": "2 strategies agreed"}

    blocks = await narrate(briefing, gateway=_gateway(_risk_response("WAIT")), agents=LEGACY_AGENTS)
    document = attach_narration(diagnostics, blocks)

    assert document["fired"] is True
    assert document["winning_direction"] == "LONG"
    assert document["reason"] == "2 strategies agreed"


async def test_the_panel_shows_the_wait_beside_a_plan_that_is_still_there() -> None:
    """Displayed, which is the whole permitted use — next to the levels it did not change."""
    briefing = fired_briefing(execution=execution())

    blocks = await narrate(briefing, gateway=_gateway(_risk_response("WAIT")), agents=LEGACY_AGENTS)
    parsed = read_agents({"agents": blocks})

    assert parsed.discarded == ()
    assert parsed.risk_officer is not None
    assert parsed.risk_officer.proceed_recommendation == "WAIT"
    assert parsed.risk_officer.text
    assert briefing.plan is not None and briefing.plan.take_profit > briefing.plan.entry_price


async def test_narrate_returns_no_signal_a_caller_could_mistake_for_a_gate() -> None:
    """There is no boolean, no veto and no filtered plan in what this function returns.

    A caller cannot accidentally honour a WAIT, because `narrate` gives it nothing to honour:
    the return value is one text block per agent and nothing else.
    """
    blocks = await narrate(
        fired_briefing(execution=execution()),
        gateway=_gateway(_risk_response("WAIT")),
        agents=LEGACY_AGENTS,
    )

    assert set(blocks) == {CHARTIST, HISTORIAN, RISK_OFFICER}
    for block in blocks.values():
        assert isinstance(block, dict)
        assert not any(isinstance(value, bool) for value in block.values())


# -- structural ------------------------------------------------------------------------------


def _sources() -> list[Path]:
    return sorted(path for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _named_in_code(path: Path) -> list[int]:
    """Line numbers where the field is named in *code* — never in a docstring or a comment.

    An AST walk rather than a substring search, and deliberately: several modules discuss the
    field in prose because its advisory status is the thing most worth explaining, and a test
    that punished them for saying so would be an argument for saying less. A docstring is one
    `ast.Constant` holding the whole string, so it never compares equal to the field name.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[int] = []
    for node in ast.walk(tree):
        named = (
            (isinstance(node, ast.Attribute) and node.attr == FIELD)
            or (isinstance(node, ast.Name) and node.id == FIELD)
            or (isinstance(node, ast.arg) and node.arg == FIELD)
            or (isinstance(node, ast.keyword) and node.arg == FIELD)
            or (isinstance(node, ast.Constant) and node.value == FIELD)
        )
        if named:
            found.append(getattr(node, "lineno", 0))
    return found


def test_there_are_sources_to_check() -> None:
    """Guards the guard: an empty glob would make every assertion below pass vacuously."""
    sources = _sources()

    assert len(sources) >= 30
    assert any(_named_in_code(path) for path in sources), (
        "the field is not named in code anywhere, so this test is checking nothing"
    )


@pytest.mark.parametrize(
    "path", _sources(), ids=lambda p: str(p.relative_to(PACKAGE_ROOT)).replace("\\", "/")
)
def test_only_the_producer_and_the_panel_name_the_recommendation(path: Path) -> None:
    relative = path.relative_to(PACKAGE_ROOT)
    if relative in PERMITTED:
        return

    assert not _named_in_code(path), (
        f"{relative} names {FIELD} in code. It is advisory only: it is displayed and logged, "
        "and the deterministic permission layer gates execution. An agent that can stop a "
        "trade can stop the right trade, and that failure leaves no trace in an equity curve."
    )


def _branch_lines(path: Path) -> list[tuple[int, str]]:
    """Every branch condition that reads the field, with the function it sits in.

    Walks `if`, `while`, ternaries, comprehension guards and boolean operators. Naming the
    field is one thing; *deciding* on it is the thing that must not exist.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    enclosing: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                enclosing.setdefault(getattr(inner, "lineno", 0), node.name)

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        tests: list[ast.expr] = []
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            tests.append(node.test)
        elif isinstance(node, ast.BoolOp):
            tests.extend(node.values)
        elif isinstance(node, ast.comprehension):
            tests.extend(node.ifs)

        for test in tests:
            for inner in ast.walk(test):
                if (isinstance(inner, ast.Attribute) and inner.attr == FIELD) or (
                    isinstance(inner, ast.Constant) and inner.value == FIELD
                ):
                    line = getattr(node, "lineno", 0)
                    found.append((line, enclosing.get(line, "<module>")))
    return found


#: The one branch on the field that is not a gate, and is required: the schema rejecting a
#: value outside the three-word vocabulary. Pinned by name so it cannot grow siblings.
VALIDATOR = "_recommendation_is_one_of_three"


def test_no_branch_anywhere_decides_on_the_recommendation() -> None:
    """Exempts nothing by file. The one permitted branch is named, and checked below."""
    offenders = [
        f"{path.relative_to(PACKAGE_ROOT)}:{line} in {where}"
        for path in _sources()
        for line, where in _branch_lines(path)
        if where != VALIDATOR
    ]

    assert not offenders, (
        f"a branch decides on {FIELD} at {offenders}. It is advisory only — displayed and "
        "logged, and read by nothing."
    )


def test_the_one_permitted_branch_only_rejects_a_word_outside_the_vocabulary() -> None:
    """The exemption above, pinned. A validator refusing an unknown string is the opposite of a
    gate: it narrows what the field may say, and still nothing reads what it says."""
    schemas = PACKAGE_ROOT / "agents" / "schemas.py"
    branches = _branch_lines(schemas)

    assert branches, "the vocabulary check has gone; the field can now be any string"
    assert {where for _, where in branches} == {VALIDATOR}


def test_the_field_is_prose_with_three_values_and_no_ordering_a_gate_could_use() -> None:
    """No numeric weight, so there is nothing to threshold and nothing to sum."""
    assert PROCEED_RECOMMENDATIONS == ("PROCEED", "CAUTION", "WAIT")
    assert all(isinstance(value, str) for value in PROCEED_RECOMMENDATIONS)


def test_the_advisory_status_is_stated_in_the_text_the_agent_itself_writes() -> None:
    """Carried by the data, not applied by a renderer. A caption can be styled away."""
    from fxagent.agents import templates

    block = risk_officer.fallback(fired_briefing(execution=execution()))

    assert templates.ADVISORY_ONLY in block["reasoning"]
    assert "gates nothing" in templates.ADVISORY_ONLY
    assert "ADVISORY" in risk_officer.SYSTEM
