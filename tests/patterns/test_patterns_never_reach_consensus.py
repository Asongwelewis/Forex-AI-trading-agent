"""`consensus.py` must not be able to see a candle formation. Enforced by import, not by review.

Two studies found candlestick formations produce no net positive return on EUR/USD after costs.
CLAUDE.md marks the package "CONTEXT ONLY — must not reach consensus.py" for that reason: a
formation that reached the vote would put a measured-worthless input into the one calculation
that decides whether money moves.

**The check is transitive.** A direct `import fxagent.patterns` in `consensus.py` is the version
of this mistake nobody makes. The one that happens is three commits later, when a helper module
consensus already imports grows a formation lookup — and a test that only read consensus.py's
own import list would stay green through it. So the closure is walked, and the whole of
`fxagent.regime` is checked rather than consensus alone: the router deciding which strategies
may speak is as much the decision path as the tally that follows it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

import fxagent
import fxagent.regime

PACKAGE_ROOT = Path(fxagent.__file__).parent
REGIME = Path(fxagent.regime.__file__).parent
CONSENSUS = REGIME / "consensus.py"

#: The decision path. Nothing under here may reach a formation, directly or through anything it
#: imports. `strategies` is included because a strategy's signal is what consensus counts.
DECISION_PATH = (
    "fxagent.regime",
    "fxagent.strategies",
)

FORBIDDEN = "fxagent.patterns"


def _imported_modules(path: Path) -> set[str]:
    """Every `fxagent.*` module name this file imports, from both statement forms."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return {name for name in names if name.startswith("fxagent")}


def _path_for(module: str) -> Path | None:
    """Where a dotted `fxagent.*` name lives on disk, module or package."""
    relative = Path(*module.split(".")[1:])
    candidate = PACKAGE_ROOT / relative.with_suffix(".py")
    if candidate.is_file():
        return candidate
    package = PACKAGE_ROOT / relative / "__init__.py"
    return package if package.is_file() else None


def _closure(start: Path) -> set[str]:
    """Every `fxagent` module reachable from `start` by following imports.

    A `from fxagent.x import Y` where Y is a name rather than a module resolves to `fxagent.x`,
    which is the module whose imports matter — so nothing is missed by not distinguishing them.
    """
    seen: set[str] = set()
    pending = [start]
    while pending:
        for module in _imported_modules(pending.pop()):
            if module in seen:
                continue
            seen.add(module)
            resolved = _path_for(module)
            if resolved is not None:
                pending.append(resolved)
    return seen


def _decision_path_modules() -> Iterator[Path]:
    for package in DECISION_PATH:
        root = PACKAGE_ROOT / Path(*package.split(".")[1:])
        yield from sorted(root.glob("*.py"))


DECISION_MODULES = list(_decision_path_modules())


def test_the_files_under_test_exist() -> None:
    """Guards the guard: a renamed module would make every assertion below pass vacuously."""
    assert CONSENSUS.is_file(), f"{CONSENSUS} is missing; this test is checking nothing"
    assert len(DECISION_MODULES) >= 6, (
        f"expected the decision path to have modules, got {DECISION_MODULES}"
    )
    assert (PACKAGE_ROOT / "patterns" / "__init__.py").is_file(), "fxagent.patterns is missing"


def test_consensus_does_not_import_patterns() -> None:
    """The requirement, stated directly."""
    assert FORBIDDEN not in _imported_modules(CONSENSUS), (
        "consensus.py imports fxagent.patterns. Candle formations are display context — two "
        "studies found no net positive return on EUR/USD after costs — and must never enter "
        "the consensus score."
    )


def test_consensus_cannot_reach_patterns_through_anything_it_imports() -> None:
    """The version of the mistake that actually happens: an indirect reach, three commits later."""
    reachable = _closure(CONSENSUS)
    offending = sorted(name for name in reachable if name.startswith(FORBIDDEN))
    assert not offending, (
        f"consensus.py can reach {offending} through its imports. The formations must stay "
        "unreachable from the decision path, not merely unimported by one file."
    )


@pytest.mark.parametrize("path", DECISION_MODULES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_module_on_the_decision_path_reaches_patterns(path: Path) -> None:
    reachable = _closure(path) | _imported_modules(path)
    offending = sorted(name for name in reachable if name.startswith(FORBIDDEN))
    assert not offending, (
        f"{path.parent.name}/{path.name} can reach {offending}. A strategy's signal is what "
        "consensus counts, so the formations must be as unreachable from a strategy as they "
        "are from the tally."
    )


def test_the_closure_walker_actually_walks() -> None:
    """Guards the guard again: a `_closure` that returned nothing would pass every test above.

    `consensus.py` imports `strategies.base` directly and `adapters.base` only through it, so
    finding the second proves the walk followed an edge rather than reading one import list.
    """
    reachable = _closure(CONSENSUS)

    assert "fxagent.strategies.base" in reachable
    assert "fxagent.adapters.base" in reachable
    assert "fxagent.indicators" in reachable  # reached via regime.classifier


def test_a_formation_is_labelled_context_by_the_data_and_not_by_the_renderer() -> None:
    """The other half of the rule: what does reach a screen says what it is."""
    from fxagent.patterns import CONTEXT_ONLY, PatternHit
    from tests.patterns.builders import MOMENT

    hit = PatternHit(name="doji", bar_index=3, timestamp=MOMENT)

    assert hit.label == CONTEXT_ONLY
    assert "NOT A SIGNAL" in hit.model_dump()["label"]
