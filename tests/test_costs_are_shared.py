"""The cost model must stay importable by both consumers, and owned by neither.

Card 16's backtest and card 22's paper resolver charge the same spread, slippage and swap or
they are not comparable — and comparing them is the whole reason for running both. The way that
guarantee dies is not by someone editing one of two copies; it is by `fxagent.costs` acquiring
an import from `fxagent.backtest`, at which point the execution path can no longer take it
without dragging the harness along, and somebody writes a small local version instead.

So this is an import test: `fxagent.costs` reaches into no consumer, and the consumers reach
into it. It will fail the moment the module stops being neutral, which is well before the fork
that failure would otherwise cause.
"""

from __future__ import annotations

import ast
from pathlib import Path

import fxagent
import fxagent.costs

COSTS = Path(fxagent.costs.__file__)
PACKAGE_ROOT = Path(fxagent.__file__).parent

#: Importing any of these would make the cost model belong to one caller.
FORBIDDEN_PREFIXES = (
    "fxagent.backtest",
    "fxagent.executor",
    "fxagent.dashboard",
    "fxagent.agents",
    "fxagent.store",
)


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_the_cost_model_does_not_live_inside_the_backtest_package() -> None:
    """A module the executor must reach through `backtest/` is a module it will eventually copy."""
    assert COSTS.parent == PACKAGE_ROOT, (
        f"fxagent/costs.py has moved to {COSTS.parent}. It sits at the top level so that neither "
        "the backtest nor the paper resolver owns it; see its module docstring."
    )


def test_the_cost_model_reaches_into_no_consumer() -> None:
    reached = {name for name in _imported_names(COSTS) if name.startswith(FORBIDDEN_PREFIXES)}
    assert not reached, (
        f"fxagent/costs.py imports {sorted(reached)}. It must stay importable by the execution "
        "path without pulling a consumer in behind it."
    )


def test_the_backtest_uses_it_rather_than_its_own() -> None:
    """The invariant stated positively: the replay engine imports the shared module."""
    replay = PACKAGE_ROOT / "backtest" / "replay.py"
    assert "fxagent.costs" in _imported_names(replay)


def test_no_second_cost_implementation_has_appeared() -> None:
    """A `costs.py` anywhere else is the fork this file exists to catch."""
    others = [
        path
        for path in PACKAGE_ROOT.rglob("costs.py")
        if path != COSTS and "__pycache__" not in path.parts
    ]
    assert not others, (
        f"a second cost model exists at {[str(p) for p in others]}. Backtest and live results "
        "stop being comparable the moment these two disagree by a single pip."
    )
