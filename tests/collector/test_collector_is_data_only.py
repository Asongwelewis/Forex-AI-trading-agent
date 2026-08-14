"""The collector must not be able to analyse, and must not be able to trade.

Enforced by inspecting the package's own imports rather than by review. The split only earns
its complexity if the collector stays the dumbest process in the system — the moment it grows
an indicator call it acquires a way to crash that takes the data record down with it, and a
missed collection window cannot be recovered the way a missed analysis can.

An import test rather than a behavioural one because this is about what the code *can* reach,
not what it happens to do today.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import fxagent.collector

PACKAGE = Path(fxagent.collector.__file__).parent
MODULES = sorted(PACKAGE.glob("*.py"))

#: Importing any of these would give the collector a way to compute, decide, or trade.
FORBIDDEN_PREFIXES = (
    "fxagent.indicators",
    "fxagent.strategies",
    "fxagent.regime",
    "fxagent.risk",
    "fxagent.memory",
    "fxagent.agents",
    "fxagent.llm",
)

#: Third-party analysis and model libraries.
FORBIDDEN_TOP_LEVEL = (
    "pandas",
    "numpy",
    "sklearn",
    "scipy",
    "vectorbt",
    "google",
    "groq",
    "openai",
    "anthropic",
    "litellm",
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


def test_the_package_has_modules_to_check() -> None:
    """Guards the guard: an empty glob would make every test below pass vacuously."""
    assert len(MODULES) >= 3, f"expected the collector package to have modules, found {MODULES}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_imports_analysis_code(path: Path) -> None:
    offending = sorted(
        name for name in _imported_names(path) if name.startswith(FORBIDDEN_PREFIXES)
    )
    assert not offending, (
        f"{path.name} imports {offending}. The collector fetches and stores; it never computes. "
        "Analysis belongs in the analyst service, which can re-run over stored data."
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_imports_a_maths_or_model_library(path: Path) -> None:
    offending = sorted(
        name for name in _imported_names(path) if name.split(".")[0] in FORBIDDEN_TOP_LEVEL
    )
    assert not offending, f"{path.name} imports {offending}, which the collector has no use for"


def test_the_data_source_protocol_cannot_reach_execution() -> None:
    """The collector accepts a narrow DataSource, not a full BrokerAdapter.

    An execution-capable adapter can still be passed, but nothing in the collector can call the
    parts of it that trade, because the protocol it programs against has no such members.
    """
    from fxagent.collector.service import DataSource

    members = {name for name in dir(DataSource) if not name.startswith("_")}
    for forbidden in ("place_order", "close_position", "get_account", "get_positions"):
        assert forbidden not in members, f"DataSource exposes {forbidden}"

    assert "bars_ending_at" in members
    assert "source" in members
