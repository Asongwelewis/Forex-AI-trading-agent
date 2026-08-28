"""The permission layer authorises. It does not execute, and it must not be able to.

Two things stay separate on purpose. A module that both decides whether an order is allowed and
then places it has no seam a test can sit in — the only way to check the refusal path would be
to let it try. Keeping the authoriser incapable of acting means every refusal is testable
without a broker anywhere near it.

It also bounds the blast radius. `fxagent/permission/` is the code most likely to be edited
under pressure ("why won't it trade?"), and that is exactly the code that should not have an
`order_send` in scope while somebody is editing it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import fxagent.permission

PACKAGE = Path(fxagent.permission.__file__).parent
MODULES = sorted(PACKAGE.glob("*.py"))

EXECUTION_CALLS = ("place_order", "close_position", "order_send")

#: Importing any of these would put a broker connection inside the authoriser.
FORBIDDEN_IMPORTS = (
    "fxagent.adapters.mt5_local",
    "fxagent.adapters.mock",
    "fxagent.adapters.twelvedata",
    "MetaTrader5",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_the_package_has_modules_to_check() -> None:
    """Guards the guard: `permission/` was an empty docstring for months."""
    assert len(MODULES) >= 4, f"expected grant, triggers, preflight and __init__, got {MODULES}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_can_place_an_order(path: Path) -> None:
    offending = sorted(
        {node.attr for node in ast.walk(_tree(path)) if isinstance(node, ast.Attribute)}
        & set(EXECUTION_CALLS)
    )
    assert not offending, (
        f"{path.name} reaches {offending}. This package answers yes or no; the caller that "
        "holds the broker connection is the one that acts on the answer."
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_imports_a_broker_adapter(path: Path) -> None:
    offending = sorted(_imported(_tree(path)) & set(FORBIDDEN_IMPORTS))
    assert not offending, (
        f"{path.name} imports {offending}. The authoriser needs the order's *shape* "
        "(OrderRequest) and never a connection that could send one."
    )


def test_no_public_function_takes_an_adapter() -> None:
    """A connection passed in is a connection that can be used.

    `check_order` takes an `OrderRequest` — a description of the order — and a snapshot of
    facts somebody else gathered. It never takes the thing that would place it.
    """
    from fxagent.permission import check_order, evaluate_triggers

    for function in (check_order, evaluate_triggers):
        parameters = set(inspect.signature(function).parameters)
        for forbidden in ("adapter", "broker", "terminal", "mt5", "connection", "client"):
            assert forbidden not in parameters, (
                f"{function.__name__} takes {forbidden!r}; the authoriser must not hold a "
                "connection"
            )


def test_the_triggers_are_pure_functions_of_a_snapshot() -> None:
    """No clock, no socket, no file. That is what lets each trigger be tested exactly.

    A trigger that read `datetime.now()` could not be tested at a boundary — every test would
    have to arrange the world into the condition instead of describing it.
    """
    import fxagent.permission.triggers as triggers

    source = Path(triggers.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # `datetime.now` appears nowhere; `TriggerSnapshot.now` is a field the caller supplies.
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "now" not in calls, "a trigger read a clock; the snapshot carries the time"

    assert _imported(tree) & {"httpx", "requests", "socket", "asyncio"} == set(), (
        "the triggers reached for the network"
    )


def test_advisory_is_the_absence_of_a_grant_not_a_setting() -> None:
    """There is no field, argument or method that turns execution on without a grant object."""
    from fxagent.permission.grant import GrantManager, PermissionGrant

    manager_members = {name for name in vars(GrantManager) if not name.startswith("_")}
    grant_fields = set(PermissionGrant.model_fields)

    for forbidden in ("state_override", "force", "enabled", "allow_all", "bypass"):
        assert forbidden not in manager_members
        assert forbidden not in grant_fields
