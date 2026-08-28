"""The trader decides and records. It must not be able to place an order.

This is GATE A expressed as a test rather than as a promise. `MT5LocalAdapter.place_order`
works, and ADR-005 has just moved the decision into the same process as the adapter that can
call it — so the thing that used to keep them apart (they ran on different machines) is gone,
and something has to replace it.

An import test rather than a behavioural one, for the same reason the collector has one: this
is about what the code *can* reach, not what today's control flow happens to do. A behavioural
test passes right up until someone adds the branch.

When `fxagent/permission/` lands, this test does not get deleted. It gets one entry added to
`ALLOWED_EXECUTION_PATH` naming the module that is allowed to reach an order, so that the set
of code able to trade stays enumerable and stays small.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import fxagent.trader

PACKAGE = Path(fxagent.trader.__file__).parent
MODULES = sorted(PACKAGE.glob("*.py"))

#: Attribute names that put an order on the wire.
EXECUTION_CALLS = ("place_order", "close_position", "order_send")

#: Modules permitted to reach execution. Empty until the permission layer exists, and it is
#: meant to stay nearly empty afterwards: an enumerable list of files that can trade is the
#: point, and a long one defeats it.
ALLOWED_EXECUTION_PATH: set[str] = set()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _attribute_names(tree: ast.Module) -> set[str]:
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_the_package_has_modules_to_check() -> None:
    """Guards the guard: an empty glob would make every test below pass vacuously."""
    assert len(MODULES) >= 3, f"expected the trader package to have modules, found {MODULES}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_calls_an_execution_method(path: Path) -> None:
    if path.stem in ALLOWED_EXECUTION_PATH:
        return
    offending = sorted(_attribute_names(_tree(path)) & set(EXECUTION_CALLS))
    assert not offending, (
        f"{path.name} reaches {offending}. The trader is advisory: it decides and records, and "
        "the permission layer that would authorise an order does not exist yet (GATE A). If "
        "this is deliberate, add the module to ALLOWED_EXECUTION_PATH so the set of files that "
        "can trade stays enumerable."
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_imports_the_broker_protocol_itself(path: Path) -> None:
    """Importing the adapter for a constant is fine; importing BrokerAdapter is not.

    `__main__` reads `SOURCE` off the MT5 adapter, which is a string. What must not appear is
    the execution-capable protocol, because a module holding one has an order-placing call in
    scope and only convention keeping it unused.
    """
    imported = _imported_names(_tree(path))
    assert "fxagent.adapters.base.BrokerAdapter" not in imported, (
        f"{path.name} imports BrokerAdapter. The trader has no use for an execution-capable "
        "interface; if it needs bars, it reads them from the store."
    )


def test_the_notifier_protocol_is_send_only() -> None:
    """An outbound channel, not a control channel.

    A notifier with a callback would be a second authorisation path beside the permission
    grant, authenticated by whatever the messaging app happens to check — which is how the
    Telegram approve/reject buttons in the original Phase 8 plan would have worked, and why
    they were dropped.
    """
    from fxagent.trader.service import Notifier

    members = {name for name in dir(Notifier) if not name.startswith("_")}
    assert members == {"send"}, f"Notifier exposes more than send: {sorted(members)}"


def test_run_cycle_takes_no_argument_that_could_enable_execution() -> None:
    """Advisory is the absence of an execution path, not a flag set to False.

    A `execute=False` default is one keyword argument away from an order. There is deliberately
    nothing here to set.
    """
    import inspect

    from fxagent.trader.cycle import run_cycle

    parameters = set(inspect.signature(run_cycle).parameters)
    for forbidden in ("execute", "live", "place", "adapter", "broker", "dry_run"):
        assert forbidden not in parameters, (
            f"run_cycle takes {forbidden!r}; it should have no way to express 'and trade it'"
        )
