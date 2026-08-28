"""Every module a workflow names must exist, or be written down as deliberately pending.

`collect-and-analyse.yml` named `fxagent.analyst` and `fxagent.resolve` for months. Neither
module has ever existed. The pass would not have gone red: `.github/actions/stage` checks for a
`__main__` submodule and, when it is absent, emits a `::warning` annotation and exits 0 — chosen
deliberately so that a cron waiting on unlanded phases does not sit permanently red, because a
permanently red cron is indistinguishable from a broken one.

That reasoning is sound and the escape hatch stays. What it cannot do is tell the difference
between "this phase lands on Thursday" and "nobody has thought about this since March", and the
second is what actually happened. So the difference is recorded here instead: a stage naming a
module that does not exist passes only if its name is in `PENDING_STAGES` below, with a note
saying what it is waiting for.

Adding a name to that set is cheap and takes five seconds. Not noticing for six months that a
scheduled job has been reporting success while doing nothing is the failure this prevents.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is needed to read the workflow files")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

#: Modules a workflow may name before they exist. Each entry needs a reason, and an entry that
#: has outlived its reason is a finding in its own right — the point is that the exception is
#: written down, not that it is forbidden.
PENDING_STAGES: dict[str, str] = {
    # Empty. `fxagent.analyst` and `fxagent.resolve` did not land here — ADR-005 moved
    # analysis and resolution onto the desktop, where MT5 runs, and out of Actions entirely.
}


def _workflow_files() -> list[Path]:
    if not WORKFLOW_DIR.is_dir():  # pragma: no cover - the repo always has these
        return []
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def _walk(node: Any) -> list[Any]:
    """Every mapping anywhere in the document, however deeply nested."""
    found: list[Any] = []
    if isinstance(node, dict):
        found.append(node)
        for value in node.values():
            found.extend(_walk(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_walk(value))
    return found


def _named_modules(path: Path) -> list[str]:
    """The `module:` inputs handed to `./.github/actions/stage`, wherever they appear.

    Walks the whole document rather than the known job/step path: a stage nested inside a
    matrix, a reusable workflow or a future job shape is exactly the one that would be missed.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for mapping in _walk(document):
        inputs = mapping.get("with")
        if isinstance(inputs, dict) and isinstance(inputs.get("module"), str):
            modules.append(inputs["module"])
    return modules


def _is_runnable(module: str) -> bool:
    """Mirrors the stage action: find_spec on the `__main__` submodule, not the package.

    A package can exist as a directory with no entry point, and `python -m` on that fails
    confusingly. A missing parent raises rather than returning None, hence the catch.
    """
    try:
        return importlib.util.find_spec(f"{module}.__main__") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def test_at_least_one_workflow_names_a_stage() -> None:
    """Guards the guard: a walker that silently found nothing would pass everything below."""
    named = [module for path in _workflow_files() for module in _named_modules(path)]
    assert named, "no workflow names a stage module; this test would assert nothing"


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_named_module_exists_or_is_declared_pending(path: Path) -> None:
    for module in _named_modules(path):
        if _is_runnable(module):
            continue
        assert module in PENDING_STAGES, (
            f"{path.name} runs `python -m {module}`, which has no __main__ module. "
            f"The stage action skips it with a ::warning and the run still reports success, "
            f"so this would never surface as a failure. Either build it, remove the stage, or "
            f"add {module!r} to PENDING_STAGES with a note saying what it is waiting for."
        )


def test_pending_stages_are_still_pending() -> None:
    """An exception that has outlived its reason is a finding too."""
    landed = [module for module in PENDING_STAGES if _is_runnable(module)]
    assert not landed, (
        f"{landed} now exist but are still listed in PENDING_STAGES. Remove them — a stale "
        f"exception quietly re-opens the hole it was written to close."
    )
