"""The MetaTrader 5 MCP is a developer tool. `fxagent/` must not know it exists.

ADR-006 accepts MetaQuotes' built-in MCP support for measurement — reading swap points, filling
modes, symbol specs and the real spread coverage off the live terminal — and refuses it as a
trading path. Three reasons, all in the ADR: it is an LLM holding an order ticket (hard rule 4),
it is a second authorisation path beside `fxagent/permission/`, and it is not replayable so
nothing it does can appear in an expectancy interval.

The distinction is only worth anything if it survives contact with a hurry, so it is asserted
rather than documented. "Just this once, read the account over MCP" is how the package acquires
a dependency on a transport that can also place orders.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import fxagent

PACKAGE = Path(fxagent.__file__).parent
MODULES = sorted(PACKAGE.rglob("*.py"))

#: Any of these appearing in `fxagent/` means the package reaches an MCP transport.
MCP_MARKERS = (
    "mcp",
    "modelcontextprotocol",
    "model_context_protocol",
    "metatrader_mcp",
    "metatrader5_mcp",
    "mcp_metatrader5",
)

#: Words that would appear if something started shelling out to an MCP CLI.
SHELL_MARKERS = ("subprocess", "os.system", "popen")


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
    """Guards the guard: an empty glob would make everything below pass vacuously."""
    assert len(MODULES) > 50, f"expected the whole package, found {len(MODULES)} modules"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_no_module_imports_an_mcp_client(path: Path) -> None:
    offending = sorted(
        name
        for name in _imported(_tree(path))
        if any(marker in name.lower().split(".") for marker in MCP_MARKERS)
    )
    assert not offending, (
        f"{path.name} imports {offending}. ADR-006: the MT5 MCP is a measurement tool for a "
        "Claude Code session, not a transport this package uses. It can place orders, which "
        "would be a second authorisation path beside fxagent/permission/, and nothing it did "
        "would be replayable."
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_no_module_shells_out_to_an_mcp_command(path: Path) -> None:
    """An import guard alone is bypassed by one `subprocess.run(["metatrader-mcp-server", ...])`."""
    source = path.read_text(encoding="utf-8").lower()
    if not any(marker in source for marker in SHELL_MARKERS):
        return
    assert "mcp" not in source, (
        f"{path.name} both shells out and mentions MCP. If that is a coincidence, rename the "
        "variable; if it is not, read ADR-006."
    )


def test_the_decision_is_written_down() -> None:
    """A rule with no recorded reason gets relitigated, usually by whoever is in a hurry."""
    adr = Path(fxagent.__file__).resolve().parents[1] / "docs" / "ADR-006-mt5-mcp.md"
    assert adr.exists(), "ADR-006 is missing; this test enforces a decision nobody can read"

    text = adr.read_text(encoding="utf-8")
    for required in ("prohibit", "hard rule 4", "permission", "replay"):
        assert required.lower() in text.lower(), f"ADR-006 no longer explains {required!r}"


def test_no_mcp_dependency_was_added_to_the_package() -> None:
    """The MCP is a property of the developer's machine, not of this distribution.

    A checkout on a machine without it must behave identically, which is what keeps the
    backtest reproducible somewhere other than one Windows desktop.
    """
    pyproject = Path(fxagent.__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8").lower()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped.startswith('"'):
            continue
        assert "mcp" not in stripped, f"a dependency mentions MCP: {stripped}"
