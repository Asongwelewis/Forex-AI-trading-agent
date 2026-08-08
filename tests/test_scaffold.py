"""Placeholder test proving the package tree is importable and the layout is intact.

Replaced by real tests from Phase 2 onward; the import check is worth keeping.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PACKAGES = [
    "src",
    "src.adapters",
    "src.strategies",
    "src.regime",
    "src.risk",
    "src.permission",
    "src.journal",
    "src.llm",
]

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_env_example_has_no_filled_values() -> None:
    """.env.example names variables but never carries a value."""
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assignments = [ln for ln in lines if "=" in ln and not ln.lstrip().startswith("#")]
    secrets = [
        ln for ln in assignments if ln.split("=", 1)[0].endswith(("KEY", "TOKEN", "PASSWORD"))
    ]

    assert secrets, "expected secret-shaped variables in .env.example"
    for line in secrets:
        assert line.split("=", 1)[1] == "", f"secret has a value in .env.example: {line}"


def test_gitignore_excludes_secrets() -> None:
    patterns = set((REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split())
    for required in (".env", "*.key", "*credentials*"):
        assert required in patterns, f"missing .gitignore pattern: {required}"
