"""Safety checks for the operator-facing terminal tools."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_cost_measurement_tool_cannot_place_orders() -> None:
    assert "order_send" not in _calls(ROOT / "scripts" / "measure_mt5_costs.py")


def test_backfill_closes_database_engine_after_writes() -> None:
    assert "dispose" in _calls(ROOT / "scripts" / "backfill_mt5.py")
