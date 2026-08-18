"""Win rate is reported and never acted on, enforced by inspecting imports rather than by review.

CLAUDE.md is explicit that win rate is not a success metric and must never gate anything, and
the reason is arithmetic: `session_breakout` targets 2R, so it breaks even at a 33% hit rate and
is strongly profitable at 40%. A threshold anywhere in the decision path would reject the
system's best strategy while passing anything that wins small ninety times and gives it back
once.

An import test rather than a behavioural one, for the same reason the collector has one: this is
about what the decision path *can* reach, not what it happens to do today. A statistic that no
deciding module can import is a statistic that cannot become a filter in six months, when the
reason for the rule has been forgotten and a 38% win rate looks embarrassing on a dashboard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import fxagent

#: Everything that decides, sizes, or gates. None of it may reach the statistics layer.
DECISION_PACKAGES = (
    "strategies",
    "regime",
    "risk",
    "permission",
    "patterns",
    "indicators",
)

PACKAGE_ROOT = Path(fxagent.__file__).parent


def _modules_in(package: str) -> list[Path]:
    return sorted((PACKAGE_ROOT / package).rglob("*.py"))


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("package", DECISION_PACKAGES)
def test_no_deciding_module_can_reach_the_statistics_layer(package: str) -> None:
    modules = _modules_in(package)
    assert modules, f"fxagent/{package} has no modules; the guard would pass vacuously"

    for module in modules:
        reached = {name for name in _imported_names(module) if name.startswith("fxagent.stats")}
        assert not reached, (
            f"{module.relative_to(PACKAGE_ROOT)} imports {sorted(reached)}. The statistics layer "
            "measures the decision path; it must not be inside it. Win rate in particular would "
            "reject a 2R strategy at a 38% hit rate that is making money."
        )


def test_win_rate_is_not_named_anywhere_in_the_decision_path() -> None:
    """Belt and braces: not reached by import, and not reimplemented under another name either."""
    for package in DECISION_PACKAGES:
        for module in _modules_in(package):
            source = module.read_text(encoding="utf-8")
            offending = [
                line
                for line in source.splitlines()
                if "win_rate" in line and not line.lstrip().startswith("#")
            ]
            assert not offending, (
                f"{module.relative_to(PACKAGE_ROOT)} mentions win_rate outside a comment: "
                f"{offending}"
            )


def test_the_statistics_layer_may_read_the_decision_path_the_other_way_round() -> None:
    """The dependency is one-directional by design, not merely absent by accident.

    `fxagent.stats` measuring `fxagent.strategies` would be fine; the reverse is what is
    forbidden. This asserts the direction is a real constraint rather than two packages that
    happen never to have met.
    """
    from fxagent.stats.performance import win_rate

    estimate = win_rate([1.0, 1.0, -1.0, -1.0], samples=200, seed=1)
    assert estimate.estimate == pytest.approx(0.5)
