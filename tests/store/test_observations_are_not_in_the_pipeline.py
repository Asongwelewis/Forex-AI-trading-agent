"""Statistical observations must stay outside the analysis pipeline.

These rows have no publication timestamp — no primary source publishes one — so there is no
honest answer to "what was knowable at time T" for them. That is precisely why they live in
their own table: inside `events` they would be visible to `events_visible_at()` and therefore
to `build_context`, and a row with a guessed publication time inside the point-in-time gate is
worse than no row, because it looks audited.

Convention will not hold this line on its own. Somebody will one day need a number and reach
for the nearest table. These tests make that reach fail loudly at the point it is made.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import text

from fxagent.store.repositories.observations import ObservationRepository

from .conftest import requires_postgres


def _imported_modules(path: Path) -> set[str]:
    """Every module named by an import in one file, without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_context_does_not_import_the_observation_repository() -> None:
    """`build_context` is the pipeline's only door into the store. Keep it shut to this table."""
    imported = _imported_modules(Path("fxagent/fundamentals/context.py"))
    offenders = {m for m in imported if "observation" in m or "statistics" in m}
    assert not offenders, (
        f"fxagent/fundamentals/context.py imports {offenders}. Statistical observations have no "
        f"publication time and must not reach the point-in-time pipeline — see migration 0010."
    )


@pytest.mark.subprocess
def test_the_observations_module_cannot_reach_the_analysis_layer() -> None:
    """`fxagent.observations` must stay importable without dragging analysis code in.

    It lived in `fxagent/fundamentals/` first, and importing it pulled in strategies, regime and
    indicators — because any submodule import runs the package `__init__`, which imports
    `context.py`, which imports `fxagent.strategies.base`. The collector is the natural home for
    the poller and its purity test is AST-based, so it would have seen one innocent import line
    and missed the whole transitive reach. This asserts the actual runtime graph instead.

    **A blocked spawn skips rather than fails.** Measuring the real import graph needs a fresh
    interpreter, and some sandboxes and CI runners refuse to start one — which arrives as
    `PermissionError` from `CreateProcess` and has nothing to say about the import graph. A
    check that is red in every full run is a check people stop reading, and then the day it goes
    red for a real reason nobody looks. So an unrunnable check reports "could not verify" and
    says so out loud; a check that ran and found a forbidden import still fails, loudly.
    """
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, fxagent.observations; "
                "print(','.join(sorted(m for m in sys.modules if m.startswith('fxagent.'))))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError as exc:
        pytest.skip(
            f"could not verify: this environment refused to start a subprocess ({exc}). The "
            "import graph is unchecked in this run — it was not checked and found clean. Run "
            "this file on its own, or anywhere a fresh interpreter can be spawned."
        )
    pulled = set(result.stdout.strip().split(","))
    forbidden = {
        m
        for m in pulled
        if m.startswith(
            ("fxagent.strategies", "fxagent.regime", "fxagent.indicators", "fxagent.fundamentals")
        )
    }
    assert not forbidden, (
        f"importing fxagent.observations reaches {sorted(forbidden)}. Keep it free of the "
        f"analysis layer so the collector can poll it without gaining a path to strategies."
    )


def test_no_analysis_module_touches_the_observations_table() -> None:
    """Nothing under strategies/, regime/ or fundamentals/context may name the table."""
    watched = [
        *Path("fxagent/strategies").glob("*.py"),
        *Path("fxagent/regime").glob("*.py"),
        Path("fxagent/fundamentals/context.py"),
    ]
    offenders = [
        path for path in watched if "statistical_observations" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these reference the observations table directly: {offenders}"


def test_the_repository_offers_no_as_of_read() -> None:
    """An `as_of` here would be answering a point-in-time question with a guess.

    Discovered by reflection rather than a hand-kept list, so a method added later is covered
    without anyone remembering to update this test.
    """
    reads = [
        name
        for name, member in inspect.getmembers(ObservationRepository, inspect.isfunction)
        if not name.startswith("_")
    ]
    with_as_of = [
        name
        for name in reads
        if "as_of" in inspect.signature(getattr(ObservationRepository, name)).parameters
    ]
    assert not with_as_of, (
        f"{with_as_of} take an as_of, implying point-in-time semantics this table cannot "
        f"honestly provide — it has no publication timestamp."
    )


@pytest.mark.db
@requires_postgres
async def test_the_gate_function_does_not_expose_observations(session) -> None:  # noqa: ANN001
    """`events_visible_at()` returns setof events; observations must not appear through it."""
    columns = (
        (
            await session.execute(
                text(
                    "select column_name from information_schema.columns where table_name = 'events'"
                )
            )
        )
        .scalars()
        .all()
    )
    assert "series_id" not in columns
    assert "reference_period" not in columns


@pytest.mark.db
@requires_postgres
async def test_observations_land_in_their_own_table(session) -> None:  # noqa: ANN001
    """A stored observation is invisible to the events table entirely."""
    repo = ObservationRepository(session)
    await repo.upsert_many(
        [
            {
                "source": "bls",
                "series_id": "CES0000000001",
                "reference_period": "2026-07",
                "period_start": date(2026, 7, 1),
                "value": 158858.0,
            }
        ]
    )
    assert await repo.count() == 1

    events = (await session.execute(text("select count(*) from events"))).scalar_one()
    assert events == 0, "an observation leaked into the events table"
