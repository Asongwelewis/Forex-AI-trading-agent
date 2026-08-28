"""Static guardrails for the point-in-time refusal-ledger migration.

The executable checks run against Postgres when ``TEST_DATABASE_URL`` is configured. These
lightweight assertions still protect the two invariants in every environment: an as-of API exists
and realised outcomes are gated by the label span rather than joined unconditionally.
"""

from __future__ import annotations

from pathlib import Path

MIGRATION = (
    Path(__file__).parents[2] / "fxagent" / "store" / "migrations" / "0013_refusal_ledger.sql"
)


def test_refusal_ledger_has_an_as_of_function() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "create or replace function refusal_ledger_visible_at(as_of timestamptz)" in sql
    assert "where e.ts_utc <= as_of" in sql


def test_refusal_ledger_does_not_expose_unresolved_outcomes() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "trade.label_span_end <= as_of" in sql
    assert "left join lateral" in sql
    assert "create or replace view refusal_ledger" in sql
