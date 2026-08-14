"""LIVE mode is refused at both layers, and the database layer is the one that matters.

The app-layer check is a courtesy that produces a readable error. The constraint is the actual
guard: it survives a refactor that deletes the Python check, and lifting it requires someone to
write `alter table trades drop constraint trades_no_live_mode;` in a reviewed migration.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from fxagent.store.repositories import EvaluationRepository, TradeRepository
from fxagent.store.repositories.trades import MODES, PERMITTED_MODES
from fxagent.store.schema import trades

from .conftest import requires_postgres

pytestmark = [pytest.mark.db, requires_postgres]

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


def _trade_kwargs(evaluation_id: int, mode: str) -> dict[str, object]:
    """Arguments `open_trade` accepts. `created_at` is set by the repository, not the caller."""
    return {
        "evaluation_id": evaluation_id,
        "symbol": "EURUSD",
        "direction": "LONG",
        "volume": 0.1,
        "entry_price": 1.0850,
        "entry_time_utc": NOW,
        "stop_price": 1.0800,
        "target_price": 1.0950,
        "label_span_start": NOW,
        "label_span_end": NOW + timedelta(hours=8),
        "mode": mode,
    }


def _trade_row(evaluation_id: int, mode: str) -> dict[str, object]:
    """A full row for a raw INSERT that deliberately bypasses the repository."""
    return {**_trade_kwargs(evaluation_id, mode), "created_at": NOW}


async def _evaluation(repo: EvaluationRepository) -> int:
    return await repo.record(
        cycle_id=uuid4(),
        ts_utc=NOW,
        symbol="EURUSD",
        regime={},
        votes={},
        consensus_score=1.0,
        fired=True,
        reason="fixture",
    )


def test_live_is_still_in_the_column_type_but_not_permitted() -> None:
    """Keeping the enum value is what makes v2 a one-line migration rather than a type change."""
    assert "LIVE" in MODES
    assert "LIVE" not in PERMITTED_MODES


async def test_repository_refuses_live_with_a_readable_error(
    evaluations_repo: EvaluationRepository, trades_repo: TradeRepository
) -> None:
    evaluation_id = await _evaluation(evaluations_repo)

    with pytest.raises(ValueError, match="demo accounts only"):
        await trades_repo.open_trade(**_trade_kwargs(evaluation_id, "LIVE"))  # type: ignore[arg-type]


async def test_the_database_refuses_live_even_bypassing_the_repository(
    evaluations_repo: EvaluationRepository, session: AsyncSession
) -> None:
    """The test that matters: deleting the Python guard does not enable live trading."""
    evaluation_id = await _evaluation(evaluations_repo)

    with pytest.raises(IntegrityError, match="trades_no_live_mode"):
        await session.execute(insert(trades).values(_trade_row(evaluation_id, "LIVE")))


@pytest.mark.parametrize("mode", PERMITTED_MODES)
async def test_permitted_modes_still_insert(
    mode: str, evaluations_repo: EvaluationRepository, trades_repo: TradeRepository
) -> None:
    evaluation_id = await _evaluation(evaluations_repo)
    trade_id = await trades_repo.open_trade(**_trade_kwargs(evaluation_id, mode))  # type: ignore[arg-type]

    stored = await trades_repo.get(trade_id)
    assert stored is not None and stored.mode == mode


async def test_the_constraint_is_separate_from_the_enum_check(session: AsyncSession) -> None:
    """Two constraints, so the diff that lifts the ban touches only the ban."""
    result = await session.execute(
        text(
            "select conname from pg_constraint "
            "where conrelid = 'trades'::regclass and contype = 'c' "
            "and conname in ('trades_mode_known', 'trades_no_live_mode')"
        )
    )
    names = {row.conname for row in result}
    assert names == {"trades_mode_known", "trades_no_live_mode"}
