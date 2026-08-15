"""`StoreSource` against a real Postgres. Skipped without `TEST_DATABASE_URL`.

Everything else in the dashboard suite runs against a stub, which is the point of the seam. But
the seam has an implementation, and it contains the two things a stub cannot check: the SQL
compiles, and the source-resolution rule does what the docstring says. A `group by` that is
wrong is wrong at runtime and nowhere else.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.dashboard.source import StoreSource, ViewRequest
from fxagent.store.engine import Database
from fxagent.store.repositories import BarRepository, EvaluationRepository, TradeRepository
from tests.conftest import requires_postgres
from tests.dashboard.builders import bar_series

pytestmark = [pytest.mark.db, requires_postgres]

START = datetime(2026, 1, 12, 0, 0, tzinfo=UTC)


async def _seed(database: Database, *, source: str = "twelvedata", count: int = 48) -> int:
    """Bars, one evaluation on them, and one trade from it. Returns the evaluation id."""
    from uuid import uuid4

    series = bar_series(start=START, count=count)
    async with database.begin() as session:
        await BarRepository(session).upsert_series(series, source=source)

        evaluation_id = await EvaluationRepository(session).record(
            cycle_id=uuid4(),
            ts_utc=START + timedelta(hours=9),
            symbol="EURUSD",
            regime={"sessions": ["LONDON"], "market_open": True, "is_trending": True},
            votes={
                "votes": [
                    {
                        "strategy": "session_breakout",
                        "weight": 1.0,
                        "direction": "LONG",
                        "confidence": 0.8,
                        "participated": True,
                        "reason": "counted toward LONG",
                    }
                ]
            },
            consensus_score=0.8,
            fired=True,
            reason="2 strategies agreed on LONG",
        )

        await TradeRepository(session).open_trade(
            evaluation_id=evaluation_id,
            symbol="EURUSD",
            direction="LONG",
            volume=0.02,
            entry_price=1.1050,
            entry_time_utc=START + timedelta(hours=9),
            stop_price=1.1020,
            target_price=1.1110,
            label_span_start=START + timedelta(hours=9),
            label_span_end=START + timedelta(days=2),
            mode="DEMO_AUTO",
        )
    return evaluation_id


async def test_the_switchers_see_a_series_that_has_bars(database: Database) -> None:
    await _seed(database)
    options = await StoreSource(database).options()

    assert len(options) == 1
    assert (options[0].symbol, options[0].timeframe, options[0].source) == (
        "EURUSD",
        "H1",
        "twelvedata",
    )
    assert options[0].bars == 48
    assert options[0].latest is not None


async def test_a_view_reads_the_bars_the_evaluations_and_the_trades_together(
    database: Database,
) -> None:
    evaluation_id = await _seed(database)
    data = await StoreSource(database).load(ViewRequest(symbol="EURUSD", timeframe="H1"))

    assert len(data.bars) == 48
    assert data.source == "twelvedata"
    assert [record.id for record in data.evaluations] == [evaluation_id]
    assert [record.evaluation_id for record in data.trades] == [evaluation_id]


async def test_the_evaluation_window_follows_the_bars_actually_returned(
    database: Database,
) -> None:
    """Ask for fewer bars and the feed narrows with the chart, rather than describing a
    stretch of time that is no longer on screen."""
    await _seed(database)
    data = await StoreSource(database).load(ViewRequest(symbol="EURUSD", timeframe="H1", bars=4))

    assert len(data.bars) == 4
    # The seeded evaluation is at hour 9, and the last four bars are hours 44-47.
    assert data.evaluations == ()


async def test_the_source_with_the_most_bars_wins_when_none_is_named(
    database: Database,
) -> None:
    """A bar's identity includes its source, so the choice has to be made somewhere. Making it
    explicitly beats making it implicitly in a query that interleaves two providers."""
    await _seed(database, source="twelvedata", count=48)
    async with database.begin() as session:
        await BarRepository(session).upsert_series(bar_series(start=START, count=5), source="mt5")

    data = await StoreSource(database).load(ViewRequest(symbol="EURUSD"))
    assert data.source == "twelvedata"

    named = await StoreSource(database).load(ViewRequest(symbol="EURUSD", source="mt5"))
    assert named.source == "mt5"
    assert len(named.bars) == 5


async def test_a_symbol_with_no_bars_is_an_empty_view_with_a_note(database: Database) -> None:
    await _seed(database)
    data = await StoreSource(database).load(ViewRequest(symbol="AUDNZD"))

    assert data.bars.bars == ()
    assert any("No stored bars" in note for note in data.notes)


async def test_a_trade_open_across_the_window_is_included(database: Database) -> None:
    """Entered before the visible stretch and still open, so its stop is still live and
    belongs on the chart."""
    await _seed(database)
    data = await StoreSource(database).load(ViewRequest(symbol="EURUSD", bars=10))

    # Bars 38-47; the trade entered at hour 9 and never closed.
    assert len(data.trades) == 1
