"""Repository round-trips, constraint enforcement, and transactional composition."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from fxagent.adapters.base import Bar, BarSeries
from fxagent.store.engine import Database
from fxagent.store.repositories import (
    BarRepository,
    EvaluationRepository,
    TradeRepository,
    WindowRepository,
)
from fxagent.store.schema import EMBEDDING_DIMENSIONS

from .conftest import requires_postgres

pytestmark = [pytest.mark.db, requires_postgres]

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


def _series(symbol: str = "EURUSD", timeframe: str = "H1", count: int = 5) -> BarSeries:
    return BarSeries(
        symbol=symbol,
        timeframe=timeframe,
        bars=tuple(
            Bar(
                timestamp=NOW + timedelta(hours=i),
                open=1.0800 + i * 0.001,
                high=1.0820 + i * 0.001,
                low=1.0790 + i * 0.001,
                close=1.0810 + i * 0.001,
                volume=1000 + i,
            )
            for i in range(count)
        ),
    )


def _embedding(seed: float) -> list[float]:
    """A vector whose *direction* varies with `seed`.

    Not `[seed] * N` — every constant vector points the same way, so cosine distance between
    any two of them is zero and a nearest-neighbour test on them passes vacuously.
    """
    return [
        1.0 if index == 0 else seed * (1.0 if index % 2 else -1.0) / (index + 1)
        for index in range(EMBEDDING_DIMENSIONS)
    ]


# -- bars ---------------------------------------------------------------------


async def test_bars_round_trip(bars_repo: BarRepository) -> None:
    await bars_repo.upsert_series(_series(), source="oanda")

    stored = await bars_repo.latest_bars("EURUSD", "H1", 5, source="oanda")

    assert len(stored) == 5
    assert stored.bars[0].timestamp == NOW, "oldest first"
    assert stored.bars[-1].timestamp == NOW + timedelta(hours=4)


async def test_reingesting_the_same_range_does_not_duplicate(
    bars_repo: BarRepository,
) -> None:
    """The collector re-fetches overlapping ranges after every restart."""
    await bars_repo.upsert_series(_series(), source="oanda")
    await bars_repo.upsert_series(_series(), source="oanda")

    assert len(await bars_repo.latest_bars("EURUSD", "H1", 50, source="oanda")) == 5


async def test_the_same_bar_from_two_sources_is_kept_separately(
    bars_repo: BarRepository,
) -> None:
    """OANDA and MT5 disagree on the last decimal; overwriting one loses reproducibility."""
    await bars_repo.upsert_series(_series(), source="oanda")
    await bars_repo.upsert_series(_series(), source="mt5")

    assert len(await bars_repo.latest_bars("EURUSD", "H1", 50, source="oanda")) == 5
    assert len(await bars_repo.latest_bars("EURUSD", "H1", 50, source="mt5")) == 5
    assert sorted(await bars_repo.sources_for("EURUSD")) == ["mt5", "oanda"]


async def test_a_series_read_requires_a_source(bars_repo: BarRepository) -> None:
    """Without it a read would interleave two providers' prices into one indicator input."""
    with pytest.raises(TypeError, match="source"):
        await bars_repo.latest_bars("EURUSD", "H1", 5)  # type: ignore[call-arg]


async def test_a_reingested_forming_bar_updates_rather_than_collides(
    bars_repo: BarRepository,
) -> None:
    """The last bar of a fetch is still forming, so its close legitimately changes."""
    await bars_repo.upsert_series(_series(count=1), source="oanda")
    await bars_repo.upsert_rows(
        [
            {
                "symbol": "EURUSD",
                "timeframe": "H1",
                "ts_utc": NOW,
                "open": 1.0800,
                "high": 1.0850,
                "low": 1.0790,
                "close": 1.0845,
                "volume": 5000,
                "source": "oanda",
            }
        ]
    )

    stored = await bars_repo.latest_bars("EURUSD", "H1", 5, source="oanda")
    assert len(stored) == 1
    assert stored.bars[0].close == pytest.approx(1.0845)


async def test_bars_between_is_inclusive(bars_repo: BarRepository) -> None:
    await bars_repo.upsert_series(_series(), source="oanda")

    window = await bars_repo.bars_between(
        "EURUSD", "H1", start=NOW + timedelta(hours=1), end=NOW + timedelta(hours=3), source="oanda"
    )
    assert len(window) == 3


async def test_latest_timestamp_tells_the_collector_where_to_resume(
    bars_repo: BarRepository,
) -> None:
    assert await bars_repo.latest_timestamp("EURUSD", "H1") is None
    await bars_repo.upsert_series(_series(), source="oanda")
    assert await bars_repo.latest_timestamp("EURUSD", "H1") == NOW + timedelta(hours=4)


async def test_naive_timestamps_are_rejected(bars_repo: BarRepository) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        await bars_repo.upsert_rows(
            [
                {
                    "symbol": "EURUSD",
                    "timeframe": "H1",
                    "ts_utc": datetime(2026, 3, 10, 12, 0),
                    "open": 1.08,
                    "high": 1.09,
                    "low": 1.07,
                    "close": 1.085,
                    "volume": 10,
                    "source": "oanda",
                }
            ]
        )


async def test_unknown_timeframe_is_rejected(bars_repo: BarRepository) -> None:
    with pytest.raises(ValueError, match="unknown timeframe"):
        await bars_repo.latest_bars("EURUSD", "H2", 5, source="oanda")


# -- evaluations and trades ---------------------------------------------------


async def _an_evaluation(repo: EvaluationRepository, *, fired: bool = True) -> int:
    return await repo.record(
        cycle_id=uuid4(),
        ts_utc=NOW,
        symbol="EURUSD",
        regime={"trend": "up", "adx": 27.4},
        votes={"session_breakout": "LONG", "range_reversion": "FLAT"},
        consensus_score=0.66,
        fired=fired,
        reason="2 of 3 agree" if fired else "router blocked range_reversion",
    )


async def test_no_signal_evaluations_are_stored(
    evaluations_repo: EvaluationRepository,
) -> None:
    """Disagreement is the training data; a store of only fired signals cannot answer it."""
    await _an_evaluation(evaluations_repo, fired=False)

    recent = await evaluations_repo.recent()
    assert len(recent) == 1
    assert recent[0].fired is False
    assert recent[0].reason


async def test_an_evaluation_without_a_reason_is_refused(
    evaluations_repo: EvaluationRepository,
) -> None:
    with pytest.raises(ValueError, match="reason is required"):
        await evaluations_repo.record(
            cycle_id=uuid4(),
            ts_utc=NOW,
            symbol="EURUSD",
            regime={},
            votes={},
            consensus_score=0.0,
            fired=False,
            reason="",
        )


async def test_a_retried_cycle_updates_rather_than_double_counting(
    evaluations_repo: EvaluationRepository,
) -> None:
    cycle = uuid4()
    common = {
        "cycle_id": cycle,
        "ts_utc": NOW,
        "symbol": "EURUSD",
        "regime": {"adx": 20.0},
        "votes": {},
        "consensus_score": 0.1,
    }
    first = await evaluations_repo.record(**common, fired=False, reason="first pass")
    second = await evaluations_repo.record(**common, fired=True, reason="after retry")

    assert first == second
    assert len(await evaluations_repo.for_cycle(cycle)) == 1
    stored = await evaluations_repo.get(first)
    assert stored is not None and stored.fired is True


async def test_trade_round_trip_and_close(
    evaluations_repo: EvaluationRepository, trades_repo: TradeRepository
) -> None:
    evaluation_id = await _an_evaluation(evaluations_repo)
    trade_id = await trades_repo.open_trade(
        evaluation_id=evaluation_id,
        symbol="EURUSD",
        direction="LONG",
        volume=0.1,
        entry_price=1.0850,
        entry_time_utc=NOW,
        stop_price=1.0800,
        target_price=1.0950,
        label_span_start=NOW,
        label_span_end=NOW + timedelta(hours=8),
        mode="DEMO_AUTO",
    )

    assert [t.id for t in await trades_repo.open_trades()] == [trade_id]

    closed = await trades_repo.close_trade(
        trade_id,
        exit_price=1.0950,
        exit_time_utc=NOW + timedelta(hours=3),
        barrier_touched="TARGET",
        pnl=10.0,
        r_multiple=2.0,
    )

    assert closed is True
    assert await trades_repo.open_trades() == []
    stored = await trades_repo.get(trade_id)
    assert stored is not None and stored.r_multiple == pytest.approx(2.0)


async def test_a_stop_on_the_wrong_side_is_refused_by_the_database(
    evaluations_repo: EvaluationRepository, trades_repo: TradeRepository
) -> None:
    """Hard rule 3, restated where the data lives."""
    evaluation_id = await _an_evaluation(evaluations_repo)

    with pytest.raises(IntegrityError, match="trades_protection_sides"):
        await trades_repo.open_trade(
            evaluation_id=evaluation_id,
            symbol="EURUSD",
            direction="LONG",
            volume=0.1,
            entry_price=1.0850,
            entry_time_utc=NOW,
            stop_price=1.0900,  # above entry on a long
            target_price=1.0950,
            label_span_start=NOW,
            label_span_end=NOW + timedelta(hours=8),
            mode="DEMO_AUTO",
        )


async def test_a_half_closed_trade_cannot_be_stored(
    evaluations_repo: EvaluationRepository, trades_repo: TradeRepository, session: object
) -> None:
    """An exit price with no barrier reads as closed but cannot be labelled."""
    from sqlalchemy import update

    from fxagent.store.schema import trades

    evaluation_id = await _an_evaluation(evaluations_repo)
    trade_id = await trades_repo.open_trade(
        evaluation_id=evaluation_id,
        symbol="EURUSD",
        direction="LONG",
        volume=0.1,
        entry_price=1.0850,
        entry_time_utc=NOW,
        stop_price=1.0800,
        target_price=1.0950,
        label_span_start=NOW,
        label_span_end=NOW + timedelta(hours=8),
        mode="ADVISORY",
    )

    with pytest.raises(IntegrityError, match="trades_exit_consistent"):
        await session.execute(  # type: ignore[attr-defined]
            update(trades).where(trades.c.id == trade_id).values(exit_price=1.09)
        )


async def test_purge_finds_labels_overlapping_a_test_window(
    evaluations_repo: EvaluationRepository, trades_repo: TradeRepository
) -> None:
    """Overlap, not containment: a straddling label leaks as much as a contained one."""
    evaluation_id = await _an_evaluation(evaluations_repo)
    common = {
        "evaluation_id": evaluation_id,
        "symbol": "EURUSD",
        "direction": "LONG",
        "volume": 0.1,
        "entry_price": 1.0850,
        "stop_price": 1.0800,
        "target_price": 1.0950,
        "mode": "ADVISORY",
    }
    await trades_repo.open_trade(
        **common,
        entry_time_utc=NOW,
        label_span_start=NOW - timedelta(days=3),
        label_span_end=NOW - timedelta(days=2),
    )
    straddling = await trades_repo.open_trade(
        **common,
        entry_time_utc=NOW + timedelta(hours=1),
        label_span_start=NOW - timedelta(hours=6),
        label_span_end=NOW + timedelta(hours=6),
    )

    overlapping = await trades_repo.labels_overlapping(start=NOW, end=NOW + timedelta(hours=1))
    assert [t.id for t in overlapping] == [straddling]


async def test_evaluation_and_trade_commit_or_roll_back_together(
    database: Database,
) -> None:
    """The reason this store speaks Postgres rather than PostgREST."""
    cycle = uuid4()
    with pytest.raises(IntegrityError):
        async with database.begin() as session:
            evaluations = EvaluationRepository(session)
            trades = TradeRepository(session)
            evaluation_id = await evaluations.record(
                cycle_id=cycle,
                ts_utc=NOW,
                symbol="EURUSD",
                regime={},
                votes={},
                consensus_score=1.0,
                fired=True,
                reason="both or neither",
            )
            await trades.open_trade(
                evaluation_id=evaluation_id,
                symbol="EURUSD",
                direction="LONG",
                volume=0.1,
                entry_price=1.0850,
                entry_time_utc=NOW,
                stop_price=1.0900,  # invalid: rolls the evaluation back too
                target_price=1.0950,
                label_span_start=NOW,
                label_span_end=NOW + timedelta(hours=8),
                mode="DEMO_AUTO",
            )

    async with database.session() as session:
        assert await EvaluationRepository(session).for_cycle(cycle) == []


# -- windows ------------------------------------------------------------------


async def test_window_search_returns_nearest_first(windows_repo: WindowRepository) -> None:
    resolved = NOW - timedelta(days=1)
    for index, seed in enumerate((0.1, 0.5, 0.9)):
        await windows_repo.upsert(
            symbol="EURUSD",
            ts_utc=NOW - timedelta(days=10 + index),
            timeframe="H1",
            embedding=_embedding(seed),
            normalised_ohlc={"o": 0.0},
            forward_outcome={"r": 1.0},
            outcome_resolved_at=resolved,
        )

    neighbours = await windows_repo.search(NOW, embedding=_embedding(0.5), limit=3)

    assert len(neighbours) == 3
    distances = [n.distance for n in neighbours]
    assert distances == sorted(distances)


async def test_window_search_hides_unresolved_outcomes(
    windows_repo: WindowRepository,
) -> None:
    """forward_outcome is future data; an unresolved analogue is the future."""
    await windows_repo.upsert(
        symbol="EURUSD",
        ts_utc=NOW - timedelta(days=5),
        timeframe="H1",
        embedding=_embedding(0.5),
        normalised_ohlc={"o": 0.0},
    )

    assert await windows_repo.search(NOW, embedding=_embedding(0.5)) == []
    assert await windows_repo.count_retrievable(NOW) == 0


async def test_window_search_hides_outcomes_that_resolve_after_as_of(
    windows_repo: WindowRepository,
) -> None:
    window_id = await windows_repo.upsert(
        symbol="EURUSD",
        ts_utc=NOW - timedelta(days=5),
        timeframe="H1",
        embedding=_embedding(0.5),
        normalised_ohlc={"o": 0.0},
        forward_outcome={"r": 2.0},
        outcome_resolved_at=NOW + timedelta(days=1),
    )

    assert await windows_repo.search(NOW, embedding=_embedding(0.5)) == []

    later = await windows_repo.search(NOW + timedelta(days=2), embedding=_embedding(0.5))
    assert [n.window.id for n in later] == [window_id]


async def test_outcome_and_resolution_time_must_arrive_together(
    windows_repo: WindowRepository,
) -> None:
    with pytest.raises(ValueError, match="must be set together"):
        await windows_repo.upsert(
            symbol="EURUSD",
            ts_utc=NOW,
            timeframe="H1",
            embedding=_embedding(0.5),
            normalised_ohlc={},
            forward_outcome={"r": 1.0},
        )


async def test_wrong_embedding_dimension_is_rejected(
    windows_repo: WindowRepository,
) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        await windows_repo.upsert(
            symbol="EURUSD",
            ts_utc=NOW,
            timeframe="H1",
            embedding=[0.1] * 7,
            normalised_ohlc={},
        )


async def test_resolving_an_outcome_makes_a_window_retrievable(
    windows_repo: WindowRepository,
) -> None:
    window_id = await windows_repo.upsert(
        symbol="EURUSD",
        ts_utc=NOW - timedelta(days=5),
        timeframe="H1",
        embedding=_embedding(0.4),
        normalised_ohlc={"o": 0.0},
    )
    assert await windows_repo.count_retrievable(NOW) == 0

    await windows_repo.resolve_outcome(
        window_id, forward_outcome={"r": 1.5}, outcome_resolved_at=NOW - timedelta(hours=1)
    )

    assert await windows_repo.count_retrievable(NOW) == 1
