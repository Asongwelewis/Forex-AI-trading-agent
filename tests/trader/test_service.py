"""The loop: fetch, decide, journal, notify — and what happens when one of those fails.

The store and the notifier are faked at the session boundary rather than mocked per call, so
these exercise the real repository call signatures. A test that stubbed `_journal` would pass
while the row shape was wrong, which is the failure mode most likely to reach production here.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from fxagent.adapters.base import BarSeries
from fxagent.risk.symbols import SymbolSpec
from fxagent.trader.service import TraderConfig, TraderService
from tests.strategies.builders import flat_bar, h1_series

SYMBOL = "EURUSD"
SOURCE = "mt5_exness"
LONDON = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


def _series(count: int = 300, *, timeframe: str = "H1") -> BarSeries:
    start = LONDON - timedelta(hours=count - 1)
    bars = [flat_bar(start + timedelta(hours=i)) for i in range(count)]
    return h1_series(bars, timeframe=timeframe) if timeframe == "H1" else BarSeries(
        symbol=SYMBOL,
        timeframe=timeframe,
        bars=tuple(
            flat_bar(LONDON - timedelta(days=count - 1 - i)) for i in range(count)
        ),
    )


class FakeSession:
    """Stands in for an AsyncSession. The repositories are patched, so it is only a token."""


class FakeDatabase:
    """Records what was written, and can be told to fail on demand."""

    def __init__(self) -> None:
        self.begins = 0
        self.sessions = 0
        self.fail_on_begin = False

    @contextlib.asynccontextmanager
    async def session(self):
        self.sessions += 1
        yield FakeSession()

    @contextlib.asynccontextmanager
    async def begin(self):
        self.begins += 1
        if self.fail_on_begin:
            raise RuntimeError("store is unreachable")
        yield FakeSession()


class RecordingNotifier:
    def __init__(self, *, ok: bool = True, boom: bool = False) -> None:
        self.sent: list[str] = []
        self._ok = ok
        self._boom = boom

    async def send(self, text: str) -> bool:
        if self._boom:
            raise RuntimeError("telegram exploded")
        self.sent.append(text)
        return self._ok


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch):
    """A service whose store reads return bars and whose writes are captured."""
    rows: list[dict[str, Any]] = []
    beats: list[dict[str, Any]] = []
    series: dict[str, BarSeries] = {"H1": _series(), "D1": _series(120, timeframe="D1")}

    class FakeBarRepository:
        def __init__(self, session: Any) -> None:
            pass

        async def latest_bars(
            self, symbol: str, timeframe: str, count: int, *, source: str
        ) -> BarSeries:
            assert source == SOURCE, "the trader must read the series it was configured for"
            stored = series.get(timeframe)
            if stored is None:
                return BarSeries(symbol=symbol, timeframe=timeframe, bars=())
            # Re-labelled for the symbol asked. The bar *values* are shared on purpose — the
            # assembly is under test, not whether two symbols happen to differ.
            return BarSeries(symbol=symbol, timeframe=timeframe, bars=stored.bars)

    class FakeEvaluationRepository:
        def __init__(self, session: Any) -> None:
            pass

        async def record(self, **row: Any) -> int:
            rows.append(row)
            return len(rows)

    class FakeHeartbeatRepository:
        def __init__(self, session: Any) -> None:
            pass

        async def beat(self, service: str, **kwargs: Any) -> int:
            beats.append({"service": service, **kwargs})
            return len(beats)

    monkeypatch.setattr("fxagent.trader.service.BarRepository", FakeBarRepository)
    monkeypatch.setattr("fxagent.trader.service.EvaluationRepository", FakeEvaluationRepository)
    monkeypatch.setattr("fxagent.trader.service.HeartbeatRepository", FakeHeartbeatRepository)

    database = FakeDatabase()

    def build(**overrides: Any) -> TraderService:
        config = TraderConfig(
            symbols=(SYMBOL,),
            timeframe="H1",
            source=SOURCE,
            specs={SYMBOL: SymbolSpec.forex(SYMBOL)},
            **overrides,
        )
        notifier = overrides.pop("notifier", None)
        return TraderService(database=database, config=config, notifier=notifier)

    return {
        "build": build,
        "rows": rows,
        "beats": beats,
        "series": series,
        "database": database,
    }


# --- the pass runs end to end ---------------------------------------------------


async def test_a_pass_journals_one_row_per_symbol_even_when_nothing_fires(wired) -> None:
    service = wired["build"]()

    results = await service.pass_once()

    assert len(results) == 1
    assert len(wired["rows"]) == 1
    row = wired["rows"][0]
    assert row["symbol"] == SYMBOL
    assert row["reason"], "the reason is required, including when nothing fired"


async def test_one_cycle_id_covers_the_whole_pass(wired, monkeypatch) -> None:
    """The evaluations table is unique on (cycle_id, symbol), so a pass is one identifier.

    Sharing it is what makes a retried pass update its rows rather than double-count the
    refusal statistics the router is meant to be learned from.
    """
    config_extras = {"symbols": (SYMBOL, "GBPUSD")}
    service = TraderService(
        database=wired["database"],
        config=TraderConfig(
            symbols=(SYMBOL, "GBPUSD"),
            timeframe="H1",
            source=SOURCE,
            specs={s: SymbolSpec.forex(s) for s in (SYMBOL, "GBPUSD")},
        ),
    )

    await service.pass_once()

    assert len({row["cycle_id"] for row in wired["rows"]}) == 1
    assert {row["symbol"] for row in wired["rows"]} == {SYMBOL, "GBPUSD"}
    assert config_extras  # keeps the intent of the two-symbol setup visible


async def test_a_dry_run_decides_but_writes_nothing(wired) -> None:
    """It must not short-circuit the decision — that would prove only that the flags parse."""
    service = wired["build"]()

    results = await service.pass_once(dry_run=True)

    assert len(results) == 1, "the decision still happened"
    assert results[0].selection.diagnostics["votes"], "the full ledger was still computed"
    assert wired["rows"] == []


async def test_an_empty_series_is_reported_rather_than_journalled_as_a_decision(wired) -> None:
    """No bars is not a verdict. Recording one would put a fabricated refusal in the ledger."""
    wired["series"]["H1"] = BarSeries(symbol=SYMBOL, timeframe="H1", bars=())
    service = wired["build"]()

    results = await service.pass_once()

    assert results == []
    assert wired["rows"] == []


# --- failure containment ----------------------------------------------------------


async def test_one_symbols_failure_does_not_end_the_pass(wired, monkeypatch) -> None:
    """Otherwise the completeness of the ledger depends on alphabetical order."""
    calls = {"n": 0}
    original = wired["build"]

    service = TraderService(
        database=wired["database"],
        config=TraderConfig(
            symbols=("AAAUSD", SYMBOL),
            timeframe="H1",
            source=SOURCE,
            specs={s: SymbolSpec.forex(s) for s in ("AAAUSD", SYMBOL)},
        ),
    )

    real_evaluate = service._evaluate

    async def flaky(symbol: str, **kwargs: Any):
        calls["n"] += 1
        if symbol == "AAAUSD":
            raise RuntimeError("this symbol is broken")
        return await real_evaluate(symbol, **kwargs)

    monkeypatch.setattr(service, "_evaluate", flaky)

    results = await service.pass_once()

    assert calls["n"] == 2, "the second symbol was still attempted"
    assert len(results) == 1
    assert service.stats.errors == 1
    assert "AAAUSD" in service.stats.last_error
    assert original  # the fixture builder stays referenced for readability


async def test_a_failing_daily_read_does_not_lose_the_intraday_evaluation(
    wired, monkeypatch
) -> None:
    """The bias then reports that it had no daily view, which is recorded rather than assumed."""
    service = wired["build"]()
    original = service._fetch

    async def fetch(symbol: str, timeframe: str, count: int) -> BarSeries:
        if timeframe == "D1":
            raise RuntimeError("daily read failed")
        return await original(symbol, timeframe, count)

    monkeypatch.setattr(service, "_fetch", fetch)

    results = await service.pass_once()

    assert len(results) == 1
    assert results[0].bias.direction is None
    assert len(wired["rows"]) == 1


async def test_a_journal_failure_is_not_swallowed(wired) -> None:
    """A decision that was made and not recorded is an observation lost.

    Everything after the ledger is best-effort; the ledger itself is not, so this must surface
    as a counted error rather than a silent success.
    """
    service = wired["build"]()
    wired["database"].fail_on_begin = True

    results = await service.pass_once()

    assert results == [], "the symbol is counted as failed, not as evaluated"
    assert service.stats.errors == 1


# --- notification -----------------------------------------------------------------


async def test_only_actionable_cycles_are_sent(wired) -> None:
    """Every cycle is journalled; a message on every silent bar is a message nobody reads."""
    notifier = RecordingNotifier()
    service = TraderService(
        database=wired["database"],
        config=TraderConfig(
            symbols=(SYMBOL,),
            timeframe="H1",
            source=SOURCE,
            specs={SYMBOL: SymbolSpec.forex(SYMBOL)},
        ),
        notifier=notifier,
    )

    await service.pass_once()

    assert notifier.sent == [], "nothing fired on a flat series, so nothing was announced"
    assert len(wired["rows"]) == 1, "and it was still recorded"


async def test_a_notifier_that_raises_does_not_cost_the_row(wired) -> None:
    service = TraderService(
        database=wired["database"],
        config=TraderConfig(
            symbols=(SYMBOL,),
            timeframe="H1",
            source=SOURCE,
            specs={SYMBOL: SymbolSpec.forex(SYMBOL)},
        ),
        notifier=RecordingNotifier(boom=True),
    )

    results = await service.pass_once()

    assert len(results) == 1
    assert len(wired["rows"]) == 1
    assert service.stats.errors == 0, "a notifier failure is not an evaluation failure"


# --- configuration guards -----------------------------------------------------------


def test_a_symbol_without_a_spec_is_refused_at_construction(wired) -> None:
    """Guessing a contract size is wrong quietly, and in the direction of larger."""
    with pytest.raises(ValueError, match="no SymbolSpec"):
        TraderService(
            database=wired["database"],
            config=TraderConfig(symbols=("EURUSD", "GBPUSD"), source=SOURCE, specs={}),
        )


def test_an_unset_source_is_refused(wired) -> None:
    """A trader reading one feed while the backtest measured another is a different system."""
    with pytest.raises(ValueError, match="source is required"):
        TraderService(
            database=wired["database"],
            config=TraderConfig(symbols=(SYMBOL,), specs={SYMBOL: SymbolSpec.forex(SYMBOL)}),
        )


# --- uptime ---------------------------------------------------------------------------


async def test_a_heartbeat_records_that_the_desktop_was_awake(wired) -> None:
    """ADR-005 accepted that this only runs while the machine is on.

    A journal that does not say when the agent was awake reads as though every unrecorded hour
    was a quiet market, and every metric derived from it is silently conditioned on uptime.
    """
    service = wired["build"]()
    await service._heartbeat()

    assert wired["beats"], "the trader must record its own uptime"
    beat = wired["beats"][0]
    assert beat["service"] == "trader"
    assert beat["detail"]["symbols"] == [SYMBOL]


async def test_a_heartbeat_failure_does_not_stop_the_loop(wired, monkeypatch) -> None:
    service = wired["build"]()
    wired["database"].fail_on_begin = True

    await service._heartbeat()  # must not raise
