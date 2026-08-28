"""Closing out paper trades: what settles, what waits, and what is never guessed.

The store is faked at the session boundary so these exercise the real repository signatures.
The bars are hand-built with arithmetic simple enough to check on paper, so an assertion about
a stop being hit is an assertion about a price, not about the code agreeing with itself.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from fxagent.adapters.base import Bar, BarSeries
from fxagent.costs import CostConfig
from fxagent.resolve.service import ResolveConfig, resolve_open_trades
from fxagent.risk.symbols import SymbolSpec
from fxagent.store.repositories.trades import TradeRecord

SYMBOL = "EURUSD"
SOURCE = "mt5_exness"
ENTRY_TIME = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
ENTRY = 1.1000
STOP = 1.0980
TARGET = 1.1040


def _bar(index: int, *, high: float, low: float, close: float | None = None) -> Bar:
    price = close if close is not None else (high + low) / 2
    return Bar(
        timestamp=ENTRY_TIME + timedelta(hours=index),
        open=price,
        high=max(high, price),
        low=min(low, price),
        close=price,
        volume=1000,
    )


def _series(bars: list[Bar]) -> BarSeries:
    return BarSeries(symbol=SYMBOL, timeframe="H1", bars=tuple(bars))


def _trade(**overrides: Any) -> TradeRecord:
    base = {
        "id": 1,
        "evaluation_id": 10,
        "symbol": SYMBOL,
        "direction": "LONG",
        "volume": 0.1,
        "entry_price": ENTRY,
        "entry_time_utc": ENTRY_TIME,
        "exit_price": None,
        "exit_time_utc": None,
        "stop_price": STOP,
        "target_price": TARGET,
        "barrier_touched": None,
        "label_span_start": ENTRY_TIME,
        "label_span_end": ENTRY_TIME + timedelta(hours=24),
        "pnl": None,
        "r_multiple": None,
        "mode": "ADVISORY",
        "created_at": ENTRY_TIME,
    }
    return TradeRecord(**{**base, **overrides})


class FakeDatabase:
    def __init__(self) -> None:
        self.closed: list[dict[str, Any]] = []

    @contextlib.asynccontextmanager
    async def session(self):
        yield object()

    @contextlib.asynccontextmanager
    async def begin(self):
        yield object()


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch):
    state: dict[str, Any] = {
        "open": [_trade()],
        "bars": _series([]),
        "quotes": {},
        "closed": [],
        "close_returns": True,
    }

    class FakeTradeRepository:
        def __init__(self, session: Any) -> None:
            pass

        async def open_trades(self, *, symbol: str | None = None) -> list[TradeRecord]:
            return list(state["open"])

        async def close_trade(self, trade_id: int, **fields: Any) -> bool:
            state["closed"].append({"trade_id": trade_id, **fields})
            return bool(state["close_returns"])

    class FakeBarRepository:
        def __init__(self, session: Any) -> None:
            pass

        async def bars_between(self, symbol, timeframe, *, start, end, source) -> BarSeries:
            assert source == SOURCE
            series: BarSeries = state["bars"]
            return BarSeries(
                symbol=symbol,
                timeframe=timeframe,
                bars=tuple(b for b in series.bars if start <= b.timestamp <= end),
            )

        async def quotes_between(self, symbol, timeframe, *, start, end, source):
            return dict(state["quotes"])

    monkeypatch.setattr("fxagent.resolve.service.TradeRepository", FakeTradeRepository)
    monkeypatch.setattr("fxagent.resolve.service.BarRepository", FakeBarRepository)
    return state


def _config(**overrides: Any) -> ResolveConfig:
    base: dict[str, Any] = {
        "source": SOURCE,
        "timeframe": "H1",
        "max_bars_held": 24,
        "costs": CostConfig(fixed_spread_pips=1.0, slippage_pips=0.5),
        "specs": {SYMBOL: SymbolSpec.forex(SYMBOL)},
    }
    return ResolveConfig(**{**base, **overrides})


LATER = ENTRY_TIME + timedelta(days=5)


# --- it closes what the bars settled ------------------------------------------


async def test_a_touched_target_closes_the_trade(wired) -> None:
    wired["bars"] = _series(
        [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY), _bar(1, high=1.1050, low=1.1000)]
    )

    stats = await resolve_open_trades(FakeDatabase(), _config(), now=LATER)

    assert stats.closed == 1
    assert stats.outcomes == {"TARGET": 1}
    closed = wired["closed"][0]
    assert closed["barrier_touched"] == "TARGET"
    assert closed["r_multiple"] is not None and closed["r_multiple"] > 0


async def test_a_touched_stop_closes_the_trade(wired) -> None:
    wired["bars"] = _series(
        [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY), _bar(1, high=1.1005, low=1.0970)]
    )

    stats = await resolve_open_trades(FakeDatabase(), _config(), now=LATER)

    assert stats.outcomes == {"STOP": 1}
    assert wired["closed"][0]["r_multiple"] < 0


async def test_a_bar_touching_both_resolves_to_stop(wired) -> None:
    """The only choice whose error has a known sign, and the replay makes the same one.

    A different rule here would make the paper run flatter itself relative to the very thing it
    is being compared against.
    """
    wired["bars"] = _series(
        [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY), _bar(1, high=1.1050, low=1.0970)]
    )

    await resolve_open_trades(FakeDatabase(), _config(), now=LATER)

    assert wired["closed"][0]["barrier_touched"] == "STOP"


# --- it does not close what the bars have not settled --------------------------


async def test_a_trade_with_no_bar_after_entry_stays_open(wired) -> None:
    """The normal case for a signal from the current hour."""
    wired["bars"] = _series([_bar(0, high=ENTRY, low=ENTRY, close=ENTRY)])

    stats = await resolve_open_trades(FakeDatabase(), _config(), now=LATER)

    assert stats.still_running == 1
    assert stats.closed == 0
    assert wired["closed"] == []


async def test_running_out_of_bars_is_not_a_time_exit(wired) -> None:
    """`resolve_barriers` says TIME both when the horizon is reached and when bars run out.

    Those are different facts: one means "ask again tomorrow", the other means "this trade is
    over". Closing the first as a TIME exit stamps a horizon outcome on a live position, and
    the label would be wrong in the direction of more data than existed.
    """
    quiet = [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY)]
    quiet += [_bar(i, high=1.1005, low=1.0995) for i in range(1, 5)]
    wired["bars"] = _series(quiet)

    stats = await resolve_open_trades(FakeDatabase(), _config(max_bars_held=24), now=LATER)

    assert stats.still_running == 1
    assert stats.closed == 0


async def test_reaching_the_horizon_does_close_as_time(wired) -> None:
    quiet = [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY)]
    quiet += [_bar(i, high=1.1005, low=1.0995) for i in range(1, 30)]
    wired["bars"] = _series(quiet)

    stats = await resolve_open_trades(FakeDatabase(), _config(max_bars_held=24), now=LATER)

    assert stats.outcomes == {"TIME": 1}


# --- it never guesses -----------------------------------------------------------


async def test_a_symbol_without_a_spec_is_skipped_not_resolved(wired) -> None:
    """An R multiple computed against a guessed contract size is wrong in a way nothing catches."""
    wired["bars"] = _series(
        [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY), _bar(1, high=1.1050, low=1.1000)]
    )

    stats = await resolve_open_trades(FakeDatabase(), _config(specs={}), now=LATER)

    assert stats.skipped == 1
    assert stats.closed == 0


async def test_a_missing_entry_bar_leaves_the_trade_open(wired) -> None:
    """An outcome measured from the wrong entry is worse than one not yet measured."""
    wired["bars"] = _series([_bar(3, high=1.1050, low=1.1000), _bar(4, high=1.1060, low=1.1010)])

    stats = await resolve_open_trades(FakeDatabase(), _config(), now=LATER)

    assert stats.skipped == 1
    assert wired["closed"] == []


# --- costs ------------------------------------------------------------------------


async def test_the_exit_is_filled_on_the_wrong_side_of_the_spread(wired) -> None:
    """A long closes by selling, and a sell hits the bid. Never the bar close.

    The costed exit is what makes the paper run comparable with the replay, which charges the
    same way through the same module.
    """
    wired["bars"] = _series(
        [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY), _bar(1, high=1.1050, low=1.1000)]
    )

    await resolve_open_trades(FakeDatabase(), _config(), now=LATER)

    exit_price = wired["closed"][0]["exit_price"]
    assert exit_price < TARGET, "a long exits below the touched target once costs are charged"


async def test_a_stored_quote_is_preferred_over_the_configured_spread(wired) -> None:
    wired["bars"] = _series(
        [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY), _bar(1, high=1.1050, low=1.1000)]
    )
    exit_time = ENTRY_TIME + timedelta(hours=1)
    tight = {exit_time: (1.10395, 1.10405)}
    wide = {exit_time: (1.1030, 1.1050)}

    wired["quotes"] = tight
    await resolve_open_trades(FakeDatabase(), _config(), now=LATER)
    with_tight = wired["closed"][-1]["exit_price"]

    wired["closed"].clear()
    wired["quotes"] = wide
    await resolve_open_trades(FakeDatabase(), _config(), now=LATER)
    with_wide = wired["closed"][-1]["exit_price"]

    assert with_wide < with_tight, "a wider stored spread must produce a worse exit"


# --- containment and idempotence ----------------------------------------------------


async def test_one_bad_trade_does_not_end_the_run(wired, monkeypatch) -> None:
    wired["open"] = [_trade(id=1, symbol="ZZZZZZ"), _trade(id=2)]
    wired["bars"] = _series(
        [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY), _bar(1, high=1.1050, low=1.1000)]
    )

    stats = await resolve_open_trades(FakeDatabase(), _config(), now=LATER)

    assert stats.examined == 2
    assert stats.closed == 1, "the good trade still resolved"


async def test_a_row_already_closed_by_a_concurrent_run_is_not_counted_twice(wired) -> None:
    """A double-closed trade would appear twice in the expectancy denominator."""
    wired["bars"] = _series(
        [_bar(0, high=ENTRY, low=ENTRY, close=ENTRY), _bar(1, high=1.1050, low=1.1000)]
    )
    wired["close_returns"] = False

    stats = await resolve_open_trades(FakeDatabase(), _config(), now=LATER)

    assert stats.closed == 0
    assert stats.outcomes == {}


async def test_nothing_open_is_a_clean_run(wired) -> None:
    wired["open"] = []

    stats = await resolve_open_trades(FakeDatabase(), _config(), now=LATER)

    assert stats.examined == 0
    assert stats.errors == 0


# --- configuration guards ------------------------------------------------------------


def test_an_unset_source_is_refused() -> None:
    with pytest.raises(ValueError, match="source is required"):
        ResolveConfig(source="")


def test_an_unknown_timeframe_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown timeframe"):
        ResolveConfig(source=SOURCE, timeframe="H3")
