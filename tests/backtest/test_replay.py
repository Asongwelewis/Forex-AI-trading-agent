"""The replay loop: point-in-time, costed, and one position at a time.

`test_a_prefix_replays_identically` is the look-ahead test. Replaying the first N bars and
replaying all of them must produce the same decisions over the overlap — if any part of the
pipeline could see a future bar, the longer run would decide differently, and no amount of
reading the loop would prove it does not.
"""

from __future__ import annotations

import pytest

from fxagent.adapters.base import BarSeries
from fxagent.backtest.replay import ReplayConfig, default_strategies, replay
from fxagent.costs import CostConfig, SpreadSource
from fxagent.risk.sizing import RiskConfig
from fxagent.risk.symbols import SymbolSpec
from fxagent.strategies.base import SignalDirection
from fxagent.strategies.carry_divergence import CarryDivergence
from fxagent.strategies.range_reversion import RangeReversion
from fxagent.strategies.session_breakout import SessionBreakout
from tests.backtest.builders import (
    FullWeightRouter,
    ScriptedStrategy,
    agreeing_pair,
    bar,
    series,
)

#: Weights every strategy fully regardless of regime, so a fixture does not have to manufacture
#: a trending London morning before the loop can be tested at all. See `FullWeightRouter`.
OPEN_ROUTER = FullWeightRouter()

CONFIG = ReplayConfig(
    spec=SymbolSpec.forex("EURUSD"),
    risk=RiskConfig(reference_equity=100_000.0),
    costs=CostConfig(fixed_spread_pips=1.0, slippage_pips=0.5),
    history_bars=50,
    max_bars_held=10,
)


def rising(count: int, *, step: float = 0.0005) -> object:
    """A steadily rising series — a long fired anywhere in it reaches its target."""
    return series(
        [
            bar(
                index,
                open_=1.1000 + index * step,
                high=1.1000 + index * step + 0.0006,
                low=1.1000 + index * step - 0.0001,
                close=1.1000 + index * step,
            )
            for index in range(count)
        ]
    )


def run(bars, fire_at, **kwargs):  # noqa: ANN001, ANN003 - local helper
    return replay(
        bars,
        kwargs.pop("config", CONFIG),
        strategies=kwargs.pop("strategies", agreeing_pair(fire_at)),
        router=OPEN_ROUTER,
        **kwargs,
    )


class TestTheDefaultsAreTheRealPipeline:
    def test_the_defaults_are_the_production_pipeline(self) -> None:
        """The injection points configure the loop; they do not let the harness fork."""
        strategies = default_strategies()
        assert set(strategies) == {"session_breakout", "range_reversion", "carry_divergence"}
        assert set(default_strategies("H1")) == {"session_breakout", "range_reversion"}
        assert isinstance(strategies["session_breakout"], SessionBreakout)
        assert isinstance(strategies["range_reversion"], RangeReversion)
        assert isinstance(strategies["carry_divergence"], CarryDivergence)


class TestPointInTime:
    def test_a_strategy_never_sees_more_history_than_configured(self) -> None:
        scripted = ScriptedStrategy("session_breakout", set())
        replay(rising(200), CONFIG, strategies={scripted.name: scripted}, router=OPEN_ROUTER)
        assert scripted.seen_lengths
        assert max(scripted.seen_lengths) <= CONFIG.history_bars

    def test_a_prefix_replays_identically(self) -> None:
        """The look-ahead test. A future bar in the window would change the earlier decisions."""
        full = rising(320)
        prefix = series(list(full.bars[:250]))

        long_run = run(full, {200, 230, 260})
        short_run = run(prefix, {200, 230, 260})

        overlap = [t for t in long_run.trades if t.exit_time <= prefix.bars[-1].timestamp]
        assert overlap, "the fixture produced no comparable trades; it proves nothing"
        for from_long, from_short in zip(overlap, short_run.trades, strict=False):
            assert from_long.entry_time == from_short.entry_time
            assert from_long.entry_price == pytest.approx(from_short.entry_price)
            assert from_long.barrier is from_short.barrier
            assert from_long.r_multiple == pytest.approx(from_short.r_multiple)


class TestTrading:
    def test_agreeing_strategies_produce_a_trade(self) -> None:
        result = run(rising(200), {160})
        assert result.fired == 1
        assert len(result.trades) == 1
        assert result.trades[0].direction is SignalDirection.LONG

    def test_a_lone_weighted_sleeve_trades(self) -> None:
        """Inverted when agreement was removed, and this is the whole point of the change.

        This test used to assert that one strategy could never trade. That rule produced zero
        trades in 12,341 decisions on real 2024-25 data, because the router never weights two
        sleeves at once. One weighted sleeve is now sufficient.
        """
        scripted = ScriptedStrategy("session_breakout", {160})
        result = replay(
            rising(200), CONFIG, strategies={scripted.name: scripted}, router=OPEN_ROUTER
        )
        assert result.fired == 1

    def test_only_one_position_runs_at_a_time(self) -> None:
        """Signals arriving mid-trade are counted as skipped, never stacked."""
        result = run(rising(200), {160, 161, 162, 163})
        assert result.fired == 1
        assert result.skipped_in_position >= 1

    def test_a_new_trade_can_open_once_the_last_one_closed(self) -> None:
        result = run(rising(260), {160, 220})
        assert result.fired == 2
        assert result.trades[1].entry_time > result.trades[0].exit_time

    def test_the_ledger_accounts_for_every_bar(self) -> None:
        result = run(rising(200), {160})
        assert result.decisions > 0
        assert result.bars_replayed > 0


class TestCosts:
    def test_a_long_enters_above_the_bar_close(self) -> None:
        """Half the spread plus slippage, on the way in."""
        result = run(rising(200), {160})
        trade = result.trades[0]
        entry_bar = next(b for b in rising(200).bars if b.timestamp == trade.entry_time)
        assert trade.entry_price > entry_bar.close
        assert trade.entry_price == pytest.approx(entry_bar.close + 0.00005 + 0.00005)

    def test_costs_make_the_net_r_worse_than_the_gross(self) -> None:
        """If these were equal the cost model would not be wired in at all."""
        trade = run(rising(200), {160}).trades[0]
        assert trade.r_multiple < trade.gross_r
        assert trade.cost_r > 0

    def test_a_free_run_and_a_costed_run_differ(self) -> None:
        free = ReplayConfig(
            spec=CONFIG.spec,
            risk=CONFIG.risk,
            costs=CostConfig(fixed_spread_pips=0.0, slippage_pips=0.0),
            history_bars=CONFIG.history_bars,
            max_bars_held=CONFIG.max_bars_held,
        )
        costed = run(rising(200), {160}).trades[0]
        gratis = run(rising(200), {160}, config=free).trades[0]
        assert gratis.r_multiple > costed.r_multiple

    def test_stored_quotes_are_used_and_reported(self) -> None:
        bars = rising(200)
        quotes = {b.timestamp: (b.close - 0.0001, b.close + 0.0001) for b in bars.bars}
        result = run(bars, {160}, quotes=quotes)
        assert result.trades[0].spread_source is SpreadSource.STORED
        assert result.spread_sources[SpreadSource.STORED] == 1

    def test_a_feed_without_quotes_falls_back_and_reports_that(self) -> None:
        result = run(rising(200), {160})
        assert result.trades[0].spread_source is SpreadSource.FIXED
        assert "fixed" in result.describe()


class TestLabelsAndDiagnostics:
    def test_every_trade_carries_a_label_span(self) -> None:
        """Purging needs it, and a trade without one silently breaks the folds."""
        for trade in run(rising(260), {160, 220}).trades:
            assert trade.label_span_end > trade.label_span_start
            assert trade.label_span_start == trade.entry_time

    def test_the_ambiguity_rate_is_reported(self) -> None:
        result = run(rising(200), {160})
        assert 0.0 <= result.ambiguity_rate <= 1.0
        assert "ambiguity" in result.describe()

    def test_an_unsizeable_setup_is_counted_not_traded(self) -> None:
        """A tiny account against a wide stop: the signal stands, the trade cannot be placed."""
        tiny = ReplayConfig(
            spec=CONFIG.spec,
            risk=RiskConfig(reference_equity=50.0),
            costs=CONFIG.costs,
            history_bars=CONFIG.history_bars,
            max_bars_held=CONFIG.max_bars_held,
        )
        result = run(rising(200), {160}, config=tiny)
        assert result.fired == 0
        assert result.not_sizeable == 1

    def test_a_strategy_off_its_timeframe_is_excluded_and_named(self) -> None:
        """carry_divergence RAISES on H1 rather than staying silent, so it must not be asked."""
        result = replay(rising(200), CONFIG, router=OPEN_ROUTER)
        assert result.excluded_strategies == ("carry_divergence",)
        assert "not asked on this timeframe" in result.describe()
        assert not result.carry_is_inert  # it was never asked, which is not the same as silent

    def test_an_inert_carry_strategy_is_declared_on_its_own_timeframe(self) -> None:
        """On D1 carry IS asked, and a neutral context means it never votes. Say so."""
        daily = series(
            [bar(index, open_=1.1000 + index * 0.0005) for index in range(200)],
        )
        daily = BarSeries(symbol="EURUSD", timeframe="D1", bars=daily.bars)
        result = replay(daily, CONFIG, router=OPEN_ROUTER)
        assert "carry_divergence" not in result.excluded_strategies
        assert result.carry_is_inert
        assert "carry_divergence is inert" in result.describe()
