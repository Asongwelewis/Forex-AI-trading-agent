"""Triple-barrier resolution, and the pessimistic rule that makes it trustworthy.

`test_a_bar_touching_both_barriers_resolves_as_stop` is the one that matters. Assuming TARGET
there is the single most common way a backtest shows an edge that does not exist, and no other
test in this file would catch it — the ambiguous bar produces a plausible-looking win either
way.
"""

from __future__ import annotations

import pytest

from fxagent.adapters.base import OrderSide
from fxagent.backtest.barriers import Barrier, resolve_barriers
from tests.backtest.builders import bar, series

#: Long from 1.1000: stop 1.0975, target 1.1050.
LONG = {"side": OrderSide.BUY, "stop_price": 1.0975, "target_price": 1.1050, "max_bars": 10}
SHORT = {"side": OrderSide.SELL, "stop_price": 1.1025, "target_price": 1.0950, "max_bars": 10}


class TestWhichBarrierIsTouched:
    def test_a_target_touch_is_reported_as_target(self) -> None:
        bars = series([bar(0), bar(1, high=1.1010), bar(2, high=1.1060)])
        outcome = resolve_barriers(bars, 0, **LONG)
        assert outcome.touched is Barrier.TARGET
        assert outcome.exit_index == 2
        assert outcome.exit_price == pytest.approx(1.1050)

    def test_a_stop_touch_is_reported_as_stop(self) -> None:
        bars = series([bar(0), bar(1, low=1.0990), bar(2, low=1.0970)])
        outcome = resolve_barriers(bars, 0, **LONG)
        assert outcome.touched is Barrier.STOP
        assert outcome.exit_price == pytest.approx(1.0975)

    def test_a_short_reads_the_barriers_the_other_way_round(self) -> None:
        bars = series([bar(0), bar(1, low=1.0940)])
        outcome = resolve_barriers(bars, 0, **SHORT)
        assert outcome.touched is Barrier.TARGET
        assert outcome.exit_price == pytest.approx(1.0950)

    def test_touching_the_level_exactly_counts(self) -> None:
        """A broker's stop triggers at the level, not a tick past it."""
        bars = series([bar(0), bar(1, low=1.0975)])
        assert resolve_barriers(bars, 0, **LONG).touched is Barrier.STOP

    def test_time_is_a_real_outcome_and_exits_at_a_printed_price(self) -> None:
        bars = series([bar(index, close=1.1000 + index * 0.0001) for index in range(6)])
        outcome = resolve_barriers(bars, 0, **{**LONG, "max_bars": 3})
        assert outcome.touched is Barrier.TIME
        assert outcome.exit_index == 3
        assert outcome.exit_price == pytest.approx(bars.bars[3].close)

    def test_the_first_barrier_wins_not_the_deepest(self) -> None:
        bars = series([bar(0), bar(1, high=1.1060), bar(2, low=1.0900)])
        assert resolve_barriers(bars, 0, **LONG).touched is Barrier.TARGET


class TestIntrabarAmbiguity:
    def test_a_bar_touching_both_barriers_resolves_as_stop(self) -> None:
        """OHLC cannot say which came first. The only error with a known sign is pessimism."""
        bars = series([bar(0), bar(1, high=1.1060, low=1.0970)])
        outcome = resolve_barriers(bars, 0, **LONG)
        assert outcome.touched is Barrier.STOP
        assert outcome.ambiguous_resolution
        assert not outcome.observed

    def test_an_unambiguous_exit_is_not_flagged(self) -> None:
        bars = series([bar(0), bar(1, high=1.1060)])
        outcome = resolve_barriers(bars, 0, **LONG)
        assert not outcome.ambiguous_resolution
        assert outcome.observed

    def test_a_short_is_equally_pessimistic(self) -> None:
        bars = series([bar(0), bar(1, high=1.1030, low=1.0940)])
        outcome = resolve_barriers(bars, 0, **SHORT)
        assert outcome.touched is Barrier.STOP
        assert outcome.ambiguous_resolution

    def test_wider_barriers_produce_less_ambiguity(self) -> None:
        """The diagnostic the rate is for: a high rate means the stops are too tight."""
        bars = series([bar(0), bar(1, high=1.1030, low=1.0980)])
        tight = resolve_barriers(
            bars, 0, side=OrderSide.BUY, stop_price=1.0990, target_price=1.1020, max_bars=10
        )
        wide = resolve_barriers(
            bars, 0, side=OrderSide.BUY, stop_price=1.0900, target_price=1.1100, max_bars=10
        )
        assert tight.ambiguous_resolution
        assert wide.touched is Barrier.TIME


class TestLookAhead:
    def test_the_entry_bar_is_not_scanned(self) -> None:
        """The entry fills at that bar's close, so its range is already history."""
        bars = series([bar(0, high=1.1060, low=1.0970), bar(1), bar(2)])
        outcome = resolve_barriers(bars, 0, **LONG)
        assert outcome.touched is Barrier.TIME
        assert not outcome.ambiguous_resolution

    def test_resolution_never_looks_past_the_time_barrier(self) -> None:
        bars = series([bar(0), bar(1), bar(2), bar(3, high=1.1060)])
        outcome = resolve_barriers(bars, 0, **{**LONG, "max_bars": 2})
        assert outcome.touched is Barrier.TIME
        assert outcome.exit_index == 2

    def test_the_series_running_out_ends_the_trade(self) -> None:
        bars = series([bar(0), bar(1)])
        outcome = resolve_barriers(bars, 0, **{**LONG, "max_bars": 50})
        assert outcome.touched is Barrier.TIME
        assert outcome.exit_index == 1


class TestLabelSpan:
    def test_the_span_runs_from_entry_bar_to_exit_bar(self) -> None:
        """Purging needs this; a label without a span cannot be removed from a training fold."""
        bars = series([bar(0), bar(1), bar(2, high=1.1060)])
        outcome = resolve_barriers(bars, 0, **LONG)
        assert outcome.label_span_start == bars.bars[0].timestamp
        assert outcome.label_span_end == bars.bars[2].timestamp
        assert outcome.bars_held == 2

    def test_the_span_is_never_a_single_point(self) -> None:
        bars = series([bar(0), bar(1, high=1.1060)])
        outcome = resolve_barriers(bars, 0, **LONG)
        assert outcome.label_span_end > outcome.label_span_start


class TestValidation:
    def test_a_zero_time_barrier_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_bars must be at least 1"):
            resolve_barriers(series([bar(0), bar(1)]), 0, **{**LONG, "max_bars": 0})

    def test_an_entry_outside_the_series_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside a series"):
            resolve_barriers(series([bar(0)]), 5, **LONG)
