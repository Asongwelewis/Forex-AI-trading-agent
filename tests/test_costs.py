"""The cost model, against hand-computed fills and a hand-counted calendar.

The Wednesday cases carry their weekday in the test name because getting the triple-swap day
wrong understates a held position by two nights a week, and that error is invisible in any test
that only checks "some nights were charged".
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxagent.adapters.base import OrderSide
from fxagent.costs import (
    ROLLOVER_HOUR_UTC,
    TRIPLE_SWAP_WEEKDAY,
    CostConfig,
    Quote,
    SpreadSource,
    fill,
    rollover_nights,
    swap_cost,
)
from fxagent.risk.symbols import SymbolSpec

EURUSD = SymbolSpec.forex("EURUSD")
USDJPY = SymbolSpec.forex("USDJPY")
DEFAULTS = CostConfig()

#: 18 Aug 2026 is a Tuesday. Every date below is stated relative to it.
TUESDAY = datetime(2026, 8, 18, tzinfo=UTC)
WEDNESDAY = datetime(2026, 8, 19, tzinfo=UTC)


def at(day: datetime, hour: int, minute: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute)


class TestFills:
    def test_a_buy_pays_the_ask_side_of_a_fixed_spread(self) -> None:
        """1 pip spread means half a pip each side, then half a pip of slippage on top."""
        result = fill(1.1000, OrderSide.BUY, EURUSD, DEFAULTS)
        assert result.price == pytest.approx(1.1000 + 0.00005 + 0.00005)
        assert result.spread_source is SpreadSource.FIXED

    def test_a_sell_pays_the_bid_side(self) -> None:
        result = fill(1.1000, OrderSide.SELL, EURUSD, DEFAULTS)
        assert result.price == pytest.approx(1.1000 - 0.00005 - 0.00005)

    def test_stored_quotes_are_used_when_the_feed_carried_them(self) -> None:
        quote = Quote(bid=1.09990, ask=1.10010)
        buy = fill(1.1000, OrderSide.BUY, EURUSD, DEFAULTS, quote)
        assert buy.price == pytest.approx(1.10010 + 0.00005)
        assert buy.spread_source is SpreadSource.STORED
        assert buy.spread == pytest.approx(0.0002)

    def test_a_half_populated_quote_falls_back_rather_than_inventing_the_other_side(self) -> None:
        result = fill(1.1000, OrderSide.BUY, EURUSD, DEFAULTS, Quote(bid=1.09990, ask=None))
        assert result.spread_source is SpreadSource.FIXED

    def test_an_inverted_quote_is_not_trusted(self) -> None:
        result = fill(1.1000, OrderSide.BUY, EURUSD, DEFAULTS, Quote(bid=1.1005, ask=1.0995))
        assert result.spread_source is SpreadSource.FIXED

    def test_slippage_is_adverse_on_both_sides(self) -> None:
        """There is no path through this module that improves a fill."""
        buy = fill(1.1000, OrderSide.BUY, EURUSD, DEFAULTS)
        sell = fill(1.1000, OrderSide.SELL, EURUSD, DEFAULTS)
        assert buy.price > 1.1000
        assert sell.price < 1.1000
        assert buy.total_cost == pytest.approx(sell.total_cost)

    def test_a_yen_pip_is_a_hundredth_so_the_same_config_costs_a_hundred_times_more_in_price(
        self,
    ) -> None:
        euro = fill(1.1000, OrderSide.BUY, EURUSD, DEFAULTS).total_cost
        yen = fill(157.00, OrderSide.BUY, USDJPY, DEFAULTS).total_cost
        assert yen == pytest.approx(euro * 100)

    def test_negative_slippage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="improve a fill"):
            CostConfig(slippage_pips=-0.5)

    def test_the_default_slippage_is_half_a_pip(self) -> None:
        assert DEFAULTS.slippage_pips == 0.5


class TestRollover:
    def test_an_intraday_trade_crosses_no_rollover(self) -> None:
        assert rollover_nights(at(TUESDAY, 9), at(TUESDAY, 17)) == 0

    def test_a_trade_opened_after_the_roll_and_closed_before_the_next_pays_nothing(self) -> None:
        assert rollover_nights(at(TUESDAY, 21, 30), at(TUESDAY, 22)) == 0

    def test_an_hour_spanning_the_roll_pays_a_night(self) -> None:
        """A night is a boundary crossed, not a duration held."""
        assert rollover_nights(at(TUESDAY, 20, 30), at(TUESDAY, 21, 30)) == 1

    def test_the_roll_instant_itself_is_inclusive_at_the_exit(self) -> None:
        assert rollover_nights(at(TUESDAY, 20), at(TUESDAY, ROLLOVER_HOUR_UTC)) == 1

    def test_a_position_opened_exactly_at_the_roll_does_not_pay_for_it(self) -> None:
        assert rollover_nights(at(TUESDAY, ROLLOVER_HOUR_UTC), at(TUESDAY, 23)) == 0

    def test_wednesday_night_counts_three(self) -> None:
        """T+2 settlement carries Wednesday's value date over the weekend."""
        assert WEDNESDAY.weekday() == TRIPLE_SWAP_WEEKDAY
        assert rollover_nights(at(WEDNESDAY, 20), at(WEDNESDAY, 22)) == 3

    def test_tuesday_night_counts_one(self) -> None:
        assert rollover_nights(at(TUESDAY, 20), at(TUESDAY, 22)) == 1

    def test_tuesday_evening_to_thursday_evening_is_five_nights(self) -> None:
        """Tuesday 1, Wednesday 3, Thursday 1. Hand-counted."""
        assert rollover_nights(at(TUESDAY, 20), at(TUESDAY.replace(day=20), 22)) == 5

    def test_a_naive_datetime_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            rollover_nights(datetime(2026, 8, 18, 20), at(TUESDAY, 22))  # noqa: DTZ001

    def test_an_exit_before_the_entry_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="is before entry"):
            rollover_nights(at(TUESDAY, 22), at(TUESDAY, 20))


class TestSwap:
    def test_an_unconfigured_swap_charges_nothing_and_says_so(self) -> None:
        """A wrong swap number is worse than an absent one, so the default is visibly absent."""
        assert not DEFAULTS.swap_is_configured
        assert swap_cost(OrderSide.BUY, 0.1, at(TUESDAY, 20), at(TUESDAY, 22), DEFAULTS) == 0.0

    def test_a_configured_rate_is_charged_per_lot_per_night(self) -> None:
        config = CostConfig(swap_long_per_lot=-7.0)
        assert config.swap_is_configured
        charge = swap_cost(OrderSide.BUY, 0.1, at(TUESDAY, 20), at(TUESDAY, 22), config)
        assert charge == pytest.approx(-0.7)

    def test_wednesday_triples_the_charge(self) -> None:
        config = CostConfig(swap_long_per_lot=-7.0)
        tuesday = swap_cost(OrderSide.BUY, 1.0, at(TUESDAY, 20), at(TUESDAY, 22), config)
        wednesday = swap_cost(OrderSide.BUY, 1.0, at(WEDNESDAY, 20), at(WEDNESDAY, 22), config)
        assert wednesday == pytest.approx(tuesday * 3)

    def test_the_two_sides_read_different_rates(self) -> None:
        config = CostConfig(swap_long_per_lot=-7.0, swap_short_per_lot=2.0)
        window = (at(TUESDAY, 20), at(TUESDAY, 22))
        assert swap_cost(OrderSide.BUY, 1.0, *window, config) == pytest.approx(-7.0)
        assert swap_cost(OrderSide.SELL, 1.0, *window, config) == pytest.approx(2.0)
