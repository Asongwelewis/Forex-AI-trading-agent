"""Total open risk, the currency clusters underneath it, and annotations that stop nothing.

Half of this file is about what `assess_exposure` does *not* do. It has no way to reject a
book, so the tests assert that a wildly over-cap book still comes back whole, with every
position present and an annotation attached — because the failure mode worth guarding against
is not a missing warning, it is someone later making the warning gate something.
"""

from __future__ import annotations

import math

import pytest

from fxagent.risk.exposure import (
    CORRELATION_WARNING,
    EXPOSURE_WARNING,
    MAX_TOTAL_RISK,
    UNBOUNDED_RISK,
    OpenRisk,
    assess_exposure,
    currency_exposure,
    total_open_risk,
)
from fxagent.risk.sizing import RiskConfig, position_size
from fxagent.risk.symbols import SymbolSpec
from fxagent.strategies.base import SignalDirection

LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT

CONFIG = RiskConfig(reference_equity=1000.0)


def bet(symbol: str, direction: SignalDirection, risk: float | None = 5.0) -> OpenRisk:
    """One open position risking `risk` account-currency units. `None` means no stop."""
    spec = SymbolSpec.forex(symbol)
    return OpenRisk(
        symbol=symbol,
        base=spec.base,
        quote=spec.quote,
        direction=direction,
        volume=0.02,
        risk_amount=risk,
    )


class TestTotalOpenRisk:
    def test_an_empty_book_risks_nothing(self) -> None:
        assert total_open_risk([]) == 0.0

    def test_risk_sums_across_positions(self) -> None:
        assert total_open_risk([bet("EURUSD", LONG), bet("GBPUSD", LONG)]) == pytest.approx(10.0)

    def test_a_position_without_a_stop_is_infinite_not_zero(self) -> None:
        """`Position.stop_loss` is optional at the adapter boundary. It must never sum as 0."""
        assert total_open_risk([bet("EURUSD", LONG, risk=None)]) == math.inf

    def test_one_unstopped_position_makes_the_whole_book_unbounded(self) -> None:
        book = [bet("EURUSD", LONG), bet("GBPUSD", LONG, risk=None)]
        assert total_open_risk(book) == math.inf

    def test_a_flat_direction_is_not_a_position(self) -> None:
        with pytest.raises(ValueError, match="abstention is not a position"):
            bet("EURUSD", SignalDirection.FLAT)

    def test_a_sized_trade_lifts_into_the_book_unchanged(self) -> None:
        size = position_size(1000.0, 0.005, 1.1000, 1.0980, SymbolSpec.forex("EURUSD"))
        assert size is not None
        risk = OpenRisk.from_size(size)
        assert risk.risk_amount == pytest.approx(size.risk_amount)
        assert risk.base == "EUR"
        assert risk.quote == "USD"
        assert risk.direction is LONG


class TestCurrencyClusters:
    def test_a_pair_is_two_currency_legs(self) -> None:
        exposures = {e.currency: e for e in currency_exposure([bet("EURUSD", LONG)])}
        assert exposures["EUR"].long_risk == pytest.approx(5.0)
        assert exposures["USD"].short_risk == pytest.approx(5.0)

    def test_three_usd_quoted_longs_are_one_dollar_position(self) -> None:
        """The finding CLAUDE.md asks for, stated as an aggregate rather than three rows."""
        book = [bet("EURUSD", LONG), bet("GBPUSD", LONG), bet("AUDUSD", LONG)]
        usd = next(e for e in currency_exposure(book) if e.currency == "USD")
        assert usd.short_risk == pytest.approx(15.0)
        assert usd.short_symbols == ("EURUSD", "GBPUSD", "AUDUSD")
        assert usd.clustered_side is SHORT

    def test_opposing_legs_are_reported_gross_and_never_netted(self) -> None:
        """Long USD 5 and short USD 5 is two live positions, not a flat book."""
        usd = next(
            e
            for e in currency_exposure([bet("EURUSD", LONG), bet("USDJPY", LONG)])
            if e.currency == "USD"
        )
        assert usd.short_risk == pytest.approx(5.0)
        assert usd.long_risk == pytest.approx(5.0)
        assert usd.gross_risk == pytest.approx(10.0)
        assert not hasattr(usd, "net")

    def test_two_unrelated_pairs_cluster_on_nothing(self) -> None:
        book = [bet("EURUSD", LONG), bet("AUDCAD", SHORT)]
        assert all(e.clustered_side is None for e in currency_exposure(book))

    def test_the_output_order_is_stable(self) -> None:
        """The journal stores these; a report that reorders itself cannot be diffed."""
        book = [bet("GBPUSD", LONG), bet("AUDCAD", SHORT), bet("EURUSD", LONG)]
        assert [e.currency for e in currency_exposure(book)] == [
            "AUD",
            "CAD",
            "EUR",
            "GBP",
            "USD",
        ]


class TestAnnotations:
    def test_a_book_inside_both_caps_says_nothing(self) -> None:
        report = assess_exposure([bet("EURUSD", LONG)], CONFIG)
        assert report.annotations == ()
        assert not report.over_cap

    def test_past_two_percent_the_plan_is_annotated(self) -> None:
        book = [bet(symbol, LONG, risk=8.0) for symbol in ("EURUSD", "GBPUSD", "AUDNZD")]
        report = assess_exposure(book, CONFIG)
        assert report.total_risk_fraction == pytest.approx(0.024)
        assert report.over_cap
        assert EXPOSURE_WARNING in report.codes

    def test_exactly_at_the_cap_is_not_over_it(self) -> None:
        book = [bet("EURUSD", LONG, risk=20.0)]
        report = assess_exposure(book, CONFIG)
        assert report.total_risk_fraction == pytest.approx(MAX_TOTAL_RISK)
        assert EXPOSURE_WARNING not in report.codes

    def test_the_cluster_is_annotated_with_its_aggregate(self) -> None:
        book = [bet("EURUSD", LONG), bet("GBPUSD", LONG), bet("AUDUSD", LONG)]
        report = assess_exposure(book, CONFIG)
        correlation = next(a for a in report.annotations if a.code == CORRELATION_WARNING)
        assert "3 open signals are SHORT USD" in correlation.detail
        assert "$15" in correlation.detail

    def test_an_unstopped_position_is_annotated_and_makes_the_total_unbounded(self) -> None:
        report = assess_exposure([bet("EURUSD", LONG, risk=None)], CONFIG)
        assert report.unbounded_symbols == ("EURUSD",)
        assert UNBOUNDED_RISK in report.codes
        assert EXPOSURE_WARNING in report.codes
        assert math.isinf(report.total_risk_fraction)
        assert "unbounded" in report.describe()

    def test_an_empty_book_is_a_report_not_a_none(self) -> None:
        report = assess_exposure([], CONFIG)
        assert report.open_count == 0
        assert report.total_risk_amount == 0.0
        assert report.annotations == ()

    def test_describe_names_the_equity_the_percentages_are_of(self) -> None:
        report = assess_exposure([bet("EURUSD", LONG)], CONFIG)
        assert "$1,000 account" in report.describe()
        assert "0.50%" in report.describe()


class TestItCannotBlock:
    def test_a_wildly_over_cap_book_still_comes_back_whole(self) -> None:
        """There is no broker connection in this process, so there is nothing to block."""
        book = [bet(symbol, LONG, risk=100.0) for symbol in ("EURUSD", "GBPUSD", "AUDUSD")]
        report = assess_exposure(book, CONFIG)
        assert report.total_risk_fraction == pytest.approx(0.30)
        assert report.open_count == 3
        assert len(report.currencies) == 4

    def test_assessing_an_over_cap_book_does_not_raise(self) -> None:
        assess_exposure([bet("EURUSD", LONG, risk=10_000.0)], CONFIG)

    def test_the_annotations_carry_no_number_to_total_up(self) -> None:
        """A severity that could be summed becomes a score, and a score becomes a threshold."""
        report = assess_exposure([bet("EURUSD", LONG, risk=100.0)], CONFIG)
        annotation = report.annotations[0]
        assert set(vars(annotation)) == {"code", "detail"}
        assert isinstance(annotation.detail, str)
