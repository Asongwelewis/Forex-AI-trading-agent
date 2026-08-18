"""Sizing arithmetic, the lot step, and the pairs that break a pip-based formula.

The USDJPY cases are here because they are where the units go wrong. A EURUSD test passes
against a formula that has quietly assumed a 0.0001 pip and a USD quote, and says nothing at
all about whether the code handles a yen cross — so the yen cases carry hand-computed
expectations rather than the same expression the implementation uses.
"""

from __future__ import annotations

import math

import pytest

from fxagent.risk.sizing import MAX_RISK_PER_TRADE, PositionSize, position_size
from fxagent.risk.symbols import SymbolSpec
from fxagent.strategies.base import SignalDirection

EURUSD = SymbolSpec.forex("EURUSD")
USDJPY = SymbolSpec.forex("USDJPY")
EURGBP = SymbolSpec.forex("EURGBP")


def _sized(
    *,
    reference_equity: float = 1000.0,
    risk_fraction: float = 0.005,
    entry: float = 1.1000,
    stop_loss: float = 1.0975,
    symbol_spec: SymbolSpec = EURUSD,
    account_currency: str = "USD",
    quote_to_account_rate: float | None = None,
) -> PositionSize:
    """A EURUSD long with a 25-pip stop on a $1,000 account. Asserted non-None for typing."""
    size = position_size(
        reference_equity,
        risk_fraction,
        entry,
        stop_loss,
        symbol_spec,
        account_currency=account_currency,
        quote_to_account_rate=quote_to_account_rate,
    )
    assert size is not None
    return size


class TestTheArithmetic:
    def test_volume_comes_from_the_stop_distance(self) -> None:
        """$5 risked over a 0.0025 stop is $250 of price value per lot, which is 0.02 lots."""
        size = _sized()
        assert size.volume == pytest.approx(0.02)
        assert size.risk_amount == pytest.approx(5.0)
        assert size.stop_distance == pytest.approx(0.0025)

    def test_a_wider_stop_gives_a_smaller_volume_for_the_same_risk(self) -> None:
        """The whole reason sizing is in risk units: the money risked does not move."""
        tight = _sized(stop_loss=1.0975)
        wide = _sized(stop_loss=1.0950)
        assert wide.volume < tight.volume
        assert wide.risk_amount == pytest.approx(tight.risk_amount, rel=0.05)

    def test_direction_is_read_off_the_stop_not_taken_on_trust(self) -> None:
        assert _sized(stop_loss=1.0975).direction is SignalDirection.LONG
        assert _sized(stop_loss=1.1025).direction is SignalDirection.SHORT

    def test_a_stop_at_the_entry_is_rejected_rather_than_divided_by(self) -> None:
        with pytest.raises(ValueError, match="zero-distance stop"):
            position_size(1000.0, 0.005, 1.1000, 1.1000, EURUSD)

    def test_equity_scales_the_size_linearly(self) -> None:
        small = _sized(reference_equity=1_000.0)
        large = _sized(reference_equity=10_000.0)
        assert large.volume == pytest.approx(small.volume * 10)


class TestJpyPairs:
    def test_a_yen_quote_is_sized_through_the_contract_not_through_a_pip(self) -> None:
        """USDJPY 157.00 with a 0.30 stop: 30,000 JPY per lot, ~$191 at 1/157. $5 buys 0.02."""
        size = position_size(1000.0, 0.005, 157.00, 156.70, USDJPY)
        assert size is not None
        assert size.quote_to_account_rate == pytest.approx(1 / 157.00)
        assert size.volume == pytest.approx(0.02)
        assert size.risk_amount == pytest.approx(0.02 * 30_000 / 157.00)
        assert size.risk_amount <= 5.0

    def test_the_yen_pip_is_a_hundredth_and_only_labels_read_it(self) -> None:
        size = position_size(1000.0, 0.005, 157.00, 156.70, USDJPY)
        assert size is not None
        assert size.stop_pips == pytest.approx(30.0)
        assert USDJPY.pip == 0.01

    def test_a_yen_base_does_not_get_the_yen_pip(self) -> None:
        """`JPY in symbol` would call this a two-decimal pair. The quote leg decides."""
        assert SymbolSpec(symbol="JPYCHF", base="JPY", quote="CHF").pip == 0.0001

    def test_a_cross_on_a_third_currency_account_demands_a_rate(self) -> None:
        with pytest.raises(ValueError, match="quote_to_account_rate"):
            position_size(1000.0, 0.005, 0.8500, 0.8480, EURGBP, account_currency="USD")

    def test_an_explicit_rate_is_used_for_that_cross(self) -> None:
        size = position_size(1000.0, 0.005, 0.8500, 0.8480, EURGBP, quote_to_account_rate=1.27)
        assert size is not None
        assert size.risk_amount == pytest.approx(size.volume * 0.0020 * 100_000 * 1.27)


class TestTheLotStep:
    def test_volume_rounds_down_to_the_step_never_up(self) -> None:
        """0.029 lots on a 0.01 step is 0.02, and the risk actually taken falls with it."""
        size = _sized(reference_equity=1_450.0)
        assert size.volume == pytest.approx(0.02)
        assert size.risk_amount < 1_450.0 * 0.005

    def test_the_actual_fraction_is_reported_beside_the_requested_one(self) -> None:
        size = _sized(reference_equity=1_450.0)
        assert size.requested_risk_fraction == pytest.approx(0.005)
        assert size.risk_fraction < size.requested_risk_fraction
        assert size.risk_fraction == pytest.approx(size.risk_amount / 1_450.0)

    def test_rounding_never_produces_more_risk_than_requested(self) -> None:
        """Swept rather than spot-checked: one rounding-up bug anywhere is a breached cap."""
        for equity in range(500, 20_000, 137):
            size = position_size(float(equity), 0.005, 1.1000, 1.0980, EURUSD)
            if size is not None:
                assert size.risk_amount <= equity * 0.005 + 1e-9

    def test_below_the_minimum_lot_returns_none_rather_than_the_minimum(self) -> None:
        """$1 of budget over a 25-pip stop needs 0.004 lots. The floor is 0.01 — no trade."""
        assert position_size(200.0, 0.005, 1.1000, 1.0975, EURUSD) is None

    def test_the_brokers_ceiling_caps_the_volume(self) -> None:
        capped = SymbolSpec.forex("EURUSD", volume_max=1.0)
        size = position_size(10_000_000.0, 0.005, 1.1000, 1.0980, capped)
        assert size is not None
        assert size.volume == pytest.approx(1.0)


class TestTheCap:
    def test_position_size_clamps_a_request_above_the_cap(self) -> None:
        """The last line before a volume. It must be impossible to get an oversized one out."""
        over = position_size(1000.0, 0.05, 1.1000, 1.0980, EURUSD)
        at_cap = position_size(1000.0, MAX_RISK_PER_TRADE, 1.1000, 1.0980, EURUSD)
        assert over is not None and at_cap is not None
        assert over.volume == at_cap.volume
        assert over.requested_risk_fraction == pytest.approx(MAX_RISK_PER_TRADE)

    def test_the_clamp_is_logged_rather_than_swallowed(self, caplog) -> None:  # noqa: ANN001
        with caplog.at_level("WARNING", logger="fxagent.risk.sizing"):
            position_size(1000.0, 0.05, 1.1000, 1.0980, EURUSD)
        assert "exceeds the absolute cap" in caplog.text

    def test_a_zero_or_negative_fraction_is_an_error_not_a_zero_size(self) -> None:
        with pytest.raises(ValueError, match="risk_fraction must be positive"):
            position_size(1000.0, 0.0, 1.1000, 1.0980, EURUSD)

    def test_no_input_produces_more_risk_than_the_cap(self) -> None:
        """Every fraction anyone could pass, including absurd ones, lands at or under 0.5%."""
        for requested in (0.001, 0.005, 0.01, 0.5, 1.0, 100.0, math.inf):
            size = position_size(50_000.0, requested, 1.1000, 1.0980, EURUSD)
            assert size is not None
            assert size.risk_fraction <= MAX_RISK_PER_TRADE + 1e-12


class TestTheLabel:
    def test_the_size_names_the_equity_it_was_computed_against(self) -> None:
        """There is no broker here, so a size without its reference equity is unreadable."""
        assert _sized().label == "0.02 lots at 0.50% of a $1,000 account"

    def test_a_non_dollar_account_reads_in_its_own_currency(self) -> None:
        size = _sized(account_currency="CHF", quote_to_account_rate=0.9)
        assert "1,000 CHF account" in size.label

    def test_describe_carries_the_levels_the_size_came_from(self) -> None:
        described = _sized().describe()
        assert "EURUSD LONG" in described
        assert "25.0 pips" in described
        assert "$5" in described
