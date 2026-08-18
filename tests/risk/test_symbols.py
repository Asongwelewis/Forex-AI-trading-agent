"""Volume quantisation, and the contract terms sizing reads.

`test_the_step_survives_binary_floating_point` is the one that matters. `math.floor(0.29/0.01)`
is 28 on this hardware, so a naive implementation drops a third of a position on exactly one
input in twenty and is correct on all the others — which is a bug that never appears in a test
written with round numbers and always appears in production.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fxagent.risk.symbols import STANDARD_LOT, SymbolSpec, round_down_to_step


class TestRounding:
    @pytest.mark.parametrize(
        ("volume", "expected"),
        [
            (0.29, 0.29),
            (0.299, 0.29),
            (0.2999999, 0.29),
            (0.01, 0.01),
            (0.009, 0.0),
            (1.005, 1.0),
            (12.34999, 12.34),
        ],
    )
    def test_the_step_survives_binary_floating_point(self, volume: float, expected: float) -> None:
        assert round_down_to_step(volume, 0.01) == pytest.approx(expected)

    def test_float_noise_is_not_a_smaller_position(self) -> None:
        """`1.1000 - 1.0975` is 0.0025000000000000022, and a bare floor halves the trade.

        Both of these are exact multiples that binary subtraction landed a quintillionth
        under. Flooring them raw gives 0.01 and 0.0 — half a position, and no position at all.
        """
        assert round_down_to_step(0.019999999999999983, 0.01) == pytest.approx(0.02)
        assert round_down_to_step(0.00999999999999999, 0.01) == pytest.approx(0.01)

    def test_a_genuine_remainder_is_still_floored(self) -> None:
        """The settle is nine places of a step, so it cannot reach a real 0.9-of-a-step."""
        assert round_down_to_step(0.0199, 0.01) == pytest.approx(0.01)
        assert round_down_to_step(0.00999, 0.01) == 0.0

    def test_it_never_rounds_up(self) -> None:
        """Swept, because one rounding-up input anywhere is a breached cap."""
        for thousandths in range(1, 5000):
            volume = thousandths / 1000
            assert round_down_to_step(volume, 0.01) <= volume + 1e-12

    def test_a_zero_or_negative_volume_is_zero(self) -> None:
        assert round_down_to_step(0.0, 0.01) == 0.0
        assert round_down_to_step(-1.0, 0.01) == 0.0

    def test_a_non_positive_step_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="volume_step must be positive"):
            round_down_to_step(1.0, 0.0)

    def test_a_coarser_step_is_honoured(self) -> None:
        assert round_down_to_step(1.7, 0.5) == pytest.approx(1.5)


class TestUsableVolume:
    def test_below_the_minimum_lot_is_none(self) -> None:
        assert SymbolSpec.forex("EURUSD").usable_volume(0.004) is None

    def test_the_minimum_lot_itself_is_placeable(self) -> None:
        assert SymbolSpec.forex("EURUSD").usable_volume(0.01) == pytest.approx(0.01)

    def test_it_never_nudges_up_to_the_minimum(self) -> None:
        """0.009 lots is a setup that cannot be taken at this risk, not a 0.01-lot trade."""
        assert SymbolSpec.forex("EURUSD").usable_volume(0.009) is None

    def test_the_ceiling_caps_and_then_steps(self) -> None:
        spec = SymbolSpec.forex("EURUSD", volume_max=1.05, volume_step=0.1)
        assert spec.usable_volume(500.0) == pytest.approx(1.0)


class TestTheSpec:
    def test_forex_splits_the_symbol_into_its_two_legs(self) -> None:
        spec = SymbolSpec.forex("EURUSD")
        assert (spec.base, spec.quote) == ("EUR", "USD")
        assert spec.contract_size == STANDARD_LOT

    def test_the_exness_suffix_is_tolerated(self) -> None:
        assert SymbolSpec.forex("EURUSDm").quote == "USD"

    def test_money_per_lot_does_not_branch_on_the_pair(self) -> None:
        """One expression for both, which is the whole reason there is no pip in the maths."""
        assert SymbolSpec.forex("EURUSD").money_per_lot(0.0020) == pytest.approx(200.0)
        assert SymbolSpec.forex("USDJPY").money_per_lot(0.30) == pytest.approx(30_000.0)

    def test_a_pair_cannot_quote_a_currency_against_itself(self) -> None:
        with pytest.raises(ValidationError, match="against itself"):
            SymbolSpec(symbol="USDUSD", base="USD", quote="USD")

    def test_an_unplaceable_volume_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exceeds volume_max"):
            SymbolSpec.forex("EURUSD", volume_min=1.0, volume_max=0.5)

    def test_the_spec_is_frozen(self) -> None:
        spec = SymbolSpec.forex("EURUSD")
        with pytest.raises(ValidationError):
            spec.contract_size = 1.0
