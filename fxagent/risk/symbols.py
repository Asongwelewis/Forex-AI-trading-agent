"""What a broker will actually accept for one instrument, and the volume arithmetic on it.

**There is no pip in the sizing maths, and that is the point.** A pip is a display convention
that differs between EURUSD and USDJPY by a factor of a hundred, and every "handle JPY pairs"
bug in this domain is someone who wrote one pip-value formula and special-cased the yen into
it. One lot is `contract_size` units of the base currency, so a price move of `d` — in quote
currency per unit of base — is worth `d * contract_size` quote currency per lot, whatever the
pair quotes to. USDJPY moving 0.30 is 30,000 JPY per lot; EURUSD moving 0.0030 is 300 USD per
lot. Same expression, no branch. `pip` exists on this class for *labels* and nothing else.

**Volume rounds through `Decimal`.** `math.floor(0.29 / 0.01)` is 28 in binary floating point,
which silently drops a third of a position, and the same expression on a different pair is
correct — so the bug appears once in production and never in a test that happened to pick nice
numbers. Rounding is always DOWN, per hard rule 8: a volume rounded up risks more than the
caller asked for, and "slightly over the cap" is still over the cap.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Imported, not re-implemented. `split_symbol` already knows about the Exness 'm' suffix and
# the slashed feed form, and a second six-letter splitter in this package would be a second
# place for "EURUSDm" to be got wrong.
from fxagent.fundamentals.context import split_symbol

__all__ = ["SymbolSpec", "round_down_to_step"]

#: Units of base currency in one standard lot. The FX default; metals and indices differ, which
#: is why it is a field on the spec rather than a constant in the sizing formula.
STANDARD_LOT = 100_000.0


#: The step count is settled to this many places before it is floored. Nine, which is eleven
#: orders of magnitude finer than a 0.01 lot step and so cannot move a real answer, and coarse
#: enough to absorb the double-rounding in a subtraction of two prices.
_STEP_EPSILON = Decimal("1e-9")


def round_down_to_step(volume: float, step: float) -> float:
    """Largest whole multiple of `step` not exceeding `volume`. Never meaningfully rounds up.

    Exact in decimal rather than approximate in binary: the inputs are broker quantities
    written as decimal literals, so `Decimal(str(x))` reproduces what the human meant instead
    of the nearest double to it.

    **The step count is settled before it is floored, and that is not a softening of the rule.**
    `1.1000 - 1.0975` is `0.0025000000000000022` in binary, so a $5 budget over that stop comes
    out as `0.019999999999999983` lots — and flooring *that* gives 0.01, half the position, over
    an error of two parts in a quintillion. Worse, a 50-pip stop on this account sizes to
    `0.00999999999999999` lots and floors to zero, so the trade is reported unsizeable when it
    is exactly one minimum lot. Float noise is not a smaller position. The settle is nine
    decimal places of a step; a genuine 0.29-of-a-step remainder is untouched by it, and the
    most it can ever add is 1e-11 lots, which no broker can represent and no cap can notice.
    """
    if step <= 0:
        raise ValueError(f"volume_step must be positive, got {step}")
    if volume <= 0:
        return 0.0
    ratio = (Decimal(str(volume)) / Decimal(str(step))).quantize(
        _STEP_EPSILON, rounding=ROUND_HALF_EVEN
    )
    return float(ratio.to_integral_value(rounding=ROUND_FLOOR) * Decimal(str(step)))


class SymbolSpec(BaseModel):
    """The broker's contract terms for one symbol — everything sizing needs and nothing else.

    Frozen and closed, like every other contract in this codebase. The defaults are the
    standard FX ones and are a convenience for tests and for the paper path; on the execution
    path these come from `symbol_info()`, because a spec that disagrees with the broker
    produces an order the broker rejects, or worse, accepts at the wrong size.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    base: str = Field(min_length=3, max_length=3, description="Currency one lot is denominated in")
    quote: str = Field(min_length=3, max_length=3, description="Currency the price is expressed in")
    contract_size: float = Field(default=STANDARD_LOT, gt=0)
    volume_step: float = Field(default=0.01, gt=0)
    volume_min: float = Field(default=0.01, gt=0)
    volume_max: float = Field(default=500.0, gt=0)

    @model_validator(mode="after")
    def _check_spec(self) -> SymbolSpec:
        if not (self.base.isalpha() and self.base.isupper()):
            raise ValueError(f"base must be an upper-case currency code, got {self.base!r}")
        if not (self.quote.isalpha() and self.quote.isupper()):
            raise ValueError(f"quote must be an upper-case currency code, got {self.quote!r}")
        if self.base == self.quote:
            raise ValueError(f"{self.symbol} cannot quote {self.base} against itself")
        if self.volume_min > self.volume_max:
            raise ValueError(
                f"volume_min {self.volume_min} exceeds volume_max {self.volume_max}; "
                "no volume would ever be placeable"
            )
        return self

    @classmethod
    def forex(cls, symbol: str, **overrides: float) -> SymbolSpec:
        """Build a spec from the symbol alone, splitting it into its two currency legs."""
        base, quote = split_symbol(symbol)
        return cls(symbol=symbol, base=base, quote=quote, **overrides)

    @property
    def pip(self) -> float:
        """One pip in price terms. **Display only — no sizing arithmetic reads this.**

        Keyed on the *quote* currency rather than on "JPY" appearing anywhere in the symbol,
        because it is the quote leg that sets the number of decimals. A substring test calls
        JPYUSD a two-decimal pair, and the yen is the base there.
        """
        return 0.01 if self.quote == "JPY" else 0.0001

    def money_per_lot(self, price_distance: float) -> float:
        """What `price_distance` is worth on one lot, **in the quote currency**.

        The whole of the JPY question lives in this one line and does not branch on it.
        """
        return abs(price_distance) * self.contract_size

    def usable_volume(self, raw_volume: float) -> float | None:
        """`raw_volume` clamped to the broker's ceiling and floored to its step, or `None`.

        `None` means the broker cannot place this — the requested size rounds below the minimum
        lot. Returning it rather than nudging up to `volume_min` is hard rule 8 read literally:
        the minimum lot at this stop distance risks more than the caller allowed, and a system
        that quietly takes the trade anyway has no risk cap, it has a risk suggestion.
        """
        if raw_volume <= 0:
            return None
        stepped = round_down_to_step(min(raw_volume, self.volume_max), self.volume_step)
        if stepped < self.volume_min:
            return None
        return stepped
