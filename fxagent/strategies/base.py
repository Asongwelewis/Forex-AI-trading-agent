"""The contract every strategy speaks: `Signal`, `MarketContext`, and `Strategy`.

**LONG/SHORT/FLAT is not BUY/SELL.** `SignalDirection` is what a strategy *believes*;
`OrderSide` is what a broker *executes*. They are deliberately separate enums, and the one
place they meet is `order_side_for`. Collapsing them would look like a simplification and
would quietly erase the distinction that FLAT carries — an opinion with no order behind it.
A strategy can hold a view without there being a trade to place, and `OrderSide` has no
vocabulary for that.

**Strategies are pure.** `generate` reads bars and context, and nothing else. No clock, no
network, no adapter calls. Every timestamp on a `Signal` comes from the bar that produced
it, so replaying the same bars in a backtest gives the same signal it gave live — which is
the only thing that makes a backtest worth reading.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

# `UtcDatetime` is imported, not redefined. It used to be declared here as well, character for
# character, which is two definitions of "reject naive datetimes" that would have had to be
# corrected in two places. The adapter layer owns it because that is where a timestamp first
# enters the system.
from fxagent.adapters.base import BarSeries, OrderSide, UtcDatetime

if TYPE_CHECKING:
    # Type-checking only, and deliberately so: `regime.classifier` imports this module, so a
    # runtime import here would close the loop. `from __future__ import annotations` keeps the
    # annotation a string, and no strategy needs the class itself — only attributes off an
    # instance the caller supplies.
    from fxagent.regime.classifier import Regime

__all__ = [
    "MAX_CROWDING_PENALTY",
    "POSITIONING_OFF",
    "MarketContext",
    "PositioningConfig",
    "Signal",
    "SignalDirection",
    "Strategy",
    "bars_to_frame",
    "crowding_confidence_factor",
    "order_side_for",
]


#: What a strategy may record as a diagnostic. Scalars only, so a signal stays JSON-safe
#: and the journal in Phase 8 can store `reasoning` without a custom encoder.
type ReasoningValue = float | int | str | bool | None


class SignalDirection(StrEnum):
    """What a strategy believes. Domain vocabulary — never sent to a broker."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


def order_side_for(direction: SignalDirection) -> OrderSide:
    """Translate a belief into a broker instruction. The only bridge between the two enums.

    FLAT raises rather than defaulting: "no position" is not an order, and silently
    turning it into a BUY or a SELL is exactly the class of bug the split exists to make
    impossible. Callers must decide what FLAT means to them before reaching a broker.
    """
    if direction is SignalDirection.LONG:
        return OrderSide.BUY
    if direction is SignalDirection.SHORT:
        return OrderSide.SELL
    raise ValueError(
        f"{direction} has no broker equivalent; FLAT means no order, not a default side"
    )


class MarketContext(BaseModel):
    """Inputs a strategy needs but must not go and fetch for itself.

    Everything here is injected by the caller, which is what keeps `generate` pure and
    replayable. `carry_divergence` is the only current consumer; Phase 5's macro agent
    supplies `macro_bias`, and falls back to `neutral()` whenever the LLM response fails
    validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rate_differential: float = Field(
        default=0.0,
        description="Annualised policy-rate gap in percentage points, base minus quote. "
        "Positive means holding the pair long earns carry.",
    )
    macro_bias: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Signed macro conviction: +1 fully bullish the base currency, "
        "-1 fully bearish, 0 neutral or unknown.",
    )
    positioning_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="CFTC non-commercial crowding for the pair, base leg minus quote leg. "
        "+1 means speculative longs sit at a three-year extreme. A measure of how much "
        "fuel is left, not a direction — see fxagent.fundamentals.cot.",
    )

    @classmethod
    def neutral(cls) -> MarketContext:
        """No carry, no view, no positioning — the honest default when upstream is unavailable.

        Note that neutral positioning and mid-range positioning are the same number here, and
        deliberately so: both mean "crowding is not an argument against this trade", which is
        the only thing any consumer of this field is allowed to do with it.
        """
        return cls(rate_differential=0.0, macro_bias=0.0, positioning_score=0.0)


#: The most confidence crowding may remove *once it is switched on*. A quarter is enough to
#: reorder two otherwise comparable setups and far too little to stand in for a gate, which is
#: the point: positioning is confirmation, and a number that could suppress a trade on its own
#: would be a trigger with the sign flipped.
MAX_CROWDING_PENALTY = 0.25


@dataclass(frozen=True)
class PositioningConfig:
    """Whether COT crowding may touch a confidence, and by how much. **Default off.**

    Off because an unmeasured modifier in the deterministic path corrupts the thing it would
    later be judged against. Phase 8 has not produced a baseline yet, and a first backtest run
    with crowding already applied cannot answer "what did crowding do" — there is nothing to
    difference it against, and every subsequent measurement inherits the contamination. Turning
    it on is a deliberate act that produces a measured delta against a clean baseline.

    **Disabled does not mean absent.** `positioning_score` is still computed, still travels on
    `MarketContext`, and is still written to `reasoning` and to the consensus diagnostics — it
    simply does not multiply anything. That is the whole point: the journal accumulates the
    input while the baseline stays clean, so the delta can be computed from stored data rather
    than from a second backtest run months later.

    Frozen, and passed in rather than read from a module-level global, so two strategies in one
    process can differ and a test never has to mutate shared state to get a deterministic run.
    """

    enabled: bool = False
    max_penalty: float = MAX_CROWDING_PENALTY

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_penalty < 1.0:
            raise ValueError(
                f"max_penalty must be in [0, 1), got {self.max_penalty}; a penalty of 1.0 could "
                "zero a confidence and turn confirmation into a veto"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PositioningConfig:
        """Read `FX_POSITIONING_ENABLED`. Anything but an explicit truthy value is off.

        Defaulting to off on an unrecognised value is deliberate: a typo in a deployment
        variable must not silently switch a modifier into the decision path.
        """
        source = os.environ if env is None else env
        raw = str(source.get("FX_POSITIONING_ENABLED", "")).strip().lower()
        return cls(enabled=raw in {"1", "true", "yes", "on"})


#: The default everywhere. A named constant rather than a bare `PositioningConfig()` at each
#: call site, so "which paths have crowding off" is one grep.
POSITIONING_OFF = PositioningConfig(enabled=False)


def crowding_confidence_factor(
    direction: SignalDirection,
    positioning_score: float,
    *,
    config: PositioningConfig = POSITIONING_OFF,
) -> float:
    """How much of a signal's confidence survives the crowd, in [1 - max_penalty, 1].

    Returns exactly 1.0 while `config.enabled` is false, which is the default — so every caller
    that has not opted in multiplies by one and produces bit-identical output to the code that
    existed before crowding did. The validation below still runs when disabled: an out-of-range
    score is an upstream bug either way, and a flag that also suppressed error checking would
    hide it until the day someone flipped the flag.

    Crowding is only ever an argument *against*. Buying what everyone already owns leaves fewer
    marginal buyers and a thinner exit, so a LONG into a +0.9 percentile is worth less than the
    same LONG into a neutral one. The reverse is not true in the same way — an uncrowded trade
    is normal, not an edge — so this scales down and never up, which is the same asymmetry hard
    rule 9 imposes on position sizing and for the same reason: a multiplier that can exceed 1.0
    is a multiplier that eventually gets pointed at size.

    FLAT has no exposure to be crowded into and is returned unmodified.
    """
    if not -1.0 <= positioning_score <= 1.0:
        raise ValueError(f"positioning_score must be in [-1, 1], got {positioning_score}")

    if not config.enabled:
        return 1.0

    max_penalty = config.max_penalty

    if direction is SignalDirection.LONG:
        crowding = max(0.0, positioning_score)
    elif direction is SignalDirection.SHORT:
        crowding = max(0.0, -positioning_score)
    else:
        return 1.0

    return 1.0 - max_penalty * crowding


class Signal(BaseModel):
    """One strategy's view of one symbol at one bar.

    `stop_loss` and `take_profit` are optional *here* and mandatory downstream: a FLAT
    signal has no protection because it has no position, while a LONG or SHORT without
    both is rejected at construction. The validator also pins them to the correct sides of
    entry, so a stop above entry on a long cannot reach the risk layer to be sized.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    entry_price: float = Field(gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    strategy_name: str = Field(min_length=1)
    timestamp: UtcDatetime
    reasoning: dict[str, ReasoningValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_protection(self) -> Signal:
        if self.direction is SignalDirection.FLAT:
            if self.stop_loss is not None or self.take_profit is not None:
                raise ValueError("a FLAT signal carries no position, so no stop or target")
            return self

        if self.stop_loss is None or self.take_profit is None:
            raise ValueError(
                f"a {self.direction} signal must carry both a stop_loss and a take_profit"
            )

        if self.direction is SignalDirection.LONG:
            if self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"LONG stop_loss {self.stop_loss} must be below entry {self.entry_price}"
                )
            if self.take_profit <= self.entry_price:
                raise ValueError(
                    f"LONG take_profit {self.take_profit} must be above entry {self.entry_price}"
                )
        else:
            if self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"SHORT stop_loss {self.stop_loss} must be above entry {self.entry_price}"
                )
            if self.take_profit >= self.entry_price:
                raise ValueError(
                    f"SHORT take_profit {self.take_profit} must be below entry {self.entry_price}"
                )
        return self

    @property
    def risk_distance(self) -> float:
        """Entry-to-stop distance. Zero for FLAT, which carries no risk to size."""
        if self.stop_loss is None:
            return 0.0
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_risk(self) -> float:
        """Target distance as a multiple of risk distance. Zero when there is no position."""
        if self.take_profit is None or self.risk_distance == 0.0:
            return 0.0
        return abs(self.take_profit - self.entry_price) / self.risk_distance


def bars_to_frame(bars: BarSeries) -> pd.DataFrame:
    """Lay a `BarSeries` out as the OHLCV frame the indicator layer expects.

    Indexed by bar OPEN time in UTC, oldest first — the ordering `BarSeries` already
    guarantees, so indicators inherit their no-look-ahead property unchanged.
    """
    return pd.DataFrame(
        {
            "open": [bar.open for bar in bars.bars],
            "high": [bar.high for bar in bars.bars],
            "low": [bar.low for bar in bars.bars],
            "close": [bar.close for bar in bars.bars],
            "volume": [bar.volume for bar in bars.bars],
        },
        index=pd.DatetimeIndex([bar.timestamp for bar in bars.bars], name="timestamp"),
    )


class Strategy(ABC):
    """A pure function from (bars, context, regime) to an opinion.

    `generate` returns `None` for "this setup is not present", which is distinct from a
    FLAT `Signal` meaning "I have looked and I want no exposure". The regime router
    needs both: silence does not count towards consensus, an explicit FLAT does.

    **`regime` is optional in the contract and mandatory to whoever reads it.** A strategy
    that gates on market state must not measure that state itself — `range_reversion` asking
    "is this a range" with its own ADX gave a second definition of ranging that could drift
    from the one the router gates on. Taking the classifier's answer instead means the two
    read one boolean. Strategies that gate on nothing the classifier measures simply ignore
    the argument, which is why it defaults rather than being required of all three; the ones
    that do read it raise on a missing or mismatched regime rather than going quietly silent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier, recorded on every signal and in the journal."""

    @property
    @abstractmethod
    def required_bars(self) -> int:
        """Bars needed before this strategy can produce anything at all.

        Derived from the longest indicator warm-up it depends on, so a caller can size its
        history request without knowing the internals.
        """

    @abstractmethod
    def generate(
        self, bars: BarSeries, context: MarketContext, regime: Regime | None = None
    ) -> Signal | None:
        """Produce a signal, or `None` if the setup or its gates are not satisfied."""

    def _has_enough_history(self, bars: BarSeries) -> bool:
        return len(bars) >= self.required_bars
