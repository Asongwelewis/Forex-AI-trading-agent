"""The daily directional bias, and what an intraday signal that opposes it is worth.

**This is a filter, not a sleeve.** It never originates a trade. It reads D1 evidence — the rate
differential, the daily EMA slope, the injected macro view — and produces a direction that the
intraday sleeves are then measured against. A signal agreeing with it passes untouched; a signal
opposing it is suppressed or downsized, depending on policy.

The reason for the demotion is the 2024–25 replay. `carry_divergence` was a voter, and consensus
required two strategies to agree; but carry reads D1 and the intraday sleeves read H1, so it was
never even asked on an H1 run, and the two intraday sleeves are gated on mutually exclusive
regimes. Zero trades in 12,341 decisions. A multi-day carry view and an hourly breakout are not
two opinions about the same question — one is the weather and the other is the hour — and asking
them to agree was a category error dressed as prudence.

**Opposition is a spectrum, and the policy is a choice.** `SUPPRESS` refuses the trade outright;
`DOWNSIZE` lets it through with its confidence cut. Downsizing is the default because a daily
carry view is weak evidence about the next four hours — strong enough to size against, not
strong enough to veto — and because a veto keyed on a slow signal stands the system down for
weeks at a time, which is the failure mode that produced the zero-trade run in the first place.

**Neutral is not agreement.** A bias of `NONE` — no differential, no slope, warming up — leaves
every signal untouched. Absence of a view must never read as a view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from fxagent.adapters.base import BarSeries
from fxagent.indicators import ema
from fxagent.strategies.base import MarketContext, Signal, SignalDirection, bars_to_frame

__all__ = [
    "CARRY_TIMEFRAME",
    "BiasMode",
    "BiasPolicy",
    "DirectionalBias",
    "apply_bias",
    "carry_bias",
]

#: The only timeframe this filter reads. An H1 series produces `DirectionalBias.none()`, never
#: an exception — the intraday path must be able to ask for a bias it may not have.
CARRY_TIMEFRAME: Final = "D1"

EMA_PERIOD: Final = 50
SLOPE_LOOKBACK: Final = 1

#: Below this the differential is noise rather than carry. Matches `carry_divergence`.
MIN_DIFFERENTIAL: Final = 0.25


class BiasMode(StrEnum):
    """What happens to a signal that opposes the daily view."""

    SUPPRESS = "SUPPRESS"
    DOWNSIZE = "DOWNSIZE"


@dataclass(frozen=True)
class BiasPolicy:
    """How hard the daily view pushes back on an intraday signal.

    `DOWNSIZE` by default. A daily carry view is genuinely weak evidence about the next few
    hours: it is worth sizing against and not worth vetoing on, and a veto on a slow signal
    stands the system down for weeks — which is exactly how the previous design reached zero
    trades in two years.
    """

    mode: BiasMode = BiasMode.DOWNSIZE
    #: Confidence multiplier applied to an opposing signal under `DOWNSIZE`.
    opposed_factor: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.opposed_factor <= 1.0:
            raise ValueError(
                f"opposed_factor must be in (0, 1], got {self.opposed_factor}. Zero would make "
                "DOWNSIZE a silent SUPPRESS, and the two must stay distinguishable in the log."
            )


@dataclass(frozen=True)
class DirectionalBias:
    """A daily view, its strength, and the evidence behind it.

    `direction` is `None` for "no view", which is distinct from a view that happens to be weak.
    `reason` is carried so the journal records why the filter did what it did on a bar where
    nothing traded — the diagnostics are the product as much as the trades are.
    """

    direction: SignalDirection | None
    strength: float
    reason: str
    rate_differential: float = 0.0
    ema_slope: float | None = None
    macro_bias: float = 0.0

    @classmethod
    def none(cls, reason: str) -> DirectionalBias:
        return cls(direction=None, strength=0.0, reason=reason)

    @property
    def has_view(self) -> bool:
        return self.direction is not None

    def opposes(self, direction: SignalDirection) -> bool:
        """Whether this view argues against `direction`. No view opposes nothing."""
        if self.direction is None or direction is SignalDirection.FLAT:
            return False
        return self.direction is not direction

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe, for the diagnostics ledger."""
        return {
            "direction": str(self.direction) if self.direction is not None else None,
            "strength": self.strength,
            "reason": self.reason,
            "rate_differential": self.rate_differential,
            "ema_slope": self.ema_slope,
            "macro_bias": self.macro_bias,
        }


def carry_bias(bars: BarSeries, context: MarketContext) -> DirectionalBias:
    """The daily view from carry, daily trend and the injected macro brief.

    **Never raises on the wrong timeframe.** An H1 series returns `none()` with a reason, because
    the intraday path asks every bar and a filter that threw on the common case would have to be
    wrapped in a try — which is how a filter becomes silently absent.

    Two evidence families must align for a view to exist: the rate differential says which side
    pays, and the daily EMA slope says whether price is fighting it. Carry trades die in
    drawdowns, not in flat markets, so a differential pointing one way against a daily trend
    pointing the other is not a weak view — it is no view.
    """
    if bars.timeframe != CARRY_TIMEFRAME:
        return DirectionalBias.none(
            f"no daily view: this is a {bars.timeframe} series and carry reads {CARRY_TIMEFRAME}"
        )
    if len(bars) < EMA_PERIOD + SLOPE_LOOKBACK:
        return DirectionalBias.none(f"no daily view: fewer than {EMA_PERIOD + SLOPE_LOOKBACK} bars")

    differential = context.rate_differential
    if abs(differential) < MIN_DIFFERENTIAL:
        return DirectionalBias.none(
            f"no daily view: rate differential {differential:+.2f} is inside the "
            f"{MIN_DIFFERENTIAL} noise band"
        )

    carry_side = SignalDirection.LONG if differential > 0 else SignalDirection.SHORT

    trend = ema(bars_to_frame(bars)["close"], EMA_PERIOD)
    slope = float(trend.iloc[-1] - trend.iloc[-1 - SLOPE_LOOKBACK])
    if math.isnan(slope):
        return DirectionalBias.none("no daily view: the daily EMA has not warmed up")

    trend_side = SignalDirection.LONG if slope > 0 else SignalDirection.SHORT
    if trend_side is not carry_side:
        return DirectionalBias.none(
            f"no daily view: carry points {carry_side} but the daily trend points {trend_side}; "
            "two families disagreeing is no view, not a weak one"
        )

    strength = min(1.0, abs(differential) / 2.0) * min(1.0, 0.5 + 0.5 * abs(context.macro_bias))
    return DirectionalBias(
        direction=carry_side,
        strength=strength,
        reason=(
            f"daily view {carry_side}: differential {differential:+.2f} with the daily EMA "
            f"sloping {'up' if slope > 0 else 'down'}"
        ),
        rate_differential=differential,
        ema_slope=slope,
        macro_bias=context.macro_bias,
    )


def apply_bias(
    signal: Signal, bias: DirectionalBias, policy: BiasPolicy | None = None
) -> tuple[Signal | None, dict[str, Any]]:
    """Filter an intraday signal through the daily view. Returns the signal and what happened.

    `None` means suppressed. The diagnostics are returned either way and on the untouched path
    too, so a bar where the filter did nothing still records that it looked.
    """
    active = policy or BiasPolicy()
    note: dict[str, Any] = {
        "bias": bias.as_dict(),
        "policy": str(active.mode),
        "opposed": False,
        "action": "none",
        "confidence_before": signal.confidence,
        "confidence_after": signal.confidence,
    }

    if not bias.opposes(signal.direction):
        note["action"] = "not opposed"
        return signal, note

    note["opposed"] = True
    if active.mode is BiasMode.SUPPRESS:
        note["action"] = f"suppressed: {signal.direction} opposes a {bias.direction} daily view"
        note["confidence_after"] = 0.0
        return None, note

    reduced = signal.confidence * active.opposed_factor
    note["action"] = (
        f"downsized {signal.confidence:.2f} -> {reduced:.2f}: {signal.direction} opposes a "
        f"{bias.direction} daily view"
    )
    note["confidence_after"] = reduced
    return signal.model_copy(update={"confidence": reduced}), note
