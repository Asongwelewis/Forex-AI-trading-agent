"""The router selects one sleeve; strategies do not vote.

**This replaces cross-strategy agreement, which was unsatisfiable by construction.** The old rule
was "≥2 of 3 agree AND the router permits that strategy now". The 2024–25 replay showed what that
meant in practice: over 12,341 decisions the router gave positive weight to two strategies on
**zero** bars, and the system took **zero** trades. It could not have taken one. `session_breakout`
is gated on ADX > 25 and `range_reversion` on ADX < 20 — the gates are mutually exclusive, so the
second vote could never arrive — and `carry_divergence` reads D1 and was never asked on an H1 run
at all. Even ignoring the router, over a 2,985-bar sample the two intraday sleeves signalled
together 13 times and agreed on direction 0 times. They are *designed* to disagree: one buys
strength and the other sells it.

So agreement is gone. The three strategies are independent **sleeves**, and the regime router
already answers the only question agreement was standing in for — which sleeve is valid now.
Requiring two contradictory gates to open simultaneously was not a safety margin; it was an
off switch.

**What replaces it, so that a single sleeve is not a single point of failure:**

*Confirmation moved inside each strategy.* A sleeve now requires its own evidence families to
align before it speaks — a breakout wants the break and the close location, a reversion wants
the band touch and the rejection. That is real confirmation, because the families are independent
of each other and both describe the same setup. Two strategies looking at different setups on
different timeframes never were.

*A daily directional bias filters what survives.* `regime.bias` reads the D1 picture and
suppresses or downsizes an intraday signal that opposes it. It never originates a trade.

**The diagnostics ledger is unchanged and non-negotiable.** Every strategy in the router's slate
gets a line on every bar, including the ones that were silent or gated, and the reason is
recorded on the rejection path exactly as on the firing path. That ledger is what produced this
turn's counterfactual — the finding that the system was structurally dead came out of the
rejection records, not the trade records — and losing it would lose the ability to notice the
next such failure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fxagent.adapters.base import UtcDatetime
from fxagent.regime.bias import BiasPolicy, DirectionalBias, apply_bias
from fxagent.regime.classifier import Regime
from fxagent.strategies.base import (
    POSITIONING_OFF,
    PositioningConfig,
    Signal,
    SignalDirection,
    crowding_confidence_factor,
)

__all__ = [
    "Contribution",
    "SelectedSignal",
    "SelectionConfig",
    "SelectionResult",
    "SleeveSelector",
]


@dataclass(frozen=True)
class SelectionConfig:
    """The one threshold left. There is deliberately no agreement count.

    `min_weight` is the router weight a sleeve must carry to be allowed to trade at all. It
    replaces `min_total_weight`, and it is per-sleeve rather than summed because there is nothing
    left to sum — a single sleeve trades, or none does.
    """

    min_weight: float = 0.5
    #: Minimum confidence a sleeve's own signal must carry after its internal confirmation and
    #: after the daily bias filter has had its say.
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.min_weight <= 1.0:
            raise ValueError(f"min_weight must be in (0, 1], got {self.min_weight}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")


class Contribution(BaseModel):
    """One sleeve's signal and the weight the router gave it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal: Signal
    weight: float = Field(gt=0.0, le=1.0)


class SelectedSignal(BaseModel):
    """The sleeve the router selected, with its own levels carried whole.

    `contributions` holds exactly one entry now. It stays a tuple, and `primary` and
    `strategy_names` stay on the class, because the agent briefing and the dashboard read them —
    and because a future multi-sleeve book would extend this rather than replace it. What it no
    longer holds is *several strategies that agreed*, and nothing here averages anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    direction: SignalDirection
    timestamp: UtcDatetime
    total_weight: float = Field(gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    contributions: tuple[Contribution, ...] = Field(min_length=1)

    @property
    def primary(self) -> Signal:
        """The selected sleeve's signal — the levels that will be traded."""
        best = max(self.contributions, key=lambda c: (c.weight, c.signal.confidence))
        return best.signal

    @property
    def strategy_names(self) -> tuple[str, ...]:
        return tuple(c.signal.strategy_name for c in self.contributions)


class SelectionResult(BaseModel):
    """The decision and the reasoning. `diagnostics` is populated on both paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal: SelectedSignal | None
    diagnostics: dict[str, Any]

    @property
    def fired(self) -> bool:
        return self.signal is not None


class SleeveSelector:
    """Picks the router-permitted sleeve, filters it through the daily view, and explains."""

    def __init__(
        self,
        config: SelectionConfig | None = None,
        *,
        positioning: PositioningConfig = POSITIONING_OFF,
        bias_policy: BiasPolicy | None = None,
    ) -> None:
        self._config = config or SelectionConfig()
        self._positioning = positioning
        self._bias_policy = bias_policy or BiasPolicy()

    @property
    def config(self) -> SelectionConfig:
        return self._config

    @property
    def positioning(self) -> PositioningConfig:
        return self._positioning

    @property
    def bias_policy(self) -> BiasPolicy:
        return self._bias_policy

    def select(
        self,
        regime: Regime,
        signals: Mapping[str, Signal | None],
        weights: Mapping[str, float],
        *,
        bias: DirectionalBias | None = None,
        positioning_score: float = 0.0,
    ) -> SelectionResult:
        """Choose a sleeve, or explain why none traded.

        `weights` drives the iteration, so a strategy the caller forgot to run still appears in
        the ledger as silent rather than vanishing from it.
        """
        daily = bias or DirectionalBias.none("no daily view was supplied")
        votes, candidates = self._tally(regime, signals, weights)

        diagnostics = self._diagnostics(regime, votes, candidates)
        diagnostics["positioning_score"] = positioning_score
        diagnostics["positioning_enabled"] = self._positioning.enabled
        diagnostics["bias"] = daily.as_dict()

        if not candidates:
            diagnostics["reason"] = self._explain_silence(votes)
            return SelectionResult(signal=None, diagnostics=diagnostics)

        # Highest router weight wins, confidence breaks ties. No summing and no averaging: the
        # selected sleeve's own levels are traded whole, because a blend of two strategies'
        # stops is a level neither of them argued for.
        chosen = max(candidates, key=lambda c: (c.weight, c.signal.confidence))
        others = [c for c in candidates if c is not chosen]
        if others:
            diagnostics["competing_sleeves"] = [c.signal.strategy_name for c in others]
            diagnostics["competing_directions"] = sorted(
                {str(c.signal.direction) for c in candidates}
            )

        filtered, bias_note = apply_bias(chosen.signal, daily, self._bias_policy)
        diagnostics["bias_filter"] = bias_note

        if filtered is None:
            diagnostics["reason"] = bias_note["action"]
            return SelectionResult(signal=None, diagnostics=diagnostics)

        crowding = crowding_confidence_factor(
            filtered.direction, positioning_score, config=self._positioning
        )
        confidence = filtered.confidence * crowding

        if confidence < self._config.min_confidence:
            diagnostics["reason"] = (
                f"{chosen.signal.strategy_name} cleared its gates but its confidence "
                f"{confidence:.2f} is below the {self._config.min_confidence:.2f} floor"
            )
            return SelectionResult(signal=None, diagnostics=diagnostics)

        diagnostics["fired"] = True
        diagnostics["winning_direction"] = str(filtered.direction)
        diagnostics["selected_sleeve"] = chosen.signal.strategy_name
        diagnostics["crowding_factor"] = crowding
        diagnostics["confidence_before_crowding"] = filtered.confidence
        diagnostics["reason"] = (
            f"router selected {chosen.signal.strategy_name} at weight {chosen.weight:.2f}; "
            f"{bias_note['action']}"
        )

        return SelectionResult(
            signal=SelectedSignal(
                symbol=regime.symbol,
                direction=filtered.direction,
                timestamp=regime.timestamp,
                total_weight=chosen.weight,
                confidence=confidence,
                contributions=(Contribution(signal=filtered, weight=chosen.weight),),
            ),
            diagnostics=diagnostics,
        )

    def _tally(
        self,
        regime: Regime,
        signals: Mapping[str, Signal | None],
        weights: Mapping[str, float],
    ) -> tuple[list[dict[str, Any]], list[Contribution]]:
        """One ledger line per strategy in the slate, whatever it did. Never shortened."""
        votes: list[dict[str, Any]] = []
        candidates: list[Contribution] = []

        for name in sorted(weights):
            weight = weights[name]
            signal = signals.get(name)
            vote: dict[str, Any] = {
                "strategy": name,
                "weight": weight,
                "direction": str(signal.direction) if signal is not None else None,
                "confidence": signal.confidence if signal is not None else None,
                "participated": False,
                "reason": "",
            }

            if signal is None:
                vote["reason"] = "silent: the setup is not present"
            elif signal.symbol != regime.symbol:
                raise ValueError(
                    f"{name} produced a signal for {signal.symbol!r} while the regime describes "
                    f"{regime.symbol!r}; this is a wiring error, not a disagreement"
                )
            elif weight <= 0.0:
                vote["reason"] = "gated: the router does not permit this sleeve in this regime"
            elif weight < self._config.min_weight:
                vote["reason"] = (
                    f"underweight: {weight:.2f} is below the {self._config.min_weight:.2f} floor"
                )
            elif signal.direction is SignalDirection.FLAT:
                vote["participated"] = True
                vote["reason"] = "flat: the sleeve looked and wants no exposure"
            else:
                vote["participated"] = True
                vote["reason"] = f"selectable, pointing {signal.direction}"
                candidates.append(Contribution(signal=signal, weight=weight))

            votes.append(vote)

        return votes, candidates

    def _explain_silence(self, votes: list[dict[str, Any]]) -> str:
        signalled = [v for v in votes if v["direction"] is not None]
        if not signalled:
            return "no sleeve produced a signal"
        blocked = [f"{v['strategy']} ({v['reason']})" for v in signalled]
        return f"signals were present but none was selectable: {'; '.join(blocked)}"

    def _diagnostics(
        self, regime: Regime, votes: list[dict[str, Any]], candidates: list[Contribution]
    ) -> dict[str, Any]:
        """Everything needed to answer "why" later, JSON-safe for the journal."""
        return {
            "fired": False,
            "reason": "",
            "winning_direction": None,
            "selected_sleeve": None,
            "symbol": regime.symbol,
            "timestamp": regime.timestamp.isoformat(),
            "session": str(regime.session) if regime.session is not None else None,
            "sessions": [str(s) for s in regime.sessions],
            "trend_strength": regime.trend_strength,
            "is_trending": regime.is_trending,
            "is_ranging": regime.is_ranging,
            "min_weight": self._config.min_weight,
            "min_confidence": self._config.min_confidence,
            "candidates": len(candidates),
            "votes": votes,
        }
