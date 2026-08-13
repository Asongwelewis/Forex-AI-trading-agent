"""Combine weighted strategy votes into one decision, and explain every one that failed.

**The rejections are the product.** CLAUDE.md asks for every disagreement to be logged, and a
backtest cannot report "how often consensus rejected a signal" unless the rejection path
records as much as the acceptance path. So `evaluate` never returns a bare `None`: it returns
a `ConsensusResult` whose `signal` may be `None` and whose `diagnostics` is populated either
way, with a vote line for all three strategies on every bar.

Three states, deliberately distinguished — collapsing any two would destroy the diagnostic:

* **Silent** — the strategy returned no signal. Its setup is not present. It is not a vote.
* **Gated** — the router weighted it 0.0. It was never asked, so its opinion is not evidence
  of agreement or of disagreement.
* **Flat** — the strategy looked and wants no exposure. This *is* a vote, and it is recorded
  as one, but it can never supply weight to LONG or SHORT. An abstention is not a direction.

A trade needs both thresholds: enough summed weight *and* enough separate strategies. Weight
alone would let one fully-weighted strategy trade by itself, which is the single-signal
failure mode CLAUDE.md rules out; count alone would ignore the router entirely.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fxagent.regime.classifier import Regime
from fxagent.strategies.base import Signal, SignalDirection, UtcDatetime

__all__ = [
    "Consensus",
    "ConsensusConfig",
    "ConsensusResult",
    "ConsensusSignal",
    "Contribution",
]

#: Directions that can actually carry a trade. FLAT votes but never wins.
_TRADEABLE: tuple[SignalDirection, ...] = (SignalDirection.LONG, SignalDirection.SHORT)


@dataclass(frozen=True)
class ConsensusConfig:
    """The two thresholds a decision must clear."""

    #: Summed router weight of the agreeing strategies.
    min_total_weight: float = 1.0
    #: How many distinct strategies must agree on the direction.
    min_agreeing: int = 2

    def __post_init__(self) -> None:
        if self.min_total_weight <= 0:
            raise ValueError(f"min_total_weight must be positive, got {self.min_total_weight}")
        if self.min_agreeing < 2:
            raise ValueError(
                f"min_agreeing must be at least 2, got {self.min_agreeing}; a threshold of 1 "
                "would let a single strategy trade alone, which is the thing consensus exists "
                "to prevent"
            )


class Contribution(BaseModel):
    """One strategy's signal and the weight the router gave it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal: Signal
    weight: float = Field(gt=0.0, le=1.0)


class ConsensusSignal(BaseModel):
    """An agreed direction, with the votes that produced it kept attached.

    No price here is invented. `entry_price`, `stop_loss` and `take_profit` are taken whole
    from one contributing signal rather than blended across several, because averaging three
    strategies' stops would produce a level that none of them argued for and that no backtest
    could attribute. `primary` names which one won.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    direction: SignalDirection
    timestamp: UtcDatetime
    total_weight: float = Field(gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    contributions: tuple[Contribution, ...] = Field(min_length=2)

    @property
    def primary(self) -> Signal:
        """The heaviest contribution, breaking ties on confidence — the levels we would use."""
        best = max(self.contributions, key=lambda c: (c.weight, c.signal.confidence))
        return best.signal

    @property
    def strategy_names(self) -> tuple[str, ...]:
        return tuple(c.signal.strategy_name for c in self.contributions)


class ConsensusResult(BaseModel):
    """The decision and the reasoning, together. `diagnostics` is populated on both paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal: ConsensusSignal | None
    diagnostics: dict[str, Any]

    @property
    def fired(self) -> bool:
        return self.signal is not None


class Consensus:
    """Applies the agreement rules to a slate of votes. Pure; reads no clock and no bars."""

    def __init__(self, config: ConsensusConfig | None = None) -> None:
        self._config = config or ConsensusConfig()

    @property
    def config(self) -> ConsensusConfig:
        return self._config

    def evaluate(
        self,
        regime: Regime,
        signals: Mapping[str, Signal | None],
        weights: Mapping[str, float],
    ) -> ConsensusResult:
        """Decide, and explain.

        `weights` is the router's full slate and drives the iteration, so a strategy the
        caller forgot to run still appears in the diagnostics as silent rather than vanishing.
        """
        votes, contributions = self._tally(regime, signals, weights)
        totals = {direction: self._totals(contributions, direction) for direction in _TRADEABLE}
        qualifying = [
            direction
            for direction, (weight, count) in totals.items()
            if weight >= self._config.min_total_weight and count >= self._config.min_agreeing
        ]

        diagnostics = self._diagnostics(regime, votes, totals)

        if not qualifying:
            diagnostics["reason"] = self._explain_failure(totals)
            return ConsensusResult(signal=None, diagnostics=diagnostics)

        if len(qualifying) > 1:
            # Both sides cleared the bar. That is a contradiction, not a close call.
            diagnostics["reason"] = (
                "both LONG and SHORT met the thresholds; refusing to pick a side from a "
                "self-contradicting slate"
            )
            return ConsensusResult(signal=None, diagnostics=diagnostics)

        direction = qualifying[0]
        winners = tuple(c for c in contributions if c.signal.direction is direction)
        total_weight, _ = totals[direction]

        diagnostics["fired"] = True
        diagnostics["winning_direction"] = str(direction)
        diagnostics["reason"] = (
            f"{len(winners)} strategies agreed on {direction} with summed weight {total_weight:.2f}"
        )
        return ConsensusResult(
            signal=ConsensusSignal(
                symbol=regime.symbol,
                direction=direction,
                timestamp=regime.timestamp,
                total_weight=total_weight,
                confidence=_weighted_confidence(winners),
                contributions=winners,
            ),
            diagnostics=diagnostics,
        )

    def _tally(
        self,
        regime: Regime,
        signals: Mapping[str, Signal | None],
        weights: Mapping[str, float],
    ) -> tuple[list[dict[str, Any]], list[Contribution]]:
        votes: list[dict[str, Any]] = []
        contributions: list[Contribution] = []

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
                vote["reason"] = "gated: the router does not permit this strategy in this regime"
            elif signal.direction is SignalDirection.FLAT:
                vote["participated"] = True
                vote["reason"] = "flat: voted, but an abstention carries no direction"
            else:
                vote["participated"] = True
                vote["reason"] = f"counted toward {signal.direction}"
                contributions.append(Contribution(signal=signal, weight=weight))

            votes.append(vote)

        return votes, contributions

    def _totals(
        self, contributions: list[Contribution], direction: SignalDirection
    ) -> tuple[float, int]:
        agreeing = [c for c in contributions if c.signal.direction is direction]
        return sum(c.weight for c in agreeing), len(agreeing)

    def _explain_failure(self, totals: Mapping[SignalDirection, tuple[float, int]]) -> str:
        best = max(totals.items(), key=lambda item: (item[1][0], item[1][1]))
        direction, (weight, count) = best
        if count == 0:
            return "no strategy offered a tradeable direction"
        return (
            f"best was {direction} with {count} agreeing and summed weight {weight:.2f}; "
            f"needs {self._config.min_agreeing} agreeing and {self._config.min_total_weight:.2f}"
        )

    def _diagnostics(
        self,
        regime: Regime,
        votes: list[dict[str, Any]],
        totals: Mapping[SignalDirection, tuple[float, int]],
    ) -> dict[str, Any]:
        """Everything needed to answer "why" later, JSON-safe for the journal."""
        return {
            "fired": False,
            "reason": "",
            "winning_direction": None,
            "symbol": regime.symbol,
            "timestamp": regime.timestamp.isoformat(),
            "session": str(regime.session) if regime.session is not None else None,
            "sessions": [str(s) for s in regime.sessions],
            "trend_strength": regime.trend_strength,
            "is_trending": regime.is_trending,
            "is_ranging": regime.is_ranging,
            "min_total_weight": self._config.min_total_weight,
            "min_agreeing": self._config.min_agreeing,
            "long_weight": totals[SignalDirection.LONG][0],
            "long_votes": totals[SignalDirection.LONG][1],
            "short_weight": totals[SignalDirection.SHORT][0],
            "short_votes": totals[SignalDirection.SHORT][1],
            "votes": votes,
        }


def _weighted_confidence(contributions: tuple[Contribution, ...]) -> float:
    """Router-weighted mean of the members' own confidences."""
    total = sum(c.weight for c in contributions)
    if total <= 0.0:  # unreachable: contributions only hold positive weights
        return 0.0
    blended = sum(c.weight * c.signal.confidence for c in contributions) / total
    return min(1.0, max(0.0, blended))
