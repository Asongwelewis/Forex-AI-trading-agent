"""Which strategies are allowed to speak in the current regime, and how loudly.

The router is the only place that knows a regime implies a permission. Strategies do not
consult it and it does not consult them — it reads a `Regime` and returns a weight per
strategy name, so a strategy can be silenced without editing the strategy.

A weight of 0.0 is a gate, not a discount: `consensus` refuses to count a zero-weighted vote
at all. That distinction matters for the diagnostics, because "wrong regime, never asked" and
"asked, and it declined" are different facts and the journal needs to tell them apart.

**Every tunable is in `RouterConfig`.** The rule *shapes* are code — a dataclass cannot hold
control flow — but every threshold, session reference and weight is a field, so tuning never
means editing logic. CLAUDE.md is explicit that the rules are what get tuned, not the
parameters; keeping the parameters in one visible place is what makes it obvious when
somebody has been tuning the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass

from fxagent.regime.classifier import Regime
from fxagent.regime.sessions import SESSION_WINDOWS, Session, local_time

__all__ = [
    "CARRY_DIVERGENCE",
    "RANGE_REVERSION",
    "SESSION_BREAKOUT",
    "RouterConfig",
    "RegimeRouter",
]

#: Must match each strategy's `name` property; the weights dict is keyed on it.
SESSION_BREAKOUT = "session_breakout"
RANGE_REVERSION = "range_reversion"
CARRY_DIVERGENCE = "carry_divergence"


@dataclass(frozen=True)
class RouterConfig:
    """Every number and session reference the routing rules read."""

    #: session_breakout trades the open of this session, in this session's own local time.
    breakout_session: Session = Session.LONDON
    #: Exclusive. 12 means the last permitted bar opens at 11:00 local.
    breakout_before_local_hour: int = 12
    breakout_weight: float = 1.0

    #: range_reversion needs a quiet market, so the busiest hours of the day disqualify it.
    reversion_blocked_by: Session = Session.OVERLAP
    reversion_weight: float = 1.0

    #: carry_divergence is a multi-day view, so it is always partially on and never fully.
    carry_weight: float = 0.5

    #: A shut market permits nothing. Set False only to inspect rules outside trading hours.
    require_market_open: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.breakout_before_local_hour <= 24:
            raise ValueError(
                f"breakout_before_local_hour must be an hour of the day, got "
                f"{self.breakout_before_local_hour}"
            )
        for name, weight in (
            ("breakout_weight", self.breakout_weight),
            ("reversion_weight", self.reversion_weight),
            ("carry_weight", self.carry_weight),
        ):
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"{name} must sit in [0, 1], got {weight}")
        if self.breakout_session is Session.OVERLAP:
            raise ValueError(
                "OVERLAP has no business hours of its own, so it cannot anchor a local-time "
                "rule; name the session whose clock the rule means"
            )


class RegimeRouter:
    """Maps a `Regime` to a weight per strategy. Pure, and total: every strategy is keyed."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self._config = config or RouterConfig()

    @property
    def config(self) -> RouterConfig:
        return self._config

    def weights(self, regime: Regime) -> dict[str, float]:
        """A weight in [0, 1] for every strategy, whether or not it is permitted.

        Always returns all three keys. A caller iterating the dict sees the full slate and
        the zeros, which is what lets the rejection count in the journal be complete.
        """
        if self._config.require_market_open and not regime.market_open:
            return dict.fromkeys((SESSION_BREAKOUT, RANGE_REVERSION, CARRY_DIVERGENCE), 0.0)

        return {
            SESSION_BREAKOUT: self._breakout_weight(regime),
            RANGE_REVERSION: self._reversion_weight(regime),
            CARRY_DIVERGENCE: self._config.carry_weight,
        }

    def _breakout_weight(self, regime: Regime) -> float:
        """The session's opening hours, in that session's own local time, and only in a trend."""
        config = self._config
        if not regime.in_session(config.breakout_session):
            return 0.0
        if not regime.is_trending:
            return 0.0

        zone = SESSION_WINDOWS[config.breakout_session].zone
        if local_time(regime.timestamp, zone).hour >= config.breakout_before_local_hour:
            return 0.0
        return config.breakout_weight

    def _reversion_weight(self, regime: Regime) -> float:
        config = self._config
        if not regime.is_ranging:
            return 0.0
        if regime.in_session(config.reversion_blocked_by):
            return 0.0
        return config.reversion_weight
