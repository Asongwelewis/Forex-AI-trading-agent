"""Session clock, volatility/trend classification, and per-strategy weighting.

Decides which strategies are permitted to speak in the current regime, and combines
their votes into a consensus. All timestamps are UTC.

The layering is deliberate and one-directional: `sessions` knows only about clocks,
`classifier` measures without deciding, `router` decides permission without knowing what a
signal is, and `consensus` combines votes without knowing why they were permitted. Nothing
here reads a clock, a network or an adapter — every answer comes from the bar it was given,
which is what makes a replayed backtest agree with the live run.
"""

from __future__ import annotations

from fxagent.regime.bias import (
    BiasMode,
    BiasPolicy,
    DirectionalBias,
    apply_bias,
    carry_bias,
)
from fxagent.regime.classifier import ClassifierConfig, Regime, RegimeClassifier
from fxagent.regime.router import (
    CARRY_DIVERGENCE,
    RANGE_REVERSION,
    SESSION_BREAKOUT,
    RegimeRouter,
    RouterConfig,
)
from fxagent.regime.selection import (
    Contribution,
    SelectedSignal,
    SelectionConfig,
    SelectionResult,
    SleeveSelector,
)
from fxagent.regime.sessions import (
    LONDON_MORNING,
    SESSION_WINDOWS,
    Session,
    SessionOpening,
    SessionWindow,
    active_sessions,
    dominant_session,
    is_market_open,
    local_time,
    minutes_until_close,
    session_bounds_utc,
)

__all__ = [
    "BiasMode",
    "BiasPolicy",
    "DirectionalBias",
    "apply_bias",
    "carry_bias",
    "CARRY_DIVERGENCE",
    "LONDON_MORNING",
    "RANGE_REVERSION",
    "SESSION_BREAKOUT",
    "SESSION_WINDOWS",
    "ClassifierConfig",
    "SleeveSelector",
    "SelectionConfig",
    "SelectionResult",
    "SelectedSignal",
    "Contribution",
    "Regime",
    "RegimeClassifier",
    "RegimeRouter",
    "RouterConfig",
    "Session",
    "SessionOpening",
    "SessionWindow",
    "active_sessions",
    "dominant_session",
    "is_market_open",
    "local_time",
    "minutes_until_close",
    "session_bounds_utc",
]
