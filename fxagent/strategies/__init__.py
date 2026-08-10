"""The three uncorrelated strategies and the contract they share.

Each is a pure function of (bars, context). None of them reads a clock, opens a socket, or
touches an adapter — the regime router in Phase 5 decides which of them is allowed to speak
today, and the risk layer in Phase 6 decides how large their signals may be. A strategy's
only job is to have a defensible opinion about the bars in front of it.

They are meant to disagree. `session_breakout` buys strength, `range_reversion` sells it,
and `carry_divergence` ignores both in favour of the rate differential. Consensus between
two of the three means something precisely because agreement is not the default.
"""

from __future__ import annotations

from fxagent.strategies.base import (
    MarketContext,
    Signal,
    SignalDirection,
    Strategy,
    bars_to_frame,
    order_side_for,
)
from fxagent.strategies.carry_divergence import CarryDivergence
from fxagent.strategies.range_reversion import RangeReversion
from fxagent.strategies.session_breakout import SessionBreakout

__all__ = [
    "CarryDivergence",
    "MarketContext",
    "RangeReversion",
    "SessionBreakout",
    "Signal",
    "SignalDirection",
    "Strategy",
    "bars_to_frame",
    "order_side_for",
]
