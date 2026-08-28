"""Close out past decisions against the bars that have since arrived.

Without this the journal accumulates decisions and never learns their outcomes: no expectancy,
no live-versus-backtest comparison, and no labels for anything that later learns which refusals
were right. It is bookkeeping, and it is the half of the record that turns a pile of
recommendations into a measurement.

**It charges the same costs the backtest charges, through the same module.** `fxagent.costs`
sits at the top level exactly so that comparing a paper run against a replay is comparing two
things and not three. A resolver that grew its own half-pip would make every divergence between
them unattributable — read as alpha decay, or as a broken strategy, or as anything except the
arithmetic difference it would be. `tests/test_costs_are_shared.py` asserts the sharing.

**And the same barrier rule.** `backtest.barriers.resolve_barriers` decides which of TARGET,
STOP or TIME was touched, resolving an ambiguous bar to STOP always. That is the only choice
whose error has a known sign, and using a different rule here than in the replay would make the
paper run flatter itself relative to the thing it is being checked against.

**Independent of the analysis pass.** Resolving yesterday's signals against today's bars is
bookkeeping; skipping it because this morning's analysis failed only compounds the backlog. It
does need bars to have arrived, which is the one thing it checks.

**It closes rows. It closes no positions.** Nothing here talks to a broker. A trade row in
`mode="ADVISORY"` is a paper trade, and resolving it is arithmetic over stored bars.
"""

from __future__ import annotations

from fxagent.resolve.service import ResolveConfig, ResolveStats, resolve_open_trades

__all__ = ["ResolveConfig", "ResolveStats", "resolve_open_trades"]
