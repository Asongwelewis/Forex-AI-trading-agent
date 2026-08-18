"""Purged walk-forward backtest harness. Replays stored bars through the live pipeline.

Four parts, and the invariant each exists to hold:

| Module | Holds |
|---|---|
| `replay` | The pipeline is imported, never reimplemented, and never sees a future bar |
| `barriers` | Intrabar ambiguity resolves to STOP, and the rate is reported |
| `folds` | Training labels overlapping a test fold are purged, and the count is printed |
| `report` | Every metric carries its interval; both equity modes are labelled |

The cost model is **not** here. It lives at `fxagent.costs`, at the top level, because card 22's
paper resolver has to charge exactly the same spread, slippage and swap — and a module inside
`backtest/` is one the execution path would eventually copy rather than import. See that
module's docstring.

Run it with `uv run python -m fxagent.backtest --symbol EUR/USD --from ... --to ...`.
"""

from __future__ import annotations

from fxagent.backtest.barriers import Barrier, BarrierOutcome, resolve_barriers
from fxagent.backtest.folds import (
    DEFAULT_EMBARGO_FRACTION,
    Fold,
    assert_purging_is_working,
    purged_walk_forward,
)
from fxagent.backtest.replay import (
    DEFAULT_SOURCE,
    ReplayConfig,
    ReplayResult,
    ReplayTrade,
    default_strategies,
    replay,
)
from fxagent.backtest.report import BacktestReport, build_report

__all__ = [
    "DEFAULT_EMBARGO_FRACTION",
    "DEFAULT_SOURCE",
    "BacktestReport",
    "Barrier",
    "BarrierOutcome",
    "Fold",
    "ReplayConfig",
    "ReplayResult",
    "ReplayTrade",
    "assert_purging_is_working",
    "build_report",
    "default_strategies",
    "purged_walk_forward",
    "replay",
    "resolve_barriers",
]
