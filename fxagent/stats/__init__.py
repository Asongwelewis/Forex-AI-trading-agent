"""Statistics over trade sequences and return series — hand-written, numpy and pandas only.

Five modules, and the order they are meant to be read in:

| Module | Answers |
|---|---|
| `returns` | What is a return, and what is an R multiple |
| `diagnostics` | Are these observations independent (they are not) |
| `resample` | What else could this equity curve have looked like |
| `risk_metrics` | How bad is the bad tail, and how deep is the hole |
| `performance` | How large is the edge, and how sure are we |

Three conventions bind all of them.

**Log returns for arithmetic, percent only for display.** Additive and symmetric; see `returns`.

**No bare point estimates in `performance`.** Every statistic comes back with a bootstrap
interval, because a Sharpe ratio on thirty trades is noise wearing a decimal point.

**Win rate never gates.** It is computed and reported and nothing else. Expectancy in R, profit
factor and max drawdown are the metrics; below 100 trades the answer is `INSUFFICIENT_DATA`.

**Every drawdown carries its `EquityMode`.** Compounded and additive-R are different quantities,
the compounded one is shallower, and a budget reasoned without the distinction must be checked
against the additive one. `require_single_mode` refuses to let the two share a chart.

No scipy, no statsmodels. Every formula is written out in the module that owns it, for the same
reason the indicators are: a smoothing convention or a tail probability we cannot audit line by
line is one we cannot defend when a result looks too good.
"""

from __future__ import annotations

from fxagent.stats.diagnostics import (
    ArchTest,
    Autocorrelation,
    HurstResult,
    arch_test,
    autocorrelation,
    hurst_exponent,
    realised_volatility,
)
from fxagent.stats.performance import (
    INSUFFICIENT_DATA,
    MIN_TRADES_FOR_JUDGEMENT,
    Estimate,
    expectancy_r,
    judgement,
    profit_factor,
    sharpe,
    sortino,
    win_rate,
)
from fxagent.stats.resample import (
    Distribution,
    Resampler,
    ResampleResult,
    block_bootstrap,
    iid_bootstrap,
    monte_carlo,
    require_single_mode,
    suggested_block_size,
)
from fxagent.stats.returns import (
    EquityMode,
    cumulative_log_returns,
    equity_curve,
    log_returns,
    r_multiple,
    r_multiples_from_pnl,
    to_log,
    to_simple,
    total_return,
)
from fxagent.stats.risk_metrics import (
    Drawdown,
    TailRisk,
    expected_shortfall,
    max_drawdown,
    probability_of_ruin,
    value_at_risk,
)

__all__ = [
    "INSUFFICIENT_DATA",
    "MIN_TRADES_FOR_JUDGEMENT",
    "ArchTest",
    "Autocorrelation",
    "Distribution",
    "Drawdown",
    "Estimate",
    "EquityMode",
    "HurstResult",
    "ResampleResult",
    "Resampler",
    "TailRisk",
    "arch_test",
    "autocorrelation",
    "block_bootstrap",
    "cumulative_log_returns",
    "equity_curve",
    "expectancy_r",
    "expected_shortfall",
    "hurst_exponent",
    "iid_bootstrap",
    "judgement",
    "log_returns",
    "max_drawdown",
    "monte_carlo",
    "probability_of_ruin",
    "profit_factor",
    "r_multiple",
    "r_multiples_from_pnl",
    "realised_volatility",
    "require_single_mode",
    "sharpe",
    "sortino",
    "suggested_block_size",
    "to_log",
    "to_simple",
    "total_return",
    "value_at_risk",
    "win_rate",
]
