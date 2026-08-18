"""Log returns, cumulative returns, and the R-multiple conversion.

**Log returns everywhere in this layer, never percent returns.** Two properties, and both of
them are the reason:

*Additive.* `log(p2/p0) == log(p1/p0) + log(p2/p1)` exactly. Summing simple returns across
periods is wrong, so a cumulative percent return has to be built by multiplying `(1 + r)` — and
the moment anything averages, regresses, or bootstraps those numbers instead, it has silently
committed the error. Every statistic in this package is a sum or a mean of something, so the
inputs are the additive form.

*Symmetric.* A move up and the same move back down cancel to zero in logs. In percent they do
not: +10% then -10% leaves you at 0.99, and the asymmetry is a real feature of compounding that
becomes a fabricated drift the moment a mean is taken over it. A series of simple returns with
mean zero *loses money*; a series of log returns with mean zero does not.

Percent returns are the right thing to *report* to a human, which is what `to_simple` is for.
They are the wrong thing to compute with, and the conversion happens at the last possible step.

**The R multiple carries its own direction.** `(exit - entry) / (entry - stop)` is signed
correctly for a long and for a short without being told which it is, because the denominator
flips sign with the stop. There is deliberately no `direction` argument to disagree with the
prices — a caller that passes LONG alongside a stop above the entry is a caller with a bug, and
the arithmetic here refuses to have an opinion it could get wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np
import pandas as pd

from fxagent.stats._common import as_array

__all__ = [
    "EquityMode",
    "cumulative_log_returns",
    "equity_curve",
    "log_returns",
    "r_multiple",
    "r_multiples_from_pnl",
    "to_log",
    "to_simple",
    "total_return",
]


def log_returns(prices: pd.Series) -> pd.Series:
    """`log(p[i] / p[i-1])`, NaN at index 0 where there is no previous price.

    NaN rather than 0.0 at the first observation, matching the indicator layer: a zero return
    is a bar that did not move, and a bar with no predecessor is a different fact. Padding one
    into the other adds a spurious observation to every mean and every variance downstream.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices).__name__}")
    values = prices.to_numpy(dtype="float64", copy=False)
    if np.any(values <= 0):
        raise ValueError("prices must be positive; the log of a non-positive price is undefined")

    result = np.full(values.size, np.nan, dtype="float64")
    result[1:] = np.log(values[1:] / values[:-1])
    return pd.Series(result, index=prices.index, name="log_return")


def cumulative_log_returns(returns: pd.Series | Sequence[float] | np.ndarray) -> np.ndarray:
    """Running total of log returns — a plain `cumsum`, because logs are additive.

    That this is a sum and not a running product is the whole argument for the log form. The
    equivalent on simple returns is `cumprod(1 + r) - 1`, and every place that is written as a
    sum by mistake is a compounding error nobody sees until the equity curve is drawn.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy(dtype="float64", copy=False)
    return np.cumsum(as_array(returns, "returns"))


def total_return(returns: pd.Series | Sequence[float] | np.ndarray) -> float:
    """The whole period's return, expressed as a simple fraction for a human to read.

    Computed as `expm1(sum(log_returns))`, so the arithmetic stays additive and the conversion
    to percent happens once, at the end, where it cannot contaminate anything.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy(dtype="float64", copy=False)
    return float(np.expm1(np.sum(as_array(returns, "returns"))))


def to_simple(log_return: float | np.ndarray) -> float | np.ndarray:
    """Log return to simple return: `expm1(x)`. For display, never for further arithmetic."""
    result = np.expm1(log_return)
    return float(result) if np.isscalar(log_return) else result


def to_log(simple_return: float | np.ndarray) -> float | np.ndarray:
    """Simple return to log return: `log1p(x)`. Rejects a total loss, whose log is -inf."""
    array = np.asarray(simple_return, dtype="float64")
    if np.any(array <= -1.0):
        raise ValueError("a simple return of -100% or worse has no finite log equivalent")
    result = np.log1p(array)
    return float(result) if np.isscalar(simple_return) else result


def r_multiple(entry_price: float, exit_price: float, stop_loss: float) -> float:
    """Result in risk units: `(exit - entry) / (entry - stop)`.

    One expression for both directions. A long stopped out returns exactly -1; a short stopped
    out returns exactly -1; a long that ran twice its stop distance returns +2. The direction
    lives in the sign of `entry - stop` and never has to be passed in, so it cannot be passed
    in wrongly.

    R is the unit the whole system is judged in, because it is the only one that is comparable
    across pairs and across account sizes. Two dollars made on EURUSD and two made on USDJPY
    are not the same trade; +2R and +2R are.
    """
    risk = entry_price - stop_loss
    if risk == 0.0:
        raise ValueError(f"stop_loss {stop_loss} equals entry {entry_price}; R is undefined")
    return (exit_price - entry_price) / risk


def r_multiples_from_pnl(
    pnl: Sequence[float] | np.ndarray, risk_amount: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Realised P&L divided by the amount that was actually at risk on each trade.

    `risk_amount` is per-trade rather than a single figure, deliberately. Position sizing
    rounds down to the lot step, so the money risked differs from trade to trade even at a
    constant risk fraction, and dividing a varying P&L by one nominal risk figure produces R
    multiples that are wrong by exactly the rounding.
    """
    profits = as_array(pnl, "pnl")
    risks = as_array(risk_amount, "risk_amount")
    if profits.size != risks.size:
        raise ValueError(f"pnl has {profits.size} entries but risk_amount has {risks.size}")
    if np.any(risks <= 0):
        raise ValueError("every risk_amount must be positive; a trade risking nothing has no R")
    return profits / risks


class EquityMode(StrEnum):
    """How a risk unit is sized as the account moves. **The two give different drawdowns.**

    `COMPOUNDED` risks a fraction of *current* equity, which is what `fxagent.risk` actually
    does: a losing run shrinks the next position, so the run costs less than the sum of its R
    multiples and the curve bends away from zero. `ADDITIVE` risks a fraction of the *starting*
    equity throughout, so the curve is the cumulative sum of R at a constant unit and a losing
    run costs exactly what it says on the tin.

    **The two disagree, and neither is universally the conservative one.** Two effects pull in
    opposite directions. Shrinking size cushions a *sustained losing run*, so a monotone
    drawdown is shallower compounded — sixteen straight losses at 1% is 14.85% compounded and
    16.0% additive. But compounding also suffers volatility drag, so a *choppy* drawdown of the
    same net R is usually deeper compounded. On random 40-trade records at a 35% hit rate the
    compounded drawdown is the deeper one about 58% of the time. Anyone who reasons "compounding
    is gentler" and stops there has it wrong more often than right.

    **Which is why a drawdown budget has to name its mode.** Card 22's
    `MAX_ACCEPTABLE_DRAWDOWN = 0.15` was reasoned about before this distinction was drawn, so it
    is a number without a mode attached and cannot be checked without choosing one. The sustained
    losing run is the shape a ruin budget exists to guard against, and additive is the deeper
    measure of exactly that shape — so `ADDITIVE` is the defensible mode for a drawdown gate,
    and the reason is that specific worst case rather than any general conservatism.

    **Never plot both on one chart.** Two equity curves from the same trades, one visibly
    shallower, is a picture that says the strategy did two different things. `ResampleResult`
    carries its mode and `require_single_mode` refuses a mixed set for exactly this reason.
    """

    COMPOUNDED = "COMPOUNDED"
    ADDITIVE = "ADDITIVE"

    @property
    def label(self) -> str:
        """Short form for a legend, an axis title or a report line."""
        return "compounded" if self is EquityMode.COMPOUNDED else "additive-R"


def equity_curve(
    r_multiples: Sequence[float] | np.ndarray,
    *,
    risk_fraction: float,
    starting_equity: float = 1.0,
    mode: EquityMode = EquityMode.COMPOUNDED,
) -> np.ndarray:
    """Equity after each trade, sized either off current equity or off the starting equity.

        COMPOUNDED:  equity[i] = equity[i-1] * (1 + risk_fraction * R[i])
        ADDITIVE:    equity[i] = equity[i-1] + risk_fraction * starting_equity * R[i]

    Length `n + 1`, so index 0 is the account before the first trade — a drawdown measured
    against a curve that starts at the first *result* understates the first losing trade.

    The default is `COMPOUNDED` because it models what the risk layer does. It is **not** the
    mode to check a drawdown budget against; see `EquityMode`.

    Equity is floored at zero in both modes, and zero is absorbing. An account cannot go
    negative — the broker closes it — and a curve that passes through zero and recovers is
    describing trades that could not have been placed.
    """
    values = as_array(r_multiples, "r_multiples")
    if not 0.0 < risk_fraction <= 1.0:
        raise ValueError(f"risk_fraction must be in (0, 1], got {risk_fraction}")
    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be positive, got {starting_equity}")

    curve = np.empty(values.size + 1, dtype="float64")
    curve[0] = starting_equity

    if mode is EquityMode.ADDITIVE:
        unit = risk_fraction * starting_equity
        np.cumsum(values * unit, out=curve[1:])
        curve[1:] += starting_equity
        if np.all(curve > 0):
            return curve
        # Ruined somewhere. Everything from the first non-positive point on is zero, because a
        # closed account does not go on to win the next trade back.
        ruined_at = int(np.argmax(curve <= 0))
        curve[ruined_at:] = 0.0
        return curve

    factors = 1.0 + risk_fraction * values
    if np.all(factors > 0):
        # The ordinary case, and worth the branch: at a 0.5% risk fraction a factor could only
        # go non-positive on a -200R trade, so this path runs for every real backtest and for
        # every one of ten thousand resampled ones. A cumulative product is the same
        # arithmetic as the loop below, without ten thousand Python iterations per path.
        np.cumprod(factors, out=curve[1:])
        curve[1:] *= starting_equity
        return curve

    equity = starting_equity
    for index, factor in enumerate(factors, start=1):
        equity = max(0.0, equity * factor)
        curve[index] = equity
    return curve
