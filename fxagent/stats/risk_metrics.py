"""Tail risk, drawdown, and a ruin estimate that is honest about what it is not.

**VaR cannot leave this module alone.** `value_at_risk` returns a `TailRisk`, which carries the
expected shortfall beside it, because value at risk is not a coherent risk measure and reporting
it by itself is actively misleading. Two specific failures:

*It says nothing about depth.* "5% of the time you lose at least 3R" is compatible with losing
exactly 3R and with losing 40R. VaR is the threshold; the entire question of what happens past
it is the part it does not answer, and the part that closes accounts.

*It is not subadditive.* Two books can each have a VaR of 1R and combine into a book with a VaR
of 5R, which means diversification can appear to *increase* measured risk. Expected shortfall
is subadditive and does not do this. Making VaR structurally unable to travel without ES is
cheaper than trusting every future caller to remember why.

**Drawdown is measured on an equity curve, not on a return series.** A drawdown is a path
property — it depends on the order the returns arrived in — so a function handed an unordered
sample cannot compute one. `max_drawdown` takes the curve.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from fxagent.stats._common import as_array

__all__ = [
    "Drawdown",
    "TailRisk",
    "expected_shortfall",
    "max_drawdown",
    "probability_of_ruin",
    "value_at_risk",
]


class TailRisk(NamedTuple):
    """A loss threshold and the average loss beyond it. Both positive means a loss.

    A negative `value_at_risk` is not a bug: it means the alpha-quantile of the sample is
    itself a gain, which happens on a short run of good results and is worth seeing rather
    than clamping to zero.
    """

    value_at_risk: float
    expected_shortfall: float
    alpha: float

    def describe(self) -> str:
        return (
            f"VaR({self.alpha:.0%}) {self.value_at_risk:.3f}, "
            f"ES {self.expected_shortfall:.3f} — the ES is the number that matters"
        )


def _tail_cutoff(returns: np.ndarray, alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    needed = int(np.ceil(1.0 / alpha))
    if returns.size < needed:
        raise ValueError(
            f"a {alpha:.0%} tail needs at least {needed} observations to contain one, and "
            f"{returns.size} were given. An empirical tail estimated from zero observations "
            "is not a small number, it is no number."
        )
    return float(np.quantile(returns, alpha))


def expected_shortfall(returns: Sequence[float] | np.ndarray, alpha: float = 0.05) -> float:
    """Mean loss in the worst `alpha` of the sample, as a positive number.

    The coherent one. Subadditive, so it cannot punish diversification, and it answers "how bad
    is bad" rather than only "where does bad start". Non-parametric: the mean of the observed
    tail, with no distribution assumed — which matters because the whole point of a tail measure
    is the part where a normal assumption is worst.
    """
    values = as_array(returns, "returns")
    cutoff = _tail_cutoff(values, alpha)
    tail = values[values <= cutoff]
    return float(-tail.mean())


def value_at_risk(returns: Sequence[float] | np.ndarray, alpha: float = 0.05) -> TailRisk:
    """The `alpha`-quantile loss, **returned with the expected shortfall attached**.

    Returns a `TailRisk`, not a float. That is deliberate and is explained at the top of this
    module: VaR alone is incoherent and silent about tail depth, so there is no call signature
    here that produces it on its own.
    """
    values = as_array(returns, "returns")
    cutoff = _tail_cutoff(values, alpha)
    tail = values[values <= cutoff]
    return TailRisk(
        value_at_risk=float(-cutoff),
        expected_shortfall=float(-tail.mean()),
        alpha=float(alpha),
    )


@dataclass(frozen=True)
class Drawdown:
    """The deepest peak-to-trough fall on an equity curve, and how long it took to undo.

    Indices are positions in the curve that was passed in. `recovery_index` is `None` when the
    curve never regained the prior peak — which is not the same as a long recovery and must not
    be reported as one, so `trades_to_recovery` is `None` too rather than the length of the
    sample.
    """

    depth: float
    peak_index: int
    trough_index: int
    recovery_index: int | None

    @property
    def recovered(self) -> bool:
        return self.recovery_index is not None

    @property
    def trades_to_recovery(self) -> int | None:
        """Observations from the trough back to the prior peak, or `None` if it never got back."""
        if self.recovery_index is None:
            return None
        return self.recovery_index - self.trough_index

    @property
    def length(self) -> int:
        """Peak to trough, in observations. The half of the story that always has an answer."""
        return self.trough_index - self.peak_index


def max_drawdown(equity: Sequence[float] | np.ndarray) -> Drawdown:
    """Deepest fractional fall from a running peak, with its peak, trough and recovery.

    `depth` is a fraction of the peak, so 0.18 is an 18% drawdown. Fractional rather than
    absolute because a $200 fall means different things at $1,000 and at $100,000, and this
    number is compared across runs with different stated equity.
    """
    curve = as_array(equity, "equity")
    if np.any(curve < 0):
        raise ValueError("equity must not be negative; see returns.equity_curve, which floors it")

    running_peak = np.maximum.accumulate(curve)
    # A zero peak can only happen if the curve starts at zero, which equity_curve forbids.
    with np.errstate(divide="ignore", invalid="ignore"):
        falls = np.where(running_peak > 0, 1.0 - curve / running_peak, 0.0)

    trough = int(np.argmax(falls))
    depth = float(falls[trough])
    peak = int(np.argmax(curve[: trough + 1]))

    recovery: int | None = None
    if depth > 0:
        after = np.flatnonzero(curve[trough:] >= curve[peak])
        if after.size:
            recovery = trough + int(after[0])
    else:
        # No drawdown at all: the trough is the peak and it is trivially "recovered".
        peak = trough
        recovery = trough

    return Drawdown(depth=depth, peak_index=peak, trough_index=trough, recovery_index=recovery)


def probability_of_ruin(
    win_rate: float,
    payoff_ratio: float,
    risk_fraction: float,
    *,
    ruin_fraction: float = 0.5,
) -> float:
    """Chance of losing `ruin_fraction` of the account, under a fixed edge and fixed sizing.

    **This is a sanity check, not a forecast.** It assumes the win rate and the payoff ratio are
    known constants that hold forever, that every trade risks exactly the same fraction, and
    that outcomes are independent. All four are false. Real win rates drift with regime, real
    payoffs are a distribution rather than a number, real sizing rounds to a lot step, and real
    trades cluster — losing runs arrive together, which is the specific way this estimate is
    optimistic. Treat a low number as "this sizing is not obviously suicidal" and nothing more;
    if a decision would change on the difference between 2% and 6% here, that decision needs
    `resample.monte_carlo` with a block bootstrap, which keeps the clustering.

    The maths is the classic gambler's ruin. Each trade wins `payoff_ratio` units with
    probability `win_rate` and loses one unit otherwise, where a unit is `risk_fraction` of the
    starting equity, so ruin is `ruin_fraction / risk_fraction` units away. The per-unit ruin
    probability `r` is the root in (0, 1) of

        r = (1 - p) + p * r**(payoff + 1)

    — condition on the first trade: a loss ends it, a win leaves `payoff + 1` units to lose
    back — and the answer is `r` raised to the number of units. With no positive edge the only
    root is 1.0, which is the correct and unwelcome answer: negative expectancy ruins the
    account with certainty given enough trades, whatever the sizing.
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"win_rate must be in [0, 1], got {win_rate}")
    if payoff_ratio <= 0:
        raise ValueError(f"payoff_ratio must be positive, got {payoff_ratio}")
    if not 0.0 < risk_fraction <= 1.0:
        raise ValueError(f"risk_fraction must be in (0, 1], got {risk_fraction}")
    if not 0.0 < ruin_fraction <= 1.0:
        raise ValueError(f"ruin_fraction must be in (0, 1], got {ruin_fraction}")

    loss_rate = 1.0 - win_rate
    expectancy = win_rate * payoff_ratio - loss_rate
    if expectancy <= 0:
        return 1.0
    if loss_rate == 0.0:
        # A strategy that never loses cannot be ruined. Handled here rather than left to the
        # bisection, which converges towards zero without reaching it and would report a
        # ruin probability of 1e-61 — a true statement rendered as noise.
        return 0.0

    units = ruin_fraction / risk_fraction

    # Bisect for the root in (0, 1). f(0) = 1 - p > 0 and f is negative just below 1 whenever
    # the edge is positive, so the root is bracketed and the search cannot fail.
    def f(r: float) -> float:
        return loss_rate + win_rate * r ** (payoff_ratio + 1.0) - r

    low, high = 0.0, 1.0 - 1e-12
    for _ in range(200):
        middle = (low + high) / 2.0
        if f(middle) > 0:
            low = middle
        else:
            high = middle
    per_unit = (low + high) / 2.0

    return float(per_unit**units)
