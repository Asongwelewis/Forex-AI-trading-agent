"""Performance statistics, every one of them with a confidence interval attached.

**There is no function here that returns a bare number, and that is the entire design.** A
Sharpe ratio computed on thirty trades is very nearly noise: its sampling distribution is wide
enough that a true Sharpe of 0 routinely produces a measured 0.6, and a measured 0.6 is exactly
the figure that gets a strategy promoted. The point estimate is not the finding — the interval
is. So every function returns `Estimate`, a three-field tuple that unpacks as
`estimate, lower, upper` and cannot be reduced to a scalar without a caller writing `[0]` and
visibly throwing the interval away.

The intervals come from a percentile bootstrap over the trade sequence. That resampling is iid,
which understates dependence — see `resample`, where the block version lives — so these
intervals are, if anything, a little narrow. They are still the difference between "Sharpe 0.9"
and "Sharpe 0.9, and the data is consistent with anything from -0.1 to 1.8".

**`win_rate` is information only and gates nothing.** CLAUDE.md is explicit: `session_breakout`
targets 2R, so its breakeven win rate is 33% and 40% is strongly profitable. A filter on win
rate would systematically discard the profitable asymmetric strategies and keep the ones that
scalp a tenth of an R eighty times out of a hundred and give it all back once. It is computed
because it is worth looking at, it is reported because a human will ask, and
`tests/stats/test_win_rate_gates_nothing.py` asserts by import inspection that no decision
module can reach it.

**`INSUFFICIENT_DATA` below 100 trades.** Also CLAUDE.md. The functions here still compute — a
40-trade interval is informative about how little is known — but `judgement` returns the
sentinel, and a report that ranks strategies must show it rather than a number.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final, NamedTuple

import numpy as np

from fxagent.stats._common import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CONFIDENCE,
    DEFAULT_SEED,
    as_array,
    generator,
    percentile_interval,
)

__all__ = [
    "INSUFFICIENT_DATA",
    "MIN_TRADES_FOR_JUDGEMENT",
    "Estimate",
    "bootstrap_estimate",
    "expectancy_r",
    "judgement",
    "profit_factor",
    "sharpe",
    "sortino",
    "win_rate",
]

#: CLAUDE.md's reporting floor. Below this a strategy has not been measured, it has been glimpsed.
MIN_TRADES_FOR_JUDGEMENT: Final = 100

INSUFFICIENT_DATA: Final = "INSUFFICIENT_DATA"


class Estimate(NamedTuple):
    """A statistic and the interval around it. Unpacks as `estimate, lower_ci, upper_ci`.

    Exactly three fields, so the tuple form the caller expects keeps working, and the sample
    size deliberately does not live here — it belongs to the sample, not to one statistic
    computed from it, and adding it would break every `a, b, c = sharpe(...)` in the codebase.
    """

    estimate: float
    lower_ci: float
    upper_ci: float

    @property
    def width(self) -> float:
        return self.upper_ci - self.lower_ci

    @property
    def spans_zero(self) -> bool:
        """Whether the interval contains zero — i.e. whether the sign is established at all."""
        return self.lower_ci <= 0.0 <= self.upper_ci

    def describe(self, *, places: int = 2) -> str:
        return (
            f"{self.estimate:.{places}f} [{self.lower_ci:.{places}f}, {self.upper_ci:.{places}f}]"
        )


def judgement(sample_size: int) -> str | None:
    """`INSUFFICIENT_DATA` below the reporting floor, otherwise `None`.

    `None` for "go ahead and report the numbers" rather than a second sentinel, so a caller
    writes `if judgement(n): ...` and cannot accidentally treat the healthy case as a verdict.
    """
    return INSUFFICIENT_DATA if sample_size < MIN_TRADES_FOR_JUDGEMENT else None


def bootstrap_estimate(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | np.random.Generator = DEFAULT_SEED,
) -> Estimate:
    """Point estimate on the observed sample, interval from `samples` resamples of it.

    Resampled statistics that come back non-finite — a profit factor on a draw that happened to
    contain no losing trades, a Sharpe on a draw with zero variance — are dropped from the
    interval rather than propagated. Dropping them narrows the interval slightly and is the
    lesser evil: a single infinity turns the upper bound into infinity and destroys the
    reporting value of every other draw.
    """
    observed = as_array(values, "values")
    if samples < 1:
        raise ValueError(f"samples must be positive, got {samples}")

    point = float(statistic(observed))
    if observed.size < 2:
        # One observation cannot produce an interval. NaN bounds say so; zero-width bounds
        # would claim certainty from a sample of one.
        return Estimate(point, float("nan"), float("nan"))

    rng = generator(seed)
    indices = rng.integers(0, observed.size, size=(samples, observed.size))
    drawn = observed[indices]
    resampled = np.array([statistic(row) for row in drawn], dtype="float64")
    lower, upper = percentile_interval(resampled, confidence)
    return Estimate(point, lower, upper)


def _sharpe(values: np.ndarray, periods_per_year: float | None) -> float:
    deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if deviation == 0.0:
        return float("nan")
    ratio = float(np.mean(values)) / deviation
    return ratio * np.sqrt(periods_per_year) if periods_per_year else ratio


def _sortino(values: np.ndarray, target: float, periods_per_year: float | None) -> float:
    excess = values - target
    downside = np.minimum(excess, 0.0)
    # Divided by the full sample size, not by the number of losing observations. That is the
    # standard definition: a strategy with few losses should score better, and dividing by the
    # loss count alone throws that advantage away.
    deviation = float(np.sqrt(np.mean(downside**2)))
    if deviation == 0.0:
        return float("inf") if float(np.mean(excess)) > 0 else float("nan")
    ratio = float(np.mean(excess)) / deviation
    return ratio * np.sqrt(periods_per_year) if periods_per_year else ratio


def _profit_factor(values: np.ndarray) -> float:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def sharpe(
    returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: float | None = None,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | np.random.Generator = DEFAULT_SEED,
) -> Estimate:
    """Mean over standard deviation, with a bootstrap interval.

    `periods_per_year=None` means *per observation, not annualised*, and that is the default
    because a trade sequence has no time axis. Annualising it requires knowing how many trades
    a year holds, which is a property of the router's gating and not of this sample; supplying
    a plausible-looking 252 to a strategy that trades eleven times a year inflates the figure
    by a factor of five and nothing in the output would show it.

    Sample standard deviation (`ddof=1`), because this is an estimate from a sample rather than
    a population, and on 30 trades the difference is not cosmetic.
    """
    return bootstrap_estimate(
        returns,
        lambda values: _sharpe(values, periods_per_year),
        samples=samples,
        confidence=confidence,
        seed=seed,
    )


def sortino(
    returns: Sequence[float] | np.ndarray,
    *,
    target: float = 0.0,
    periods_per_year: float | None = None,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | np.random.Generator = DEFAULT_SEED,
) -> Estimate:
    """Sharpe with only the downside in the denominator, with a bootstrap interval.

    The argument for it: upside variance is not risk, and a strategy punished for its good
    months by a symmetric denominator is being measured with the wrong instrument. The argument
    against reading too much into it: the downside deviation is estimated from fewer
    observations than the full one, so it is noisier — and the interval here will be wider than
    the Sharpe interval on the same trades, which is the honest signal that it is.
    """
    return bootstrap_estimate(
        returns,
        lambda values: _sortino(values, target, periods_per_year),
        samples=samples,
        confidence=confidence,
        seed=seed,
    )


def profit_factor(
    r_multiples: Sequence[float] | np.ndarray,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | np.random.Generator = DEFAULT_SEED,
) -> Estimate:
    """Gross wins over gross losses, with a bootstrap interval.

    `inf` when a sample contains no losing trades. That is the truthful answer to the ratio and
    a warning about the sample rather than a result: a strategy with no losses in the record
    has not been observed long enough for this statistic to mean anything, which is what the
    interval is there to say.
    """
    return bootstrap_estimate(
        r_multiples,
        _profit_factor,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )


def expectancy_r(
    r_multiples: Sequence[float] | np.ndarray,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | np.random.Generator = DEFAULT_SEED,
) -> Estimate:
    """Average R per trade, with a bootstrap interval. **The metric this system is judged on.**

    Not win rate, and not total profit. Expectancy in R is comparable across pairs, across
    account sizes and across strategies with completely different hit rates, which is exactly
    what is needed to compare a 2R breakout against a mean-reversion strategy that scratches
    most of its trades.

    Read the interval first. An expectancy of +0.15R whose interval runs from -0.10 to +0.40 is
    a strategy that has not yet been shown to have an edge, however good the headline looks.
    """
    return bootstrap_estimate(
        r_multiples,
        lambda values: float(np.mean(values)),
        samples=samples,
        confidence=confidence,
        seed=seed,
    )


def win_rate(
    r_multiples: Sequence[float] | np.ndarray,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | np.random.Generator = DEFAULT_SEED,
) -> Estimate:
    """Fraction of trades closing above 0R. **Information only — this must never gate anything.**

    Kept because a human will ask for it and because it is a useful sanity check against the
    payoff ratio, and for no other reason. A `session_breakout` targeting 2R breaks even at 33%
    and is strongly profitable at 40%, so a threshold on this number would reject the system's
    best strategy while passing anything that grinds out small frequent wins ahead of one large
    loss. Expectancy in R, profit factor and max drawdown are the metrics; this is a diagnostic.

    Strictly above zero, so a scratched trade counts as neither a win nor a loss in the
    numerator — it simply is not a win.
    """
    return bootstrap_estimate(
        r_multiples,
        lambda values: float(np.mean(values > 0)),
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
