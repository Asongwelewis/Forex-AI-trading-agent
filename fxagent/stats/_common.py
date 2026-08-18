"""Shared plumbing for the statistics layer.

Three rules bind every function in this package, and all three live here.

**Non-finite input raises.** An indicator's warm-up is NaN by design, and an unlabelled trade
has no R multiple. Both are real conditions, and both must be dropped by a caller who has
decided what dropping them means — silently skipping them turns "40 trades, 12 unlabelled" into
"28 trades" and reports a confidence interval that is narrower than the data supports.

**Randomness is seeded and reproducible.** Every function that resamples takes a `seed` with a
fixed default, so running the same backtest twice produces the same interval. An interval that
moves between runs cannot be compared against last week's, which is most of what it is for.

**Nothing here is annualised unless it is asked to be.** A Sharpe ratio computed on a trade
sequence has no time axis at all, and multiplying it by the square root of a number somebody
guessed is how a strategy acquires an impressive figure nobody can reproduce. `periods_per_year`
defaults to `None` everywhere, and `None` means "per observation", stated as such.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_SEED",
    "as_array",
    "generator",
    "percentile_interval",
    "require_confidence",
]

#: Fixed so a report is reproducible. Vary it deliberately to check a result is not an artefact
#: of one draw — that is a different question from the one the default is here to answer.
DEFAULT_SEED: Final = 8_675_309

#: Enough that the 5th and 95th percentiles are stable to about three decimal places. Bootstrap
#: error falls as 1/sqrt(samples), so ten thousand is roughly the point past which more draws
#: buy less precision than collecting one more trade would.
DEFAULT_BOOTSTRAP_SAMPLES: Final = 10_000

DEFAULT_CONFIDENCE: Final = 0.95


def as_array(values: Sequence[float] | np.ndarray, name: str = "values") -> np.ndarray:
    """A one-dimensional float array, or a raised error explaining exactly what was wrong.

    Rejects non-finite entries rather than dropping them. See the module docstring: a silently
    shortened sample produces a confidence interval that claims more than the data supports.
    """
    array = np.asarray(values, dtype="float64")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(
            f"{name} contains {bad} non-finite value(s). Warm-up NaNs and unlabelled trades "
            "must be dropped deliberately by the caller — dropping them here would report an "
            "interval narrower than the data supports."
        )
    return array


def generator(seed: int | np.random.Generator) -> np.random.Generator:
    """A `Generator` from a seed, or the one already supplied.

    Accepting a live `Generator` lets a caller thread one draw sequence through several
    resampling calls, which is what makes two metrics computed from the same run comparable.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def require_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return confidence


def percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    """The two-sided percentile interval of a bootstrap distribution.

    The percentile method, plainly: the interval is the empirical quantiles of the resampled
    statistic. It is the simplest defensible choice and it is biased for skewed statistics —
    a bootstrapped profit factor is skewed, and its interval sits slightly low. BCa corrects
    for that and is not implemented here; the honest position is that these intervals are
    indicative of width, not exact coverage, and a strategy whose case rests on the third
    decimal of a lower bound does not have a case.
    """
    tail = (1.0 - require_confidence(confidence)) / 2.0
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lower = float(np.quantile(finite, tail))
    upper = float(np.quantile(finite, 1.0 - tail))
    return lower, upper
