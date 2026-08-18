"""Series diagnostics: autocorrelation, Hurst, realised volatility, and an ARCH test.

These answer the question that sits underneath every other statistic in this package — *are
these observations independent?* Almost nothing else here is valid if they are not. An iid
bootstrap assumes it. A Sharpe ratio's sampling distribution assumes it. A t-like interval on
expectancy assumes it. So the honest order of work is to run these first and let the answer
decide whether the block bootstrap is a refinement or a requirement.

**The chi-square p-value is hand-written.** scipy is not a dependency of this project and
adding one for a single tail probability would be a poor trade, so `_chi2_survival` implements
the regularised upper incomplete gamma function directly, by series below the transition point
and by continued fraction above it. It is checked against textbook critical values in
`tests/stats/test_diagnostics.py` rather than against another implementation of itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from fxagent.stats._common import as_array

__all__ = [
    "ArchTest",
    "Autocorrelation",
    "HurstResult",
    "arch_test",
    "autocorrelation",
    "hurst_exponent",
    "realised_volatility",
]

#: Above this, a Hurst estimate is reporting the shape of the regression more than the series.
MIN_HURST_OBSERVATIONS: Final = 64


@dataclass(frozen=True, eq=False)
class Autocorrelation:
    """Correlation of a series with itself at lags 1..n, and the band beyond which it matters.

    `confidence_band` is the two-sided 95% band for white noise, `1.96 / sqrt(n)`. A single
    coefficient poking outside it in twenty lags is what white noise looks like — the finding
    is a coefficient far outside, or several in a row on the same side.
    """

    lags: np.ndarray
    values: np.ndarray
    confidence_band: float
    observations: int

    @property
    def significant_lags(self) -> tuple[int, ...]:
        """Lags whose coefficient sits outside the white-noise band."""
        outside = np.flatnonzero(np.abs(self.values) > self.confidence_band)
        return tuple(int(self.lags[position]) for position in outside)

    def describe(self) -> str:
        rendered = "  ".join(
            f"lag{lag}={value:+.3f}" for lag, value in zip(self.lags, self.values, strict=True)
        )
        return f"acf (band ±{self.confidence_band:.3f}): {rendered}"


def autocorrelation(values: Sequence[float] | np.ndarray, max_lag: int = 10) -> Autocorrelation:
    """Sample autocorrelation at lags 1 through `max_lag`.

    The biased estimator — every lag divides by `n`, not by `n - k`. That is the standard
    convention and it is the one the `1.96 / sqrt(n)` band is derived for; using `n - k` inflates
    the high lags, where there is least data, and makes them appear significant against a band
    that no longer applies to them.
    """
    series = as_array(values, "values")
    if max_lag < 1:
        raise ValueError(f"max_lag must be at least 1, got {max_lag}")
    if max_lag >= series.size:
        raise ValueError(
            f"max_lag {max_lag} needs more than {series.size} observations; a lag as long as "
            "the sample has no pairs to correlate"
        )

    centred = series - series.mean()
    denominator = float(np.dot(centred, centred))
    if denominator == 0.0:
        raise ValueError("values are constant; autocorrelation is undefined with zero variance")

    lags = np.arange(1, max_lag + 1)
    coefficients = np.array(
        [float(np.dot(centred[lag:], centred[:-lag]) / denominator) for lag in lags],
        dtype="float64",
    )
    return Autocorrelation(
        lags=lags,
        values=coefficients,
        confidence_band=1.96 / math.sqrt(series.size),
        observations=int(series.size),
    )


@dataclass(frozen=True, eq=False)
class HurstResult:
    """The rescaled-range exponent, with the points the regression was fitted through.

    `exponent` near 0.5 is a random walk, above is persistent (a move tends to be followed by
    another in the same direction), below is mean-reverting.
    """

    exponent: float
    window_sizes: np.ndarray
    rescaled_ranges: np.ndarray

    @property
    def interpretation(self) -> str:
        """A label, with a deliberately wide neutral band.

        0.40 to 0.60 rather than something tighter around 0.5, because classic R/S returns
        about 0.57 on genuinely independent data of eight thousand observations. A band that
        called that "persistent" would label white noise as a trend on every run, which is
        worse than saying nothing. Read `hurst_exponent`'s docstring: the comparison that
        actually settles the question is against a shuffled copy of the same series, not
        against a constant.
        """
        if self.exponent > 0.60:
            return "persistent"
        if self.exponent < 0.40:
            return "mean-reverting"
        return "indistinguishable from a random walk at this sample size"

    def describe(self) -> str:
        return f"Hurst {self.exponent:.3f} — {self.interpretation}"


def hurst_exponent(
    values: Sequence[float] | np.ndarray, *, min_window: int = 8, max_window: int | None = None
) -> HurstResult:
    """Classic rescaled-range (R/S) Hurst exponent.

    **Pass increments, not levels.** Log returns, or the R sequence — not prices. R/S applied to
    a price level returns something close to 1 for any random walk whatsoever, because a level
    is a cumulative sum and cumulative sums trend by construction. That result then gets read as
    "the market is strongly trending", which it is not; it is a statement about integration.

    **Biased upward on short samples.** The R/S estimator returns roughly 0.55–0.60 on genuinely
    independent data of a few thousand observations, and more on less. So the number to compare
    against is not 0.5 in the abstract — it is the value this same function returns on shuffled
    copies of the same series. A shuffle destroys ordering and keeps the distribution, which
    makes it the right null, and the difference between the two is the finding.

    The estimate is the slope of `log(R/S)` regressed on `log(window)`, over non-overlapping
    windows at roughly geometric spacing.
    """
    series = as_array(values, "values")
    if series.size < MIN_HURST_OBSERVATIONS:
        raise ValueError(
            f"Hurst needs at least {MIN_HURST_OBSERVATIONS} observations to fit a slope through "
            f"more than two window sizes, got {series.size}"
        )
    if min_window < 4:
        raise ValueError(f"min_window must be at least 4, got {min_window}")

    # A quarter of the sample: the largest window still giving four independent chunks to
    # average, past which each point of the regression rests on one or two observations.
    ceiling = max_window if max_window is not None else series.size // 4
    if ceiling <= min_window:
        raise ValueError(f"max_window {ceiling} must exceed min_window {min_window}")

    sizes: list[int] = []
    window = min_window
    while window <= ceiling:
        sizes.append(window)
        window = max(window + 1, int(window * 1.5))

    measured_sizes: list[float] = []
    measured_rs: list[float] = []
    for size in sizes:
        chunks = series.size // size
        ratios: list[float] = []
        for chunk in range(chunks):
            segment = series[chunk * size : (chunk + 1) * size]
            deviations = np.cumsum(segment - segment.mean())
            spread = float(deviations.max() - deviations.min())
            scale = float(np.std(segment, ddof=1))
            # A flat segment has no range to rescale; including it as a zero would drag the
            # whole window's average down and bend the slope.
            if scale > 0 and spread > 0:
                ratios.append(spread / scale)
        if ratios:
            measured_sizes.append(float(size))
            measured_rs.append(float(np.mean(ratios)))

    if len(measured_sizes) < 2:
        raise ValueError("not enough non-degenerate windows to fit a Hurst slope")

    log_sizes = np.log(np.array(measured_sizes))
    log_rs = np.log(np.array(measured_rs))
    slope, _ = np.polyfit(log_sizes, log_rs, 1)

    return HurstResult(
        exponent=float(slope),
        window_sizes=np.array(measured_sizes),
        rescaled_ranges=np.array(measured_rs),
    )


def realised_volatility(
    log_returns: Sequence[float] | np.ndarray, *, periods_per_year: float | None = None
) -> float:
    """Square root of the summed squared log returns over the sample.

    Model-free, and deliberately **not** mean-adjusted. Over a short window the sample mean of a
    return series is almost entirely noise, and subtracting it removes real variation while
    adding estimation error — which is why the realised-variance literature does not subtract
    it. A sample standard deviation is a different quantity and belongs to a different question.

    With `periods_per_year`, returns the annualised figure `sqrt(periods_per_year * mean(r^2))`,
    which is the form quoted as "20% vol". Without it, the raw realised volatility over exactly
    the window supplied — no time axis assumed, because none was given.
    """
    values = as_array(log_returns, "log_returns")
    total = float(np.dot(values, values))
    if periods_per_year is None:
        return math.sqrt(total)
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    return math.sqrt(periods_per_year * total / values.size)


@dataclass(frozen=True)
class ArchTest:
    """Engle's LM test for volatility clustering.

    `p_value` is the chance of seeing this much dependence in squared returns if there were
    none. Small means clustering — quiet periods follow quiet periods and violent ones follow
    violent ones — which is the norm in FX and is the specific reason an iid bootstrap over a
    trade sequence understates drawdown.
    """

    statistic: float
    p_value: float
    lags: int
    observations: int

    @property
    def clustering_detected(self) -> bool:
        """At the conventional 5%. A convention, and reported beside the p-value, not instead."""
        return self.p_value < 0.05

    def describe(self) -> str:
        verdict = "clustering" if self.clustering_detected else "no clustering detected"
        return f"ARCH LM({self.lags}) = {self.statistic:.2f}, p = {self.p_value:.4f} — {verdict}"


def arch_test(returns: Sequence[float] | np.ndarray, *, lags: int = 5) -> ArchTest:
    """Regress squared residuals on their own lags; `n * R^2` is chi-square under no ARCH.

    The residual here is the return minus its sample mean, which is the usual choice when there
    is no conditional mean model to speak of — and for FX returns at any frequency shorter than
    a month the mean is indistinguishable from zero anyway.
    """
    series = as_array(returns, "returns")
    if lags < 1:
        raise ValueError(f"lags must be at least 1, got {lags}")
    if series.size <= 2 * lags + 1:
        raise ValueError(
            f"an ARCH({lags}) test needs more than {2 * lags + 1} observations, got {series.size}"
        )

    squared = (series - series.mean()) ** 2
    rows = squared.size - lags
    target = squared[lags:]
    design = np.column_stack(
        [np.ones(rows)] + [squared[lags - lag : squared.size - lag] for lag in range(1, lags + 1)]
    )

    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    centred = target - target.mean()
    total = float(np.dot(centred, centred))
    if total == 0.0:
        # Perfectly constant squared returns: no variance to explain, and no evidence of ARCH.
        return ArchTest(statistic=0.0, p_value=1.0, lags=lags, observations=int(series.size))

    r_squared = 1.0 - float(np.dot(residuals, residuals)) / total
    statistic = rows * r_squared
    return ArchTest(
        statistic=float(statistic),
        p_value=_chi2_survival(float(statistic), lags),
        lags=lags,
        observations=int(series.size),
    )


def _chi2_survival(statistic: float, degrees_of_freedom: int) -> float:
    """`P(X > statistic)` for a chi-square variable. Hand-written; see the module docstring."""
    if statistic <= 0:
        return 1.0
    return _gamma_upper(degrees_of_freedom / 2.0, statistic / 2.0)


def _gamma_upper(a: float, x: float) -> float:
    """Regularised upper incomplete gamma `Q(a, x)`.

    Series expansion below the `x < a + 1` transition and a continued fraction above it — the
    standard split, because each form converges quickly on exactly the side the other does not.
    """
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_continued_fraction(a, x)


def _gamma_series(a: float, x: float, *, iterations: int = 500, tolerance: float = 1e-14) -> float:
    """Lower `P(a, x)` by its series. Converges fast for small `x`."""
    term = 1.0 / a
    total = term
    divisor = a
    for _ in range(iterations):
        divisor += 1.0
        term *= x / divisor
        total += term
        if abs(term) < abs(total) * tolerance:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_continued_fraction(
    a: float, x: float, *, iterations: int = 500, tolerance: float = 1e-14
) -> float:
    """Upper `Q(a, x)` by the modified Lentz continued fraction. Converges fast for large `x`."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, iterations + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tolerance:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))
