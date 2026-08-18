"""Bootstrap and Monte Carlo over a **trade sequence**, never over price.

Resampling price and then re-running a strategy over it answers a question nobody asked: it
tests the strategy against a market that never existed, and it destroys exactly the structure
the strategy was built to exploit. Resampling the *trades* asks the question that matters —
"these are the results the edge produced; how differently could they have been ordered and
drawn, and what would that have done to the account?" One realised equity curve is a single
draw from a distribution, and every drawdown figure quoted from it is a sample of size one.

**Two resamplers, and the default is the pessimistic one for a reason.**

*IID* draws each trade independently with replacement. It assumes trades are independent, and
they are not. Strategies lose in runs — a regime turns, and the same setup fails five times
before the router gates it — and independent draws scatter those losses across the sequence
where they cancel instead of compounding. The measured drawdown comes out too shallow, which is
the one direction an error must not go.

*BLOCK* draws contiguous runs of trades and lays them end to end, so a losing streak that
happened stays whole and can land anywhere in the path. It preserves short-range dependence at
the cost of assuming dependence dies after `block_size` trades. That is still an assumption,
but it is the conservative one, and the difference between the two drawdown distributions is
itself the finding: if they agree, the independence assumption was harmless here; if the block
version is much worse, the iid number was never usable.

**Percentiles, not the mean.** The mean final equity of ten thousand paths is close to the
realised one by construction and tells you nothing. The 5th percentile drawdown is the number
that decides whether the sizing survives a bad year.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from fxagent.stats._common import DEFAULT_SEED, as_array, generator
from fxagent.stats.returns import EquityMode, equity_curve
from fxagent.stats.risk_metrics import max_drawdown

__all__ = [
    "DEFAULT_PATHS",
    "PERCENTILES",
    "Distribution",
    "PathOutcome",
    "ResampleResult",
    "Resampler",
    "block_bootstrap",
    "iid_bootstrap",
    "monte_carlo",
    "require_single_mode",
    "suggested_block_size",
]

#: Enough paths for a stable 5th percentile without making a report take a coffee break.
DEFAULT_PATHS: Final = 10_000

#: Reported for every metric. The tails are the point, so both ends are carried, and the median
#: rather than the mean sits in the middle — a drawdown distribution is skewed and its mean is
#: pulled by paths that will not happen.
PERCENTILES: Final[tuple[int, ...]] = (5, 25, 50, 75, 95)


class Resampler(StrEnum):
    """How a path is drawn. `BLOCK` keeps losing streaks intact; `IID` breaks them up."""

    IID = "IID"
    BLOCK = "BLOCK"


def suggested_block_size(sample_size: int) -> int:
    """`n ** (1/3)`, the usual rule of thumb, floored at 2.

    A rule of thumb and offered as one. The right block length is the horizon over which trades
    actually stay dependent, which is a property of the strategy and the regime router, not of
    the sample size — a caller who knows that a regime persists for about a fortnight should
    pass the number of trades in a fortnight instead of this.
    """
    if sample_size < 1:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    return max(2, int(round(sample_size ** (1.0 / 3.0))))


def iid_bootstrap(
    r_multiples: Sequence[float] | np.ndarray,
    *,
    paths: int = DEFAULT_PATHS,
    seed: int | np.random.Generator = DEFAULT_SEED,
) -> np.ndarray:
    """`paths` sequences drawn independently with replacement. Shape `(paths, n)`.

    Assumes trades are independent. They are not — see the module docstring — so this
    understates drawdown. It is here because the comparison against the block version is
    informative, not because it is the right default.
    """
    values = as_array(r_multiples, "r_multiples")
    rng = generator(seed)
    if paths < 1:
        raise ValueError(f"paths must be positive, got {paths}")
    indices = rng.integers(0, values.size, size=(paths, values.size))
    return values[indices]


def block_bootstrap(
    r_multiples: Sequence[float] | np.ndarray,
    *,
    block_size: int | None = None,
    paths: int = DEFAULT_PATHS,
    seed: int | np.random.Generator = DEFAULT_SEED,
) -> np.ndarray:
    """`paths` sequences built from contiguous blocks laid end to end. Shape `(paths, n)`.

    Circular moving blocks: a block may wrap past the end of the sample back to the start. The
    wrap exists so that every trade has an equal chance of appearing, including the ones near
    the ends — non-circular blocks systematically under-sample the first and last few trades,
    and the last few trades of a backtest are usually the most recent regime.

    The final block is truncated so every path has exactly the original number of trades, which
    keeps the paths comparable to each other and to the realised sequence.
    """
    values = as_array(r_multiples, "r_multiples")
    size = values.size
    length = suggested_block_size(size) if block_size is None else int(block_size)
    if length < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if length > size:
        raise ValueError(f"block_size {length} exceeds the {size} trades available")
    if paths < 1:
        raise ValueError(f"paths must be positive, got {paths}")

    rng = generator(seed)
    blocks_needed = int(np.ceil(size / length))
    starts = rng.integers(0, size, size=(paths, blocks_needed))
    # offsets broadcast a block's worth of positions onto every start, then wrap.
    offsets = np.arange(length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % size
    return values[indices.reshape(paths, -1)[:, :size]]


@dataclass(frozen=True)
class PathOutcome:
    """What one resampled sequence did to the account."""

    final_equity: float
    total_return: float
    max_drawdown: float
    longest_losing_streak: int
    trades_to_recovery: int | None


@dataclass(frozen=True, eq=False)
class Distribution:
    """One metric across every path, summarised by percentile rather than by average.

    Holds the raw samples so a caller can ask for a percentile this class did not pre-compute.
    `eq=False` because two distributions are never usefully compared with `==`, and the default
    dataclass equality on an array raises rather than answering.
    """

    name: str
    samples: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def median(self) -> float:
        return self.percentile(50)

    @property
    def worst(self) -> float:
        return float(np.min(self.samples))

    @property
    def best(self) -> float:
        return float(np.max(self.samples))

    def percentile(self, p: float) -> float:
        if not 0.0 <= p <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {p}")
        return float(np.percentile(self.samples, p))

    @property
    def percentiles(self) -> dict[int, float]:
        return {p: self.percentile(p) for p in PERCENTILES}

    def describe(self, *, places: int = 3) -> str:
        rendered = "  ".join(f"P{p}={self.percentile(p):.{places}f}" for p in PERCENTILES)
        return f"{self.name}: {rendered}"


@dataclass(frozen=True)
class ResampleResult:
    """The distributions, and the settings they came from.

    `resampler`, `block_size` and `mode` travel with the numbers because a drawdown percentile
    means a different thing under each, and a report that shows the figure without them invites
    the iid number to be quoted as though it were the block one — or, worse, a compounded
    drawdown to be checked against a budget that was reasoned in additive R.
    """

    resampler: Resampler
    block_size: int | None
    mode: EquityMode
    paths: int
    trades: int
    risk_fraction: float
    starting_equity: float
    final_equity: Distribution
    total_return: Distribution
    max_drawdown: Distribution
    longest_losing_streak: Distribution
    trades_to_recovery: Distribution
    never_recovered: float

    def describe(self) -> str:
        method = (
            f"{self.resampler} (block {self.block_size})"
            if self.resampler is Resampler.BLOCK
            else str(self.resampler)
        )
        lines = [
            f"{self.paths:,} paths of {self.trades} trades, {method}, "
            f"risking {self.risk_fraction:.1%} of a {self.starting_equity:,.0f} account "
            f"[{self.mode.label}]",
            self.final_equity.describe(places=2),
            self.max_drawdown.describe(),
            self.longest_losing_streak.describe(places=0),
        ]
        if self.never_recovered < 1.0:
            lines.append(self.trades_to_recovery.describe(places=0))
        lines.append(f"never recovered: {self.never_recovered:.1%} of paths")
        return "\n".join(lines)


def _longest_losing_streak(path: np.ndarray) -> int:
    """Longest run of strictly negative R.

    Strictly negative, so a scratch trade at exactly 0R breaks a streak rather than extending
    it. A breakeven exit is not a loss, and counting it as one would inflate every streak on a
    strategy that moves its stop to entry.
    """
    longest = 0
    current = 0
    for value in path:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _outcome(
    path: np.ndarray, risk_fraction: float, starting_equity: float, mode: EquityMode
) -> PathOutcome:
    curve = equity_curve(
        path, risk_fraction=risk_fraction, starting_equity=starting_equity, mode=mode
    )
    drawdown = max_drawdown(curve)
    return PathOutcome(
        final_equity=float(curve[-1]),
        total_return=float(curve[-1] / starting_equity - 1.0),
        max_drawdown=drawdown.depth,
        longest_losing_streak=_longest_losing_streak(path),
        trades_to_recovery=drawdown.trades_to_recovery,
    )


def monte_carlo(
    r_multiples: Sequence[float] | np.ndarray,
    *,
    risk_fraction: float,
    starting_equity: float = 10_000.0,
    resampler: Resampler = Resampler.BLOCK,
    mode: EquityMode = EquityMode.COMPOUNDED,
    block_size: int | None = None,
    paths: int = DEFAULT_PATHS,
    seed: int | np.random.Generator = DEFAULT_SEED,
) -> ResampleResult:
    """Resample the trade sequence `paths` times and report what the account did each time.

    Defaults to the block bootstrap. The iid version is available and is the wrong default:
    it assumes trades are independent, which understates drawdown, and understating drawdown is
    the failure mode that empties an account rather than the one that costs an opportunity.
    """
    values = as_array(r_multiples, "r_multiples")
    if values.size < 2:
        raise ValueError(f"resampling needs at least 2 trades, got {values.size}")

    if resampler is Resampler.IID:
        drawn = iid_bootstrap(values, paths=paths, seed=seed)
        used_block: int | None = None
    else:
        used_block = suggested_block_size(values.size) if block_size is None else int(block_size)
        drawn = block_bootstrap(values, block_size=used_block, paths=paths, seed=seed)

    outcomes = [_outcome(path, risk_fraction, starting_equity, mode) for path in drawn]

    recovered = [o.trades_to_recovery for o in outcomes if o.trades_to_recovery is not None]
    never = 1.0 - len(recovered) / len(outcomes)

    def distribution(
        name: str, extract: Callable[[PathOutcome], float], *, mode_dependent: bool = True
    ) -> Distribution:
        """Name the mode into every metric the mode changes, and into no others.

        A losing streak is a property of the R sequence and is identical either way, so
        labelling it would suggest a difference that is not there. Everything measured in
        money — including the drawdown, which is the number a budget is checked against —
        carries the label, so it cannot be read off a report and compared against a figure
        computed the other way.
        """
        samples = np.array([extract(outcome) for outcome in outcomes], dtype="float64")
        return Distribution(
            name=f"{name} ({mode.label})" if mode_dependent else name, samples=samples
        )

    return ResampleResult(
        resampler=resampler,
        block_size=used_block,
        mode=mode,
        paths=len(outcomes),
        trades=int(values.size),
        risk_fraction=risk_fraction,
        starting_equity=starting_equity,
        final_equity=distribution("final equity", lambda o: o.final_equity),
        total_return=distribution("total return", lambda o: o.total_return),
        max_drawdown=distribution("max drawdown", lambda o: o.max_drawdown),
        longest_losing_streak=distribution(
            "longest losing streak", lambda o: o.longest_losing_streak, mode_dependent=False
        ),
        trades_to_recovery=Distribution(
            name=f"trades to recovery ({mode.label})",
            samples=np.array(recovered, dtype=float) if recovered else np.array([np.nan]),
        ),
        never_recovered=never,
    )


def require_single_mode(*results: ResampleResult) -> EquityMode:
    """The one `EquityMode` every result shares, or a raised error naming the mixture.

    **Call this before drawing.** Two equity curves from the same trades, one visibly
    shallower than the other, is a chart that says the strategy did two different things; and a
    reader comparing the shallower line against a drawdown budget reasoned in additive R has
    been given the wrong number by a picture that looked complete.

    Raising rather than picking a winner is the point. There is no defensible way to merge a
    compounded drawdown with an additive one — they are different quantities, not different
    estimates of the same quantity — so the only correct behaviour is to refuse and make the
    caller draw two charts.
    """
    if not results:
        raise ValueError("no results to check; a chart with nothing on it has no mode")
    modes = {result.mode for result in results}
    if len(modes) > 1:
        found = ", ".join(sorted(mode.label for mode in modes))
        raise ValueError(
            f"these results mix equity modes ({found}). A compounded drawdown and an additive "
            "one are different quantities and must not share a chart or a report line — plot "
            "them separately, each labelled."
        )
    return modes.pop()
