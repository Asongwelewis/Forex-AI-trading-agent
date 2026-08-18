"""Total open risk, currency clustering, and annotations that cannot stop anything.

**Nothing in this module blocks a trade, and that is not an oversight.** There is no broker
connection in this process and no order to hold back — the executor is a separate service on a
separate machine, and the deterministic permission layer is the only thing that gates. What
this module produces is *annotation*: `EXPOSURE_WARNING` on a plan whose open risk has passed
2%, `CORRELATION_WARNING` on a book that is three copies of one bet, `UNBOUNDED_RISK` on a
position with no stop. A reader sees them, the journal keeps them, and the plan is otherwise
returned exactly as it came in. A function here that returned a veto would be a second risk
model sitting beside the real one and disagreeing with it on some Friday afternoon.

**Correlation is reported gross and never netted.** Three long EUR/USD-style pairs is one short
dollar position, and the honest way to say so is "three signals, 1.5% of equity, all short
USD" — not a single netted number. Netting is where a book that is one concentrated bet starts
reading as a balanced one: a long USD leg and a short USD leg of the same size cancel to zero
on paper while both of them are still live, still costing spread, and still able to lose
together when the actual risk was never the dollar in the first place. So `CurrencyExposure`
carries a long side and a short side and deliberately offers no `net` property to read.

**A position without a stop is infinite here, not zero.** `Position.stop_loss` is optional at
the adapter boundary because it reflects broker state we did not choose, and the one thing that
must not happen is for the missing number to sum as 0.0 and make an unbounded book look empty.
`total_open_risk` returns `math.inf` the moment any leg is unstopped, which fails every cap
comparison rather than passing them all.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from fxagent.risk.sizing import PositionSize, RiskConfig, format_money
from fxagent.strategies.base import SignalDirection

__all__ = [
    "CORRELATION_WARNING",
    "EXPOSURE_WARNING",
    "MAX_TOTAL_RISK",
    "UNBOUNDED_RISK",
    "CurrencyExposure",
    "ExposureReport",
    "OpenRisk",
    "RiskAnnotation",
    "assess_exposure",
    "currency_exposure",
    "total_open_risk",
]

#: Hard rule 8's second half: total open risk across the book.
MAX_TOTAL_RISK: Final = 0.02

#: Open risk has passed `MAX_TOTAL_RISK`.
EXPOSURE_WARNING: Final = "EXPOSURE_WARNING"
#: Several open signals are the same currency bet wearing different symbols.
CORRELATION_WARNING: Final = "CORRELATION_WARNING"
#: An open position has no stop, so the book's risk has no upper bound at all.
UNBOUNDED_RISK: Final = "UNBOUNDED_RISK"

#: How many signals on one side of one currency count as a cluster rather than a coincidence.
#: Two, because two is already the point at which the aggregate is what matters and the
#: individual sizes stop being the interesting number.
MIN_CLUSTER: Final = 2


@dataclass(frozen=True)
class OpenRisk:
    """One live bet, reduced to what exposure arithmetic needs.

    `risk_amount` is account currency, and `None` means *unbounded* — a position with no stop —
    never "none at risk". The two would be the same field with the same value in a sloppier
    model, which is precisely the confusion `Position` warns about at the adapter boundary.
    """

    symbol: str
    base: str
    quote: str
    direction: SignalDirection
    volume: float
    risk_amount: float | None

    def __post_init__(self) -> None:
        if self.direction is SignalDirection.FLAT:
            raise ValueError(
                f"{self.symbol} is FLAT and carries no exposure; an abstention is not a position"
            )
        if self.volume <= 0:
            raise ValueError(f"{self.symbol} volume must be positive, got {self.volume}")
        if self.risk_amount is not None and self.risk_amount < 0:
            raise ValueError(f"{self.symbol} risk_amount must not be negative")

    @classmethod
    def from_size(cls, size: PositionSize) -> OpenRisk:
        """Lift a freshly sized trade into the book it is about to join."""
        return cls(
            symbol=size.symbol,
            base=size.spec.base,
            quote=size.spec.quote,
            direction=size.direction,
            volume=size.volume,
            risk_amount=size.risk_amount,
        )

    @property
    def unbounded(self) -> bool:
        return self.risk_amount is None

    @property
    def risk(self) -> float:
        """Risk as a number that can be summed. Infinite when there is no stop."""
        return math.inf if self.risk_amount is None else self.risk_amount

    @property
    def legs(self) -> tuple[tuple[str, SignalDirection], ...]:
        """The two currency positions this pair actually is.

        Long EURUSD is long EUR and short USD, and the whole correlation finding falls out of
        saying so: three long majors are three short USD legs, which aggregate into one number.
        """
        if self.direction is SignalDirection.LONG:
            return ((self.base, SignalDirection.LONG), (self.quote, SignalDirection.SHORT))
        return ((self.base, SignalDirection.SHORT), (self.quote, SignalDirection.LONG))


def total_open_risk(open_signals: Iterable[OpenRisk]) -> float:
    """Account-currency amount at risk across the whole book.

    Returns `math.inf` if any position is unstopped. That is the honest total — an unbounded
    loss plus a bounded one is unbounded — and it makes every downstream cap comparison fail
    loudly instead of comparing against a sum that quietly left the dangerous position out.
    """
    return math.fsum(position.risk for position in open_signals)


@dataclass(frozen=True)
class CurrencyExposure:
    """How much is riding on one currency, in each direction, kept apart.

    There is no `net` property here and there will not be one. See the module docstring.
    """

    currency: str
    long_risk: float
    short_risk: float
    long_symbols: tuple[str, ...]
    short_symbols: tuple[str, ...]

    @property
    def gross_risk(self) -> float:
        """Both sides added. Not a net — a measure of how much is riding on this currency."""
        return self.long_risk + self.short_risk

    @property
    def clustered_side(self) -> SignalDirection | None:
        """The side holding `MIN_CLUSTER` or more symbols, if either does.

        Both sides clustering at once is possible and is reported as the larger one, because a
        book that is long USD three ways and short it three ways is two findings and the loud
        half is the one worth putting in front of a human first.
        """
        long_clustered = len(self.long_symbols) >= MIN_CLUSTER
        short_clustered = len(self.short_symbols) >= MIN_CLUSTER
        if long_clustered and short_clustered:
            return (
                SignalDirection.LONG if self.long_risk >= self.short_risk else SignalDirection.SHORT
            )
        if long_clustered:
            return SignalDirection.LONG
        if short_clustered:
            return SignalDirection.SHORT
        return None


def currency_exposure(open_signals: Iterable[OpenRisk]) -> tuple[CurrencyExposure, ...]:
    """Every currency the book touches, with its long and short sides reported separately.

    Sorted by currency code so two runs over the same book produce the same output — the
    journal stores these, and a report that reorders itself between runs cannot be diffed.
    """
    longs: dict[str, list[OpenRisk]] = {}
    shorts: dict[str, list[OpenRisk]] = {}
    for position in open_signals:
        for currency, side in position.legs:
            bucket = longs if side is SignalDirection.LONG else shorts
            bucket.setdefault(currency, []).append(position)

    return tuple(
        CurrencyExposure(
            currency=currency,
            long_risk=math.fsum(p.risk for p in longs.get(currency, ())),
            short_risk=math.fsum(p.risk for p in shorts.get(currency, ())),
            long_symbols=tuple(p.symbol for p in longs.get(currency, ())),
            short_symbols=tuple(p.symbol for p in shorts.get(currency, ())),
        )
        for currency in sorted(set(longs) | set(shorts))
    )


@dataclass(frozen=True)
class RiskAnnotation:
    """One thing worth saying about the book. Prose and a code — no number to total up.

    Deliberately without a severity weight or a score. The moment an annotation carries
    something summable, someone sums it, and the sum becomes a threshold nobody validated.
    """

    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True)
class ExposureReport:
    """The state of the book, annotated. Carries no verdict and gates nothing."""

    reference_equity: float
    account_currency: str
    open_count: int
    total_risk_amount: float
    max_total_risk: float
    unbounded_symbols: tuple[str, ...]
    currencies: tuple[CurrencyExposure, ...]
    annotations: tuple[RiskAnnotation, ...]

    @property
    def total_risk_fraction(self) -> float:
        """Open risk as a fraction of stated equity. Infinite if anything is unstopped."""
        return self.total_risk_amount / self.reference_equity

    @property
    def over_cap(self) -> bool:
        """Whether the 2% cap has been passed. **A fact, not a gate** — see the module docstring."""
        return self.total_risk_fraction > self.max_total_risk

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(annotation.code for annotation in self.annotations)

    def describe(self) -> str:
        """A few lines for the panel and Telegram. Annotations last, so they read as the point."""
        risked = _amount(self.total_risk_amount, self.account_currency)
        headline = (
            f"{self.open_count} open, risking {risked}"
            f" ({_percent(self.total_risk_fraction)} of a "
            f"{format_money(self.reference_equity, self.account_currency)} account, "
            f"cap {self.max_total_risk:.1%})"
        )
        lines = [headline]
        for exposure in self.currencies:
            if exposure.clustered_side is None:
                continue
            side = exposure.clustered_side
            symbols = (
                exposure.long_symbols if side is SignalDirection.LONG else exposure.short_symbols
            )
            risk = exposure.long_risk if side is SignalDirection.LONG else exposure.short_risk
            lines.append(
                f"  {len(symbols)} x {side} {exposure.currency} "
                f"({', '.join(symbols)}) — {_amount(risk, self.account_currency)}"
            )
        lines.extend(f"  {annotation}" for annotation in self.annotations)
        return "\n".join(lines)


def _amount(value: float, currency: str) -> str:
    return "unbounded" if math.isinf(value) else format_money(value, currency)


def _percent(value: float) -> str:
    return "unbounded" if math.isinf(value) else f"{value:.2%}"


def assess_exposure(
    open_signals: Sequence[OpenRisk],
    config: RiskConfig,
    *,
    max_total_risk: float = MAX_TOTAL_RISK,
) -> ExposureReport:
    """Annotate the book. Never raises on a book it dislikes, never returns a refusal.

    An empty book is a valid book and produces a report with no annotations, rather than
    `None` — the panel draws "0 open, risking $0" from it, and a caller that had to handle a
    missing report would be handling the ordinary case as an exception.
    """
    total = total_open_risk(open_signals)
    currencies = currency_exposure(open_signals)
    unbounded = tuple(p.symbol for p in open_signals if p.unbounded)
    fraction = total / config.reference_equity

    annotations: list[RiskAnnotation] = []

    if unbounded:
        annotations.append(
            RiskAnnotation(
                UNBOUNDED_RISK,
                f"{', '.join(unbounded)} carries no stop loss, so total open risk has no upper "
                "bound and is reported as unbounded rather than as the sum of the rest.",
            )
        )

    if fraction > max_total_risk:
        annotations.append(
            RiskAnnotation(
                EXPOSURE_WARNING,
                f"open risk {_percent(fraction)} of a "
                f"{format_money(config.reference_equity, config.account_currency)} account is "
                f"above the {max_total_risk:.1%} cap across {len(open_signals)} "
                f"{'position' if len(open_signals) == 1 else 'positions'}.",
            )
        )

    for exposure in currencies:
        side = exposure.clustered_side
        if side is None:
            continue
        symbols = exposure.long_symbols if side is SignalDirection.LONG else exposure.short_symbols
        risk = exposure.long_risk if side is SignalDirection.LONG else exposure.short_risk
        annotations.append(
            RiskAnnotation(
                CORRELATION_WARNING,
                f"{len(symbols)} open signals are {side} {exposure.currency} "
                f"({', '.join(symbols)}) — one {exposure.currency} position of "
                f"{_amount(risk, config.account_currency)}, "
                f"{_percent(risk / config.reference_equity)} of equity, reported gross.",
            )
        )

    return ExposureReport(
        reference_equity=config.reference_equity,
        account_currency=config.account_currency,
        open_count=len(open_signals),
        total_risk_amount=total,
        max_total_risk=max_total_risk,
        unbounded_symbols=unbounded,
        currencies=currencies,
        annotations=tuple(annotations),
    )
