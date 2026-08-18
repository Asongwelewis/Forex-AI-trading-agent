"""Turning a stop distance into a volume, under a cap that nothing can raise.

Sizing is one division, and every defect in it is a units error either side of the divide:

    volume = (equity * risk_fraction) / (stop_distance * contract_size * quote_to_account)

The numerator is account currency. The denominator is what one lot loses if the stop is hit,
converted into account currency. Both sides are named in the code below so the conversion can
be read rather than trusted.

**The 0.5% cap is applied twice, deliberately, and the two applications behave differently.**
`RiskConfig` *rejects* a fraction above `MAX_RISK_PER_TRADE` at construction, because a
deployment variable asking for 5% is a misconfiguration and a service that starts anyway is a
service that trades on it. `position_size` *clamps* instead, silently in the safe direction and
loudly in the log, because it is the last line before a volume and it must be impossible to get
an oversized order out of it — including from a caller that never built a `RiskConfig`, from a
backtest sweep, or from a future module that has not been written yet. Neither path can raise
the cap; there is no argument, no config key and no environment variable that does.

**Size never scales up.** Hard rule 9. `risk_fraction` is a ceiling that rounding-down and the
lot step can only reduce, and `PositionSize.risk_fraction` reports what was actually taken —
which is always at or below what was asked for, never above.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from fxagent.risk.symbols import SymbolSpec
from fxagent.strategies.base import SignalDirection

__all__ = [
    "MAX_RISK_PER_TRADE",
    "NOT_SIZEABLE",
    "PositionSize",
    "RiskConfig",
    "format_money",
    "position_size",
]

logger = logging.getLogger(__name__)

#: Hard rule 8. Absolute, and there is no code path that reads a larger number than this.
MAX_RISK_PER_TRADE: Final = 0.005

#: What a trade plan says when `position_size` returns `None`. A sentence rather than a code,
#: because it goes in front of a human and "the setup is not sizeable" is the whole finding:
#: nothing is wrong, the stop is simply too wide for this account at this risk level.
NOT_SIZEABLE: Final = "not sizeable at this risk level"

#: Rendered in front of the amount where the convention is unambiguous; everything else gets
#: its ISO code after the number. Cosmetic, and confined to `format_money`.
_CURRENCY_PREFIX: Final[dict[str, str]] = {"USD": "$", "EUR": "EUR ", "GBP": "GBP "}


def format_money(amount: float, currency: str) -> str:
    """`1000.0, 'USD'` -> `'$1,000'`. Labels only — never parsed, never round-tripped."""
    prefix = _CURRENCY_PREFIX.get(currency)
    rendered = f"{amount:,.2f}".rstrip("0").rstrip(".")
    return f"{prefix}{rendered}" if prefix else f"{rendered} {currency}"


@dataclass(frozen=True)
class RiskConfig:
    """Stated equity and the fraction of it one trade may risk.

    **`reference_equity` is a number from config, not a balance from a broker.** The analyst
    runs on a machine with no MT5 terminal on it, so there is nothing to ask; and a size quoted
    against an equity the reader cannot see is a size the reader cannot check. Every output of
    this module carries the figure it was computed against for exactly that reason.
    """

    reference_equity: float = 1000.0
    account_currency: str = "USD"
    risk_fraction: float = MAX_RISK_PER_TRADE

    def __post_init__(self) -> None:
        if self.reference_equity <= 0:
            raise ValueError(f"reference_equity must be positive, got {self.reference_equity}")
        if len(self.account_currency) != 3 or not self.account_currency.isalpha():
            raise ValueError(f"account_currency must be an ISO code, got {self.account_currency!r}")
        if self.risk_fraction <= 0:
            raise ValueError(f"risk_fraction must be positive, got {self.risk_fraction}")
        if self.risk_fraction > MAX_RISK_PER_TRADE:
            raise ValueError(
                f"risk_fraction {self.risk_fraction} exceeds the absolute cap "
                f"{MAX_RISK_PER_TRADE} set by hard rule 8. This cap is not raisable by "
                "configuration; lower the value or change the rule, not this check."
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RiskConfig:
        """Read `FX_REFERENCE_EQUITY`, `FX_ACCOUNT_CURRENCY` and `FX_RISK_FRACTION`.

        A malformed or over-cap value raises rather than falling back to the default. Startup
        is the right place to fail on a risk misconfiguration: the alternative is a process
        that runs all day sizing against a number nobody chose.
        """
        source = os.environ if env is None else env
        equity = source.get("FX_REFERENCE_EQUITY", "").strip()
        currency = source.get("FX_ACCOUNT_CURRENCY", "").strip().upper()
        fraction = source.get("FX_RISK_FRACTION", "").strip()
        try:
            return cls(
                reference_equity=float(equity) if equity else 1000.0,
                account_currency=currency or "USD",
                risk_fraction=float(fraction) if fraction else MAX_RISK_PER_TRADE,
            )
        except ValueError as exc:
            raise ValueError(f"invalid risk configuration in the environment: {exc}") from exc

    def size(
        self,
        entry: float,
        stop_loss: float,
        symbol_spec: SymbolSpec,
        *,
        quote_to_account_rate: float | None = None,
    ) -> PositionSize | None:
        """`position_size` with this config's equity, currency and fraction filled in."""
        return position_size(
            self.reference_equity,
            self.risk_fraction,
            entry,
            stop_loss,
            symbol_spec,
            account_currency=self.account_currency,
            quote_to_account_rate=quote_to_account_rate,
        )


@dataclass(frozen=True)
class PositionSize:
    """A placeable volume and every number that produced it.

    `requested_risk_fraction` and `risk_fraction` differ whenever the lot step bit into the
    size, which is most of the time on a small account. Both travel: the first is what the
    system asked for, the second is what it is actually exposed to, and a report that showed
    only the first would overstate the risk taken while a report that showed only the second
    would make the cap look like it moves.
    """

    spec: SymbolSpec
    direction: SignalDirection
    volume: float
    entry_price: float
    stop_loss: float
    stop_distance: float
    risk_amount: float
    risk_fraction: float
    requested_risk_fraction: float
    reference_equity: float
    account_currency: str
    quote_to_account_rate: float

    @property
    def symbol(self) -> str:
        return self.spec.symbol

    @property
    def stop_pips(self) -> float:
        """Stop distance in pips. For reading, not for sizing — see `SymbolSpec.pip`."""
        return self.stop_distance / self.spec.pip

    @property
    def label(self) -> str:
        """`'0.12 lots at 0.5% of a $1,000 account'` — the sentence a human checks."""
        return (
            f"{self.volume:g} lots at {self.risk_fraction:.2%} of a "
            f"{format_money(self.reference_equity, self.account_currency)} account"
        )

    def describe(self) -> str:
        """The label plus the levels it came from, for the panel and for Telegram."""
        return (
            f"{self.symbol} {self.direction}: {self.label}. "
            f"Stop {self.stop_pips:.1f} pips at {self.stop_loss:g} from {self.entry_price:g}, "
            f"risking {format_money(self.risk_amount, self.account_currency)}."
        )


def _direction_from_stop(entry: float, stop_loss: float) -> SignalDirection:
    """Which way the position points, read off the stop rather than taken on trust.

    A caller who passes a direction *and* a stop can pass a pair that disagree, and the
    disagreement sizes a short as though it were a long. There is only one consistent answer
    here, so it is derived. A stop at the entry is rejected: it is a division by zero wearing
    a plausible-looking price, and `Signal` and `OrderRequest` both already refuse to carry one.
    """
    if stop_loss < entry:
        return SignalDirection.LONG
    if stop_loss > entry:
        return SignalDirection.SHORT
    raise ValueError(
        f"stop_loss {stop_loss} equals entry {entry}; a zero-distance stop has no size"
    )


def _resolve_conversion(
    spec: SymbolSpec,
    entry: float,
    account_currency: str,
    quote_to_account_rate: float | None,
) -> float:
    """Account-currency units per one unit of the symbol's quote currency.

    Three cases are arithmetic and are handled; the fourth is a market price this module has no
    business inventing, and it raises.

    * An explicit rate always wins.
    * Quote leg *is* the account currency (USD account, EURUSD) — the rate is 1.
    * Base leg is the account currency (USD account, USDJPY) — the rate is `1 / entry`, which
      is the identity USD/JPY = 1 / (JPY per USD), not an approximation of one.
    * Anything else (USD account, EURGBP) needs a GBPUSD quote that is not in this call. A
      default of 1.0 there would understate a yen-quoted risk by roughly a hundred and one, so
      the caller is made to supply it.
    """
    if quote_to_account_rate is not None:
        if quote_to_account_rate <= 0:
            raise ValueError(f"quote_to_account_rate must be positive, got {quote_to_account_rate}")
        return quote_to_account_rate
    if spec.quote == account_currency:
        return 1.0
    if spec.base == account_currency:
        return 1.0 / entry
    raise ValueError(
        f"{spec.symbol} is quoted in {spec.quote} on a {account_currency} account and neither "
        "leg matches; pass quote_to_account_rate. Assuming 1.0 would misprice the risk by the "
        "exchange rate itself."
    )


def position_size(
    reference_equity: float,
    risk_fraction: float,
    entry: float,
    stop_loss: float,
    symbol_spec: SymbolSpec,
    *,
    account_currency: str = "USD",
    quote_to_account_rate: float | None = None,
) -> PositionSize | None:
    """Volume for this setup, or `None` when the setup is not sizeable at this risk level.

    `risk_fraction` is clamped down to `MAX_RISK_PER_TRADE` — see the module docstring for why
    this clamps where `RiskConfig` raises. It is never scaled up, by anything, ever.

    `None` is not an error and is not a rejection of the setup. It means the stop is wide
    enough, or the account small enough, that the smallest lot the broker will accept would
    risk more than the cap allows. The trade plan reports `NOT_SIZEABLE`; the signal itself
    stands, and the same setup on a larger stated equity sizes fine.
    """
    if reference_equity <= 0:
        raise ValueError(f"reference_equity must be positive, got {reference_equity}")
    if risk_fraction <= 0:
        raise ValueError(f"risk_fraction must be positive, got {risk_fraction}")
    if entry <= 0 or stop_loss <= 0:
        raise ValueError(f"entry {entry} and stop_loss {stop_loss} must both be positive prices")

    direction = _direction_from_stop(entry, stop_loss)
    capped_fraction = min(risk_fraction, MAX_RISK_PER_TRADE)
    if risk_fraction > MAX_RISK_PER_TRADE:
        logger.warning(
            "requested risk_fraction %s exceeds the absolute cap %s; sizing at the cap",
            risk_fraction,
            MAX_RISK_PER_TRADE,
        )

    rate = _resolve_conversion(symbol_spec, entry, account_currency, quote_to_account_rate)
    stop_distance = abs(entry - stop_loss)

    risk_budget = reference_equity * capped_fraction
    loss_per_lot = symbol_spec.money_per_lot(stop_distance) * rate
    volume = symbol_spec.usable_volume(risk_budget / loss_per_lot)

    if volume is None:
        logger.info(
            "%s %s is %s: %s at %s%% would need %.4f lots, below the %s minimum",
            symbol_spec.symbol,
            direction,
            NOT_SIZEABLE,
            format_money(risk_budget, account_currency),
            capped_fraction * 100,
            risk_budget / loss_per_lot,
            symbol_spec.volume_min,
        )
        return None

    risk_amount = volume * loss_per_lot
    return PositionSize(
        spec=symbol_spec,
        direction=direction,
        volume=volume,
        entry_price=entry,
        stop_loss=stop_loss,
        stop_distance=stop_distance,
        risk_amount=risk_amount,
        risk_fraction=risk_amount / reference_equity,
        requested_risk_fraction=capped_fraction,
        reference_equity=reference_equity,
        account_currency=account_currency,
        quote_to_account_rate=rate,
    )
