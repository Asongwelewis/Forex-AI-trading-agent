"""The `BrokerAdapter` protocol and the Pydantic contracts crossing it.

Every model here is frozen and forbids extra fields: a boundary contract that can be
mutated after validation is not a contract. All timestamps are timezone-aware UTC;
naive datetimes are rejected rather than assumed.

The central invariant is in `OrderRequest`. `stop_loss` and `take_profit` are
non-optional `float`, so an order without protection cannot be constructed at all —
not merely rejected later by the broker. A model validator additionally rejects a
stop or target on the wrong side of the entry, and rejects zero distance (which
would otherwise reach Phase 6 position sizing as a division by zero).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Final, Protocol, runtime_checkable

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "TIMEFRAMES",
    "AccountState",
    "Bar",
    "BarSeries",
    "BrokerAdapter",
    "OrderRequest",
    "OrderResult",
    "OrderSide",
    "Position",
    "Tick",
]


def _ensure_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalise aware ones to UTC.

    A naive timestamp is ambiguous, and silently assuming UTC is how session logic
    ends up an hour wrong on a Sunday open. Callers must be explicit.
    """
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware; naive datetimes are rejected")
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]

#: Timeframes the adapters understand, and the bar interval each denotes.
TIMEFRAMES: Final[dict[str, timedelta]] = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
    "D1": timedelta(days=1),
}


class OrderSide(StrEnum):
    """Direction of an order or an open position."""

    BUY = "BUY"
    SELL = "SELL"


class _Contract(BaseModel):
    """Base for every model crossing the adapter boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Bar(_Contract):
    """A single OHLC bar. `timestamp` is the bar's OPEN time."""

    timestamp: UtcDatetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_ohlc_consistency(self) -> Bar:
        if self.high < self.low:
            raise ValueError(f"high {self.high} is below low {self.low}")
        if self.high < max(self.open, self.close):
            raise ValueError(f"high {self.high} is below open/close")
        if self.low > min(self.open, self.close):
            raise ValueError(f"low {self.low} is above open/close")
        return self


class BarSeries(_Contract):
    """An ordered run of bars for one symbol and timeframe, oldest first."""

    symbol: str = Field(min_length=1)
    timeframe: str
    bars: tuple[Bar, ...]

    @model_validator(mode="after")
    def _check_series(self) -> BarSeries:
        if self.timeframe not in TIMEFRAMES:
            raise ValueError(
                f"unknown timeframe {self.timeframe!r}; expected one of {sorted(TIMEFRAMES)}"
            )
        timestamps = [bar.timestamp for bar in self.bars]
        if any(
            later <= earlier for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        ):
            raise ValueError("bars must be strictly increasing in time, oldest first")
        return self

    def __len__(self) -> int:
        return len(self.bars)


class Tick(_Contract):
    """A quote snapshot. Spread is derived, never stored, so it cannot disagree."""

    symbol: str = Field(min_length=1)
    timestamp: UtcDatetime
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    point: float = Field(gt=0, description="Value of one point, e.g. 1e-5 on a 5-digit EURUSD")

    @model_validator(mode="after")
    def _check_quote(self) -> Tick:
        if self.ask < self.bid:
            raise ValueError(f"ask {self.ask} is below bid {self.bid}")
        return self

    @property
    def spread_points(self) -> int:
        """Current spread in points. Derived from bid/ask so it is always consistent."""
        return round((self.ask - self.bid) / self.point)


class AccountState(_Contract):
    """Account snapshot.

    `is_demo` exists to enforce CLAUDE.md hard rule 1 structurally rather than by
    convention: a caller can refuse to trade an account that does not assert it.
    """

    balance: float
    equity: float
    margin: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    is_demo: bool


class Position(_Contract):
    """An open position as reported by the broker.

    `stop_loss` is optional here — unlike `OrderRequest` — because this reflects
    observed broker state we do not control. A position with `stop_loss=None` carries
    UNBOUNDED risk and Phase 6 exposure maths must treat it as such, never as zero.
    """

    ticket: int = Field(gt=0)
    symbol: str = Field(min_length=1)
    side: OrderSide
    volume: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    opened_at: UtcDatetime


class OrderRequest(_Contract):
    """An order to place. Cannot be constructed without a stop loss and take profit.

    Both are plain non-optional `float`, so `None` is a validation error and the
    frozen model forbids stripping them afterwards.
    """

    symbol: str = Field(min_length=1)
    side: OrderSide
    volume: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    comment: str = Field(default="", max_length=31)

    @model_validator(mode="after")
    def _check_protection_sides(self) -> OrderRequest:
        """Reject a stop or target on the wrong side of the entry, or at zero distance."""
        if self.side is OrderSide.BUY:
            if self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"BUY stop_loss {self.stop_loss} must be below entry {self.entry_price}"
                )
            if self.take_profit <= self.entry_price:
                raise ValueError(
                    f"BUY take_profit {self.take_profit} must be above entry {self.entry_price}"
                )
        else:
            if self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"SELL stop_loss {self.stop_loss} must be above entry {self.entry_price}"
                )
            if self.take_profit >= self.entry_price:
                raise ValueError(
                    f"SELL take_profit {self.take_profit} must be below entry {self.entry_price}"
                )
        return self

    @property
    def risk_distance(self) -> float:
        """Absolute entry-to-stop distance. Guaranteed non-zero by validation."""
        return abs(self.entry_price - self.stop_loss)


class OrderResult(_Contract):
    """Outcome of a place or close request. Failures are values, not exceptions."""

    success: bool
    timestamp: UtcDatetime
    ticket: int | None = Field(default=None, gt=0)
    retcode: int | None = None
    message: str = ""
    filled_price: float | None = Field(default=None, gt=0)
    filled_volume: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_success_shape(self) -> OrderResult:
        if self.success and self.ticket is None:
            raise ValueError("a successful order result must carry a ticket")
        return self


@runtime_checkable
class BrokerAdapter(Protocol):
    """What strategies may assume about a broker.

    Deliberately narrow. Nothing above this layer imports a broker SDK, so the
    Windows-only MT5 adapter can be replaced without touching strategy code.
    """

    def get_bars(self, symbol: str, timeframe: str, count: int) -> BarSeries: ...

    def get_tick(self, symbol: str) -> Tick: ...

    def get_account(self) -> AccountState: ...

    def get_positions(self) -> list[Position]: ...

    def place_order(self, order: OrderRequest) -> OrderResult: ...

    def close_position(self, ticket: int) -> OrderResult: ...
