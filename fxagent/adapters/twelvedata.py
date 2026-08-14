"""Read-only `BrokerAdapter` over the Twelve Data REST API.

Data only. `place_order` and `close_position` raise `NotImplementedError` rather than
pretending — an adapter that silently accepted an order and did nothing would be the single
worst failure mode available to this system.

Used for two jobs: historical backfill deeper than the broker feed reaches, and a daily
sanity check comparing an independent close against ours.

Four things about this API are easy to get wrong and are handled explicitly:

**Errors arrive with HTTP 200.** A blown rate limit returns status 200 and a body of
``{"code": 429, "message": "...", "status": "error"}``. Code that checks `raise_for_status()`
and moves on will parse the error as data and store nothing, or worse, store garbage.

**Values are newest-first.** `BarSeries` requires oldest-first and rejects anything else, so
the response is reversed on the way in.

**FX pairs carry no volume.** The field is absent for currency pairs. `Bar.volume` is
non-optional, so it is recorded as 0 — meaning "not reported", not "no trading happened".
Nothing downstream may read a Twelve Data volume as a real quantity.

**Symbols are slashed.** `EURUSD` is `EUR/USD` here, and neither form carries the Exness `m`
suffix. Translation happens at this boundary so nothing above the adapter sees a
venue-specific name.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Final

import httpx

from fxagent.adapters.base import (
    TIMEFRAMES,
    AccountState,
    Bar,
    BarSeries,
    OrderRequest,
    OrderResult,
    Position,
    Tick,
)
from fxagent.adapters.credits import CreditLedger

__all__ = [
    "TwelveDataAdapter",
    "TwelveDataError",
    "TwelveDataResponseError",
    "TwelveDataUnsupported",
]

logger = logging.getLogger(__name__)

BASE_URL: Final = "https://api.twelvedata.com"

#: Our timeframe names to Twelve Data's `interval` values.
INTERVALS: Final[dict[str, str]] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1day",
}

#: Twelve Data caps a single time_series response at 5000 bars.
MAX_OUTPUTSIZE: Final = 5000

#: Identifies rows this adapter writes. Part of the bars unique key, so it must never change
#: without a migration — two values would silently double every historical bar.
SOURCE: Final = "twelvedata"


class TwelveDataError(RuntimeError):
    """Base for every Twelve Data failure."""


class TwelveDataResponseError(TwelveDataError):
    """The API returned an error payload. Carries the provider's own code."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class TwelveDataUnsupported(NotImplementedError):
    """An execution method was called on a data-only adapter."""


def to_provider_symbol(symbol: str) -> str:
    """`EURUSD` -> `EUR/USD`. Accepts an already-slashed symbol unchanged."""
    cleaned = symbol.strip().upper()
    if "/" in cleaned:
        return cleaned
    if len(cleaned) != 6:
        raise ValueError(
            f"cannot infer a currency pair from {symbol!r}; pass it slashed, e.g. 'EUR/USD'"
        )
    return f"{cleaned[:3]}/{cleaned[3:]}"


def from_provider_symbol(symbol: str) -> str:
    """`EUR/USD` -> `EURUSD`, the canonical form used everywhere above the adapters."""
    return symbol.replace("/", "").strip().upper()


class TwelveDataAdapter:
    """Read-only market data. Satisfies `BrokerAdapter` structurally; refuses to trade."""

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient | None = None,
        ledger: CreditLedger | None = None,
        base_url: str = BASE_URL,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required; set TWELVEDATA_API_KEY")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._ledger = ledger or CreditLedger()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **kwargs: Any) -> TwelveDataAdapter:
        import os

        source = os.environ if env is None else env
        key = (source.get("TWELVEDATA_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "TWELVEDATA_API_KEY is not set. Get a free key at twelvedata.com and add it to "
                ".env under DATA."
            )
        return cls(api_key=key, **kwargs)

    def __repr__(self) -> str:
        """Never renders the key."""
        return f"TwelveDataAdapter(credits={self._ledger!r})"

    @property
    def source(self) -> str:
        """Value written to `bars.source` for rows from this adapter."""
        return SOURCE

    @property
    def ledger(self) -> CreditLedger:
        return self._ledger

    async def __aenter__(self) -> TwelveDataAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- data ------------------------------------------------------------------

    async def get_bars(self, symbol: str, timeframe: str, count: int) -> BarSeries:
        """The `count` most recent bars, oldest first."""
        return await self.bars_ending_at(symbol, timeframe, count, end=None)

    async def bars_ending_at(
        self, symbol: str, timeframe: str, count: int, *, end: datetime | None
    ) -> BarSeries:
        """Bars ending at `end`, for walking backwards through a backfill."""
        if timeframe not in TIMEFRAMES:
            raise ValueError(
                f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}"
            )
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        if count > MAX_OUTPUTSIZE:
            raise ValueError(
                f"count {count} exceeds Twelve Data's {MAX_OUTPUTSIZE}-bar response cap; "
                "page the request instead of asking for more"
            )

        params: dict[str, Any] = {
            "symbol": to_provider_symbol(symbol),
            "interval": INTERVALS[timeframe],
            "outputsize": count,
            # Without this the API returns exchange-local times and the offset is not stated.
            "timezone": "UTC",
            "format": "JSON",
        }
        if end is not None:
            params["end_date"] = _format_query_time(end)

        payload = await self._get("time_series", params, description=f"{symbol} {timeframe}")
        values = payload.get("values")
        if not values:
            raise TwelveDataResponseError(
                f"no {timeframe} bars returned for {symbol}. The pair may not be covered by the "
                "free tier, or the window may fall entirely outside market hours."
            )

        # Newest-first from the API; BarSeries requires oldest-first and enforces it.
        bars = tuple(_to_bar(row) for row in reversed(values))
        return BarSeries(symbol=from_provider_symbol(symbol), timeframe=timeframe, bars=bars)

    async def get_tick(self, symbol: str) -> Tick:
        """Latest quote.

        Twelve Data's free tier exposes a price, not a two-sided book, so bid and ask are both
        that price and `spread_points` is 0. That is honest — this feed genuinely does not know
        the spread — and it is why execution decisions must never be taken from this adapter.
        """
        payload = await self._get(
            "price", {"symbol": to_provider_symbol(symbol)}, description=f"{symbol} price"
        )
        raw = payload.get("price")
        if raw is None:
            raise TwelveDataResponseError(f"no price returned for {symbol}")

        price = float(raw)
        return Tick(
            symbol=from_provider_symbol(symbol),
            timestamp=datetime.now(UTC),
            bid=price,
            ask=price,
            point=_point_for(price),
        )

    # -- execution: refused ----------------------------------------------------

    def get_account(self) -> AccountState:
        raise TwelveDataUnsupported(
            "TwelveDataAdapter is a market-data feed and has no account. Use the execution "
            "adapter for account state."
        )

    def get_positions(self) -> list[Position]:
        raise TwelveDataUnsupported(
            "TwelveDataAdapter is a market-data feed and holds no positions."
        )

    def place_order(self, order: OrderRequest) -> OrderResult:
        raise TwelveDataUnsupported(
            "TwelveDataAdapter cannot place orders — it is a read-only data feed. Route "
            "execution through the broker adapter. This raises rather than returning a failed "
            "OrderResult so the mistake cannot be mistaken for a rejected trade."
        )

    def close_position(self, ticket: int) -> OrderResult:
        raise TwelveDataUnsupported(
            "TwelveDataAdapter cannot close positions — it is a read-only data feed."
        )

    # -- transport -------------------------------------------------------------

    async def _get(
        self, endpoint: str, params: dict[str, Any], *, description: str
    ) -> dict[str, Any]:
        """One credited GET, refused locally before it is sent if the quota is spent."""
        self._ledger.spend(1, description=f"{endpoint} ({description})")

        url = f"{self._base_url}/{endpoint}"
        response = await self._client.get(url, params={**params, "apikey": self._api_key})

        if response.status_code == 429:
            raise TwelveDataResponseError(
                "Twelve Data returned 429 despite the local credit ledger allowing the call. "
                "The ledger and the provider disagree — check for another process sharing this "
                "API key.",
                code=429,
            )
        response.raise_for_status()

        payload = response.json()
        if isinstance(payload, list):  # batch responses are lists; we never request one
            raise TwelveDataResponseError(f"unexpected list response from {endpoint}")

        # The trap: an error is HTTP 200 with status="error" in the body.
        if payload.get("status") == "error":
            code = payload.get("code")
            raise TwelveDataResponseError(
                f"Twelve Data rejected {description}: {payload.get('message', 'no message')} "
                f"(code {code})",
                code=int(code) if isinstance(code, int | str) and str(code).isdigit() else None,
            )
        return payload


def _format_query_time(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("end must be timezone-aware; naive datetimes are rejected")
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _parse_datetime(raw: str) -> datetime:
    """Parse a Twelve Data timestamp as UTC.

    Intraday values look like `2026-03-10 12:00:00`; daily values are bare dates. Neither
    carries an offset, which is safe only because `timezone=UTC` is sent on every request —
    attaching UTC here without that parameter would be assuming what we asked for.
    """
    text = raw.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TwelveDataResponseError(f"unparseable datetime {raw!r} in response") from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _to_bar(row: dict[str, Any]) -> Bar:
    try:
        return Bar(
            timestamp=_parse_datetime(row["datetime"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            # FX pairs report no volume. 0 means "not reported" — never read it as a quantity.
            volume=int(float(row.get("volume") or 0)),
        )
    except KeyError as exc:
        raise TwelveDataResponseError(f"bar is missing field {exc.args[0]!r}: {row}") from exc
    except (TypeError, ValueError) as exc:
        raise TwelveDataResponseError(f"malformed bar {row}: {exc}") from exc


def _point_for(price: float) -> float:
    """JPY pairs quote to 3 decimals, everything else to 5."""
    return 1e-3 if price > 20 else 1e-5
