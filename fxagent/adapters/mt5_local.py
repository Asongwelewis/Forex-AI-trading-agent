"""`BrokerAdapter` over a locally running MetaTrader 5 terminal.

Windows-only, and requires the terminal to be running and logged in. The `MetaTrader5`
package is imported lazily so this module stays importable — and testable — anywhere.

Three things here are easy to get wrong and are handled explicitly:

**Server time is not UTC.** MT5 reports bar and tick times on the *broker's* clock,
encoded as if it were a Unix timestamp. Taking those values at face value would shift every
timestamp by hours and silently corrupt the session logic in Phase 5. The offset is detected
on connect, or set explicitly via `MT5_SERVER_UTC_OFFSET_HOURS`, and subtracted from every
timestamp we return. Detection needs a *live* quote: offset and staleness are the same
measurement seen from one tick, so a quote that no timezone explains is rejected rather than
read as a timezone. Do not assume a value per broker — read the clock in Market Watch.

**A freshly subscribed symbol serves a stale tick first.** `symbol_select` returns before any
data flows, and until it does the terminal replays its last cached tick for that symbol — days
old if the symbol was not already in Market Watch. Detection therefore polls for a live quote
instead of judging the first one it is handed, and `MT5_CLOCK_SYMBOL` can point it at a 24/7
instrument so the offset is still readable at the weekend.

**numpy scalars are not Python scalars.** `copy_rates_*` returns a structured array whose
fields are `numpy.float64` / `numpy.uint64`. Those do not serialise to JSON, so every value
is coerced at the boundary.

**Demo accounts only.** Connect refuses any account whose `trade_mode` is not DEMO, and
pins the terminal's login against the configured one — enforcing CLAUDE.md hard rule 1
structurally rather than by convention.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from types import ModuleType, TracebackType
from typing import Any

from fxagent.adapters.base import (
    TIMEFRAMES,
    AccountState,
    Bar,
    BarSeries,
    OrderRequest,
    OrderResult,
    OrderSide,
    Position,
    Tick,
)

logger = logging.getLogger(__name__)

#: Beyond this, a detected server offset is not a timezone — it is stale data.
MAX_PLAUSIBLE_OFFSET = timedelta(hours=14)

#: Server offsets are whole or half hours; detection snaps to this grid.
OFFSET_GRANULARITY = timedelta(minutes=30)

#: How far a quote may sit from the nearest half-hour offset before detection refuses it.
#:
#: A single tick cannot separate "the broker clock is two hours ahead" from "this quote is
#: two hours old" — they are the same number. What it *can* show is a quote landing nowhere
#: near a half-hour boundary, because no timezone explains that; such a quote is stale and
#: the offset read from it would shift every timestamp we return. Must stay far below half
#: of OFFSET_GRANULARITY, or a genuine offset would be rejected as staleness.
MAX_QUOTE_LAG = timedelta(seconds=30)

#: How long detection waits for the first *live* tick after subscribing a symbol.
#:
#: `symbol_select` returns as soon as the subscription is registered, not when data starts
#: flowing. Until the first live tick arrives the terminal serves whatever it last cached for
#: that symbol, and on a cold Market Watch that can be days old. Reading it once and failing
#: on the staleness check rejects a healthy connection in the middle of the London session —
#: measured doing exactly that on a Thursday at 17:21 UTC, against a three-day-old cached
#: EURUSDm tick. So poll, and let the staleness guard judge a tick that had a chance to
#: arrive. The guard itself is unchanged: this fixes what it is shown, not what it accepts.
TICK_WARMUP = timedelta(seconds=3)

#: Gap between polls while waiting for that first live tick.
TICK_POLL_INTERVAL = timedelta(milliseconds=250)


class MT5Error(RuntimeError):
    """Base for every MetaTrader 5 failure that is not an order rejection."""


class MT5DependencyError(MT5Error):
    """The Windows-only `MetaTrader5` package is not installed."""


class MT5ConnectionError(MT5Error):
    """The terminal could not be attached, or is not usable as configured."""


class MT5AccountGuardError(MT5Error):
    """The attached account is not the configured demo account. Never proceed."""


def _require_mt5() -> ModuleType:
    """Import the Windows-only `MetaTrader5` package, or explain how to get it."""
    try:
        import MetaTrader5  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise MT5DependencyError(
            "MetaTrader5 is not installed (Windows-only). Install it with "
            "`uv sync --extra mt5`, and run a local MT5 terminal logged into the demo."
        ) from exc
    return MetaTrader5


def _utc_from_server_epoch(epoch: float, server_utc_offset: timedelta) -> datetime:
    """Convert an MT5 timestamp (broker clock) into a true timezone-aware UTC instant."""
    return datetime.fromtimestamp(int(epoch), UTC) - server_utc_offset


def _optional_price(value: float) -> float | None:
    """MT5 reports an absent stop or target as 0.0; absent is not the same as zero."""
    price = float(value)
    return price if price > 0 else None


def _bar_from_rate(rate: Any, server_utc_offset: timedelta) -> Bar:
    """Convert one row of a `copy_rates_*` structured array into a `Bar`.

    Every field is coerced to a native Python type: numpy scalars do not serialise.
    """
    return Bar(
        timestamp=_utc_from_server_epoch(rate["time"], server_utc_offset),
        open=float(rate["open"]),
        high=float(rate["high"]),
        low=float(rate["low"]),
        close=float(rate["close"]),
        volume=int(rate["tick_volume"]),
    )


def _position_from_mt5(raw: Any, symbol: str, server_utc_offset: timedelta) -> Position:
    """Convert an MT5 position tuple into a `Position`, stripped of the broker suffix."""
    return Position(
        ticket=int(raw.ticket),
        symbol=symbol,
        side=OrderSide.BUY if int(raw.type) == 0 else OrderSide.SELL,
        volume=float(raw.volume),
        entry_price=float(raw.price_open),
        stop_loss=_optional_price(raw.sl),
        take_profit=_optional_price(raw.tp),
        opened_at=_utc_from_server_epoch(raw.time, server_utc_offset),
    )


def _snap_offset(delta: timedelta) -> timedelta:
    """Round a raw clock difference to the nearest half hour."""
    units = round(delta / OFFSET_GRANULARITY)
    return OFFSET_GRANULARITY * units


def _describe_lag(lag: timedelta) -> str:
    """Render a quote lag as an operator reads it: seconds when small, minutes when not."""
    seconds = abs(lag.total_seconds())
    return f"{seconds:.1f}s" if seconds < 120 else f"{seconds / 60:.1f} minutes"


class MT5LocalAdapter:
    """`BrokerAdapter` backed by a local MetaTrader 5 terminal.

    Use as a context manager so the terminal connection is always released:

        with MT5LocalAdapter.from_env() as adapter:
            print(adapter.get_account().balance)
    """

    def __init__(
        self,
        *,
        login: int,
        password: str,
        server: str,
        terminal_path: str | None = None,
        symbol_suffix: str = "",
        server_utc_offset: timedelta | None = None,
        reference_symbol: str = "EURUSD",
        clock_symbol: str | None = None,
        deviation_points: int = 20,
        tick_warmup: timedelta = TICK_WARMUP,
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._terminal_path = terminal_path
        self._symbol_suffix = symbol_suffix
        self._configured_offset = server_utc_offset
        self._reference_symbol = reference_symbol
        #: Read the broker clock from here. A 24/7 instrument keeps detection working at
        #: weekends, when every FX symbol's last tick is by definition stale.
        self._clock_symbol = clock_symbol or reference_symbol
        self._deviation_points = deviation_points
        self._tick_warmup = tick_warmup

        self._mt5: ModuleType | None = None
        self._server_utc_offset: timedelta | None = None

    def __repr__(self) -> str:
        """Never leaks the password — this object ends up in logs and tracebacks."""
        return (
            f"MT5LocalAdapter(login={self._login}, server={self._server!r}, "
            f"symbol_suffix={self._symbol_suffix!r}, connected={self._mt5 is not None})"
        )

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> MT5LocalAdapter:
        """Build from environment variables. Credentials come from nowhere else."""
        source = os.environ if env is None else env
        missing = [
            key for key in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER") if not source.get(key)
        ]
        if missing:
            raise MT5ConnectionError(
                f"missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in your Exness DEMO credentials."
            )

        raw_login = source["MT5_LOGIN"]
        try:
            login = int(raw_login)
        except ValueError:
            raise MT5ConnectionError(
                f"MT5_LOGIN must be the numeric account id, got {raw_login!r}"
            ) from None

        offset_hours = source.get("MT5_SERVER_UTC_OFFSET_HOURS", "").strip()
        offset = timedelta(hours=float(offset_hours)) if offset_hours else None

        symbols = source.get("MT5_SYMBOLS", "").split(",")
        reference = symbols[0].strip() if symbols and symbols[0].strip() else "EURUSD"

        return cls(
            login=login,
            password=source["MT5_PASSWORD"],
            server=source["MT5_SERVER"],
            terminal_path=source.get("MT5_TERMINAL_PATH") or None,
            symbol_suffix=source.get("MT5_SYMBOL_SUFFIX", ""),
            server_utc_offset=offset,
            reference_symbol=reference,
            clock_symbol=source.get("MT5_CLOCK_SYMBOL", "").strip() or None,
            deviation_points=int(source.get("MT5_DEVIATION_POINTS", "20")),
        )

    # -- connection lifecycle --------------------------------------------------

    def __enter__(self) -> MT5LocalAdapter:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disconnect()

    def connect(self) -> None:
        """Attach to the terminal and refuse to proceed unless everything checks out."""
        mt5 = _require_mt5()

        kwargs: dict[str, Any] = {
            "login": self._login,
            "password": self._password,
            "server": self._server,
        }
        if self._terminal_path:
            kwargs["path"] = self._terminal_path

        if not mt5.initialize(**kwargs):
            code, description = mt5.last_error()
            raise MT5ConnectionError(
                f"mt5.initialize() failed ({code}: {description}). Check that the "
                "MetaTrader 5 terminal is running and logged into the demo account, that "
                "MT5_SERVER matches the server shown in the terminal exactly, and that no "
                "other process holds the terminal."
            )

        self._mt5 = mt5
        try:
            self._verify_terminal()
            self._verify_account()
            self._server_utc_offset = self._resolve_server_offset()
        except Exception:
            self.disconnect()
            raise

        logger.info(
            "connected to MT5 server=%s login=%s server_utc_offset=%s",
            self._server,
            self._login,
            self._server_utc_offset,
        )

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None
            self._server_utc_offset = None

    def _terminal(self) -> ModuleType:
        if self._mt5 is None:
            raise MT5ConnectionError("not connected; call connect() or use the context manager")
        return self._mt5

    def _verify_terminal(self) -> None:
        mt5 = self._terminal()
        info = mt5.terminal_info()
        if info is None:
            raise MT5ConnectionError(
                "terminal_info() returned nothing; the terminal is not attached"
            )
        if not info.trade_allowed:
            raise MT5ConnectionError(
                "Algo Trading is disabled in the terminal. Enable the 'Algo Trading' "
                "toolbar button (or Tools > Options > Expert Advisors) and reconnect."
            )

    def _verify_account(self) -> None:
        """Pin the login and hard-reject anything that is not a demo account."""
        mt5 = self._terminal()
        account = mt5.account_info()
        if account is None:
            raise MT5ConnectionError(
                "account_info() returned nothing; the terminal is not logged in"
            )

        if int(account.login) != self._login:
            raise MT5AccountGuardError(
                f"terminal is logged into account {account.login}, but MT5_LOGIN is "
                f"{self._login}. Refusing to operate on an account we were not told to use."
            )

        if int(account.trade_mode) != int(mt5.ACCOUNT_TRADE_MODE_DEMO):
            raise MT5AccountGuardError(
                f"account {account.login} is not a DEMO account "
                f"(trade_mode={account.trade_mode}). This project connects to demo accounts "
                "only — see CLAUDE.md hard rule 1."
            )

    def _resolve_server_offset(self) -> timedelta:
        """Use the configured offset, else detect it from the broker clock.

        Polls for a live quote rather than trusting the first one returned. See `TICK_WARMUP`:
        subscribing a symbol does not mean data has arrived, and the cached tick served in the
        meantime can be days old even while the market is busy.
        """
        if self._configured_offset is not None:
            return self._configured_offset

        symbol = self._select(self._clock_symbol)
        best = self._poll_for_clock(symbol)
        if best is None:
            raise MT5ConnectionError(
                f"cannot read a quote for {symbol} to detect the server time offset. Set "
                "MT5_SERVER_UTC_OFFSET_HOURS explicitly to skip detection."
            )

        offset, lag = best
        if abs(offset) > MAX_PLAUSIBLE_OFFSET:
            raise MT5ConnectionError(
                f"detected an implausible server time offset of {offset} from the last "
                f"{symbol} quote. The quote is stale, most likely because the market is "
                f"closed — it did not go live within {_describe_lag(self._tick_warmup)} of "
                "subscribing. Set MT5_SERVER_UTC_OFFSET_HOURS explicitly — read the clock in "
                "the terminal's Market Watch rather than assuming a value for the broker. "
                "MT5_CLOCK_SYMBOL can point detection at a 24/7 instrument instead."
            )

        if abs(lag) > MAX_QUOTE_LAG:
            raise MT5ConnectionError(
                f"the last {symbol} quote is {_describe_lag(lag)} away from the nearest "
                f"half-hour offset ({offset}), more than the {_describe_lag(MAX_QUOTE_LAG)} "
                "allowed. No timezone explains that gap, so the quote is stale and any offset "
                "inferred from it would shift every timestamp. It stayed stale for "
                f"{_describe_lag(self._tick_warmup)} after subscribing, so the market is most "
                "likely closed. Reconnect when it is open, set MT5_CLOCK_SYMBOL to a 24/7 "
                "instrument, or set MT5_SERVER_UTC_OFFSET_HOURS explicitly to skip detection."
            )

        logger.debug(
            "detected server offset %s from %s (quote lag %s)", offset, symbol, _describe_lag(lag)
        )
        return offset

    def _poll_for_clock(self, symbol: str) -> tuple[timedelta, timedelta] | None:
        """Read `symbol` until a tick passes both guards, or the warm-up budget runs out.

        Returns the `(offset, lag)` of the tick that passed, or of the last one read if none
        did — the latest read being the one the caller should explain the failure with. None
        if no tick was served at all.

        Deliberately not "the tick with the smallest lag": a badly stale quote snaps to an
        offset three days wide and lands within a second of that grid point, so lag alone
        ranks it as good as a live tick. Plausibility is what separates them, and that is the
        caller's check.
        """
        mt5 = self._terminal()
        deadline = time.monotonic() + self._tick_warmup.total_seconds()
        latest: tuple[timedelta, timedelta] | None = None

        while True:
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None and int(tick.time) > 0:
                raw = datetime.fromtimestamp(int(tick.time), UTC) - datetime.now(UTC)
                offset = _snap_offset(raw)
                latest = (offset, offset - raw)
                if abs(offset) <= MAX_PLAUSIBLE_OFFSET and abs(latest[1]) <= MAX_QUOTE_LAG:
                    return latest

            if time.monotonic() >= deadline:
                return latest
            time.sleep(TICK_POLL_INTERVAL.total_seconds())

    @property
    def reference_symbol(self) -> str:
        """First entry of MT5_SYMBOLS. The default clock symbol, but not necessarily it."""
        return self._reference_symbol

    @property
    def clock_symbol(self) -> str:
        """Symbol the server offset is read from — MT5_CLOCK_SYMBOL, else the reference.

        Worth pointing at a crypto CFD: they quote 24/7, so detection still works at the
        weekend when every FX symbol's most recent tick is hours or days old and the
        staleness guard can only refuse.
        """
        return self._clock_symbol

    @property
    def server_utc_offset(self) -> timedelta:
        if self._server_utc_offset is None:
            raise MT5ConnectionError("not connected; server time offset is unknown")
        return self._server_utc_offset

    # -- symbols ---------------------------------------------------------------

    def _broker_symbol(self, symbol: str) -> str:
        """Apply the broker's account-type suffix, e.g. EURUSD -> EURUSDm on Exness."""
        return f"{symbol}{self._symbol_suffix}"

    def _select(self, symbol: str) -> str:
        mt5 = self._terminal()
        broker_symbol = self._broker_symbol(symbol)
        if not mt5.symbol_select(broker_symbol, True):
            raise MT5ConnectionError(
                f"symbol {broker_symbol!r} is not available. Check its exact name in Market "
                "Watch — Exness suffixes symbols by account type (EURUSDm, EURUSDz). Set "
                "MT5_SYMBOL_SUFFIX accordingly."
            )
        return broker_symbol

    # -- BrokerAdapter ---------------------------------------------------------

    def get_bars(self, symbol: str, timeframe: str, count: int) -> BarSeries:
        if timeframe not in TIMEFRAMES:
            raise ValueError(
                f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}"
            )
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        mt5 = self._terminal()
        broker_symbol = self._select(symbol)
        rates = mt5.copy_rates_from_pos(broker_symbol, _mt5_timeframe(mt5, timeframe), 0, count)
        if rates is None or len(rates) == 0:
            code, description = mt5.last_error()
            raise MT5ConnectionError(
                f"no {timeframe} bars returned for {broker_symbol} ({code}: {description})"
            )

        offset = self.server_utc_offset
        return BarSeries(
            symbol=symbol,
            timeframe=timeframe,
            bars=tuple(_bar_from_rate(rate, offset) for rate in rates),
        )

    def get_tick(self, symbol: str) -> Tick:
        mt5 = self._terminal()
        broker_symbol = self._select(symbol)

        info = mt5.symbol_info(broker_symbol)
        tick = mt5.symbol_info_tick(broker_symbol)
        if info is None or tick is None:
            code, description = mt5.last_error()
            raise MT5ConnectionError(
                f"no quote available for {broker_symbol} ({code}: {description})"
            )

        result = Tick(
            symbol=symbol,
            timestamp=_utc_from_server_epoch(tick.time, self.server_utc_offset),
            bid=float(tick.bid),
            ask=float(tick.ask),
            point=float(info.point),
        )
        self._warn_on_spread_disagreement(broker_symbol, result, int(info.spread))
        return result

    def _warn_on_spread_disagreement(self, broker_symbol: str, tick: Tick, reported: int) -> None:
        """A large disagreement means `point` is wrong, not that the spread moved."""
        if abs(tick.spread_points - reported) > 1:
            logger.warning(
                "spread mismatch on %s: derived %s points from bid/ask, terminal reports %s. "
                "Check the symbol's point/digits configuration.",
                broker_symbol,
                tick.spread_points,
                reported,
            )

    def get_account(self) -> AccountState:
        mt5 = self._terminal()
        account = mt5.account_info()
        if account is None:
            raise MT5ConnectionError("account_info() returned nothing")
        return AccountState(
            balance=float(account.balance),
            equity=float(account.equity),
            margin=float(account.margin),
            currency=str(account.currency),
            is_demo=int(account.trade_mode) == int(mt5.ACCOUNT_TRADE_MODE_DEMO),
        )

    def get_positions(self) -> list[Position]:
        mt5 = self._terminal()
        raw_positions = mt5.positions_get()
        if raw_positions is None:
            return []
        offset = self.server_utc_offset
        return [
            _position_from_mt5(raw, self._strip_suffix(str(raw.symbol)), offset)
            for raw in raw_positions
        ]

    def _strip_suffix(self, broker_symbol: str) -> str:
        if self._symbol_suffix and broker_symbol.endswith(self._symbol_suffix):
            return broker_symbol[: -len(self._symbol_suffix)]
        return broker_symbol

    def place_order(self, order: OrderRequest) -> OrderResult:
        """Send a market order with its stop and target attached to the same request."""
        mt5 = self._terminal()
        broker_symbol = self._select(order.symbol)

        info = mt5.symbol_info(broker_symbol)
        tick = mt5.symbol_info_tick(broker_symbol)
        if info is None or tick is None:
            return self._failure(f"no quote available for {broker_symbol}")

        is_buy = order.side is OrderSide.BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "volume": float(order.volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": float(tick.ask if is_buy else tick.bid),
            "sl": float(order.stop_loss),
            "tp": float(order.take_profit),
            "deviation": self._deviation_points,
            "comment": order.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _filling_mode(mt5, info),
        }
        return self._send(request, fallback_volume=order.volume)

    def close_position(self, ticket: int) -> OrderResult:
        mt5 = self._terminal()
        found = mt5.positions_get(ticket=ticket)
        if not found:
            return self._failure(f"no open position with ticket {ticket}")

        position = found[0]
        broker_symbol = str(position.symbol)
        info = mt5.symbol_info(broker_symbol)
        tick = mt5.symbol_info_tick(broker_symbol)
        if info is None or tick is None:
            return self._failure(f"no quote available for {broker_symbol}")

        closing_a_buy = int(position.type) == 0
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "position": int(ticket),
            "volume": float(position.volume),
            "type": mt5.ORDER_TYPE_SELL if closing_a_buy else mt5.ORDER_TYPE_BUY,
            "price": float(tick.bid if closing_a_buy else tick.ask),
            "deviation": self._deviation_points,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _filling_mode(mt5, info),
        }
        return self._send(request, fallback_volume=float(position.volume))

    # -- order plumbing --------------------------------------------------------

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _failure(self, message: str, retcode: int | None = None) -> OrderResult:
        logger.warning("order failed: %s", message)
        return OrderResult(success=False, timestamp=self._now(), retcode=retcode, message=message)

    def _send(self, request: dict[str, Any], fallback_volume: float) -> OrderResult:
        mt5 = self._terminal()
        result = mt5.order_send(request)
        if result is None:
            code, description = mt5.last_error()
            return self._failure(f"order_send returned nothing ({code}: {description})", code)

        retcode = int(result.retcode)
        if retcode != int(mt5.TRADE_RETCODE_DONE):
            return self._failure(f"broker rejected the order: {result.comment}", retcode)

        volume = float(result.volume) or fallback_volume
        return OrderResult(
            success=True,
            timestamp=self._now(),
            ticket=int(result.order),
            retcode=retcode,
            message=str(result.comment),
            filled_price=float(result.price) or None,
            filled_volume=volume or None,
        )


def _mt5_timeframe(mt5: ModuleType, timeframe: str) -> int:
    """Map our timeframe names onto MT5's constants."""
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return int(mapping[timeframe])


def _filling_mode(mt5: ModuleType, info: Any) -> int:
    """Pick a filling mode the symbol actually supports.

    `symbol_info().filling_mode` is a *bitmask* of permitted modes, not a single value —
    Exness EURUSDm reports 3, meaning FOK and IOC are both allowed. Passing that 3 straight
    through as `type_filling` would be rejected outright, because the two numbering schemes
    do not line up: the mask runs FOK=1, IOC=2, while the `ORDER_FILLING_*` enum sent back
    in the request runs FOK=0, IOC=1, RETURN=2.

    The 1 and 2 below are written as literals because the MetaTrader5 package ships no
    `SYMBOL_FILLING_FOK` / `SYMBOL_FILLING_IOC` constants to name them (confirmed absent on
    5.0.6090); they are the MQL5 `SYMBOL_FILLING_MODE` flags.
    """
    allowed = int(info.filling_mode)
    if allowed & 2:  # SYMBOL_FILLING_IOC — accepts a partial fill
        return int(mt5.ORDER_FILLING_IOC)
    if allowed & 1:  # SYMBOL_FILLING_FOK — all or nothing
        return int(mt5.ORDER_FILLING_FOK)
    raise MT5ConnectionError(
        f"symbol reports filling_mode={allowed}, which permits neither IOC nor FOK. "
        "ORDER_FILLING_RETURN is not valid for TRADE_ACTION_DEAL, so there is no filling "
        "mode we can legally send — refusing to place an order that the broker would "
        "reject. Check the symbol's trade settings in Market Watch."
    )
