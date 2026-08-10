"""Hand-constructed bar sequences with arithmetic simple enough to check on paper.

The baseline bar is deliberately degenerate — `open == close`, symmetric wick — so that
every true range is identical and ATR converges to an exact constant. That is what lets
the strategy tests assert real numbers for stops and targets instead of asserting that
the code agrees with itself.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fxagent.adapters.base import Bar, BarSeries

#: A Monday, so the H1 timestamps below sit inside a normal trading week.
WEEK_START = datetime(2026, 1, 5, tzinfo=UTC)

BASE_PRICE = 1.1000


def bar(
    timestamp: datetime,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 1_000,
) -> Bar:
    return Bar(timestamp=timestamp, open=open_, high=high, low=low, close=close, volume=volume)


def flat_bar(timestamp: datetime, *, price: float = BASE_PRICE, band: float = 0.0010) -> Bar:
    """A bar that opens and closes at `price` with a symmetric `band` wick.

    Its true range is exactly `2 * band` regardless of the previous close, and it produces
    no directional movement at all when repeated — so a run of these has ATR == 2 * band
    and ADX == 0, both exactly.
    """
    return bar(
        timestamp,
        open_=price,
        high=price + band,
        low=price - band,
        close=price,
    )


def h1_series(bars: list[Bar], *, symbol: str = "EURUSD", timeframe: str = "H1") -> BarSeries:
    return BarSeries(symbol=symbol, timeframe=timeframe, bars=tuple(bars))


def flat_run(
    *,
    end: datetime,
    count: int,
    step: timedelta = timedelta(hours=1),
    price: float = BASE_PRICE,
    band: float = 0.0010,
) -> list[Bar]:
    """`count` identical flat bars, the last of them opening at `end`."""
    start = end - step * (count - 1)
    return [flat_bar(start + step * index, price=price, band=band) for index in range(count)]


def replace_at(bars: list[Bar], timestamp: datetime, replacement: Bar) -> list[Bar]:
    """Swap in one bar by timestamp, leaving the rest of the sequence untouched."""
    swapped = [replacement if existing.timestamp == timestamp else existing for existing in bars]
    if swapped == bars:
        raise AssertionError(f"no bar at {timestamp} to replace; the fixture is wrong")
    return swapped


def rising_daily(count: int, *, start: float = BASE_PRICE, step: float = 0.0010) -> list[Bar]:
    """A clean daily uptrend: every bar closes `step` above the last.

    True range is a constant 1.5 * step on every bar after the first, so ATR(14) settles on
    exactly that and the carry stop can be asserted to the pip.
    """
    return _daily_ramp(count, start=start, step=step)


def falling_daily(count: int, *, start: float = BASE_PRICE, step: float = 0.0010) -> list[Bar]:
    """The mirror of `rising_daily`, for testing that a slope gate blocks the wrong side."""
    return _daily_ramp(count, start=start, step=-step)


def _daily_ramp(count: int, *, start: float, step: float) -> list[Bar]:
    wick = abs(step) / 2.0
    return [
        bar(
            WEEK_START + timedelta(days=index),
            open_=start + step * index,
            high=start + step * index + wick,
            low=start + step * index - wick,
            close=start + step * index,
        )
        for index in range(count)
    ]


def daily_series(bars: list[Bar], *, symbol: str = "EURUSD") -> BarSeries:
    return BarSeries(symbol=symbol, timeframe="D1", bars=tuple(bars))
