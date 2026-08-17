"""Bar sequences with a known, exact ATR, so a threshold can be asserted rather than eyeballed.

The warm-up run is `flat_bar`s with a symmetric wick: every true range is exactly `2 * band`,
so ATR settles on exactly `2 * band` and every ATR-relative threshold becomes a number that can
be worked out on paper. A fixture whose ATR is "about right" turns a threshold test into a test
that the code agrees with itself.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fxagent.adapters.base import Bar, BarSeries

MOMENT = datetime(2026, 1, 12, 9, 0, tzinfo=UTC)

#: Enough flat bars for ATR(14) to have converged with room to spare.
WARMUP = 40

#: Half the flat bar's range, so ATR == 2 * BAND == 0.0020 at the default scale.
BAND = 0.0010
PRICE = 1.1000


def atr_for(band: float = BAND) -> float:
    """The exact ATR a warm-up run of `band`-wicked flat bars converges to."""
    return 2 * band


def series(
    *tail: tuple[float, float, float, float], band: float = BAND, price: float = PRICE
) -> BarSeries:
    """A warm-up run of flat bars followed by `tail`, given as (open, high, low, close).

    The tail bars are appended one hour apart after the warm-up, so the bar under test is
    always the last one and `detect_latest` is the natural way to ask about it.
    """
    bars: list[Bar] = [
        Bar(
            timestamp=MOMENT - timedelta(hours=WARMUP + len(tail) - index),
            open=price,
            high=price + band,
            low=price - band,
            close=price,
            volume=1_000,
        )
        for index in range(WARMUP)
    ]
    for offset, (open_, high, low, close) in enumerate(tail):
        bars.append(
            Bar(
                timestamp=MOMENT - timedelta(hours=len(tail) - 1 - offset),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=1_000,
            )
        )
    return BarSeries(symbol="EURUSD", timeframe="H1", bars=tuple(bars))


def names_at_latest(bars: BarSeries) -> set[str]:
    from fxagent.patterns import detect_latest

    return {hit.name for hit in detect_latest(bars)}


def scaled(
    tail: tuple[tuple[float, float, float, float], ...], factor: float
) -> tuple[tuple[float, float, float, float], ...]:
    """The same shapes with every distance from `PRICE` multiplied by `factor`.

    Used to prove the thresholds are volatility-relative: scale the fixture and the verdicts
    must not move. A detector holding a fixed pip cut-off fails the moment `factor` is 10.
    """
    return tuple(
        tuple(PRICE + (value - PRICE) * factor for value in candle)  # type: ignore[misc]
        for candle in tail
    )
