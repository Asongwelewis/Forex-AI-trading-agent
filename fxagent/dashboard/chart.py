"""Build the left pane: candles, our own overlays, DST-correct session shading, markers.

Pure. Everything here is a function of the bars, evaluations and trades it is handed, so the
whole pane can be built and asserted on in a test without a database, a browser or a clock.

Three things this module refuses to do, each because the alternative was available and worse:

**It computes no market state of its own.** The EMA, the bands and the Asian range come from
`fxagent.indicators` and `fxagent.strategies.session_breakout` — the same code the strategies
read. A chart drawn from a second implementation is a chart that agrees with the system until
one of them is tuned, and then disagrees without either looking wrong.

**It derives no session boundary.** The bands come from `session_bounds_utc`, so London shades
08:00-17:00 UTC in January and 07:00-16:00 in July because `zoneinfo` says so. Fixed UTC
constants here would put the shading an hour off the strategy's own window for half the year,
which is precisely the bug the session module was written to prevent — reintroducing it in the
picture of the system would be worse than never drawing it, because the picture is what gets
believed.

**It invents no price.** Overlays are NaN where the indicator has not warmed up, and those
become whitespace, not zero. A gap in a line is honest; a line at the bottom of the pane is a
measurement that never happened.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from fxagent.adapters.base import BarSeries
from fxagent.dashboard.contract import read_votes
from fxagent.dashboard.models import (
    Candle,
    ChartPayload,
    LinePoint,
    Marker,
    MarkerKind,
    Overlay,
    SessionBand,
    TradeLevels,
)
from fxagent.indicators import ema
from fxagent.indicators.volatility import BOLLINGER_DEVIATIONS, BOLLINGER_PERIOD, bollinger_bands
from fxagent.regime.router import CARRY_DIVERGENCE, RANGE_REVERSION, SESSION_BREAKOUT
from fxagent.regime.sessions import Session, session_bounds_utc
from fxagent.store.repositories.evaluations import EvaluationRecord
from fxagent.store.repositories.trades import TradeRecord
from fxagent.strategies.base import bars_to_frame
from fxagent.strategies.session_breakout import TIMEFRAME as ASIAN_RANGE_TIMEFRAME
from fxagent.strategies.session_breakout import asian_range

__all__ = ["ChartConfig", "build_chart", "price_precision", "session_bands"]

logger = logging.getLogger(__name__)

#: One colour per strategy, so a glance at a marker answers "which one fired" without a legend
#: lookup. Keyed on the router's own constants: a strategy renamed there loses its colour here
#: loudly, at import, rather than silently falling through to the unknown-strategy grey.
STRATEGY_COLOURS: dict[str, str] = {
    SESSION_BREAKOUT: "#e0a03c",
    RANGE_REVERSION: "#3ca7a0",
    CARRY_DIVERGENCE: "#9b7fd4",
}
UNKNOWN_STRATEGY_COLOUR = "#8a8f98"

#: Directions that put a marker on the chart. FLAT is a vote and not a level, so it has nowhere
#: to point; silence has neither.
_DIRECTIONAL = ("LONG", "SHORT")

#: Most trades drawn at once. Each is three price lines, and past a few dozen the pane stops
#: being readable — which is a display limit, so it is enforced here and *reported* in `notes`
#: rather than applied quietly.
MAX_TRADES_DRAWN = 40


@dataclass(frozen=True)
class ChartConfig:
    """Which overlays are drawn and with what parameters. Every number is here, none inline."""

    ema_periods: tuple[int, ...] = (20, 50)
    bollinger_period: int = BOLLINGER_PERIOD
    bollinger_deviations: float = BOLLINGER_DEVIATIONS
    show_asian_range: bool = True
    ema_colours: tuple[str, ...] = ("#4c8dd6", "#d67f4c")
    bollinger_colour: str = "#6d7581"
    asian_range_colour: str = "#c2b280"
    shaded_sessions: tuple[Session, ...] = (
        Session.TOKYO,
        Session.LONDON,
        Session.NEW_YORK,
        Session.OVERLAP,
    )


def price_precision(symbol: str) -> int:
    """Decimal places to quote this pair to: 3 for a yen cross, 5 for everything else.

    Derived from the symbol rather than measured from the data, because the data cannot tell
    the difference between a pair quoted to three places and a five-place pair that happened to
    print two trailing zeros.
    """
    canonical = "".join(character for character in symbol.upper() if character.isalpha())
    return 3 if "JPY" in canonical else 5


def _seconds(moment: datetime) -> int:
    """A UTC instant as the Unix seconds Lightweight Charts calls a `UTCTimestamp`."""
    return int(moment.astimezone(UTC).timestamp())


def session_bands(
    start: datetime,
    end: datetime,
    sessions: Iterable[Session] = (
        Session.TOKYO,
        Session.LONDON,
        Session.NEW_YORK,
        Session.OVERLAP,
    ),
) -> tuple[SessionBand, ...]:
    """Every shaded stretch between `start` and `end`, clipped to that window.

    Each session's bounds are asked of `session_bounds_utc` per *local* date, which is what
    makes the shading follow daylight saving. A day either side of the window is probed as
    well, so a session that opened before the first bar still shades the part of itself that is
    on screen.

    Bands are additive, exactly as `active_sessions` is: the overlap is drawn *over* London and
    New York rather than instead of them, so the busiest hours of the day read as darker rather
    than as a different session that London somehow left.
    """
    if end < start:
        raise ValueError(f"end {end.isoformat()} is before start {start.isoformat()}")

    first_day = (start - timedelta(days=1)).date()
    last_day = (end + timedelta(days=1)).date()

    bands: list[SessionBand] = []
    day: date = first_day
    while day <= last_day:
        for session in sessions:
            bounds = session_bounds_utc(session, day)
            if bounds is None:
                continue
            opens, closes = max(bounds[0], start), min(bounds[1], end)
            if opens < closes:
                bands.append(
                    SessionBand(session=session, start=_seconds(opens), end=_seconds(closes))
                )
        day += timedelta(days=1)

    return tuple(sorted(bands, key=lambda band: (band.start, band.session)))


def _points(times: Sequence[int], values: Sequence[float]) -> tuple[LinePoint, ...]:
    """Pair an indicator's values with bar times, turning NaN into whitespace."""
    return tuple(
        LinePoint(time=time, value=None if math.isnan(value) else float(value))
        for time, value in zip(times, values, strict=True)
    )


def _indicator_overlays(
    bars: BarSeries, times: Sequence[int], config: ChartConfig
) -> list[Overlay]:
    frame = bars_to_frame(bars)
    close = frame["close"]
    overlays: list[Overlay] = []

    for index, period in enumerate(config.ema_periods):
        colour = config.ema_colours[index % len(config.ema_colours)]
        overlays.append(
            Overlay(
                key=f"ema_{period}",
                label=f"EMA {period}",
                colour=colour,
                points=_points(times, ema(close, period).to_numpy()),
            )
        )

    bands = bollinger_bands(close, config.bollinger_period, config.bollinger_deviations)
    label = f"BB {config.bollinger_period} / {config.bollinger_deviations:g}σ"
    for key, series, band_label in (
        ("bb_upper", bands.upper, f"{label} upper"),
        ("bb_middle", bands.middle, f"{label} mid"),
        ("bb_lower", bands.lower, f"{label} lower"),
    ):
        overlays.append(
            Overlay(
                key=key,
                label=band_label,
                colour=config.bollinger_colour,
                points=_points(times, series.to_numpy()),
            )
        )

    return overlays


def _asian_range_overlays(
    bars: BarSeries, times: Sequence[int], config: ChartConfig
) -> tuple[list[Overlay], list[str]]:
    """Two step lines holding each day's Asian high and low, broken where there is no range.

    The range is `session_breakout.asian_range`, not a reimplementation of it, so the box on
    the chart is the box the strategy breaks out of. Days whose Asian session did not fully
    arrive get whitespace on both lines — the same days the strategy refuses to trade.
    """
    if bars.timeframe != ASIAN_RANGE_TIMEFRAME:
        return [], [
            f"Asian session range is measured on {ASIAN_RANGE_TIMEFRAME} bars and is not drawn "
            f"on {bars.timeframe}."
        ]

    ranges: dict[date, tuple[float, float] | None] = {}
    for bar in bars.bars:
        day = bar.timestamp.date()
        if day not in ranges:
            ranges[day] = asian_range(bars, day)

    highs: list[LinePoint] = []
    lows: list[LinePoint] = []
    for bar, time in zip(bars.bars, times, strict=True):
        measured = ranges[bar.timestamp.date()]
        highs.append(LinePoint(time=time, value=measured[0] if measured else None))
        lows.append(LinePoint(time=time, value=measured[1] if measured else None))

    incomplete = sum(1 for measured in ranges.values() if measured is None)
    notes = (
        [f"{incomplete} day(s) had an incomplete Asian session, so no range is drawn for them."]
        if incomplete
        else []
    )

    colour = config.asian_range_colour
    return [
        Overlay(
            key="asian_high",
            label="Asian range high",
            colour=colour,
            points=tuple(highs),
            style="step",
        ),
        Overlay(
            key="asian_low",
            label="Asian range low",
            colour=colour,
            points=tuple(lows),
            style="step",
        ),
    ], notes


def _signal_markers(evaluations: Iterable[EvaluationRecord]) -> list[Marker]:
    """One marker per strategy that named a direction, whether or not it was allowed to vote.

    A gated signal is drawn dimmer rather than dropped. "session_breakout wanted to go long and
    the router had already weighted it to zero" is the disagreement the journal exists to
    record; a chart that showed only the votes that counted would hide the half of the story
    that explains the other half.
    """
    markers: list[Marker] = []
    for record in evaluations:
        time = _seconds(record.ts_utc)
        for vote in read_votes(record.votes):
            if vote.direction not in _DIRECTIONAL:
                continue

            colour = STRATEGY_COLOURS.get(vote.strategy, UNKNOWN_STRATEGY_COLOUR)
            long = vote.direction == "LONG"
            gated = not vote.participated
            confidence = f" {vote.confidence:.2f}" if vote.confidence is not None else ""

            markers.append(
                Marker(
                    time=time,
                    kind=MarkerKind.GATED if gated else MarkerKind.SIGNAL,
                    position="belowBar" if long else "aboveBar",
                    shape="circle" if gated else ("arrowUp" if long else "arrowDown"),
                    colour=_dim(colour) if gated else colour,
                    text=(
                        f"{vote.strategy} {vote.direction}{confidence}{' (gated)' if gated else ''}"
                    ),
                    strategy=vote.strategy,
                    direction=vote.direction,
                )
            )
    return markers


def _dim(colour: str) -> str:
    """A #rrggbb colour at 40% alpha, for a signal the router did not let vote."""
    return f"rgba({int(colour[1:3], 16)}, {int(colour[3:5], 16)}, {int(colour[5:7], 16)}, 0.40)"


def _trade_markers_and_levels(
    trades: Sequence[TradeRecord], window_start: datetime, window_end: datetime
) -> tuple[list[Marker], list[TradeLevels], list[str]]:
    """Entry and exit markers, and each trade's three price lines, clipped to the window.

    **Clipping is not cosmetic.** A price line carries its own times, and Lightweight Charts
    widens the time scale to fit every series on the chart — so a trade opened three days
    before the first candle would drag the axis back three days and leave the candles squeezed
    into the right-hand quarter of a pane that is otherwise empty. Found by running the panel
    over a real store with an open position older than the visible window.

    A trade that entered off-screen therefore gets its lines clipped to the left edge and
    **no entry marker**: an arrow at the first bar would be claiming the position opened there.
    The dropped markers are counted into `notes`, because a trade drawn without its entry is
    exactly the kind of thing a reader would otherwise misread.
    """
    notes: list[str] = []
    drawn = list(trades)
    if len(drawn) > MAX_TRADES_DRAWN:
        notes.append(
            f"{len(drawn) - MAX_TRADES_DRAWN} older trade(s) in this window are not drawn; "
            f"the chart shows the most recent {MAX_TRADES_DRAWN}."
        )
        drawn = drawn[-MAX_TRADES_DRAWN:]

    markers: list[Marker] = []
    levels: list[TradeLevels] = []
    entered_earlier = 0

    for trade in drawn:
        long = trade.direction == "LONG"

        if trade.entry_time_utc < window_start:
            entered_earlier += 1
        else:
            markers.append(
                Marker(
                    time=_seconds(trade.entry_time_utc),
                    kind=MarkerKind.TRADE_ENTRY,
                    position="belowBar" if long else "aboveBar",
                    shape="arrowUp" if long else "arrowDown",
                    colour="#e8e8e8",
                    text=(
                        f"{trade.mode} {trade.direction} {trade.volume:g} @ {trade.entry_price:g}"
                    ),
                    direction=trade.direction,
                )
            )

        if trade.exit_time_utc is not None and window_start <= trade.exit_time_utc <= window_end:
            outcome = f" {trade.r_multiple:+.2f}R" if trade.r_multiple is not None else ""
            markers.append(
                Marker(
                    time=_seconds(trade.exit_time_utc),
                    kind=MarkerKind.TRADE_EXIT,
                    position="aboveBar" if long else "belowBar",
                    shape="square",
                    colour="#e8e8e8",
                    text=f"exit {trade.barrier_touched or '?'}{outcome}",
                    direction=trade.direction,
                )
            )

        # An open trade's lines run to the right-hand edge, because that is where the position
        # still is. Stopping them at the entry bar would draw a live stop as historical.
        end = trade.exit_time_utc or window_end
        levels.append(
            TradeLevels(
                trade_id=trade.id,
                direction=trade.direction,
                entry_price=trade.entry_price,
                stop_price=trade.stop_price,
                target_price=trade.target_price,
                start=_seconds(max(trade.entry_time_utc, window_start)),
                end=_seconds(min(max(end, window_start), window_end)),
                open=trade.exit_time_utc is None,
                volume=trade.volume,
                mode=trade.mode,
                barrier_touched=trade.barrier_touched,
                r_multiple=trade.r_multiple,
            )
        )

    if entered_earlier:
        notes.append(
            f"{entered_earlier} trade(s) opened before this window; their lines start at the "
            "left edge and their entry markers are off-screen."
        )

    return markers, levels, notes


def build_chart(
    bars: BarSeries,
    *,
    source: str,
    evaluations: Sequence[EvaluationRecord] = (),
    trades: Sequence[TradeRecord] = (),
    config: ChartConfig | None = None,
) -> ChartPayload:
    """Assemble the whole left pane from one series of bars and what happened on them.

    An empty series produces an empty payload rather than raising: a symbol the collector has
    not reached yet is an ordinary state, and the panel says "no bars" far more usefully than a
    stack trace does.
    """
    settings = config or ChartConfig()

    if not bars.bars:
        return ChartPayload(
            symbol=bars.symbol,
            timeframe=bars.timeframe,
            source=source,
            price_precision=price_precision(bars.symbol),
            candles=(),
            overlays=(),
            session_bands=(),
            markers=(),
            trades=(),
            notes=("No bars stored for this symbol and timeframe.",),
        )

    times = [_seconds(bar.timestamp) for bar in bars.bars]
    candles = tuple(
        Candle(
            time=time,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
        )
        for bar, time in zip(bars.bars, times, strict=True)
    )

    notes: list[str] = []
    overlays = _indicator_overlays(bars, times, settings)
    if settings.show_asian_range:
        range_overlays, range_notes = _asian_range_overlays(bars, times, settings)
        overlays.extend(range_overlays)
        notes.extend(range_notes)

    window_start = bars.bars[0].timestamp
    window_end = bars.bars[-1].timestamp

    markers = _signal_markers(evaluations)
    trade_markers, levels, trade_notes = _trade_markers_and_levels(trades, window_start, window_end)
    markers.extend(trade_markers)
    notes.extend(trade_notes)

    return ChartPayload(
        symbol=bars.symbol,
        timeframe=bars.timeframe,
        source=source,
        price_precision=price_precision(bars.symbol),
        candles=candles,
        overlays=tuple(overlays),
        session_bands=session_bands(window_start, window_end, settings.shaded_sessions),
        # Lightweight Charts requires markers in ascending time order and silently misplaces
        # them otherwise. Sorted on the way out so no caller has to remember.
        markers=tuple(sorted(markers, key=lambda marker: (marker.time, marker.kind))),
        trades=tuple(levels),
        notes=tuple(notes),
    )
