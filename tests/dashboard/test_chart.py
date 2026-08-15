"""The left pane, asserted without a browser.

The load-bearing test in this file is `test_london_shading_follows_daylight_saving`. Everything
else here would still pass if the session bands were hardcoded UTC constants, and the whole
reason `regime.sessions` exists is that hardcoded UTC constants are wrong for half the year and
wrong silently.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.dashboard.chart import (
    STRATEGY_COLOURS,
    ChartConfig,
    build_chart,
    price_precision,
    session_bands,
)
from fxagent.dashboard.models import MarkerKind
from fxagent.indicators import ema
from fxagent.regime.router import RANGE_REVERSION, SESSION_BREAKOUT
from fxagent.regime.sessions import Session
from fxagent.strategies.base import bars_to_frame
from tests.dashboard.builders import bar_series, evaluation, trade, vote

WINTER = datetime(2026, 1, 14, tzinfo=UTC)
SUMMER = datetime(2026, 7, 15, tzinfo=UTC)


def _band(bands, session: Session):
    return next(band for band in bands if band.session is session)


def _hours(band) -> tuple[int, int]:
    return (
        datetime.fromtimestamp(band.start, UTC).hour,
        datetime.fromtimestamp(band.end, UTC).hour,
    )


# --- session shading ----------------------------------------------------------


def test_london_shading_follows_daylight_saving() -> None:
    """08:00-17:00 UTC in January, 07:00-16:00 in July, from the same code path.

    This is the whole justification for building the bands server-side out of
    `session_bounds_utc` instead of drawing them in JavaScript from two constants. A front end
    that shaded 08:00-17:00 all year would put the London band an hour off the window
    `session_breakout` actually trades for the entire summer, and the picture would be the
    thing everybody believed.
    """
    winter = _band(session_bands(WINTER, WINTER + timedelta(days=1)), Session.LONDON)
    summer = _band(session_bands(SUMMER, SUMMER + timedelta(days=1)), Session.LONDON)

    assert _hours(winter) == (8, 17)
    assert _hours(summer) == (7, 16)


def test_tokyo_does_not_move_because_japan_has_no_daylight_saving() -> None:
    winter = _band(session_bands(WINTER, WINTER + timedelta(days=1)), Session.TOKYO)
    summer = _band(session_bands(SUMMER, SUMMER + timedelta(days=1)), Session.TOKYO)

    assert _hours(winter) == _hours(summer) == (0, 9)


def test_the_overlap_is_drawn_over_london_and_new_york_not_instead_of_them() -> None:
    """Additive, exactly as `active_sessions` is. All three bands exist for the same hours."""
    bands = session_bands(WINTER, WINTER + timedelta(days=1))
    sessions = {band.session for band in bands}

    assert {Session.LONDON, Session.NEW_YORK, Session.OVERLAP} <= sessions

    overlap = _band(bands, Session.OVERLAP)
    london = _band(bands, Session.LONDON)
    new_york = _band(bands, Session.NEW_YORK)

    assert overlap.start == max(london.start, new_york.start)
    assert overlap.end == min(london.end, new_york.end)


def test_a_weekend_window_shades_nothing() -> None:
    saturday = datetime(2026, 1, 17, tzinfo=UTC)
    assert session_bands(saturday, saturday + timedelta(hours=23)) == ()


def test_bands_are_clipped_to_the_window_they_are_asked_for() -> None:
    """A session that opened before the first bar still shades the part that is on screen."""
    start = datetime(2026, 1, 14, 12, 0, tzinfo=UTC)
    end = datetime(2026, 1, 14, 14, 0, tzinfo=UTC)

    for band in session_bands(start, end):
        assert band.start >= int(start.timestamp())
        assert band.end <= int(end.timestamp())


def test_bands_come_back_in_time_order() -> None:
    bands = session_bands(WINTER, WINTER + timedelta(days=3))
    assert [band.start for band in bands] == sorted(band.start for band in bands)


def test_an_inverted_window_is_a_wiring_error() -> None:
    with pytest.raises(ValueError, match="is before"):
        session_bands(WINTER + timedelta(days=1), WINTER)


# --- overlays -----------------------------------------------------------------


def test_overlays_are_the_indicator_layers_own_numbers() -> None:
    """Not recomputed, not rounded, not smoothed for display."""
    bars = bar_series(count=60)
    payload = build_chart(bars, source="twelvedata")

    overlay = next(item for item in payload.overlays if item.key == "ema_20")
    expected = ema(bars_to_frame(bars)["close"], 20)

    assert [point.time for point in overlay.points] == [
        int(bar.timestamp.timestamp()) for bar in bars.bars
    ]
    assert overlay.points[-1].value == pytest.approx(float(expected.iloc[-1]))


def test_indicator_warm_up_is_a_hole_not_a_zero() -> None:
    """A line at the bottom of the pane is a measurement that never happened."""
    payload = build_chart(bar_series(count=60), source="twelvedata")
    overlay = next(item for item in payload.overlays if item.key == "ema_50")

    assert overlay.points[0].value is None
    assert all(point.value is None for point in overlay.points[:49])
    assert overlay.points[49].value is not None


def test_the_bollinger_envelope_is_drawn_around_its_own_middle() -> None:
    payload = build_chart(bar_series(count=60), source="twelvedata")
    keys = {overlay.key for overlay in payload.overlays}

    assert {"bb_upper", "bb_middle", "bb_lower"} <= keys

    upper, middle, lower = (
        next(overlay for overlay in payload.overlays if overlay.key == key)
        for key in ("bb_upper", "bb_middle", "bb_lower")
    )
    for high, mid, low in zip(upper.points, middle.points, lower.points, strict=True):
        if mid.value is None:
            assert high.value is None and low.value is None
        else:
            assert high.value >= mid.value >= low.value


def test_the_asian_range_overlay_is_the_strategys_own_range() -> None:
    """Same function `session_breakout` breaks out of, so the box and the trade agree."""
    from fxagent.strategies.session_breakout import asian_range

    bars = bar_series(count=48)
    payload = build_chart(bars, source="twelvedata")

    high = next(overlay for overlay in payload.overlays if overlay.key == "asian_high")
    low = next(overlay for overlay in payload.overlays if overlay.key == "asian_low")
    measured = asian_range(bars, bars.bars[0].timestamp.date())

    assert measured is not None
    assert high.points[0].value == pytest.approx(measured[0])
    assert low.points[0].value == pytest.approx(measured[1])
    assert high.style == "step"


def test_a_day_with_an_incomplete_asian_session_gets_a_hole_and_a_note() -> None:
    """The days the strategy refuses to trade are the days the chart refuses to draw."""
    # Starts at 04:00, so the first day is missing hours 0-3 of its Asian session.
    bars = bar_series(start=datetime(2026, 1, 12, 4, 0, tzinfo=UTC), count=30)
    payload = build_chart(bars, source="twelvedata")

    high = next(overlay for overlay in payload.overlays if overlay.key == "asian_high")

    assert high.points[0].value is None
    assert any("incomplete Asian session" in note for note in payload.notes)


def test_the_asian_range_is_not_drawn_on_a_timeframe_it_is_not_defined_on() -> None:
    bars = bar_series(timeframe="H4", count=40)
    payload = build_chart(bars, source="twelvedata")

    assert not any(overlay.key.startswith("asian_") for overlay in payload.overlays)
    assert any("H1" in note for note in payload.notes)


def test_overlays_can_be_switched_off_without_touching_the_builder() -> None:
    payload = build_chart(
        bar_series(count=60),
        source="twelvedata",
        config=ChartConfig(ema_periods=(10,), show_asian_range=False),
    )
    keys = {overlay.key for overlay in payload.overlays}

    assert "ema_10" in keys
    assert "ema_20" not in keys
    assert not any(key.startswith("asian_") for key in keys)


# --- markers ------------------------------------------------------------------


def test_a_marker_is_coloured_by_the_strategy_that_produced_it() -> None:
    payload = build_chart(
        bar_series(count=48),
        source="twelvedata",
        evaluations=[
            evaluation(
                votes=[
                    vote(SESSION_BREAKOUT, direction="LONG", confidence=0.7, participated=True),
                    vote(RANGE_REVERSION, direction="SHORT", confidence=0.4, participated=True),
                ]
            )
        ],
    )

    by_strategy = {marker.strategy: marker for marker in payload.markers}

    assert by_strategy[SESSION_BREAKOUT].colour == STRATEGY_COLOURS[SESSION_BREAKOUT]
    assert by_strategy[RANGE_REVERSION].colour == STRATEGY_COLOURS[RANGE_REVERSION]
    assert by_strategy[SESSION_BREAKOUT].shape == "arrowUp"
    assert by_strategy[RANGE_REVERSION].shape == "arrowDown"


def test_a_gated_signal_is_drawn_dimmed_rather_than_dropped() -> None:
    """ "It wanted to and the router said no" is the disagreement the journal exists to keep."""
    payload = build_chart(
        bar_series(count=48),
        source="twelvedata",
        evaluations=[
            evaluation(
                votes=[
                    vote(
                        SESSION_BREAKOUT,
                        weight=0.0,
                        direction="LONG",
                        participated=False,
                        reason="gated: the router does not permit this strategy in this regime",
                    )
                ]
            )
        ],
    )

    marker = payload.markers[0]
    assert marker.kind is MarkerKind.GATED
    assert marker.colour.startswith("rgba(")
    assert "gated" in marker.text


def test_silent_and_flat_strategies_put_nothing_on_the_chart() -> None:
    """A FLAT vote is an opinion with no level to point at, and silence is not even that."""
    payload = build_chart(
        bar_series(count=48),
        source="twelvedata",
        evaluations=[
            evaluation(
                votes=[
                    vote(SESSION_BREAKOUT, direction=None, reason="silent"),
                    vote(RANGE_REVERSION, direction="FLAT", participated=True),
                ]
            )
        ],
    )

    assert payload.markers == ()


def test_markers_come_back_in_ascending_time_order() -> None:
    """Lightweight Charts misplaces unsorted markers instead of complaining about them."""
    base = datetime(2026, 1, 12, 8, 0, tzinfo=UTC)
    payload = build_chart(
        bar_series(count=48),
        source="twelvedata",
        evaluations=[
            evaluation(
                identifier=2,
                ts_utc=base + timedelta(hours=3),
                votes=[vote(SESSION_BREAKOUT, direction="LONG", participated=True)],
            ),
            evaluation(
                identifier=1,
                ts_utc=base,
                votes=[vote(RANGE_REVERSION, direction="SHORT", participated=True)],
            ),
        ],
    )

    times = [marker.time for marker in payload.markers]
    assert times == sorted(times)


# --- trades -------------------------------------------------------------------


def test_a_trade_carries_its_entry_stop_and_target_together() -> None:
    payload = build_chart(
        bar_series(count=48),
        source="twelvedata",
        trades=[trade(entry_price=1.1050, stop_price=1.1020, target_price=1.1110)],
    )

    levels = payload.trades[0]
    assert (levels.entry_price, levels.stop_price, levels.target_price) == (1.1050, 1.1020, 1.1110)
    assert levels.open is True


def test_an_open_trades_lines_run_to_the_right_hand_edge() -> None:
    """A live stop drawn as a two-hour stub reads as a position that already closed."""
    bars = bar_series(count=48)
    payload = build_chart(bars, source="twelvedata", trades=[trade()])

    assert payload.trades[0].end == int(bars.bars[-1].timestamp.timestamp())


def test_a_closed_trade_stops_where_it_closed_and_marks_its_barrier() -> None:
    exit_at = datetime(2026, 1, 12, 15, 0, tzinfo=UTC)
    payload = build_chart(
        bar_series(count=48),
        source="twelvedata",
        trades=[trade(exit_time=exit_at, barrier="TARGET", r_multiple=2.0)],
    )

    assert payload.trades[0].end == int(exit_at.timestamp())
    assert payload.trades[0].open is False

    exits = [marker for marker in payload.markers if marker.kind is MarkerKind.TRADE_EXIT]
    assert len(exits) == 1
    assert "TARGET" in exits[0].text
    assert "+2.00R" in exits[0].text


def test_dropping_older_trades_is_reported_rather_than_silent() -> None:
    from fxagent.dashboard.chart import MAX_TRADES_DRAWN

    base = datetime(2026, 1, 12, 1, 0, tzinfo=UTC)
    trades = [
        trade(identifier=index, entry_time=base + timedelta(minutes=index))
        for index in range(MAX_TRADES_DRAWN + 5)
    ]
    payload = build_chart(bar_series(count=48), source="twelvedata", trades=trades)

    assert len(payload.trades) == MAX_TRADES_DRAWN
    assert any("not drawn" in note for note in payload.notes)


# --- odds and ends -------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("EURUSD", 5), ("EUR_USD", 5), ("USDJPY", 3), ("EURJPY", 3), ("gbpusd", 5)],
)
def test_price_precision_follows_the_quote_convention(symbol: str, expected: int) -> None:
    assert price_precision(symbol) == expected


def test_an_empty_series_produces_an_empty_pane_and_says_so() -> None:
    """A symbol the collector has not reached yet is an ordinary state, not an exception."""
    from fxagent.adapters.base import BarSeries

    payload = build_chart(BarSeries(symbol="EURUSD", timeframe="H1", bars=()), source="twelvedata")

    assert payload.candles == ()
    assert payload.overlays == ()
    assert payload.notes == ("No bars stored for this symbol and timeframe.",)


def test_candles_are_keyed_on_the_bar_open_time() -> None:
    bars = bar_series(count=5)
    payload = build_chart(bars, source="twelvedata")

    assert [candle.time for candle in payload.candles] == [
        int(bar.timestamp.timestamp()) for bar in bars.bars
    ]
    assert payload.candles[0].close == pytest.approx(bars.bars[0].close)


def test_a_trade_opened_before_the_window_is_clipped_to_the_left_edge() -> None:
    """A price line carries its own times, and Lightweight Charts widens the axis to fit every
    series — so an unclipped old position drags the chart back days and squeezes the candles
    into a corner. Found by running the panel over a real store."""
    bars = bar_series(count=48)
    first = bars.bars[0].timestamp

    payload = build_chart(
        bars,
        source="twelvedata",
        trades=[trade(entry_time=first - timedelta(days=3))],
    )

    assert payload.trades[0].start == int(first.timestamp())
    assert not [m for m in payload.markers if m.kind is MarkerKind.TRADE_ENTRY]
    assert any("opened before this window" in note for note in payload.notes)


def test_a_trade_inside_the_window_keeps_its_entry_marker() -> None:
    bars = bar_series(count=48)
    payload = build_chart(
        bars,
        source="twelvedata",
        trades=[trade(entry_time=bars.bars[10].timestamp)],
    )

    entries = [m for m in payload.markers if m.kind is MarkerKind.TRADE_ENTRY]
    assert len(entries) == 1
    assert entries[0].time == int(bars.bars[10].timestamp.timestamp())
    assert payload.notes == ()


def test_no_trade_line_extends_beyond_the_candles() -> None:
    """The axis must be decided by the bars, never by a position drawn over them."""
    bars = bar_series(count=48)
    first, last = bars.bars[0].timestamp, bars.bars[-1].timestamp

    payload = build_chart(
        bars,
        source="twelvedata",
        trades=[
            trade(identifier=1, entry_time=first - timedelta(days=2)),
            trade(
                identifier=2,
                entry_time=bars.bars[5].timestamp,
                exit_time=last + timedelta(days=5),
                barrier="TARGET",
            ),
        ],
    )

    for levels in payload.trades:
        assert int(first.timestamp()) <= levels.start <= levels.end <= int(last.timestamp())
