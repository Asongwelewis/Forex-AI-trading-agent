"""Session breakout: one clean entry, and four reasons to stay out.

Every fixture is a run of identical flat bars with one bar swapped in, so the Asian range,
the true range and the ATR are all exact constants. Each negative test changes exactly one
thing away from a setup that is proven to fire in the same test, which is what makes it
evidence that the gate blocked the trade rather than a fixture that never worked.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.strategies import MarketContext, SignalDirection
from fxagent.strategies.session_breakout import SessionBreakout
from tests.strategies.builders import bar, flat_run, h1_series, replace_at

CONTEXT = MarketContext.neutral()
BARS = 60
BAND = 0.0010

#: The flat baseline puts the Asian range here, exactly.
ASIAN_HIGH = 1.1010
ASIAN_LOW = 1.0990

#: Baseline true range is 2 * BAND on every bar; the breakout bar below spans 0.0040.
BASELINE_TR = 2 * BAND
BREAKOUT_TR = 0.0040
#: Wilder's step from a settled ATR of BASELINE_TR onto one wider bar.
EXPECTED_ATR = (BASELINE_TR * 13 + BREAKOUT_TR) / 14

UP_BREAK = {"open_": 1.1000, "high": 1.1035, "low": 1.0995, "close": 1.1030}
DOWN_BREAK = {"open_": 1.1000, "high": 1.1005, "low": 1.0965, "close": 1.0970}


def _at(hour: int, *, day: int = 6) -> datetime:
    return datetime(2026, 1, day, hour, tzinfo=UTC)


def _series(
    *,
    hour: int = 8,
    shape: dict[str, float] | None = None,
    band: float = BAND,
    count: int = BARS,
    extra_breaks: dict[int, dict[str, float]] | None = None,
    drop_hours: tuple[int, ...] = (),
    timeframe: str = "H1",
):
    """Flat bars ending at `hour`, with a breakout bar swapped in at that hour."""
    end = _at(hour)
    bars = flat_run(end=end, count=count, band=band)

    for break_hour, break_shape in (extra_breaks or {}).items():
        bars = replace_at(bars, _at(break_hour), bar(_at(break_hour), **break_shape))
    if shape is not None:
        bars = replace_at(bars, end, bar(end, **shape))
    if drop_hours:
        dropped = {_at(dropped_hour) for dropped_hour in drop_hours}
        bars = [existing for existing in bars if existing.timestamp not in dropped]

    return h1_series(bars, timeframe=timeframe)


# --- it fires ------------------------------------------------------------------


def test_upward_break_signals_long_with_an_atr_stop_below_entry() -> None:
    signal = SessionBreakout().generate(_series(shape=UP_BREAK), CONTEXT)

    assert signal is not None
    assert signal.direction is SignalDirection.LONG
    assert signal.entry_price == pytest.approx(1.1030)
    # 1.5 * ATR is nearer to entry than the range floor at 1.0990, so it wins.
    assert signal.stop_loss == pytest.approx(1.1030 - 1.5 * EXPECTED_ATR)
    assert signal.reasoning["stop_source"] == "atr"
    assert signal.take_profit == pytest.approx(1.1030 + 2 * 1.5 * EXPECTED_ATR)
    assert signal.reward_risk == pytest.approx(2.0)
    assert signal.strategy_name == "session_breakout"
    assert signal.timestamp == _at(8)


def test_downward_break_signals_short_with_the_stop_above_entry() -> None:
    signal = SessionBreakout().generate(_series(shape=DOWN_BREAK), CONTEXT)

    assert signal is not None
    assert signal.direction is SignalDirection.SHORT
    assert signal.entry_price == pytest.approx(1.0970)
    assert signal.stop_loss == pytest.approx(1.0970 + 1.5 * EXPECTED_ATR)
    assert signal.take_profit == pytest.approx(1.0970 - 2 * 1.5 * EXPECTED_ATR)


@pytest.mark.parametrize(
    ("shape", "direction"),
    [(UP_BREAK, SignalDirection.LONG), (DOWN_BREAK, SignalDirection.SHORT)],
    ids=["long", "short"],
)
def test_stop_is_always_on_the_losing_side_of_entry(
    shape: dict[str, float], direction: SignalDirection
) -> None:
    """The single assertion that matters most: a stop on the wrong side is a runaway loss."""
    signal = SessionBreakout().generate(_series(shape=shape), CONTEXT)

    assert signal is not None
    if direction is SignalDirection.LONG:
        assert signal.stop_loss < signal.entry_price < signal.take_profit
    else:
        assert signal.take_profit < signal.entry_price < signal.stop_loss


def test_a_range_nearer_than_the_atr_stop_is_used_instead() -> None:
    """Narrow Asian range, so its far side is the tighter stop and the ATR is ignored.

    Range 1.0998-1.1002, entry 1.1004: the floor is 0.0006 away, while 1.5 * ATR spans
    0.000642..., so the range wins and the risk is exactly 6 pips.
    """
    narrow = {"open_": 1.1000, "high": 1.1006, "low": 1.0998, "close": 1.1004}
    signal = SessionBreakout().generate(_series(shape=narrow, band=0.0002), CONTEXT)

    assert signal is not None
    assert signal.reasoning["stop_source"] == "range"
    assert signal.stop_loss == pytest.approx(1.0998)
    assert signal.take_profit == pytest.approx(1.1016)


def test_reasoning_records_the_range_it_measured() -> None:
    signal = SessionBreakout().generate(_series(shape=UP_BREAK), CONTEXT)

    assert signal is not None
    assert signal.reasoning["asian_high"] == pytest.approx(ASIAN_HIGH)
    assert signal.reasoning["asian_low"] == pytest.approx(ASIAN_LOW)
    assert signal.reasoning["break_distance"] == pytest.approx(1.1030 - ASIAN_HIGH)
    assert signal.reasoning["breakout_hour_utc"] == 8


# --- it refuses ----------------------------------------------------------------


@pytest.mark.parametrize("hour", [8, 9, 10, 11])
def test_the_whole_london_window_is_tradeable(hour: int) -> None:
    assert SessionBreakout().generate(_series(hour=hour, shape=UP_BREAK), CONTEXT) is not None


@pytest.mark.parametrize("hour", [7, 12, 13, 17])
def test_an_identical_break_outside_the_window_is_ignored(hour: int) -> None:
    """Same bar, same range, only the hour differs — and the hour is the whole reason."""
    assert SessionBreakout().generate(_series(hour=hour, shape=UP_BREAK), CONTEXT) is None


def test_a_second_break_in_the_same_window_is_not_chased() -> None:
    """08:00 already closed outside, so the 09:00 close is not the first and does not trade."""
    already_broken = _series(hour=9, shape=UP_BREAK, extra_breaks={8: UP_BREAK})
    assert SessionBreakout().generate(already_broken, CONTEXT) is None

    # Identical fixture minus the earlier break: now 09:00 *is* the first, and it fires.
    first_break = _series(hour=9, shape=UP_BREAK)
    assert SessionBreakout().generate(first_break, CONTEXT) is not None


def test_a_close_that_stays_inside_the_range_is_not_a_breakout() -> None:
    inside = {"open_": 1.1000, "high": 1.1008, "low": 1.0995, "close": 1.1005}
    assert SessionBreakout().generate(_series(shape=inside), CONTEXT) is None


def test_a_close_exactly_on_the_range_high_is_not_beyond_it() -> None:
    touching = {"open_": 1.1000, "high": 1.1012, "low": 1.0995, "close": ASIAN_HIGH}
    assert SessionBreakout().generate(_series(shape=touching), CONTEXT) is None


def test_an_incomplete_asian_session_yields_no_range_and_no_signal() -> None:
    """One missing hour means the range was never observed, so it is not guessed at."""
    gapped = _series(shape=UP_BREAK, drop_hours=(3,))
    assert SessionBreakout().generate(gapped, CONTEXT) is None

    # The same series with that hour restored does fire, so the gap is what stopped it.
    assert SessionBreakout().generate(_series(shape=UP_BREAK), CONTEXT) is not None


def test_too_little_history_returns_none_rather_than_guessing() -> None:
    strategy = SessionBreakout()
    short = _series(shape=UP_BREAK, count=strategy.required_bars - 1)
    assert strategy.generate(short, CONTEXT) is None


def test_the_wrong_timeframe_is_a_wiring_error_not_a_missing_setup() -> None:
    with pytest.raises(ValueError, match="reads H1 bars"):
        SessionBreakout().generate(_series(shape=UP_BREAK, timeframe="M15"), CONTEXT)


def test_the_previous_days_range_is_not_reused() -> None:
    """The range is measured on the signal bar's own UTC day, never carried over."""
    end = _at(8)
    bars = flat_run(end=end, count=BARS, band=BAND)
    # Wipe today's Asian session entirely; yesterday's identical bars must not stand in.
    bars = [
        existing
        for existing in bars
        if not (existing.timestamp.date() == end.date() and existing.timestamp.hour < 7)
    ]
    bars = replace_at(bars, end, bar(end, **UP_BREAK))

    assert SessionBreakout().generate(h1_series(bars), CONTEXT) is None


def test_required_bars_covers_the_atr_warm_up_and_a_full_session() -> None:
    strategy = SessionBreakout()
    assert strategy.required_bars >= 15  # ATR(14) needs 15 bars to exist at all
    assert strategy.required_bars >= 12  # 00:00 through 11:00 of one day
    assert strategy.name == "session_breakout"


def test_the_signal_timestamp_comes_from_the_bar_not_a_clock() -> None:
    """Shifting the fixture a week later must move the signal timestamp with it."""
    signal = SessionBreakout().generate(_series(shape=UP_BREAK), CONTEXT)
    assert signal is not None
    assert signal.timestamp == _at(8)
    assert signal.timestamp - timedelta(hours=8) == datetime(2026, 1, 6, tzinfo=UTC)
