"""Each detector on a bar it should find, and on a near-miss it should not.

Every fixture sits on a warm-up run whose ATR is exactly 0.0020, so each threshold below is a
number that can be checked on paper rather than a value the code agreed with itself about.
"""

from __future__ import annotations

import pytest

from fxagent.patterns import (
    DEFAULT_CONFIG,
    DEFINITIONS,
    PATTERN_NAMES,
    detect,
    detect_at,
    detect_latest,
)
from fxagent.patterns.base import Candle, PatternConfig
from tests.patterns.builders import BAND, PRICE, atr_for, names_at_latest, series

ATR = atr_for()  # 0.0020


def test_the_fixture_atr_is_the_number_every_threshold_below_assumes() -> None:
    """Guards every other test in this file: a drifting ATR would silently move the thresholds."""
    hit = detect_latest(series((PRICE, PRICE + 0.0009, PRICE - 0.0009, PRICE)))
    assert hit, "the reference bar produced no detection at all"
    assert hit[0].criteria["atr"] == pytest.approx(ATR)


# -- single-bar formations -------------------------------------------------------------------


def test_doji_fires_when_the_body_is_under_a_tenth_of_atr() -> None:
    # body 0.00005, threshold 0.1 * 0.0020 = 0.0002.
    names = names_at_latest(series((1.1000, 1.1008, 1.0992, 1.10005)))

    assert "doji" in names


def test_doji_does_not_fire_on_a_body_just_over_the_threshold() -> None:
    # body 0.0003, threshold 0.0002.
    names = names_at_latest(series((1.1000, 1.1008, 1.0992, 1.1003)))

    assert "doji" not in names


def test_doji_does_not_fire_on_a_bar_that_barely_moved() -> None:
    """`min_span_atr` exists so the quietest bar of the week is not the cleanest doji of it."""
    # span 0.0004, threshold 0.5 * 0.0020 = 0.0010. Body is a doji body; the bar is nothing.
    names = names_at_latest(series((1.1000, 1.1002, 1.0998, 1.10001)))

    assert "doji" not in names
    assert "pin_bar" not in names
    # It is still an inside bar, and correctly so — it sits within the flat bar before it.
    assert names == {"inside_bar"}


def test_hammer_fires_on_a_long_lower_shadow_under_a_small_body() -> None:
    # body 0.0003, lower shadow 0.0010 (3.3 bodies), upper 0.0001 (0.05 ATR).
    hits = {hit.name: hit for hit in detect_latest(series((1.1005, 1.1009, 1.0995, 1.1008)))}

    assert "hammer" in hits
    assert hits["hammer"].criteria["dominant_shadow_in_bodies"] == pytest.approx(10 / 3)
    assert hits["hammer"].criteria["min_shadow_in_bodies"] == 2.0


def test_hammer_does_not_fire_when_the_opposite_shadow_is_large() -> None:
    """A long wick on both sides is not a hammer; it is a bar that went nowhere twice."""
    # upper shadow 0.0008, cap 0.15 * 0.0020 = 0.0003.
    names = names_at_latest(series((1.1005, 1.1016, 1.0995, 1.1008)))

    assert "hammer" not in names


def test_shooting_star_is_the_hammer_inverted() -> None:
    # body 0.0003 at the bottom, upper shadow 0.0010, lower 0.0001.
    names = names_at_latest(series((1.0995, 1.1005, 1.0991, 1.0992)))

    assert "shooting_star" in names
    assert "hammer" not in names


def test_marubozu_fires_on_a_bar_that_is_nearly_all_body() -> None:
    # body 0.0018 (0.9 ATR, threshold 0.8), both shadows 0.
    hits = {hit.name: hit for hit in detect_latest(series((1.1000, 1.1018, 1.1000, 1.1018)))}

    assert "marubozu" in hits
    assert hits["marubozu"].criteria["body_in_atr"] == pytest.approx(0.9)


def test_marubozu_does_not_fire_when_a_shadow_is_present() -> None:
    # upper shadow 0.0003, cap 0.05 * 0.0020 = 0.0001.
    names = names_at_latest(series((1.1000, 1.1021, 1.1000, 1.1018)))

    assert "marubozu" not in names


def test_pin_bar_fires_when_one_shadow_is_most_of_the_bar() -> None:
    # lower shadow 0.0015 of a 0.0017 span = 0.88, threshold 0.66.
    hits = {hit.name: hit for hit in detect_latest(series((1.1000, 1.1002, 1.0985, 1.10005)))}

    assert "pin_bar" in hits
    assert hits["pin_bar"].criteria["dominant_shadow_fraction"] == pytest.approx(15 / 17)


def test_a_body_less_pin_bar_is_not_reported_as_a_hammer() -> None:
    """`hammer_min_body_atr` is why. The shadow-to-body ratio diverges as the body goes to zero,
    so without a body floor every doji with a wick would be a textbook hammer."""
    names = names_at_latest(series((1.1000, 1.1002, 1.0985, 1.10005)))

    assert {"pin_bar", "doji"} <= names
    assert "hammer" not in names


# -- two-bar formations ----------------------------------------------------------------------


def test_bullish_engulfing_fires_when_an_up_body_covers_the_previous_down_body() -> None:
    hits = {
        hit.name: hit
        for hit in detect_latest(
            series(
                (1.1010, 1.1030, 1.0998, 1.1000),  # down bar, body 1.1000-1.1010
                (1.0998, 1.1016, 1.0998, 1.1014),  # up bar, body 1.0998-1.1014
            )
        )
    }

    assert "bullish_engulfing" in hits
    assert hits["bullish_engulfing"].criteria["body_over_previous"] == pytest.approx(1.6)
    # The previous bar's tall upper wick keeps this from also being an outside bar, which is a
    # different formation about ranges rather than bodies.
    assert "outside_bar" not in hits


def test_bearish_engulfing_fires_when_a_down_body_covers_the_previous_up_body() -> None:
    names = names_at_latest(
        series(
            (1.1000, 1.1012, 1.0970, 1.1010),
            (1.1012, 1.1013, 1.0995, 1.0996),
        )
    )

    assert "bearish_engulfing" in names
    assert "bullish_engulfing" not in names


def test_engulfing_does_not_fire_when_the_engulfing_body_is_trivial() -> None:
    """A two-pip body swallowing a one-pip body in a dead market is not a reversal."""
    # Current body 0.0004, threshold 0.5 * 0.0020 = 0.0010.
    names = names_at_latest(
        series(
            (1.1002, 1.1012, 1.0998, 1.1000),
            (1.1000, 1.1013, 1.0997, 1.1004),
        )
    )

    assert "bullish_engulfing" not in names


def test_engulfing_needs_the_two_bars_to_disagree() -> None:
    """Two up bars, the second covering the first, is continuation and not an engulfing."""
    names = names_at_latest(
        series(
            (1.1000, 1.1012, 1.0998, 1.1004),
            (1.0998, 1.1020, 1.0996, 1.1016),
        )
    )

    assert "bullish_engulfing" not in names
    assert "bearish_engulfing" not in names


def test_inside_bar_fires_when_the_whole_range_sits_within_the_previous_one() -> None:
    hits = {
        hit.name: hit
        for hit in detect_latest(
            series(
                (1.1000, 1.1025, 1.0995, 1.1020),
                (1.1005, 1.1015, 1.1000, 1.1010),
            )
        )
    }

    assert "inside_bar" in hits
    assert hits["inside_bar"].criteria["high_room"] == pytest.approx(0.0010)
    assert hits["inside_bar"].criteria["low_room"] == pytest.approx(0.0005)


def test_inside_bar_does_not_fire_when_one_side_breaks_out() -> None:
    names = names_at_latest(
        series(
            (1.1000, 1.1025, 1.0995, 1.1020),
            (1.1005, 1.1030, 1.1000, 1.1010),
        )
    )

    assert "inside_bar" not in names


def test_inside_bar_does_not_fire_inside_a_bar_that_barely_moved() -> None:
    """Two quiet bars are not a compression."""
    # Previous span 0.0008, threshold 0.5 * 0.0020 = 0.0010.
    names = names_at_latest(
        series(
            (1.1000, 1.1004, 1.0996, 1.1002),
            (1.1001, 1.1003, 1.0997, 1.1002),
        )
    )

    assert "inside_bar" not in names


def test_outside_bar_fires_when_the_range_covers_the_previous_one_on_both_sides() -> None:
    names = names_at_latest(
        series(
            (1.1000, 1.1008, 1.0998, 1.1005),
            (1.1004, 1.1015, 1.0990, 1.1002),
        )
    )

    assert "outside_bar" in names
    assert "inside_bar" not in names


def test_outside_bar_does_not_fire_when_only_one_side_extends() -> None:
    names = names_at_latest(
        series(
            (1.1000, 1.1008, 1.0998, 1.1005),
            (1.1004, 1.1015, 1.0999, 1.1002),
        )
    )

    assert "outside_bar" not in names


def test_a_two_bar_formation_cannot_fire_on_the_first_bar_of_a_series() -> None:
    """There is no previous bar, and inventing one is how a scan reports a formation that
    depends on data it does not have."""
    from fxagent.patterns.candles import DETECTORS

    candle = Candle(open=1.1000, high=1.1020, low=1.0980, close=1.1015)
    for name, detector in DETECTORS:
        if name in {"inside_bar", "outside_bar", "bullish_engulfing", "bearish_engulfing"}:
            assert detector(candle, None, ATR, DEFAULT_CONFIG) is None, name


# -- warm-up and scanning --------------------------------------------------------------------


def test_no_formation_is_reported_before_atr_has_a_value() -> None:
    """Warm-up is reported as nothing. A scale of zero would make every bar a marubozu."""
    from datetime import timedelta

    from fxagent.adapters.base import Bar, BarSeries
    from tests.patterns.builders import MOMENT

    short = BarSeries(
        symbol="EURUSD",
        timeframe="H1",
        bars=tuple(
            Bar(
                timestamp=MOMENT - timedelta(hours=4 - index),
                open=1.1000,
                high=1.1018,
                low=1.1000,
                close=1.1018,
                volume=1_000,
            )
            for index in range(5)
        ),
    )

    assert detect(short) == ()


def test_a_scan_reports_hits_oldest_first_with_their_own_bar_index() -> None:
    bars = series(
        (1.1000, 1.1018, 1.1000, 1.1018),  # marubozu
        (1.1000, 1.1008, 1.0992, 1.10005),  # doji
    )

    hits = detect(bars, last=2)
    indices = [hit.bar_index for hit in hits]

    assert indices == sorted(indices)
    assert {hit.name for hit in hits if hit.bar_index == len(bars.bars) - 1} >= {"doji"}
    assert all(hit.timestamp == bars.bars[hit.bar_index].timestamp for hit in hits)


def test_limiting_the_window_never_changes_a_verdict() -> None:
    """The volatility scale is computed over the whole series, so `last` only trims output."""
    bars = series(
        (1.1000, 1.1018, 1.1000, 1.1018),
        (1.1000, 1.1008, 1.0992, 1.10005),
    )

    everything = detect(bars)
    trimmed = detect(bars, last=1)

    assert trimmed == tuple(hit for hit in everything if hit.bar_index == len(bars.bars) - 1)


def test_an_empty_series_is_not_an_error() -> None:
    from fxagent.adapters.base import BarSeries

    assert detect(BarSeries(symbol="EURUSD", timeframe="H1", bars=())) == ()


def test_detect_at_rejects_an_index_off_the_end() -> None:
    with pytest.raises(IndexError, match="outside a series"):
        detect_at(series((1.1000, 1.1018, 1.1000, 1.1018)), 9_999)


# -- the registry ----------------------------------------------------------------------------


def test_every_detector_has_a_definition_and_every_definition_a_detector() -> None:
    """A formation reaching the panel without a definition would print a bare name at a reader."""
    assert set(PATTERN_NAMES) == set(DEFINITIONS)
    assert len(PATTERN_NAMES) == 9


def test_no_definition_claims_a_formation_predicts_anything() -> None:
    """The evidence says they do not. A definition is where that claim would sneak back in."""
    forbidden = ("signal", "reversal", "predict", "bullish sign", "bearish sign", "buy", "sell")
    for name, definition in DEFINITIONS.items():
        lowered = definition.lower()
        for word in forbidden:
            assert word not in lowered, f"{name}'s definition says {word!r}"


def test_the_thresholds_refuse_a_configuration_that_makes_everything_a_pattern() -> None:
    with pytest.raises(ValueError, match="hammer_shadow_bodies"):
        PatternConfig(hammer_shadow_bodies=1.0)
    with pytest.raises(ValueError, match="pin_shadow_fraction"):
        PatternConfig(pin_shadow_fraction=1.5)
    with pytest.raises(ValueError, match="doji_body_atr"):
        PatternConfig(doji_body_atr=0.9, marubozu_body_atr=0.8)


def test_the_default_band_and_price_are_what_the_fixtures_assume() -> None:
    assert (BAND, PRICE) == (0.0010, 1.1000)
