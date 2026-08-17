"""Thresholds are multiples of ATR, never price distances. This is the test that proves it.

A ten-pip body is a doji on EUR/USD in a dead August hour and an ordinary bar on GBP/JPY at the
New York open. A detector holding a fixed cut-off would agree with itself on one pair and be
wrong on the next, and nobody would notice until the panel started reporting a doji on every
bar of a volatile session.

Two directions, both needed:

* Scale a whole fixture — bars *and* the volatility around them — and the verdicts must not
  move. A fixed pip threshold fails this the moment the factor is 10.
* Hold one bar's shape fixed and change only the surrounding volatility, and the verdicts
  *must* move. Otherwise the ATR is being computed and ignored, which passes the first test
  perfectly.
"""

from __future__ import annotations

import pytest

from fxagent.patterns import detect_latest
from tests.patterns.builders import BAND, PRICE, series

#: One fixture per formation, so the scaling claim is made about all of them rather than about
#: whichever one happened to be convenient.
SHAPES: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "doji": ((1.1000, 1.1008, 1.0992, 1.10005),),
    "hammer": ((1.1005, 1.1009, 1.0995, 1.1008),),
    "shooting_star": ((1.0995, 1.1005, 1.0991, 1.0992),),
    "marubozu": ((1.1000, 1.1018, 1.1000, 1.1018),),
    "pin_bar": ((1.1000, 1.1002, 1.0985, 1.10005),),
    "bullish_engulfing": (
        (1.1010, 1.1030, 1.0998, 1.1000),
        (1.0998, 1.1016, 1.0998, 1.1014),
    ),
    "bearish_engulfing": (
        (1.1000, 1.1012, 1.0970, 1.1010),
        (1.1012, 1.1013, 1.0995, 1.0996),
    ),
    "inside_bar": (
        (1.1000, 1.1025, 1.0995, 1.1020),
        (1.1005, 1.1015, 1.1000, 1.1010),
    ),
    "outside_bar": (
        (1.1000, 1.1008, 1.0998, 1.1005),
        (1.1004, 1.1015, 1.0990, 1.1002),
    ),
}

#: Ten times the volatility, and a tenth of it. Both, because a threshold accidentally
#: expressed as a floor behaves differently from one accidentally expressed as a ceiling.
FACTORS = (0.1, 1.0, 10.0, 250.0)


def _rescaled(
    tail: tuple[tuple[float, float, float, float], ...], factor: float
) -> tuple[tuple[float, float, float, float], ...]:
    """Every distance from `PRICE` multiplied by `factor`, shapes and proportions untouched."""
    return tuple(
        (
            PRICE + (open_ - PRICE) * factor,
            PRICE + (high - PRICE) * factor,
            PRICE + (low - PRICE) * factor,
            PRICE + (close - PRICE) * factor,
        )
        for open_, high, low, close in tail
    )


def _names(tail: tuple[tuple[float, float, float, float], ...], factor: float) -> set[str]:
    bars = series(*_rescaled(tail, factor), band=BAND * factor)
    return {hit.name for hit in detect_latest(bars)}


@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("formation", sorted(SHAPES))
def test_the_same_shape_is_detected_at_every_volatility(formation: str, factor: float) -> None:
    assert formation in _names(SHAPES[formation], factor), (
        f"{formation} was not detected with the whole fixture scaled by {factor}. A threshold "
        "expressed as a price distance rather than as a multiple of ATR fails exactly here."
    )


@pytest.mark.parametrize("formation", sorted(SHAPES))
def test_scaling_changes_nothing_about_which_formations_are_present(formation: str) -> None:
    """Not merely that the target still fires — that the whole verdict is identical.

    A detector could keep firing under scaling while a *different* one started or stopped, and
    a test that only checked its own formation would call that a pass.
    """
    baseline = _names(SHAPES[formation], 1.0)

    for factor in FACTORS:
        assert _names(SHAPES[formation], factor) == baseline, (
            f"scaling by {factor} changed the formations present from {baseline}"
        )


def test_the_same_bar_is_a_doji_only_when_the_market_around_it_is_volatile() -> None:
    """The other direction. Identical bar, different surrounding ATR, different verdict.

    Body 0.0003. Against ATR 0.0020 the doji cut is 0.0002 and this is an ordinary small bar;
    against ATR 0.0100 the cut is 0.0010 and the same bar is a doji. Without this, a detector
    that computed ATR and then ignored it would pass every test above.
    """
    shape = (1.1000, 1.1035, 1.0965, 1.1003)

    quiet = {hit.name for hit in detect_latest(series(shape, band=0.0010))}
    volatile = {hit.name for hit in detect_latest(series(shape, band=0.0050))}

    assert "doji" not in quiet
    assert "doji" in volatile


def test_the_criteria_report_the_scale_they_were_measured_against() -> None:
    """Which is what makes a stored detection re-checkable rather than something to trust."""
    for factor in (1.0, 10.0):
        hits = detect_latest(series(*_rescaled(SHAPES["hammer"], factor), band=BAND * factor))
        hammer = next(hit for hit in hits if hit.name == "hammer")

        assert hammer.criteria["atr"] == pytest.approx(2 * BAND * factor)
        # The shape is unchanged, so the ratio is too — only the absolute numbers moved.
        assert hammer.criteria["dominant_shadow_in_bodies"] == pytest.approx(10 / 3)
