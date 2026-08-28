"""Cross-feed divergence maths — the part of the smoke test that can be quietly wrong."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.adapters.base import Bar, BarSeries
from fxagent.adapters.divergence import (
    Divergence,
    compare_series,
    gap_filler_verdict,
    interpret,
    pip_size,
)

BASE = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)


def _series(
    *, offset_pips: float = 0.0, shift_hours: int = 0, count: int = 5, jpy: bool = False
) -> BarSeries:
    close = 157.50 if jpy else 1.0850
    pip = 0.01 if jpy else 0.0001
    return BarSeries(
        symbol="USDJPY" if jpy else "EURUSD",
        timeframe="H1",
        bars=tuple(
            Bar(
                timestamp=BASE + timedelta(hours=i + shift_hours),
                open=close,
                high=close + 50 * pip,
                low=close - 50 * pip,
                close=close + offset_pips * pip,
                volume=0,
            )
            for i in range(count)
        ),
    )


def test_identical_feeds_show_no_divergence() -> None:
    result = compare_series("a", _series(), "b", _series())

    assert result.overlapping == 5
    assert result.mean_abs_pips == 0.0
    assert result.max_abs_pips == 0.0


def test_a_constant_offset_is_reported_in_pips() -> None:
    result = compare_series("a", _series(), "b", _series(offset_pips=2.0))

    # approx, not exact: the offset is built by float arithmetic on the price.
    assert result.mean_abs_pips == pytest.approx(2.0)
    assert result.max_abs_pips == pytest.approx(2.0)


def test_jpy_pairs_use_the_larger_pip() -> None:
    """0.01, not 0.0001 — otherwise every JPY comparison reads 100x too divergent."""
    assert pip_size(157.5) == 0.01
    assert pip_size(1.085) == 0.0001

    result = compare_series("a", _series(jpy=True), "b", _series(jpy=True, offset_pips=3.0))
    assert result.max_abs_pips == pytest.approx(3.0)


def test_comparison_aligns_on_timestamp_not_position() -> None:
    """Index-to-index would report a huge divergence for what is only a one-bar offset."""
    result = compare_series("a", _series(count=5), "b", _series(count=5, shift_hours=2))

    assert result.overlapping == 3
    assert result.max_abs_pips == 0.0


def test_no_shared_timestamps_is_reported_rather_than_averaged() -> None:
    result = compare_series("a", _series(count=2), "b", _series(count=2, shift_hours=50))

    assert result.overlapping == 0
    assert result.worst_at is None
    assert "NO OVERLAP" in interpret(result)


def test_same_broker_feeds_are_held_to_a_tighter_standard() -> None:
    """MT5 and MetaApi read the same book; a pip apart means an adapter bug."""
    loose = Divergence("mt5", "metaapi", 10, 2.0, 3.0, BASE)
    assert "UNEXPECTED" in interpret(loose)

    tight = Divergence("mt5", "metaapi", 10, 0.1, 0.4, BASE)
    assert "within expectations" in interpret(tight)


def test_independent_feeds_tolerate_a_spread_sized_difference() -> None:
    """Twelve Data quotes mid against a broker's bid; a few pips is the expected result."""
    normal = Divergence("mt5", "twelvedata", 10, 1.5, 4.0, BASE)
    assert "within expectations" in interpret(normal)

    absurd = Divergence("mt5", "twelvedata", 10, 60.0, 120.0, BASE)
    assert "SUSPICIOUS" in interpret(absurd)


def test_the_order_of_the_pair_does_not_change_the_verdict() -> None:
    left = Divergence("mt5", "metaapi", 10, 2.0, 3.0, BASE)
    right = Divergence("metaapi", "mt5", 10, 2.0, 3.0, BASE)

    assert interpret(left) == interpret(right)


def test_render_states_both_the_mean_and_the_worst_bar() -> None:
    rendered = Divergence("a", "b", 10, 1.25, 4.5, BASE).render()

    assert "1.25 pips" in rendered
    assert "4.50 pips" in rendered
    assert "2026-03-11 12:00 UTC" in rendered


def test_report_includes_ohlc_distributions_and_barrier_flip_share() -> None:
    left = _series(count=2)
    right = BarSeries(
        symbol="EURUSD",
        timeframe="H1",
        bars=(
            Bar(
                timestamp=BASE,
                open=1.0850,
                high=1.0900,
                low=1.0840,
                close=1.0850,
                volume=0,
            ),
            Bar(
                timestamp=BASE + timedelta(hours=1),
                open=1.0850,
                high=1.0900,
                low=1.0840,
                close=1.0850,
                volume=0,
            ),
        ),
    )
    result = compare_series("mt5", left, "twelvedata", right, barrier_pips=(5, 20))

    assert len(result.ohlc_mean_abs_pips) == 4
    assert len(result.ohlc_p95_abs_pips) == 4
    assert result.barrier_touch_flip_share == pytest.approx(0.0)
    assert "mean OHLC" in result.render()


def test_gap_filler_verdict_rejects_barrier_flips() -> None:
    divergence = Divergence(
        "mt5",
        "twelvedata",
        10,
        2.0,
        3.0,
        BASE,
        barrier_touch_flip_share=0.2,
    )
    assert gap_filler_verdict(divergence).startswith("UNUSABLE")
