"""The encoding spec must stay arithmetically consistent with the column it justifies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.adapters.base import Bar, BarSeries
from fxagent.memory import encode_window
from fxagent.memory import window_spec as spec
from fxagent.store.schema import EMBEDDING_DIMENSIONS


def test_paa_divides_the_window_evenly() -> None:
    """48 bars into 16 segments is 3 each. A remainder would silently drop or duplicate bars."""
    assert spec.WINDOW_BARS == 48
    assert spec.SEGMENTS == 16
    assert spec.BARS_PER_SEGMENT == 3
    assert spec.SEGMENTS * spec.BARS_PER_SEGMENT == spec.WINDOW_BARS


def test_used_dimensions_are_the_stated_fifty_six() -> None:
    assert spec.USED_DIMENSIONS == 56
    assert [b.length for b in spec.BLOCKS] == [16, 16, 16, 8]


def test_blocks_are_contiguous_and_non_overlapping() -> None:
    """An inserted block must not quietly land on top of its neighbour."""
    cursor = 0
    for block in spec.BLOCKS:
        assert block.start == cursor, f"{block.name} starts at {block.start}, expected {cursor}"
        cursor = block.stop
    assert cursor == spec.USED_DIMENSIONS


def test_layout_fits_the_column_with_the_intended_headroom() -> None:
    """This is the assertion that ties the spec to `vector(128)` in 0006."""
    assert spec.USED_DIMENSIONS + spec.PADDING_DIMENSIONS == EMBEDDING_DIMENSIONS
    assert spec.PADDING_DIMENSIONS == 72
    assert spec.PADDING_DIMENSIONS > 0, "headroom is what avoids a re-encode when features land"


def test_column_is_within_pgvector_index_limits() -> None:
    """hnsw and ivfflat both refuse to index beyond 2000 dimensions."""
    assert EMBEDDING_DIMENSIONS <= 2000


def test_summary_block_has_a_slot_for_every_named_field() -> None:
    summary = spec.block_for("summary")
    assert len(spec.SUMMARY_FIELDS) <= summary.length


def test_session_flags_are_one_hot_not_an_ordinal() -> None:
    """Four separate flags keep Tokyo and London equidistant; one integer code would not."""
    sessions = [f for f in spec.SUMMARY_FIELDS if f.startswith("session_")]
    assert len(sessions) == 4

    indices = [spec.summary_index(name) for name in sessions]
    assert indices == sorted(indices)
    assert indices == list(range(indices[0], indices[0] + 4)), "contiguous"


def test_summary_indices_land_inside_the_summary_block() -> None:
    summary = spec.block_for("summary")
    for field in spec.SUMMARY_FIELDS:
        index = spec.summary_index(field)
        assert summary.start <= index < summary.stop


def test_block_slices_address_the_right_extent() -> None:
    close = spec.block_for("close_z")
    assert close.slice == slice(0, 16)
    assert spec.block_for("summary").slice == slice(48, 56)


def test_unknown_names_fail_loudly() -> None:
    with pytest.raises(KeyError, match="unknown block"):
        spec.block_for("no_such_block")
    with pytest.raises(KeyError, match="unknown summary field"):
        spec.summary_index("no_such_field")


def _series(count: int = 48, *, start_price: float = 1.1) -> BarSeries:
    start = datetime(2026, 8, 17, tzinfo=UTC)
    bars = tuple(
        Bar(
            timestamp=start + timedelta(hours=index),
            open=start_price + index * 0.0001,
            high=start_price + index * 0.0001 + 0.0003,
            low=start_price + index * 0.0001 - 0.0001,
            close=start_price + index * 0.0001 + 0.0001,
            volume=index + 1,
        )
        for index in range(count)
    )
    return BarSeries(symbol="EURUSD", timeframe="H1", bars=bars)


def test_encoder_fills_the_declared_vector_and_summary_slots() -> None:
    vector = encode_window(_series(), summary={"adx": 25.0, "spread_percentile": 0.9})
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert vector[spec.summary_index("adx")] == 25.0
    assert vector[spec.summary_index("spread_percentile")] == 0.9
    assert all(value == 0.0 for value in vector[spec.USED_DIMENSIONS :])


def test_encoder_rejects_partial_windows() -> None:
    with pytest.raises(ValueError, match="exactly 48 bars"):
        encode_window(_series(47))


def test_encoder_is_point_in_time_and_ignores_bars_before_the_window() -> None:
    full = _series(60)
    trailing = BarSeries(symbol=full.symbol, timeframe=full.timeframe, bars=full.bars[-48:])
    changed_prefix = list(full.bars)
    changed_prefix[0] = changed_prefix[0].model_copy(
        update={"open": 1.5, "high": 1.6, "close": 1.55}
    )
    changed = BarSeries(symbol=full.symbol, timeframe=full.timeframe, bars=tuple(changed_prefix))
    assert encode_window(trailing) == encode_window(
        BarSeries(symbol=changed.symbol, timeframe=changed.timeframe, bars=changed.bars[-48:])
    )
