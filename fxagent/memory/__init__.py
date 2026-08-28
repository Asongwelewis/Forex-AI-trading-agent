"""Window encoding, pgvector retrieval, and point-in-time rules governing both.

``window_spec`` fixes the vector(128) layout and ``encoder.encode_window`` fills it from a
bounded, point-in-time bar window. Retrieval lives in the windows repository, where the
resolved-outcome filter is enforced next to its query.
"""

from __future__ import annotations

from fxagent.memory.encoder import encode_window
from fxagent.memory.window_spec import (
    BARS_PER_SEGMENT,
    BLOCKS,
    PADDING_DIMENSIONS,
    SEGMENTS,
    SUMMARY_FIELDS,
    USED_DIMENSIONS,
    WINDOW_BARS,
    WINDOW_TIMEFRAME,
    Block,
    block_for,
    summary_index,
)

__all__ = [
    "BARS_PER_SEGMENT",
    "BLOCKS",
    "PADDING_DIMENSIONS",
    "SEGMENTS",
    "SUMMARY_FIELDS",
    "USED_DIMENSIONS",
    "WINDOW_BARS",
    "WINDOW_TIMEFRAME",
    "Block",
    "block_for",
    "summary_index",
    "encode_window",
]
