"""Window encoding, pgvector retrieval, and the point-in-time rules governing both.

`window_spec` fixes the encoding layout and is what makes the `vector(128)` in the schema a
consequence of a specification rather than a guess. The encoder that fills that layout is not
here yet — it needs the session clock from `fxagent.regime` for the session one-hot.

Retrieval itself lives in `fxagent.store.repositories.windows`, where it can enforce the
resolved-outcome filter next to the query it applies to.
"""

from __future__ import annotations

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
]
