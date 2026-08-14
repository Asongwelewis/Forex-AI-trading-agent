"""The window encoding spec. This is what makes `vector(128)` a consequence rather than a guess.

A window is **48 H1 bars** — two trading days, so a session is always seen against the day
before it rather than in isolation.

Those 48 bars are reduced to **16 segments by piecewise aggregate approximation** (PAA: the mean
of each consecutive group of 3). Retrieval is meant to find windows of the same *shape*, and at
bar-level resolution two structurally identical sessions differ everywhere by noise, so cosine
distance ends up measuring the noise. Averaging into segments is what makes "the same shape"
the dominant term.

Layout::

     0..15   normalised close        z-scored within the window
    16..31   normalised true range
    32..47   normalised volume
    48..55   summary: ADX, ATR percentile, session one-hot (4), spread percentile
    ------
        56   used
    56..127  padding, always zero

**The padding is deliberate.** Adding a feature later writes into padding and leaves every
stored embedding still comparable — cosine distance is unaffected by shared zero components. The
alternative is `alter table ... type vector(M)` plus re-encoding every row, and re-encoding means
recomputing indicators over history that may no longer be available at the same resolution.

Each block is normalised **within its own window**, never across the corpus. A z-score against a
global mean would leak the future into every historical window — the corpus includes bars that
had not happened yet at the time the window closes (hard rule 6).

This module fixes the layout only. The encoder that fills it lives with the rest of `memory/`
and needs the session clock from `fxagent.regime`, which does not exist yet: the session one-hot
at offsets 50..53 cannot be computed without it.
"""

from __future__ import annotations

from dataclasses import dataclass

from fxagent.store.schema import EMBEDDING_DIMENSIONS

__all__ = [
    "BARS_PER_SEGMENT",
    "PADDING_DIMENSIONS",
    "SEGMENTS",
    "SUMMARY_FIELDS",
    "USED_DIMENSIONS",
    "WINDOW_BARS",
    "WINDOW_TIMEFRAME",
    "Block",
    "BLOCKS",
    "block_for",
    "summary_index",
]

#: 48 H1 bars — two trading days: a session plus the prior day for context.
WINDOW_BARS = 48
WINDOW_TIMEFRAME = "H1"

#: PAA target. 48 bars / 16 segments = 3 bars averaged per segment.
SEGMENTS = 16
BARS_PER_SEGMENT = WINDOW_BARS // SEGMENTS

#: Scalars in the summary block, in slot order. The four session flags are one-hot: exactly one
#: is 1.0, so "Tokyo" and "London" stay equidistant instead of being ordered by an integer code.
SUMMARY_FIELDS = (
    "adx",
    "atr_percentile",
    "session_tokyo",
    "session_london",
    "session_new_york",
    "session_overlap",
    "spread_percentile",
)


@dataclass(frozen=True)
class Block:
    """One contiguous run of dimensions with a single meaning."""

    name: str
    start: int
    length: int

    @property
    def stop(self) -> int:
        return self.start + self.length

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


def _lay_out() -> tuple[Block, ...]:
    """Pack the blocks back to back, so an inserted block cannot silently overlap its neighbour."""
    sizes = (
        ("close_z", SEGMENTS),
        ("true_range_z", SEGMENTS),
        ("volume_z", SEGMENTS),
        ("summary", 8),
    )
    blocks: list[Block] = []
    cursor = 0
    for name, length in sizes:
        blocks.append(Block(name=name, start=cursor, length=length))
        cursor += length
    return tuple(blocks)


BLOCKS = _lay_out()

#: 16 + 16 + 16 + 8 = 56.
USED_DIMENSIONS = sum(block.length for block in BLOCKS)

#: Headroom for features added later, without a re-encode.
PADDING_DIMENSIONS = EMBEDDING_DIMENSIONS - USED_DIMENSIONS


def block_for(name: str) -> Block:
    """The block with this name, or a clear error listing the ones that exist."""
    for block in BLOCKS:
        if block.name == name:
            return block
    raise KeyError(f"unknown block {name!r}; expected one of {[b.name for b in BLOCKS]}")


def summary_index(field: str) -> int:
    """Absolute index of a named summary scalar."""
    try:
        offset = SUMMARY_FIELDS.index(field)
    except ValueError:
        raise KeyError(
            f"unknown summary field {field!r}; expected one of {list(SUMMARY_FIELDS)}"
        ) from None
    return block_for("summary").start + offset
