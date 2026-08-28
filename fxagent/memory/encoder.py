"""Point-in-time encoder for the 48-bar analogue windows.

The encoder consumes only the supplied bars and an optional summary captured at the window's
last timestamp. It does not fetch indicators or look beyond the window, so the same prefix always
produces the same vector.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from fxagent.adapters.base import BarSeries
from fxagent.memory.window_spec import BARS_PER_SEGMENT, SUMMARY_FIELDS, WINDOW_BARS
from fxagent.regime.sessions import Session, active_sessions, dominant_session
from fxagent.store.schema import EMBEDDING_DIMENSIONS

__all__ = ["encode_window"]


def _zscore(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return [0.0] * len(values) if scale == 0 else [(value - mean) / scale for value in values]


def _paa(values: list[float]) -> list[float]:
    return [
        sum(values[start : start + BARS_PER_SEGMENT]) / BARS_PER_SEGMENT
        for start in range(0, WINDOW_BARS, BARS_PER_SEGMENT)
    ]


def encode_window(
    series: BarSeries,
    *,
    summary: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    """Encode exactly the trailing 48 H1 bars into the schema's 128 dimensions."""
    if series.timeframe != "H1":
        raise ValueError(f"window timeframe must be H1, got {series.timeframe!r}")
    if len(series.bars) != WINDOW_BARS:
        raise ValueError(f"window must contain exactly {WINDOW_BARS} bars, got {len(series.bars)}")

    bars = list(series.bars)
    closes = [bar.close for bar in bars]
    ranges: list[float] = []
    volumes = [float(bar.volume) for bar in bars]
    previous = bars[0].open
    for bar in bars:
        ranges.append(max(bar.high - bar.low, abs(bar.high - previous), abs(bar.low - previous)))
        previous = bar.close

    vector = _paa(_zscore(closes)) + _paa(_zscore(ranges)) + _paa(_zscore(volumes))
    values = dict(summary or {})
    dominant = dominant_session(active_sessions(bars[-1].timestamp))
    for session in (Session.TOKYO, Session.LONDON, Session.NEW_YORK, Session.OVERLAP):
        values[f"session_{session.value.lower()}"] = float(dominant is session)
    vector.extend(float(values.get(field, 0.0)) for field in SUMMARY_FIELDS)
    vector.extend([0.0] * (EMBEDDING_DIMENSIONS - len(vector)))
    return tuple(vector)
