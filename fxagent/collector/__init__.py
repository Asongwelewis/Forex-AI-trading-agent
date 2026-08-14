"""The collector service: fetch bars, write bars, prove it was running.

Deliberately the dumbest component in the system. It never computes an indicator, evaluates a
strategy, or calls a model — a test enforces that by inspecting this package's imports.

The reason is asymmetry of loss. Analysis can be re-run over stored data any number of times;
a collection window that was missed is gone for good. So this process is kept small enough that
there is very little in it that can break, and it is the one that stays up.
"""

from __future__ import annotations

from fxagent.collector.gaps import Gap, find_gaps, is_market_open
from fxagent.collector.service import (
    SERVICE_NAME,
    CollectorConfig,
    CollectorService,
    CollectorStats,
    DataSource,
)

__all__ = [
    "SERVICE_NAME",
    "CollectorConfig",
    "CollectorService",
    "CollectorStats",
    "DataSource",
    "Gap",
    "find_gaps",
    "is_market_open",
]
