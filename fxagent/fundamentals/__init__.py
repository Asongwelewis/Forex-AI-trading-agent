"""Fundamental inputs: the economic calendar, central bank communications, and the context
assembled from them.

Every source implements `FundamentalSource` and every one of them fails soft — an unreachable
feed logs a warning and yields nothing, because fundamentals are context and an analysis cycle
that refuses to run without them is worse than one that runs with less.

Reads go through `EventRepository`, which gates on `publication_time_utc` in SQL. The sources'
job is to stamp that field honestly; see `calendar.py` for why a feed with no publication
metadata gets the fetch time and not the event date.
"""

from __future__ import annotations

from fxagent.fundamentals.base import (
    IMPORTANCE_VALUES,
    Event,
    FundamentalSource,
    fetch_all,
    fetch_safely,
)
from fxagent.fundamentals.calendar import (
    CALENDAR_URL,
    ForexFactoryCalendar,
    parse_value,
    surprise_score,
)
from fxagent.fundamentals.centralbank import DEFAULT_FEEDS, CentralBankFeed, CentralBankRss
from fxagent.fundamentals.context import (
    FundamentalContext,
    PolicyRate,
    build_context,
    rate_differential,
    split_symbol,
)

# Imported last: `cot` reaches back into `context` for `split_symbol`, so `context` must be
# fully initialised before this line runs.
from fxagent.fundamentals.cot import (  # noqa: E402 - see the comment above
    CONTRACT_CODES,
    COT_DATASET_URL,
    CftcCotSource,
    CotHistory,
    CotReport,
    PairPositioning,
    percentile_rank,
    publication_time,
)

__all__ = [
    "CALENDAR_URL",
    "CONTRACT_CODES",
    "COT_DATASET_URL",
    "DEFAULT_FEEDS",
    "IMPORTANCE_VALUES",
    "CentralBankFeed",
    "CentralBankRss",
    "CftcCotSource",
    "CotHistory",
    "CotReport",
    "Event",
    "ForexFactoryCalendar",
    "FundamentalContext",
    "FundamentalSource",
    "PairPositioning",
    "PolicyRate",
    "build_context",
    "fetch_all",
    "fetch_safely",
    "parse_value",
    "percentile_rank",
    "publication_time",
    "rate_differential",
    "split_symbol",
    "surprise_score",
]
