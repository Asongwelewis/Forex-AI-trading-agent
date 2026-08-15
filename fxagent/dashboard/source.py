"""Where the panel's rows come from, behind a seam the tests can stand in for.

`DashboardSource` is the only thing in the dashboard that touches a database. Everything
downstream — the chart builder, the feed builder, the socket, the routes — takes `ViewData` and
is therefore testable without Postgres, which is the difference between a UI whose behaviour is
asserted and one that is only ever eyeballed.

**The dashboard reads and never writes.** Only `select` statements are reachable from here, no
repository method that mutates is called, and `Database.session()` is used rather than
`begin()`, so no transaction is even open to commit. That is not an accident of the current
routes: it is the property that makes it safe to expose this process on a LAN, which
`tests/dashboard/test_app.py` asserts by checking the app declares no mutating route at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from fxagent.adapters.base import BarSeries
from fxagent.dashboard.models import SeriesOption
from fxagent.store.engine import Database
from fxagent.store.repositories import (
    BarRepository,
    EvaluationRecord,
    EvaluationRepository,
    TradeRecord,
    TradeRepository,
)

__all__ = ["DashboardSource", "StoreSource", "ViewData", "ViewRequest", "utc_now"]

logger = logging.getLogger(__name__)

#: How many bars the chart asks for by default. Enough to see a full week of H1 with the
#: hundred-bar warm-up the classifier needs already behind the left edge.
DEFAULT_BAR_COUNT = 400
#: How many evaluations the feed shows. One London session is 4-5 hours of H1 per symbol, so
#: this is comfortably a week of them.
DEFAULT_FEED_LIMIT = 120

MAX_BAR_COUNT = 5000
MAX_FEED_LIMIT = 500


@dataclass(frozen=True)
class ViewRequest:
    """One (symbol, timeframe) view, with its sizes clamped where they are read.

    Clamped rather than validated-and-rejected: these arrive from a query string, and a panel
    that 400s because someone typed `bars=99999` into the address bar is less useful than one
    that shows five thousand bars and says so.
    """

    symbol: str
    timeframe: str = "H1"
    #: None means "whichever source has bars for this series", resolved by the source.
    source: str | None = None
    bars: int = DEFAULT_BAR_COUNT
    feed_limit: int = DEFAULT_FEED_LIMIT

    def clamped(self) -> ViewRequest:
        return ViewRequest(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source=self.source,
            bars=max(1, min(self.bars, MAX_BAR_COUNT)),
            feed_limit=max(1, min(self.feed_limit, MAX_FEED_LIMIT)),
        )

    @property
    def key(self) -> str:
        """Identity for socket subscriptions. Two clients on the same view share one rebuild."""
        return f"{self.symbol}|{self.timeframe}|{self.source or '*'}|{self.bars}|{self.feed_limit}"


@dataclass(frozen=True)
class ViewData:
    """Everything one view needs, already read. The builders take this and nothing else."""

    request: ViewRequest
    bars: BarSeries
    source: str
    evaluations: tuple[EvaluationRecord, ...] = ()
    trades: tuple[TradeRecord, ...] = ()
    options: tuple[SeriesOption, ...] = ()
    notes: tuple[str, ...] = ()


class DashboardSource(Protocol):
    """Reads the store. Implemented by `StoreSource` and by the stub in the tests."""

    async def options(self) -> tuple[SeriesOption, ...]: ...

    async def load(self, request: ViewRequest) -> ViewData: ...


class StoreSource:
    """Reads Supabase through the existing repositories. No SQL of its own."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def options(self) -> tuple[SeriesOption, ...]:
        async with self._database.session() as session:
            summaries = await BarRepository(session).available_series()
        return tuple(
            SeriesOption(
                symbol=summary.symbol,
                timeframe=summary.timeframe,
                source=summary.source,
                bars=summary.bars,
                latest=summary.latest.isoformat() if summary.latest else None,
            )
            for summary in summaries
        )

    async def load(self, request: ViewRequest) -> ViewData:
        """Read the bars, the evaluations covering them, and the trades open across them.

        The evaluation and trade windows are derived from the bars actually returned rather
        than from a fixed lookback, so the feed always describes the stretch of time the chart
        is showing. Asking for a different number of bars moves both together.
        """
        wanted = request.clamped()
        options = await self.options()
        source = _resolve_source(wanted, options)

        if source is None:
            return ViewData(
                request=wanted,
                bars=BarSeries(symbol=wanted.symbol, timeframe=wanted.timeframe, bars=()),
                source=wanted.source or "",
                options=options,
                notes=(
                    f"No stored bars for {wanted.symbol} {wanted.timeframe}"
                    + (f" from {wanted.source}." if wanted.source else "."),
                ),
            )

        async with self._database.session() as session:
            bars = await BarRepository(session).latest_bars(
                wanted.symbol, wanted.timeframe, wanted.bars, source=source
            )
            if not bars.bars:
                return ViewData(
                    request=wanted,
                    bars=bars,
                    source=source,
                    options=options,
                    notes=("The series exists but returned no bars for this window.",),
                )

            window_start = bars.bars[0].timestamp
            window_end = bars.bars[-1].timestamp

            evaluations = await EvaluationRepository(session).recent(
                symbol=wanted.symbol, since=window_start, limit=wanted.feed_limit
            )
            trades = await TradeRepository(session).open_during(
                # The chart's right edge is the last bar's OPEN time, so a trade entered inside
                # that final, still-forming bar would fall outside a window ending there. The
                # padding is one bar's worth of slack rather than a guess.
                start=window_start,
                end=window_end + timedelta(days=1),
                symbol=wanted.symbol,
            )

        return ViewData(
            request=wanted,
            bars=bars,
            source=source,
            evaluations=tuple(evaluations),
            trades=tuple(trades),
            options=options,
        )


def _resolve_source(request: ViewRequest, options: tuple[SeriesOption, ...]) -> str | None:
    """Which source's bars to draw, given that a bar's identity includes its source.

    An unfiltered read would interleave two providers' prices into one series, which
    `BarSeries` rejects outright — so the choice has to be made, and making it explicitly here
    beats making it implicitly in a query. Named source wins; otherwise the one with the most
    bars, which is the series a human opening the page meant.
    """
    candidates = [
        option
        for option in options
        if option.symbol == request.symbol and option.timeframe == request.timeframe
    ]
    if request.source is not None:
        candidates = [option for option in candidates if option.source == request.source]
    if not candidates:
        return None
    return max(candidates, key=lambda option: (option.bars, option.source)).source


def utc_now() -> datetime:
    """The clock, in one place, so a test can patch one function instead of a module."""
    return datetime.now(UTC)
