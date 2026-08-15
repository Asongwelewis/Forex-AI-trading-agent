"""A `DashboardSource` that holds rows in memory.

The seam exists so the panel's behaviour can be asserted without Postgres. Everything the real
`StoreSource` does beyond this — resolving which provider's bars to draw, deriving the
evaluation window from the bars returned — is repository work the store suite already covers.
"""

from __future__ import annotations

from fxagent.adapters.base import BarSeries
from fxagent.dashboard.models import SeriesOption
from fxagent.dashboard.source import ViewData, ViewRequest
from fxagent.store.repositories.evaluations import EvaluationRecord
from fxagent.store.repositories.trades import TradeRecord
from tests.dashboard.builders import bar_series

__all__ = ["FailingSource", "StubSource"]


class StubSource:
    """Returns whatever it was last given. Mutate the attributes to simulate a new row."""

    def __init__(
        self,
        *,
        bars: BarSeries | None = None,
        evaluations: tuple[EvaluationRecord, ...] = (),
        trades: tuple[TradeRecord, ...] = (),
        options: tuple[SeriesOption, ...] | None = None,
        source: str = "twelvedata",
    ) -> None:
        self.bars = bars if bars is not None else bar_series(count=48)
        self.evaluations = evaluations
        self.trades = trades
        self.source = source
        self.options_returned = (
            options
            if options is not None
            else (
                SeriesOption(
                    symbol="EURUSD",
                    timeframe="H1",
                    source=source,
                    bars=len(self.bars),
                    latest=self.bars.bars[-1].timestamp.isoformat() if self.bars.bars else None,
                ),
                SeriesOption(symbol="GBPUSD", timeframe="H1", source=source, bars=10),
            )
        )
        self.loads = 0

    async def options(self) -> tuple[SeriesOption, ...]:
        return self.options_returned

    async def load(self, request: ViewRequest) -> ViewData:
        self.loads += 1
        return ViewData(
            request=request,
            bars=(
                self.bars
                if request.symbol == self.bars.symbol
                else self.bars.model_copy(update={"symbol": request.symbol})
            ),
            source=self.source,
            evaluations=self.evaluations,
            trades=self.trades,
            options=self.options_returned,
        )


class FailingSource:
    """Every read raises. Used to prove the panel degrades instead of disappearing."""

    def __init__(self, message: str = "connection refused") -> None:
        self.message = message

    async def options(self) -> tuple[SeriesOption, ...]:
        raise RuntimeError(self.message)

    async def load(self, request: ViewRequest) -> ViewData:
        raise RuntimeError(self.message)
