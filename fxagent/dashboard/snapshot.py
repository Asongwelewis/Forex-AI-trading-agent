"""Assemble one complete view and stamp it with a content hash.

The hash is what makes the socket quiet. The refresh loop rebuilds a snapshot on a timer and
compares `revision` against the last one it sent; equal means nothing has changed and nothing
is pushed. Without it, "push instead of poll" would be a poll with the direction reversed —
every client woken every few seconds to redraw a chart that has not moved.

`generated_at` is therefore deliberately outside the hash. It changes on every rebuild by
definition, and including it would make every revision unique and the comparison useless.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from fxagent.dashboard.chart import ChartConfig, build_chart
from fxagent.dashboard.feed import build_feed
from fxagent.dashboard.models import ChartPayload, FeedPayload, GrantSnapshot, Snapshot
from fxagent.dashboard.source import ViewData

__all__ = ["build_snapshot", "revision_of"]


def revision_of(chart: ChartPayload, feed: FeedPayload) -> str:
    """A short content hash over everything the browser would redraw.

    Truncated to 16 hex characters: this is a change *detector*, not a signature, and the only
    consequence of a collision is one skipped repaint of a panel that repaints every few
    seconds anyway.
    """
    digest = hashlib.sha256()
    digest.update(chart.model_dump_json().encode("utf-8"))
    digest.update(feed.model_dump_json().encode("utf-8"))
    return digest.hexdigest()[:16]


def build_snapshot(
    data: ViewData,
    grant: GrantSnapshot,
    *,
    now: datetime,
    config: ChartConfig | None = None,
    error: str | None = None,
) -> Snapshot:
    """Chart plus feed plus permission, for one (symbol, timeframe), as one pushable object."""
    chart = build_chart(
        data.bars,
        source=data.source,
        evaluations=data.evaluations,
        trades=data.trades,
        config=config,
    )
    feed = build_feed(data.request.symbol, data.evaluations, data.trades, grant)

    if data.notes:
        chart = chart.model_copy(update={"notes": chart.notes + data.notes})

    return Snapshot(
        symbol=data.request.symbol,
        timeframe=data.request.timeframe,
        revision=revision_of(chart, feed),
        generated_at=now.isoformat(),
        chart=chart,
        feed=feed,
        options=data.options,
        error=error,
    )
