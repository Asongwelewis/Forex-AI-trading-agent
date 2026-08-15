"""The push side, and the claim that makes it push rather than poll.

`test_an_unchanged_view_pushes_nothing` is the one that matters. Without it "WebSocket instead
of polling" would be a claim about which protocol is used rather than about how much traffic
there is, and a socket that resends an identical snapshot every fifteen seconds is a poll with
the direction reversed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from fxagent.dashboard.grant import AdvisoryOnly, read_grant
from fxagent.dashboard.live import LiveHub
from fxagent.dashboard.models import GrantState
from fxagent.dashboard.snapshot import build_snapshot
from fxagent.dashboard.source import ViewRequest
from tests.dashboard.builders import evaluation, vote
from tests.dashboard.stubs import FailingSource, StubSource

NOW = datetime(2026, 1, 12, 12, 0, tzinfo=UTC)
VIEW = ViewRequest(symbol="EURUSD", timeframe="H1")


def hub_for(source, **kwargs) -> LiveHub:
    return LiveHub(source, AdvisoryOnly(), refresh_seconds=1000.0, clock=lambda: NOW, **kwargs)


# --- change detection ----------------------------------------------------------


async def test_an_unchanged_view_pushes_nothing() -> None:
    source = StubSource()
    hub = hub_for(source)
    await hub.subscribe(VIEW)

    assert await hub.refresh_once() == 0
    assert await hub.refresh_once() == 0


async def test_a_new_evaluation_is_pushed_once() -> None:
    source = StubSource()
    hub = hub_for(source)
    subscriber, first = await hub.subscribe(VIEW)

    source.evaluations = (
        evaluation(votes=[vote("session_breakout", direction="LONG", participated=True)]),
    )

    assert await hub.refresh_once() == 1
    pushed = await subscriber.next()

    assert pushed.revision != first.revision
    assert len(pushed.feed.entries) == 1

    # And then it goes quiet again.
    assert await hub.refresh_once() == 0


async def test_the_revision_ignores_when_the_snapshot_was_built() -> None:
    """It has to. A hash covering `generated_at` changes on every rebuild by definition, and
    the comparison it feeds would never be equal."""
    source = StubSource()
    early = build_snapshot(
        await source.load(VIEW),
        await read_grant(AdvisoryOnly(), NOW),
        now=NOW,
    )
    later = build_snapshot(
        await source.load(VIEW),
        await read_grant(AdvisoryOnly(), NOW),
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert early.revision == later.revision
    assert early.generated_at != later.generated_at


# --- subscribers ----------------------------------------------------------------


async def test_two_clients_on_one_view_cost_one_rebuild() -> None:
    source = StubSource()
    hub = hub_for(source)

    await hub.subscribe(VIEW)
    await hub.subscribe(VIEW)
    before = source.loads

    await hub.refresh_once()

    assert hub.rooms == 1
    assert hub.watchers == 2
    assert source.loads - before == 1


async def test_two_views_are_rebuilt_separately() -> None:
    hub = hub_for(StubSource())
    await hub.subscribe(VIEW)
    await hub.subscribe(ViewRequest(symbol="GBPUSD", timeframe="H1"))

    assert hub.rooms == 2


async def test_the_last_client_leaving_stops_the_view_being_rebuilt() -> None:
    source = StubSource()
    hub = hub_for(source)
    subscriber, _ = await hub.subscribe(VIEW)

    hub.unsubscribe(VIEW, subscriber)

    assert hub.rooms == 0
    assert await hub.refresh_once() == 0


async def test_a_slow_client_gets_the_newest_snapshot_not_a_backlog() -> None:
    """A single slot, not a queue. A dashboard wants the current state; a laptop that has been
    asleep must not wake to a queue of stale ones."""
    source = StubSource()
    hub = hub_for(source)
    subscriber, _ = await hub.subscribe(VIEW)

    source.evaluations = (evaluation(identifier=1, reason="first"),)
    await hub.refresh_once()
    source.evaluations = (evaluation(identifier=2, reason="second"),)
    await hub.refresh_once()

    delivered = await subscriber.next()
    assert delivered.feed.entries[0].reason == "second"

    # Nothing else is waiting behind it.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(subscriber.next(), timeout=0.05)


async def test_the_first_snapshot_arrives_with_the_subscription() -> None:
    """Subscribing and reading are one step, so a change landing between them cannot be missed."""
    hub = hub_for(StubSource())
    _, first = await hub.subscribe(VIEW)

    assert first.symbol == "EURUSD"
    assert first.chart.candles


# --- failure ---------------------------------------------------------------------


async def test_one_unreadable_view_does_not_stop_the_others_refreshing() -> None:
    class SometimesFails(StubSource):
        def __init__(self) -> None:
            super().__init__()
            self.broken: str | None = None

        async def load(self, request):
            if request.symbol == self.broken:
                raise RuntimeError("connection reset")
            return await super().load(request)

    source = SometimesFails()
    hub = hub_for(source)
    good, _ = await hub.subscribe(VIEW)
    await hub.subscribe(ViewRequest(symbol="GBPUSD"))

    # Both views subscribed cleanly; one of them now starts failing mid-flight, which is what
    # a dropped connection to Supabase looks like from here.
    source.broken = "GBPUSD"
    source.evaluations = (evaluation(reason="still working"),)

    assert await hub.refresh_once() == 1
    assert (await good.next()).feed.entries[0].reason == "still working"


async def test_a_failing_store_raises_to_the_caller_rather_than_faking_a_snapshot() -> None:
    """An empty chart is indistinguishable from a quiet market, so it must not be invented."""
    hub = hub_for(FailingSource())

    with pytest.raises(RuntimeError, match="connection refused"):
        await hub.snapshot(VIEW)


# --- permission -------------------------------------------------------------------


async def test_the_default_grant_is_advisory_because_the_state_machine_does_not_exist() -> None:
    grant = await read_grant(AdvisoryOnly(), NOW)

    assert grant.state is GrantState.ADVISORY
    assert grant.expires_at is None
    assert "execution is impossible" in grant.reason


async def test_an_unreadable_grant_is_reported_as_advisory_not_as_granted() -> None:
    """Fail closed. There is no path here that reports GRANTED because something broke."""

    class Broken:
        async def current(self, now):
            raise RuntimeError("no row")

    grant = await read_grant(Broken(), NOW)

    assert grant.state is GrantState.ADVISORY
    assert "could not be read" in grant.reason


async def test_the_refresh_loop_starts_and_stops_cleanly() -> None:
    hub = LiveHub(StubSource(), AdvisoryOnly(), refresh_seconds=0.01, clock=lambda: NOW)
    await hub.start()
    await hub.start()  # idempotent
    await asyncio.sleep(0.03)
    await hub.stop()
    await hub.stop()
