"""The push side: one server-side refresh, many sockets, nothing repeated to a browser.

**The browser never polls.** It opens one WebSocket, receives a snapshot immediately, and then
receives another only when something has actually changed. There is no interval in the client
and no repeated `fetch`.

**The server does poll, once, and that is the honest design.** Nothing notifies this process
when a row lands: the writers are short-lived GitHub Actions jobs (ADR-002), and Postgres
`LISTEN` is not usable through Supabase's transaction pooler, where consecutive statements land
on different backends and a listening connection is not held open. So one background task
rebuilds each *subscribed* view on a timer and pushes only when the content hash moves. Ten
browsers on the same symbol cost one rebuild, not ten, and a quiet hour costs zero messages.

**A slow client cannot back the server up.** Each subscriber holds a single slot, not a queue:
a snapshot arriving while an older one is still unsent replaces it. A dashboard wants the
current state, never a backlog of stale ones, and an unbounded queue behind a laptop that has
been asleep is a memory leak with a delivery guarantee nobody asked for.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import datetime

from fxagent.dashboard.chart import ChartConfig
from fxagent.dashboard.grant import GrantReader, read_grant
from fxagent.dashboard.models import Snapshot
from fxagent.dashboard.snapshot import build_snapshot
from fxagent.dashboard.source import DashboardSource, ViewRequest, utc_now

__all__ = ["DEFAULT_REFRESH_SECONDS", "LiveHub", "Subscriber"]

logger = logging.getLogger(__name__)

#: How often a subscribed view is rebuilt. The analyst writes hourly, so this is about how
#: quickly a new bar or a manual re-run shows up, not about chasing ticks.
DEFAULT_REFRESH_SECONDS = 15.0


class Subscriber:
    """One socket's slot. Holds the newest snapshot and nothing older."""

    __slots__ = ("_latest", "_ready")

    def __init__(self) -> None:
        self._latest: Snapshot | None = None
        self._ready = asyncio.Event()

    def offer(self, snapshot: Snapshot) -> None:
        """Replace whatever was waiting. Never blocks, never grows."""
        self._latest = snapshot
        self._ready.set()

    async def next(self) -> Snapshot:
        await self._ready.wait()
        self._ready.clear()
        snapshot = self._latest
        self._latest = None
        assert snapshot is not None  # noqa: S101 - only set together with the event
        return snapshot


class _Room:
    """Everyone watching one view, and the revision they were last sent."""

    __slots__ = ("request", "revision", "subscribers")

    def __init__(self, request: ViewRequest) -> None:
        self.request = request
        self.revision: str | None = None
        self.subscribers: set[Subscriber] = set()


class LiveHub:
    """Builds snapshots, and pushes them to whoever is watching when they change."""

    def __init__(
        self,
        source: DashboardSource,
        grants: GrantReader,
        *,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        chart_config: ChartConfig | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._source = source
        self._grants = grants
        self._refresh_seconds = refresh_seconds
        self._chart_config = chart_config
        self._clock = clock
        self._rooms: dict[str, _Room] = {}
        self._task: asyncio.Task[None] | None = None

    @property
    def refresh_seconds(self) -> float:
        return self._refresh_seconds

    async def snapshot(self, request: ViewRequest) -> Snapshot:
        """Build one view now. Used by the REST route and on every socket connect."""
        now = self._clock()
        data = await self._source.load(request.clamped())
        grant = await read_grant(self._grants, now)
        return build_snapshot(data, grant, now=now, config=self._chart_config)

    async def subscribe(self, request: ViewRequest) -> tuple[Subscriber, Snapshot]:
        """Join a view and get its current state in the same step.

        Joining and the first snapshot are one operation on purpose: a client that subscribed
        and then asked separately could miss a change landing between the two, and would then
        sit on a stale chart until the *next* change — a bug that only appears under exactly
        the conditions nobody can reproduce.
        """
        wanted = request.clamped()
        room = self._rooms.setdefault(wanted.key, _Room(wanted))
        subscriber = Subscriber()
        room.subscribers.add(subscriber)

        snapshot = await self.snapshot(wanted)
        room.revision = snapshot.revision
        return subscriber, snapshot

    def unsubscribe(self, request: ViewRequest, subscriber: Subscriber) -> None:
        """Leave. The room goes with the last client, so an unwatched view stops rebuilding."""
        room = self._rooms.get(request.clamped().key)
        if room is None:
            return
        room.subscribers.discard(subscriber)
        if not room.subscribers:
            del self._rooms[room.request.key]

    @property
    def rooms(self) -> int:
        """How many distinct views are being rebuilt. Reported by /api/health."""
        return len(self._rooms)

    @property
    def watchers(self) -> int:
        return sum(len(room.subscribers) for room in self._rooms.values())

    async def refresh_once(self) -> int:
        """Rebuild every watched view, push the ones that moved, return how many were pushed.

        Returns a count rather than nothing so a test can assert the quiet path: two refreshes
        over unchanged data must push exactly once, and that is the whole claim this file makes.
        """
        pushed = 0
        for room in list(self._rooms.values()):
            try:
                snapshot = await self.snapshot(room.request)
            except Exception:
                # One unreadable view must not stop the others being refreshed, and must not
                # kill the loop. The client keeps its last good snapshot; /api/health is where
                # a persistent failure shows up.
                logger.exception("failed to rebuild %s", room.request.key)
                continue

            if snapshot.revision == room.revision:
                continue

            room.revision = snapshot.revision
            for subscriber in room.subscribers:
                subscriber.offer(snapshot)
            pushed += 1

        return pushed

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_seconds)
            await self.refresh_once()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="fxagent-dashboard-refresh")
            logger.info("refresh loop started at %.1fs", self._refresh_seconds)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info("refresh loop stopped")
