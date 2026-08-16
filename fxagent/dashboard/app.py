"""The FastAPI app: five read routes, one socket, one page.

**Every route is a GET.** There is no POST, PUT, PATCH or DELETE anywhere in this file, and
`tests/dashboard/test_app.py` asserts that by inspecting the route table rather than by trusting
this sentence. The reason is CLAUDE.md hard rule 1 and the shape of the deployment: this process
is meant to be reachable across a network, and the only safe thing to expose that way is a
window. Approving or placing a trade is a different feature with an authenticated channel
(Telegram, with a pinned chat id) and does not belong behind a web page at all.

That the password gate is *middleware* rather than a login form is the same rule holding: a
form would need a POST, and the first mutating route on this service is the one that ends the
argument above. See `auth.py`.

Two transports live here, and which one the client uses is the server's answer at `/api/config`,
not a guess. The socket is the design; polling is what a serverless host can actually run. See
`transport.py` for what the second one costs.

The page itself is served from `static/`, vendored and self-hosted — no CDN, no external font,
no analytics. A dashboard that cannot render without reaching the internet is a dashboard that
goes blank exactly when the network is the thing you are trying to debug.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from fxagent.dashboard.auth import PasswordGate, configured_password, is_authenticated
from fxagent.dashboard.chart import ChartConfig
from fxagent.dashboard.grant import AdvisoryOnly, GrantReader
from fxagent.dashboard.live import DEFAULT_REFRESH_SECONDS, LiveHub, Subscriber
from fxagent.dashboard.models import Envelope, Snapshot
from fxagent.dashboard.source import (
    DEFAULT_BAR_COUNT,
    DEFAULT_FEED_LIMIT,
    DashboardSource,
    ViewRequest,
    store_or_unavailable,
)
from fxagent.dashboard.transport import Transport, configured_transport
from fxagent.dashboard.vendored import path as vendored_chart_library
from fxagent.store.config import connection_hint

__all__ = ["STATIC_DIR", "VENDOR_SCRIPT", "create_app"]

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
#: The vendored charting library. Its absence is reported by /api/health and shown on the page,
#: because a blank chart with a 404 in the console is the least diagnosable failure here.
VENDOR_SCRIPT = vendored_chart_library()


def create_app(
    *,
    source: DashboardSource | None = None,
    grants: GrantReader | None = None,
    refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    chart_config: ChartConfig | None = None,
    transport: Transport | None = None,
    password: str | None = None,
) -> FastAPI:
    """Build the app. Everything it depends on is injectable, which is why it is testable.

    With no `source`, one is built from the environment — `SUPABASE_DB_URL`, through the same
    `Database` the collector uses. The tests pass a stub instead and never open a socket to
    anything.

    `transport` and `password` default to the environment (`FX_DASHBOARD_TRANSPORT`,
    `FX_DASHBOARD_PASSWORD`) so a deployment configures them without a code change, and are
    arguments so a test can have both without touching `os.environ`.
    """
    database = None
    if source is None:
        # An unconfigured store must not raise here. On a serverless host that turns every
        # request into an opaque 500 with the reason in a log nobody is reading; the panel
        # instead loads and prints the missing variable's name where the chart would be.
        source, database = store_or_unavailable()

    mode = transport if transport is not None else configured_transport()
    secret = password if password is not None else configured_password()

    hub = LiveHub(
        source,
        grants or AdvisoryOnly(),
        refresh_seconds=refresh_seconds,
        chart_config=chart_config,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Only the socket has subscribers to push to, and on a serverless host the loop would
        # be started on every cold start and killed with the invocation. See transport.py.
        if mode.needs_refresh_loop:
            await hub.start()
        try:
            yield
        finally:
            await hub.stop()
            if database is not None:
                await database.dispose()

    app = FastAPI(
        title="FX regime agent — instrument panel",
        description="Read-only. This process cannot place, approve or modify an order.",
        lifespan=lifespan,
        # The generated docs are the one place a reader can confirm the read-only claim for
        # themselves, so they stay on.
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.hub = hub
    app.state.source = source
    app.state.transport = mode

    if secret:
        app.add_middleware(PasswordGate, password=secret)
    else:
        logger.warning(
            "%s is not set, so the panel is open to anyone who can reach it. That is the right "
            "default on localhost and on a private network, and the wrong one on a public host.",
            "FX_DASHBOARD_PASSWORD",
        )

    def _request(
        symbol: str,
        timeframe: str,
        bar_source: str | None,
        bars: int,
        feed_limit: int,
    ) -> ViewRequest:
        return ViewRequest(
            symbol=symbol,
            timeframe=timeframe,
            source=bar_source,
            bars=bars,
            feed_limit=feed_limit,
        ).clamped()

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        """What the client needs to know before it connects. Read once, on load.

        The transport is the server's answer rather than the client's guess, because the client
        cannot tell a host that refuses WebSocket upgrades from one that is briefly down — and
        the two want opposite responses.
        """
        return {
            "transport": str(mode),
            "poll_seconds": hub.refresh_seconds,
            "read_only": True,
        }

    @app.get("/api/health")
    async def health(request: Request) -> JSONResponse:
        """Is the store readable, is the chart library present, who is watching what.

        Reports `degraded` rather than raising when the store cannot be read: the page is still
        useful with a stale snapshot on it, and a health check that 500s tells a monitor less
        than one that answers with the reason.

        Reachable without the password so an uptime check still works — but an unauthenticated
        caller gets liveness only. The full report names the store's error, and a connection
        error names the host.
        """
        detail: dict[str, Any] = {
            "status": "ok",
            "read_only": True,
            "rooms": hub.rooms,
            "watchers": hub.watchers,
            "refresh_seconds": hub.refresh_seconds,
            "transport": str(mode),
            "chart_library": "present" if VENDOR_SCRIPT.exists() else "missing",
        }
        try:
            detail["series"] = len(await source.options())
        except Exception as error:  # noqa: BLE001 - the reason is the useful part
            detail["status"] = "degraded"
            detail["store_error"] = str(error)
            hint = connection_hint(error)
            if hint:
                detail["store_hint"] = hint

        if detail["chart_library"] == "missing":
            detail["status"] = "degraded"
            detail["chart_library_hint"] = (
                "run `uv run python scripts/vendor_lightweight_charts.py`"
            )

        status = 200 if detail["status"] == "ok" else 503
        if not is_authenticated(request):
            detail = {"status": detail["status"], "read_only": True}
        return JSONResponse(detail, status_code=status)

    @app.get("/api/options")
    async def options() -> Any:
        """Every (symbol, timeframe, source) with bars behind it. The switchers read this."""
        try:
            return [option.model_dump() for option in await source.options()]
        except Exception as error:  # noqa: BLE001 - see /api/snapshot
            logger.exception("could not list the stored series")
            return _unavailable(error)

    @app.get("/api/snapshot", response_model=Snapshot)
    async def snapshot(
        symbol: str = Query(..., min_length=1),
        timeframe: str = Query("H1"),
        bar_source: str | None = Query(None, alias="source"),
        bars: int = Query(DEFAULT_BAR_COUNT),
        feed_limit: int = Query(DEFAULT_FEED_LIMIT),
        since: str | None = Query(
            None,
            description="A revision the caller already holds. Answers 304 when it is current.",
        ),
    ) -> Any:
        """One complete view. The socket sends this same object; polling clients read this one.

        **`since` is what makes polling affordable.** The view is rebuilt either way — that is
        the only way to know whether it moved — but when the content hash matches what the
        caller already has, the answer is `304 Not Modified` with no body. A quiet market then
        costs an empty round trip per interval instead of 150KB of unchanged JSON.

        An unreadable store is a 503 naming the reason, not a 500 with a traceback. The
        distinction is the difference between "the panel is broken" and "the panel is fine and
        the database is not", which is the first thing anyone looking at this needs to know.
        """
        try:
            built = await hub.snapshot(_request(symbol, timeframe, bar_source, bars, feed_limit))
        except Exception as error:  # noqa: BLE001 - the reason is the useful part
            logger.exception("could not build a snapshot for %s %s", symbol, timeframe)
            return _unavailable(error)

        if since is not None and since == built.revision:
            # 304 carries no body by definition, which is the point: the ETag is the answer.
            return Response(status_code=304, headers={"ETag": f'"{built.revision}"'})
        return JSONResponse(built.model_dump(mode="json"), headers={"ETag": f'"{built.revision}"'})

    @app.websocket("/ws")
    async def live(
        websocket: WebSocket,
        symbol: str = Query(..., min_length=1),
        timeframe: str = Query("H1"),
        bar_source: str | None = Query(None, alias="source"),
        bars: int = Query(DEFAULT_BAR_COUNT),
        feed_limit: int = Query(DEFAULT_FEED_LIMIT),
    ) -> None:
        """Subscribe to one view. A snapshot on connect, then one per change and no more.

        Switching symbol or timeframe closes this socket and opens another, rather than being a
        message on the existing one. One socket therefore means one view for its whole life,
        which removes the only race worth worrying about here — a snapshot for the old view
        arriving after the client has already redrawn for the new one.
        """
        await websocket.accept()
        request = _request(symbol, timeframe, bar_source, bars, feed_limit)

        try:
            subscriber, first = await hub.subscribe(request)
        except Exception as error:  # noqa: BLE001 - reported to the client, then closed
            logger.exception("could not build the first snapshot for %s", request.key)
            failure = Envelope(type="error", message=f"could not read the store: {error}")
            await websocket.send_text(failure.model_dump_json())
            await websocket.close()
            return

        try:
            await websocket.send_text(Envelope(type="snapshot", snapshot=first).model_dump_json())
            await _pump(websocket, subscriber)
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(request, subscriber)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def _unavailable(error: Exception) -> JSONResponse:
    """503 with the reason attached, and the fix when the reason is one we recognise.

    The panel prints this where the chart would be, so a deployment problem is legible on the
    screen rather than sixty lines down a traceback in a log nobody has open.
    """
    message = f"could not read the store: {error}"
    hint = connection_hint(error)
    if hint:
        message = f"{message} — {hint}"
    return JSONResponse({"error": message}, status_code=503)


async def _pump(websocket: WebSocket, subscriber: Subscriber) -> None:
    """Send snapshots until the client goes away.

    Two tasks, not one. Nothing is ever *expected* from the browser, but a socket that is only
    ever written to does not notice a disconnect until the next write — which on a quiet
    Sunday is hours away, and the subscriber would sit in its room being refreshed for a tab
    that closed at lunchtime. The reader exists to notice.
    """
    sending = asyncio.create_task(_send_forever(websocket, subscriber))
    watching = asyncio.create_task(_watch_for_close(websocket))

    done, pending = await asyncio.wait({sending, watching}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in done:
        # Re-raise a genuine failure; a disconnect is the ordinary way out.
        with_error = task.exception()
        if with_error is not None and not isinstance(with_error, WebSocketDisconnect):
            raise with_error


async def _send_forever(websocket: WebSocket, subscriber: Subscriber) -> None:
    while True:
        snapshot = await subscriber.next()
        await websocket.send_text(Envelope(type="snapshot", snapshot=snapshot).model_dump_json())


async def _watch_for_close(websocket: WebSocket) -> None:
    """Drain whatever the client sends. Content is ignored; arrival is not the point."""
    while True:
        await websocket.receive_text()
