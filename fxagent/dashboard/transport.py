"""How updates reach the browser: a socket where one can be held open, polling where it cannot.

The panel was built around a WebSocket and one shared server-side refresh, and on a host that
runs a process — a container on Oracle, on Fly, or on this laptop — that is still what it uses.
Ten browsers on one symbol cost one rebuild, and a quiet hour costs zero messages.

**Serverless has no process to hold either half of that.** A Vercel function is invoked per
request and cannot accept a WebSocket upgrade, and there is nowhere for a background task to
live between invocations, so the shared refresh loop has nothing to run on and no memory to
compare revisions in. Deploying there is therefore not a configuration change: it is a
different transport, and pretending otherwise would produce a panel that connects, says
"live", and never updates again.

So the client asks the server which transport to use, and the server answers from
`FX_DASHBOARD_TRANSPORT`. Both are implemented and both are tested.

**Polling here is conditional, not a re-download.** The client sends the revision it is holding;
the server rebuilds the view, compares the content hash, and answers `304 Not Modified` with no
body when nothing has moved. So a quiet market costs one empty round trip per interval instead
of 150KB of unchanged JSON, and the revision that makes that possible is the same hash the
socket already used to decide whether to push.

What is genuinely lost, and is worth writing down rather than glossing:

* **Latency** becomes the poll interval instead of "as soon as the row lands".
* **The shared rebuild is gone.** Every polling client rebuilds its own view, so ten browsers
  now cost ten reads instead of one. Fine for a personal panel; not fine for a busy one.
* **A database round trip per client per tick**, which on Supabase's free tier is the number to
  watch. The transaction pooler (port 6543) is not optional in this mode — `DatabaseConfig`
  already switches to `NullPool` when it sees that port.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum

__all__ = ["TRANSPORT_ENV", "Transport", "configured_transport"]

logger = logging.getLogger(__name__)

TRANSPORT_ENV = "FX_DASHBOARD_TRANSPORT"


class Transport(StrEnum):
    """Which mechanism the client should use to stay current."""

    #: One socket, snapshot on connect, then one message per change. Needs a long-lived process.
    SOCKET = "ws"
    #: Conditional GET on a timer. The only thing that works on a serverless host.
    POLL = "poll"

    @property
    def needs_refresh_loop(self) -> bool:
        """Whether the server-side rebuild loop is worth running.

        Only the socket has subscribers to push to. Starting the loop under polling would read
        the store on a timer for nobody — and on serverless it would be started fresh on every
        cold start and killed with the invocation, which is worse than useless.
        """
        return self is Transport.SOCKET


def configured_transport(env: dict[str, str] | None = None) -> Transport:
    """Read `FX_DASHBOARD_TRANSPORT`, defaulting to the socket.

    The default is the socket because that is the better design and the one every self-hosted
    deployment can use; polling is opted into by the host that cannot do better. An unrecognised
    value falls back to the socket *and logs*, rather than silently degrading a working
    deployment to polling because somebody typed `websocket`.
    """
    source = os.environ if env is None else env
    raw = (source.get(TRANSPORT_ENV) or "").strip().lower()

    if not raw:
        return Transport.SOCKET
    if raw in {"ws", "websocket", "socket"}:
        return Transport.SOCKET
    if raw in {"poll", "polling", "http"}:
        return Transport.POLL

    logger.warning(
        "%s=%r is not a transport; using the socket. Valid values are 'ws' and 'poll'.",
        TRANSPORT_ENV,
        raw,
    )
    return Transport.SOCKET
