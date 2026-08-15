"""Vercel's entrypoint. The whole panel, as one serverless function.

Vercel's Python runtime imports this module and serves the ASGI app it finds in `app`. That is
the only thing this file does — every decision lives in `fxagent.dashboard`, so the deployment
is a wrapper rather than a fork.

**`pyproject.toml` is hidden from this build on purpose.** Vercel's builder prefers `uv sync`
against it when it exists, and `fastapi` lives in an optional extra that `uv sync` does not
install — which builds cleanly and then fails on every request. `.vercelignore` hides it so the
generated `requirements.txt` is used instead. Read that file before changing either.

**This is a degraded deployment and that is a decision, not an accident.** Read
`fxagent/dashboard/transport.py` before changing anything here: a serverless host cannot hold a
WebSocket or run the shared refresh loop, so the panel polls, each viewer costs its own read of
the store, and updates lag by the poll interval. The socket path still exists and is still the
better one; it needs a host that runs a process.

Three environment variables have to be set in the Vercel project, and two of them are not
optional:

* `SUPABASE_DB_URL` — **the transaction pooler, port 6543.** Not the direct connection and not
  the session pooler. Every invocation opens its own connection, and `DatabaseConfig` already
  switches to `NullPool` and disables the statement cache when it sees that port. Pointing this
  at 5432 will work in testing and exhaust the connection allowance under any real use. With it
  unset the panel still loads and says so on the page, rather than returning an opaque 500.
* `FX_DASHBOARD_TRANSPORT=poll` — without it the client is told to open a WebSocket, which this
  host will refuse, and the panel falls back after two failed attempts with a visible note. It
  works either way; setting it skips the failure.
* `FX_DASHBOARD_PASSWORD` — a Vercel URL is public. Without this the panel is too, and the
  startup log says so.
"""

from __future__ import annotations

import logging
import os

from fxagent.dashboard.app import create_app

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)-7s %(name)s: %(message)s",
)

#: The name Vercel's Python runtime looks for.
app = create_app()
