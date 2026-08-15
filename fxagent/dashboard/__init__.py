"""The instrument panel: a chart of our own bars on the left, the agent's reasoning on the right.

The whole point is stated in the phase's own acceptance test — watch a full London session and
understand every decision without opening a log file. So the panel shows the refusals as
prominently as the trades, since on most bars a refusal is the only thing that happened, and it
shows *why* each one was refused rather than only that it was.

Layers, outermost last:

| Module | Job |
|---|---|
| `models` | Every shape the browser may receive. |
| `contract` | How an `evaluations` row is read, including the keys the agent layer will write. |
| `chart` | Pure: bars + evaluations + trades → the left pane. |
| `feed` | Pure: evaluations + trades + grant → the right pane. |
| `grant` | Execution permission, read-only and fail-closed. |
| `source` | The only thing here that touches Postgres. |
| `snapshot` | Assembles one view and hashes it. |
| `live` | One server-side refresh, many sockets, pushes only on change. |
| `app` | FastAPI. GET routes only. |

`fastapi` and `uvicorn` are an optional extra (`uv sync --extra dashboard`), so `chart`, `feed`
and `contract` import without them. That is deliberate: the builders are the part with the
logic worth testing, and they should not need a web framework installed to be tested.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(**kwargs: object):  # type: ignore[no-untyped-def]
    """Lazy re-export, so `import fxagent.dashboard` works without FastAPI installed."""
    from fxagent.dashboard.app import create_app as _create_app

    return _create_app(**kwargs)  # type: ignore[arg-type]
