# The dashboard container. The only long-lived process in the system.
#
# The collector and the analyst are not here and should not be: they wake, work for a minute
# and exit, and GitHub Actions cron invokes them (ADR-002). Containerising a job that runs for
# ninety seconds an hour buys a thing to keep alive and nothing else. The dashboard is
# different — it holds a socket open, so it has to be somewhere.
#
# Built on uv's image so the lockfile is the build, not a `pip install` that resolves something
# newer than what was tested. `--frozen` fails on a stale lockfile rather than quietly drifting.
#
# ARM: the target is Oracle's Always Free tier, which is aarch64. uvicorn[standard] pulls
# uvloop, httptools and websockets, all of which publish aarch64 wheels — that is why the extra
# is worth naming here rather than discovering at build time that the image needs a compiler.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    FX_DASHBOARD_HOST=0.0.0.0 \
    FX_DASHBOARD_PORT=8080 \
    # Nothing at runtime goes through `uv`. `uv run` re-resolves and syncs the environment on
    # every invocation, which needs a writable .venv and a writable cache — and this container
    # runs read-only, as a non-root user who owns neither. Putting the environment on PATH
    # means the command is just `python`, resolved to the interpreter the build produced.
    # Found the honest way: the first version of this file used `uv run` and the image built
    # fine, because the *build* runs as root on a writable layer.
    PATH="/app/.venv/bin:$PATH"

# `readme` in pyproject.toml points at README.md, so the build needs it. fxagent is copied
# before the sync because hatchling builds the package itself as part of it.
COPY pyproject.toml uv.lock README.md ./
COPY fxagent ./fxagent

RUN uv sync --frozen --no-dev --extra dashboard

# The charting library is committed under fxagent/dashboard/static/vendor/ and was copied with
# the package above. Fail the build if it is not there or does not match its pinned hash: a
# container that starts and serves a blank chart pane is a worse outcome than one that refuses
# to be built.
RUN python -c "import sys; from fxagent.dashboard.vendored import verify; sys.exit(verify() or 0)"

# Nothing here needs root, and this process is deliberately exposed to a network.
RUN useradd --create-home --uid 10001 panel
USER panel

EXPOSE 8080

# Liveness only: does the process still serve HTTP. Store reachability is *reported* at
# /api/health and deliberately not restarted on — a dashboard whose database blinked should
# show the last snapshot and say so, not disappear and take the page you were reading with it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=4)"]

CMD ["python", "-m", "fxagent.dashboard"]
