"""Entry point: `python -m fxagent.dashboard`.

Binds 0.0.0.0 by default because the point of the container is to be reachable from another
machine — the Oracle host, or a phone on the same network. That is a deliberate exposure of a
process with no authentication, which is exactly why `app.py` has no route that can change
anything: the security argument is "there is nothing here to attack", and it only holds as long
as that stays true.

Unlike the collector and the analyst this process is long-lived. It is the one thing in the
system that is, and it can be: it holds no state worth losing, and a restart costs a page
reload. Nothing schedules it, nothing depends on it, and it writes nothing — a dashboard that
is down is a dashboard that is down, not a missed collection window.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from fxagent.dashboard.app import VENDOR_SCRIPT, create_app
from fxagent.dashboard.live import DEFAULT_REFRESH_SECONDS

logger = logging.getLogger("fxagent.dashboard")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m fxagent.dashboard",
        description="Read-only instrument panel. Cannot place, approve or modify an order.",
    )
    parser.add_argument("--host", default=os.environ.get("FX_DASHBOARD_HOST", "0.0.0.0"))  # noqa: S104
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("FX_DASHBOARD_PORT", "8080"))
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=float(os.environ.get("FX_DASHBOARD_REFRESH_SECONDS", str(DEFAULT_REFRESH_SECONDS))),
        help="how often a watched view is rebuilt server-side and pushed if it changed",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        logger.error(
            "uvicorn is not installed. The dashboard is an optional extra: "
            "run `uv sync --extra dashboard`."
        )
        return 2

    if not VENDOR_SCRIPT.exists():
        # Not fatal. The page loads, says the library is missing, and the feed still reads —
        # which is more useful than refusing to start and telling nobody why.
        logger.warning(
            "the charting library is not vendored at %s; the chart pane will not render. "
            "Run `uv run python scripts/vendor_lightweight_charts.py`.",
            VENDOR_SCRIPT,
        )

    application = create_app(refresh_seconds=args.refresh_seconds)
    logger.info("serving the panel on http://%s:%d (read-only)", args.host, args.port)
    uvicorn.run(application, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
