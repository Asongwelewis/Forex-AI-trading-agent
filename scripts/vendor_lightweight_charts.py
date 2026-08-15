"""Fetch the pinned charting library into `fxagent/dashboard/static/vendor/`.

The file is committed, so this is not part of running the dashboard — it is how the committed
file is produced and how it is re-checked. The pin itself lives in `fxagent.dashboard.vendored`,
which is also what the health check and the test read, so there is one version number in the
repository rather than three that agree until one of them is edited.

    uv run python scripts/vendor_lightweight_charts.py           # fetch and verify
    uv run python scripts/vendor_lightweight_charts.py --check   # verify what is committed
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request

from fxagent.dashboard.vendored import SHA256, URL, VERSION, path, verify


def check() -> int:
    problem = verify()
    if problem is not None:
        print(problem, file=sys.stderr)
        return 1
    print(f"ok: lightweight-charts {VERSION} ({path().stat().st_size} bytes)")
    return 0


def download() -> int:
    print(f"fetching {URL}")
    with urllib.request.urlopen(URL, timeout=60) as response:  # noqa: S310 - constant https URL
        payload = response.read()

    actual = hashlib.sha256(payload).hexdigest()
    if actual != SHA256:
        print(f"refusing to write: expected sha256 {SHA256}, got {actual}", file=sys.stderr)
        return 1

    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    print(f"wrote {target} ({len(payload)} bytes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="verify the committed copy instead of fetching"
    )
    return check() if parser.parse_args(argv).check else download()


if __name__ == "__main__":
    sys.exit(main())
