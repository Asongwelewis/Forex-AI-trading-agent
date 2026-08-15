"""The pinned charting library: which version, which bytes, and where they live.

The pin is in the package rather than in the script that fetches it, because three things need
to agree about it — the fetcher, the health check, and the test that verifies the committed
copy — and a constant that lives in `scripts/` can only be reached by two of them.

**Pinned to a version and a hash, not to "latest".** A chart library that updates itself is a
chart that changes what it draws without a line of our code moving, which defeats the point of
self-hosting it. The hash is checked before the bytes are written, so a truncated or tampered
download never lands in the tree.

**v4, not v5.** v5 moved markers off the series API into a `createSeriesMarkers` plugin and
changed how series are created. `static/app.js` is written against v4. Moving major version is
a deliberate change to make with the front end open, not something a `latest` tag does under it.

Apache 2.0. The licence header inside the file is preserved, which is the attribution the
licence asks for.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["FILENAME", "SHA256", "URL", "VERSION", "path", "verify"]

VERSION = "4.2.3"
FILENAME = "lightweight-charts.standalone.production.js"
URL = f"https://unpkg.com/lightweight-charts@{VERSION}/dist/{FILENAME}"
SHA256 = "c7dda807d662a95b3d257119ed315cec669e3bdf5aaece75c480a39307f23540"


def path() -> Path:
    """Where the committed copy lives, served from `/static/vendor/`."""
    return Path(__file__).parent / "static" / "vendor" / FILENAME


def verify() -> str | None:
    """None when the committed copy is exactly the pinned bytes, else why it is not."""
    target = path()
    if not target.exists():
        return f"missing: {target}"

    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != SHA256:
        return (
            f"checksum mismatch for {FILENAME}\n  expected {SHA256}\n  actual   {actual}\n"
            "On a fresh Windows clone, check that .gitattributes still marks the vendor "
            "directory as binary — an EOL conversion rewrites every line and therefore the hash."
        )
    return None
