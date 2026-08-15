"""The front end's two structural promises: it is self-hosted, and its library is pinned.

Neither is checkable by looking at the page — a CDN link renders perfectly until the day the
network is the thing you are trying to debug, and an unpinned library changes what it draws
without changing a line of our code.
"""

from __future__ import annotations

import re

import pytest

from fxagent.dashboard.app import STATIC_DIR, VENDOR_SCRIPT
from fxagent.dashboard.vendored import SHA256, VERSION, verify

#: Anything that would make the page fetch from a host that is not this container.
EXTERNAL = re.compile(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", re.IGNORECASE)

ASSETS = ("index.html", "styles.css", "app.js")


def test_the_charting_library_is_vendored_and_matches_its_pinned_hash() -> None:
    """A mismatch on a fresh clone usually means git normalised the line endings — see the
    `binary` entry for this directory in .gitattributes."""
    assert VENDOR_SCRIPT.exists(), "run `uv run python scripts/vendor_lightweight_charts.py`"
    assert verify() is None


def test_the_vendored_library_is_the_version_the_front_end_is_written_against() -> None:
    header = VENDOR_SCRIPT.read_bytes()[:400].decode("utf-8", "replace")

    assert f"v{VERSION}" in header
    assert "Apache License 2.0" in header, "the licence header is the attribution; keep it"
    assert SHA256 and len(SHA256) == 64


@pytest.mark.parametrize("name", ASSETS)
def test_no_asset_reaches_outside_this_container(name: str) -> None:
    """Self-hosted means self-hosted. No CDN, no external font, no analytics."""
    body = (STATIC_DIR / name).read_text(encoding="utf-8")

    assert not EXTERNAL.search(body), f"{name} loads something over the network"


def test_the_page_loads_the_library_it_ships_with() -> None:
    body = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert f"/static/vendor/{VENDOR_SCRIPT.name}" in body


def test_the_front_end_renders_stored_text_as_text() -> None:
    """Agent narration is prose from a language model and reasons are stored strings. Neither
    is ever interpolated into markup, so `innerHTML` must not appear in the renderer at all."""
    body = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "innerHTML" not in body
    assert "textContent" in body


def test_the_session_colours_agree_between_the_stylesheet_and_the_painter() -> None:
    """The bands are painted onto a canvas, which cannot read a CSS custom property, so the
    four colours exist in both files. That is a real duplication and this is what keeps it
    honest — a legend swatch that disagrees with the band it labels is a chart that lies."""
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for token, key in (
        ("--tokyo", "TOKYO"),
        ("--london", "LONDON"),
        ("--newyork", "NEW_YORK"),
        ("--overlap", "OVERLAP"),
    ):
        in_css = re.search(rf"{token}:\s*(rgba\([^)]*\))", css)
        in_js = re.search(rf'{key}:\s*"(rgba\([^)]*\))"', js)

        assert in_css and in_js, f"{key} is missing from one of the two files"
        assert _colour(in_css.group(1)) == _colour(in_js.group(1)), (
            f"{key} differs: {in_css.group(1)} in CSS, {in_js.group(1)} in JS"
        )


def _colour(value: str) -> tuple[float, ...]:
    """rgba(...) as numbers, so `0.1` and `0.10` compare equal."""
    return tuple(float(part) for part in value[value.index("(") + 1 : -1].split(","))


def test_the_client_distinguishes_an_unreadable_store_from_an_empty_one() -> None:
    """They are different problems with different fixes — a missing environment variable
    versus a collector that has not run — and the server has already worked out which. The
    client reporting the first as the second sends the reader to the wrong place."""
    body = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "if (!response.ok)" in body, "the client must branch on the status, not just the body"
    assert "options.error" in body, "the server's reason must reach the screen"
    assert "reachable but holds no bars" in body
