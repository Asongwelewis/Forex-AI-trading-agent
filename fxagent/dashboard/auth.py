"""A shared-password gate for when the panel is not on a private network.

**HTTP Basic, deliberately, and not a login form.** A form needs a POST route, a session cookie
and somewhere to keep it — which would put the first mutating endpoint on a service whose entire
security argument is that it has none, and would break the route-table test that enforces it.
Basic auth adds no route at all: it is a header the browser sends and this middleware checks.

It is not a strong scheme, and it is not pretending to be. Over HTTPS it is a shared secret in a
header, which is exactly the right weight for "this panel is mine and I would rather it were not
indexed". It is not a user system, there are no accounts, and nothing behind it can be changed
by anyone who gets in — a leaked password costs visibility of the journal, not control of it.

**Absent means open, and says so.** With `FX_DASHBOARD_PASSWORD` unset the gate is off, because
that is the right default for `localhost` and for a container on a home LAN, which is how this
runs most of the time. `create_app` logs a warning at startup so an unset password on a public
host is noisy rather than silent.

`/api/health` stays reachable without the password, so an uptime check and a container
healthcheck still work — but an unauthenticated caller gets the liveness answer and nothing
else. The full report names the store's error, and a connection error names the host.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import logging
import os
import secrets

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

__all__ = [
    "COOKIE",
    "OPEN_PATHS",
    "PASSWORD_ENV",
    "PasswordGate",
    "configured_password",
    "is_authenticated",
]

logger = logging.getLogger(__name__)

PASSWORD_ENV = "FX_DASHBOARD_PASSWORD"

#: Reachable without the password. Liveness only — see the module docstring for what an
#: unauthenticated caller is *not* told.
OPEN_PATHS = frozenset({"/api/health"})

REALM = "FX regime agent"


def configured_password(env: dict[str, str] | None = None) -> str | None:
    """The shared password, or None when the gate is off.

    Whitespace-only counts as unset. A password of `" "` is almost certainly a deployment
    variable that picked up a stray space, and treating it as a real secret would mean a panel
    nobody can open with a password nobody meant to set.
    """
    source = os.environ if env is None else env
    password = (source.get(PASSWORD_ENV) or "").strip()
    return password or None


#: Name of the companion cookie. See `_ticket` for why a cookie exists at all.
COOKIE = "fxagent_panel"


def _ticket(password: str) -> str:
    """A cookie value derived from the password, for the WebSocket handshake.

    **Browsers do not attach cached Basic credentials to a WebSocket upgrade.** The JavaScript
    `WebSocket` constructor cannot set headers either, so with Basic auth alone a password-
    protected panel would authenticate every page load and then silently fail to connect its
    socket — the panel would sit there saying "reconnecting" with no way to discover why. That
    is precisely the silent-failure class this codebase keeps refusing, so it is fixed rather
    than documented.

    Cookies *are* sent on a same-origin handshake, so an authenticated HTTP response sets one
    and the socket accepts it. The cookie is an HMAC of the password rather than the password
    itself: it is derived, so it survives a restart without any stored state, and it is one-way,
    so a cookie read off a machine does not hand over the password the user typed.
    """
    return hmac.new(password.encode("utf-8"), b"fxagent-dashboard-socket", "sha256").hexdigest()


class PasswordGate:
    """Checks HTTP Basic against one shared password. Any username is accepted.

    Pure ASGI middleware rather than a FastAPI dependency, so it covers the static mount and
    the WebSocket route as well — a dependency would have to be repeated on each, and the one
    that gets forgotten is the one that matters.
    """

    def __init__(self, app: ASGIApp, password: str) -> None:
        self._app = app
        self._password = password
        self._ticket = _ticket(password)

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        by_header = self._header_matches(scope)
        authenticated = by_header or self._cookie_matches(scope)
        # Recorded either way: /api/health is reachable unauthenticated and decides how much to
        # say from this flag rather than from a second look at the header.
        scope.setdefault("state", {})["authenticated"] = authenticated

        if authenticated:
            await self._app(scope, receive, self._with_ticket(scope, send))
            return

        if scope.get("path") in OPEN_PATHS:
            await self._app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            # No handshake, no upgrade. A socket that connects and is then closed looks like a
            # server fault to the client; refusing it outright is the honest answer.
            await send({"type": "websocket.close", "code": 1008})
            return

        response = JSONResponse(
            {"error": "This panel is password protected."},
            status_code=401,
            headers={"WWW-Authenticate": f'Basic realm="{REALM}", charset="UTF-8"'},
        )
        await response(scope, receive, send)

    def _with_ticket(self, scope, send):  # type: ignore[no-untyped-def]
        """Attach the socket cookie to an authenticated HTTP response."""
        if scope["type"] != "http":
            return send

        secure = "; Secure" if _is_https(scope) else ""
        cookie = f"{COOKIE}={self._ticket}; Path=/; HttpOnly; SameSite=Strict{secure}"

        async def wrapped(message):  # type: ignore[no-untyped-def]
            if message["type"] == "http.response.start":
                headers = [*message.get("headers", []), (b"set-cookie", cookie.encode("latin-1"))]
                message = {**message, "headers": headers}
            await send(message)

        return wrapped

    def _header_matches(self, scope) -> bool:  # type: ignore[no-untyped-def]
        header = _header(scope, b"authorization")
        if header is None:
            return False

        scheme, _, encoded = header.partition(" ")
        if scheme.lower() != "basic" or not encoded:
            return False

        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return False

        _, separator, password = decoded.partition(":")
        if not separator:
            return False

        # Constant time, so a wrong password cannot be narrowed down one character at a time.
        return secrets.compare_digest(password, self._password)

    def _cookie_matches(self, scope) -> bool:  # type: ignore[no-untyped-def]
        raw = _header(scope, b"cookie")
        if raw is None:
            return False

        for part in raw.split(";"):
            name, separator, value = part.strip().partition("=")
            if separator and name == COOKIE:
                return secrets.compare_digest(value, self._ticket)
        return False


def _is_https(scope) -> bool:  # type: ignore[no-untyped-def]
    """Whether the browser's leg of this request is TLS, proxy included.

    Vercel and every other managed host terminate TLS in front of the app, so `scope["scheme"]`
    is `http` on a page the user reached over `https`. Marking the cookie `Secure` from that
    would leave it unmarked on exactly the deployments that need it.
    """
    forwarded = _header(scope, b"x-forwarded-proto")
    if forwarded:
        return forwarded.split(",")[0].strip() == "https"
    return scope.get("scheme") in ("https", "wss")


def _header(scope, name: bytes) -> str | None:  # type: ignore[no-untyped-def]
    for key, value in scope.get("headers", ()):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def is_authenticated(request: Request) -> bool:
    """Whether this request cleared the gate. True when no gate is configured."""
    return bool(getattr(request.state, "authenticated", True))
