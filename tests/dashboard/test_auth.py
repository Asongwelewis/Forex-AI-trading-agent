"""The password gate.

Two properties carry the weight here. The gate must not open by accident — a missing header, a
malformed one, a wrong password and an empty password are all closed — and it must not close by
accident either: `/api/health` stays reachable so a container healthcheck and an uptime monitor
still work, and the panel's own WebSocket must still connect, which is the one thing HTTP Basic
cannot do on its own.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from fxagent.dashboard.app import create_app
from fxagent.dashboard.auth import COOKIE, PASSWORD_ENV, configured_password
from fxagent.dashboard.transport import Transport
from tests.dashboard.stubs import StubSource

PASSWORD = "an-actual-password"


def basic(password: str, user: str = "panel") -> dict[str, str]:
    encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


@pytest.fixture
def client() -> TestClient:
    app = create_app(source=StubSource(), password=PASSWORD, refresh_seconds=1000)
    with TestClient(app) as running:
        yield running


# --- configuration ---------------------------------------------------------------


def test_no_password_configured_means_no_gate() -> None:
    assert configured_password({}) is None
    assert configured_password({PASSWORD_ENV: ""}) is None


def test_a_whitespace_password_is_treated_as_unset() -> None:
    """Almost certainly a deployment variable that picked up a stray space. Treating it as a
    real secret would mean a panel nobody can open with a password nobody meant to set."""
    assert configured_password({PASSWORD_ENV: "   "}) is None


def test_a_configured_password_is_stripped_but_kept() -> None:
    assert configured_password({PASSWORD_ENV: "  hunter2  "}) == "hunter2"


def test_the_panel_is_open_when_no_password_is_configured() -> None:
    with TestClient(create_app(source=StubSource(), password=None)) as client:
        assert client.get("/api/snapshot", params={"symbol": "EURUSD"}).status_code == 200


# --- the gate ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["/", "/api/snapshot?symbol=EURUSD", "/api/options", "/static/app.js"]
)
def test_everything_is_closed_without_the_password(client: TestClient, path: str) -> None:
    """The static mount included — the gate is ASGI middleware rather than a route dependency
    precisely so there is no route to forget it on."""
    response = client.get(path)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic")


def test_the_right_password_opens_it(client: TestClient) -> None:
    assert client.get("/api/options", headers=basic(PASSWORD)).status_code == 200


def test_any_username_is_accepted(client: TestClient) -> None:
    """There are no accounts. A username field exists because Basic has one, not because it
    means anything."""
    assert client.get("/api/options", headers=basic(PASSWORD, "anyone")).status_code == 200


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": "Basic not-base64!!"},
        {"Authorization": "Bearer " + base64.b64encode(b"panel:" + PASSWORD.encode()).decode()},
        {"Authorization": "Basic"},
        {"Authorization": ""},
    ],
)
def test_a_malformed_header_is_closed(client: TestClient, header: dict[str, str]) -> None:
    assert client.get("/api/options", headers=header).status_code == 401


def test_a_near_miss_is_closed(client: TestClient) -> None:
    assert client.get("/api/options", headers=basic(PASSWORD + " ")).status_code == 401
    assert client.get("/api/options", headers=basic(PASSWORD[:-1])).status_code == 401


# --- what stays reachable -----------------------------------------------------------


def test_health_answers_without_the_password_but_says_less(client: TestClient) -> None:
    """A healthcheck must work; a stranger must not learn the store's error, which on a
    connection failure names the host."""
    open_answer = client.get("/api/health").json()
    full_answer = client.get("/api/health", headers=basic(PASSWORD)).json()

    assert open_answer["status"] in {"ok", "degraded"}
    assert set(open_answer) == {"status", "read_only"}
    assert "series" in full_answer


def test_the_socket_is_refused_without_credentials(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws?symbol=EURUSD") as socket,
    ):
        socket.receive_json()


def test_an_authenticated_page_load_hands_out_the_socket_cookie(client: TestClient) -> None:
    """Browsers do not attach cached Basic credentials to a WebSocket upgrade and the
    JavaScript client cannot set a header, so without this the panel would authenticate on
    every page load and then silently fail to connect. Cookies *are* sent on the handshake."""
    response = client.get("/", headers=basic(PASSWORD))

    assert response.status_code == 200
    assert COOKIE in response.cookies


def test_the_cookie_is_not_the_password(client: TestClient) -> None:
    """It is an HMAC of it, so a cookie read off a machine does not hand over what was typed."""
    ticket = client.get("/", headers=basic(PASSWORD)).cookies[COOKIE]

    assert PASSWORD not in ticket
    assert len(ticket) == 64  # sha256, hex


def test_the_socket_opens_on_the_cookie_alone(client: TestClient) -> None:
    app = create_app(
        source=StubSource(), password=PASSWORD, transport=Transport.SOCKET, refresh_seconds=1000
    )
    with TestClient(app) as running:
        ticket = running.get("/", headers=basic(PASSWORD)).cookies[COOKIE]

        with running.websocket_connect(
            "/ws?symbol=EURUSD", headers={"Cookie": f"{COOKIE}={ticket}"}
        ) as socket:
            envelope = socket.receive_json()

    assert envelope["type"] == "snapshot"


def test_a_forged_cookie_does_not_open_the_socket(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(
            "/ws?symbol=EURUSD", headers={"Cookie": f"{COOKIE}={'0' * 64}"}
        ) as socket,
    ):
        socket.receive_json()


def test_the_gate_adds_no_route_at_all() -> None:
    """The reason it is Basic and not a login form: a form needs a POST, and the first mutating
    route on this service is the one that ends its whole security argument."""
    gated = {route.path for route in create_app(source=StubSource(), password=PASSWORD).routes}
    open_app = {route.path for route in create_app(source=StubSource(), password=None).routes}

    assert gated == open_app
