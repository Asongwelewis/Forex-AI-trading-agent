"""Two transports, and the conditional GET that makes the second one affordable.

The socket is the design. Polling exists because a serverless host cannot hold one open, and
this file is where the claim "polling costs an empty round trip on a quiet market" stops being
a sentence in an ADR and becomes something that fails when it stops being true.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fxagent.dashboard.app import create_app
from fxagent.dashboard.transport import TRANSPORT_ENV, Transport, configured_transport
from tests.dashboard.builders import evaluation
from tests.dashboard.stubs import StubSource

VIEW = {"symbol": "EURUSD", "timeframe": "H1"}


@pytest.fixture
def source() -> StubSource:
    return StubSource()


def polling_client(source: StubSource) -> TestClient:
    return TestClient(create_app(source=source, transport=Transport.POLL, refresh_seconds=1000))


# --- configuration -------------------------------------------------------------


def test_the_socket_is_the_default() -> None:
    """The better design, and the one every self-hosted deployment can run."""
    assert configured_transport({}) is Transport.SOCKET


@pytest.mark.parametrize("value", ["poll", "POLL", " polling ", "http"])
def test_polling_is_opted_into(value: str) -> None:
    assert configured_transport({TRANSPORT_ENV: value}) is Transport.POLL


@pytest.mark.parametrize("value", ["ws", "websocket", "socket"])
def test_the_socket_can_be_named_explicitly(value: str) -> None:
    assert configured_transport({TRANSPORT_ENV: value}) is Transport.SOCKET


def test_an_unrecognised_transport_falls_back_to_the_socket(caplog) -> None:
    """And says so. Silently degrading a working deployment to polling because somebody typed
    `web-socket` would be a performance regression nobody could find."""
    with caplog.at_level("WARNING"):
        assert configured_transport({TRANSPORT_ENV: "web-socket"}) is Transport.SOCKET

    assert "not a transport" in caplog.text


def test_only_the_socket_wants_the_refresh_loop() -> None:
    assert Transport.SOCKET.needs_refresh_loop is True
    assert Transport.POLL.needs_refresh_loop is False


# --- what the client is told ----------------------------------------------------


def test_the_client_is_told_which_transport_to_use(source: StubSource) -> None:
    """The server's answer, not the client's guess: a browser cannot tell a host that refuses
    WebSocket upgrades from one that is briefly down, and those want opposite responses."""
    with polling_client(source) as client:
        config = client.get("/api/config").json()

    assert config["transport"] == "poll"
    assert config["poll_seconds"] > 0
    assert config["read_only"] is True


def test_the_refresh_loop_does_not_run_under_polling(source: StubSource) -> None:
    """There would be nobody to push to, and on a serverless host it would be started on every
    cold start and killed with the invocation."""
    with polling_client(source) as client:
        client.get("/api/config")
        assert client.app.state.hub.rooms == 0
        # One read for the config route's hub state and nothing periodic behind it.
        before = source.loads
        client.get("/api/config")
        assert source.loads == before


# --- the conditional GET ----------------------------------------------------------


def test_a_snapshot_carries_its_revision_as_an_etag(source: StubSource) -> None:
    with polling_client(source) as client:
        response = client.get("/api/snapshot", params=VIEW)

    assert response.status_code == 200
    assert response.headers["ETag"] == f'"{response.json()["revision"]}"'


def test_an_unchanged_view_answers_304_with_no_body(source: StubSource) -> None:
    """The whole reason polling is affordable: a quiet market costs an empty round trip
    instead of 150KB of identical JSON."""
    with polling_client(source) as client:
        first = client.get("/api/snapshot", params=VIEW).json()
        again = client.get("/api/snapshot", params={**VIEW, "since": first["revision"]})

    assert again.status_code == 304
    assert again.content == b""


def test_a_changed_view_answers_with_the_whole_snapshot(source: StubSource) -> None:
    with polling_client(source) as client:
        first = client.get("/api/snapshot", params=VIEW).json()

        source.evaluations = (evaluation(reason="something happened"),)
        second = client.get("/api/snapshot", params={**VIEW, "since": first["revision"]})

    assert second.status_code == 200
    assert second.json()["revision"] != first["revision"]
    assert second.json()["feed"]["entries"][0]["reason"] == "something happened"


def test_a_stale_revision_is_not_mistaken_for_a_current_one(source: StubSource) -> None:
    with polling_client(source) as client:
        assert client.get("/api/snapshot", params={**VIEW, "since": "nonsense"}).status_code == 200


def test_the_socket_and_the_poll_agree_on_the_revision(source: StubSource) -> None:
    """Both transports compare the same hash, so a client that switches between them does not
    redraw for a change that never happened."""
    with TestClient(create_app(source=source, transport=Transport.SOCKET)) as client:
        over_http = client.get("/api/snapshot", params=VIEW).json()["revision"]
        with client.websocket_connect("/ws?symbol=EURUSD&timeframe=H1") as socket:
            over_socket = socket.receive_json()["snapshot"]["revision"]

    assert over_http == over_socket


# --- an unconfigured store ----------------------------------------------------------


def test_an_unconfigured_store_still_serves_a_page_that_says_why() -> None:
    """A serverless deploy with no SUPABASE_DB_URL used to raise during import, so every
    request became an opaque 500 with the reason in a log nobody was reading."""
    from fxagent.dashboard.source import UnavailableSource

    reason = "no database URL configured; set SUPABASE_DB_URL"
    with TestClient(create_app(source=UnavailableSource(reason), transport=Transport.POLL)) as c:
        assert c.get("/").status_code == 200
        assert c.get("/api/config").status_code == 200

        health = c.get("/api/health")
        assert health.status_code == 503
        assert reason in health.json()["store_error"]

        snapshot = c.get("/api/snapshot", params=VIEW)
        assert snapshot.status_code == 503
        assert reason in snapshot.json()["error"]


def test_a_configuration_error_falls_back_but_a_connection_error_does_not() -> None:
    """ "Supabase is having a moment" and "you forgot a variable" send you to different places,
    so only the second one is caught here."""
    from fxagent.dashboard.source import UnavailableSource, store_or_unavailable
    from fxagent.store.config import DatabaseConfigError

    def unconfigured():
        raise DatabaseConfigError("no database URL configured")

    source, database = store_or_unavailable(unconfigured)
    assert isinstance(source, UnavailableSource)
    assert database is None

    def unreachable():
        raise OSError("connection refused")

    with pytest.raises(OSError, match="connection refused"):
        store_or_unavailable(unreachable)
