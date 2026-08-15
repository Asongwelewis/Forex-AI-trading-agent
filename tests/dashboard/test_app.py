"""The routes and the socket, through Starlette's real request path.

`test_no_route_can_change_anything` is the security argument for this service written down as
an assertion. The dashboard binds 0.0.0.0 with no authentication, and the only reason that is
acceptable is that there is nothing behind it to attack. That property has to be checked
mechanically, because it is the kind of thing a single convenient POST erases forever.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fxagent.dashboard.app import STATIC_DIR, create_app
from fxagent.dashboard.grant import AdvisoryOnly
from fxagent.dashboard.models import GrantSnapshot, GrantState
from tests.dashboard.builders import evaluation, trade, vote
from tests.dashboard.stubs import FailingSource, StubSource

READ_ONLY_METHODS = {"GET", "HEAD", "OPTIONS"}


@pytest.fixture
def source() -> StubSource:
    return StubSource(
        evaluations=(
            evaluation(
                identifier=11,
                fired=True,
                reason="2 strategies agreed on LONG with summed weight 1.50",
                consensus_score=0.72,
                votes=[
                    vote("session_breakout", direction="LONG", confidence=0.8, participated=True),
                    vote("carry_divergence", weight=0.5, direction="LONG", participated=True),
                ],
                extra={
                    "agents": {
                        "chartist": {"text": "Broke the Asian high on the first London bar."}
                    },
                    "patterns": [{"name": "marubozu", "definition": "A body with no wicks."}],
                },
            ),
        ),
        trades=(trade(identifier=5, evaluation_id=11),),
    )


@pytest.fixture
def client(source: StubSource) -> TestClient:
    with TestClient(create_app(source=source, refresh_seconds=0.02)) as running:
        yield running


# --- the read-only claim -------------------------------------------------------


def test_no_route_can_change_anything() -> None:
    """Hard rule 1's perimeter. A single POST here would move this service into the blast
    radius of every network it is reachable from."""
    app = create_app(source=StubSource())

    for route in app.routes:
        methods = getattr(route, "methods", None)
        if methods is None:
            continue  # the WebSocket route, which has no HTTP methods at all
        assert methods <= READ_ONLY_METHODS, f"{route.path} exposes {methods - READ_ONLY_METHODS}"


def test_the_health_route_states_the_same_thing() -> None:
    app = create_app(source=StubSource())
    with TestClient(app) as client:
        assert client.get("/api/health").json()["read_only"] is True


# --- routes ---------------------------------------------------------------------


def test_the_page_is_served_from_this_container(client: TestClient) -> None:
    body = client.get("/").text

    assert "instrument panel" in body
    assert "/static/vendor/lightweight-charts" in body


def test_the_switchers_are_built_from_series_that_actually_have_bars(
    client: TestClient,
) -> None:
    options = client.get("/api/options").json()

    assert {option["symbol"] for option in options} == {"EURUSD", "GBPUSD"}
    assert all(option["bars"] > 0 for option in options)


def test_a_snapshot_carries_both_panes(client: TestClient) -> None:
    payload = client.get("/api/snapshot", params={"symbol": "EURUSD"}).json()

    assert payload["symbol"] == "EURUSD"
    assert payload["chart"]["candles"]
    assert payload["chart"]["session_bands"]
    assert payload["feed"]["entries"][0]["fired"] is True
    assert payload["feed"]["grant"]["state"] == "ADVISORY"


def test_a_snapshot_carries_the_markers_and_the_trade_levels(client: TestClient) -> None:
    chart = client.get("/api/snapshot", params={"symbol": "EURUSD"}).json()["chart"]

    assert {marker["strategy"] for marker in chart["markers"] if marker["strategy"]} == {
        "session_breakout",
        "carry_divergence",
    }
    levels = chart["trades"][0]
    assert levels["stop_price"] and levels["target_price"]


def test_the_panel_shows_the_agent_narration_and_the_formation(client: TestClient) -> None:
    entry = client.get("/api/snapshot", params={"symbol": "EURUSD"}).json()["feed"]["entries"][0]

    assert entry["chartist"]["text"].startswith("Broke the Asian high")
    assert entry["patterns"][0]["label"] == "CONTEXT ONLY — NOT A SIGNAL"


def test_an_absurd_bar_count_is_clamped_rather_than_rejected(client: TestClient) -> None:
    """It arrives from a query string. A 400 is less useful than a chart and a limit."""
    response = client.get("/api/snapshot", params={"symbol": "EURUSD", "bars": 10_000_000})

    assert response.status_code == 200


def test_a_missing_symbol_is_a_bad_request(client: TestClient) -> None:
    assert client.get("/api/snapshot").status_code == 422


# --- health ----------------------------------------------------------------------


def test_health_reports_who_is_watching_what(client: TestClient) -> None:
    detail = client.get("/api/health").json()

    assert detail["status"] in {"ok", "degraded"}
    assert detail["series"] == 2
    assert detail["rooms"] == 0


def test_an_unreadable_store_is_a_503_naming_the_reason_not_a_500() -> None:
    """ "The panel is broken" and "the panel is fine and the database is not" are the two
    answers a reader needs to tell apart, and a traceback tells them neither."""
    with TestClient(create_app(source=FailingSource())) as client:
        snapshot = client.get("/api/snapshot", params={"symbol": "EURUSD"})
        options = client.get("/api/options")

    assert snapshot.status_code == 503
    assert "connection refused" in snapshot.json()["error"]
    assert options.status_code == 503


def test_an_unreadable_store_is_degraded_rather_than_a_stack_trace() -> None:
    with TestClient(create_app(source=FailingSource())) as client:
        response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert "connection refused" in response.json()["store_error"]


def test_health_notices_a_missing_chart_library(client: TestClient) -> None:
    """The least diagnosable failure here is a blank pane and a 404 in a console nobody opened."""
    detail = client.get("/api/health").json()

    assert detail["chart_library"] == ("present" if _vendored() else "missing")


def _vendored() -> bool:
    return (STATIC_DIR / "vendor" / "lightweight-charts.standalone.production.js").exists()


# --- the socket --------------------------------------------------------------------


def test_a_snapshot_arrives_on_connect_without_being_asked_for(client: TestClient) -> None:
    with client.websocket_connect("/ws?symbol=EURUSD&timeframe=H1") as socket:
        envelope = socket.receive_json()

    assert envelope["type"] == "snapshot"
    assert envelope["snapshot"]["chart"]["candles"]


def test_a_new_evaluation_is_pushed_to_an_open_socket(
    client: TestClient, source: StubSource
) -> None:
    """The end-to-end push path: the browser asked once and is told about the change."""
    with client.websocket_connect("/ws?symbol=EURUSD&timeframe=H1") as socket:
        first = socket.receive_json()["snapshot"]

        source.evaluations = (
            *source.evaluations,
            evaluation(identifier=12, reason="a second look at the same hour"),
        )

        second = socket.receive_json()["snapshot"]

    assert second["revision"] != first["revision"]
    assert len(second["feed"]["entries"]) == 2


def test_a_socket_onto_an_unreadable_store_is_told_why_and_closed() -> None:
    with (
        TestClient(create_app(source=FailingSource())) as client,
        client.websocket_connect("/ws?symbol=EURUSD") as socket,
    ):
        envelope = socket.receive_json()

    assert envelope["type"] == "error"
    assert "connection refused" in envelope["message"]


def test_a_disconnected_client_stops_being_refreshed(client: TestClient) -> None:
    hub = client.app.state.hub

    with client.websocket_connect("/ws?symbol=EURUSD"):
        pass

    # The room is torn down in the endpoint's `finally`, which the context manager's close
    # triggers; give the app's loop a turn to run it.
    for _ in range(50):
        if hub.rooms == 0:
            break
        client.get("/api/health")

    assert hub.rooms == 0


# --- permission ----------------------------------------------------------------------


def test_a_granted_state_is_displayed_with_its_expiry_and_never_changed_here() -> None:
    class Granted(AdvisoryOnly):
        async def current(self, now):
            return GrantSnapshot(
                state=GrantState.GRANTED,
                reason="granted for the London session",
                granted_at="2026-01-12T08:00:00+00:00",
                expires_at="2026-01-12T12:00:00+00:00",
                symbols=("EURUSD",),
                source="test",
            )

    with TestClient(create_app(source=StubSource(), grants=Granted())) as client:
        grant = client.get("/api/snapshot", params={"symbol": "EURUSD"}).json()["feed"]["grant"]

    # The instant travels; the countdown is the browser's job, so no number here decreases.
    assert grant["state"] == "GRANTED"
    assert grant["expires_at"] == "2026-01-12T12:00:00+00:00"
