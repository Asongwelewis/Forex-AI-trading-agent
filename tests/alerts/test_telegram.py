"""The notifier sends and never raises, and there is no way to send anything back.

The second half matters more than the first. The original Phase 8 plan had inline
Approve/Reject buttons that would place an order on approve, which makes a chat ID an
authorisation credential. These tests assert the absence of that path structurally, so
re-adding it is a visible decision rather than a convenience someone adds on a Friday.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import pytest

import fxagent.alerts
from fxagent.alerts.telegram import (
    MAX_MESSAGE_CHARS,
    TelegramNotifier,
    TelegramSettings,
)

SETTINGS = TelegramSettings(bot_token="123:secret-token-value", chat_id="4242")


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"ok": True})


# --- it sends ------------------------------------------------------------------


async def test_a_card_is_posted_as_json_to_the_configured_chat() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.read().decode()
        return _ok(request)

    async with _client(handler) as client:
        assert await TelegramNotifier(SETTINGS, client=client).send("EURUSD LONG") is True

    assert "sendMessage" in str(seen["url"])
    body = json.loads(str(seen["json"]))
    assert body["chat_id"] == "4242"
    assert body["text"] == "EURUSD LONG"


async def test_no_parse_mode_is_requested() -> None:
    """A price like 1.1000 or an underscore in a reason is a Markdown parse error.

    Telegram rejects the whole message for it, so formatting would cost cards.
    """
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return _ok(request)

    async with _client(handler) as client:
        await TelegramNotifier(SETTINGS, client=client).send("stop_loss 1.0980 *not bold*")

    assert "parse_mode" not in seen["body"]


async def test_an_over_long_message_is_truncated_rather_than_lost() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return _ok(request)

    async with _client(handler) as client:
        assert await TelegramNotifier(SETTINGS, client=client).send("x" * 9000) is True

    assert seen["body"].count("x") <= MAX_MESSAGE_CHARS


# --- it never raises -------------------------------------------------------------


async def test_a_transport_error_returns_false_rather_than_propagating() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    async with _client(handler) as client:
        assert await TelegramNotifier(SETTINGS, client=client).send("hello") is False


@pytest.mark.parametrize("status", [400, 403, 429, 500])
async def test_a_refusal_returns_false(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"ok": False, "description": "chat not found"})

    async with _client(handler) as client:
        assert await TelegramNotifier(SETTINGS, client=client).send("hello") is False


async def test_a_non_json_error_page_does_not_break_the_error_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A notifier that raised while reporting a failure would swap a missing card for a crash."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>bad gateway</html>")

    async with _client(handler) as client:
        assert await TelegramNotifier(SETTINGS, client=client).send("hello") is False


async def test_an_empty_message_is_refused_before_the_network() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _ok(request)

    async with _client(handler) as client:
        assert await TelegramNotifier(SETTINGS, client=client).send("   ") is False

    assert calls == 0


# --- credentials -------------------------------------------------------------------


def test_missing_configuration_yields_none_not_an_exception() -> None:
    """Running without Telegram is a supported mode; the journal is the part that matters."""
    assert TelegramSettings.from_env({}) is None
    assert TelegramSettings.from_env({"TELEGRAM_BOT_TOKEN": "t"}) is None
    assert TelegramSettings.from_env({"TELEGRAM_BOT_TOKEN": " ", "TELEGRAM_CHAT_ID": " "}) is None
    assert TelegramNotifier.from_env({}) is None


def test_the_repr_never_contains_the_token() -> None:
    """This object ends up in logs and tracebacks."""
    assert "secret-token-value" not in repr(TelegramNotifier(SETTINGS))


async def test_the_token_is_not_written_to_the_log_on_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"ok": False, "description": "Unauthorized"})

    with caplog.at_level("WARNING"):
        async with _client(handler) as client:
            await TelegramNotifier(SETTINGS, client=client).send("hello")

    assert "secret-token-value" not in caplog.text
    assert "Unauthorized" in caplog.text, "the cause must still be named"


# --- there is no way back ------------------------------------------------------------


def test_the_package_exposes_no_reader() -> None:
    """No webhook, no long poll, no update parsing, no callback registry.

    An inbound path would make a chat ID an authorisation credential, beside — and weaker
    than — the deterministic permission layer that is supposed to be the only one.
    """
    forbidden = ("getUpdates", "get_updates", "webhook", "callback", "answer_callback_query")
    source = Path(fxagent.alerts.telegram.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    for name in forbidden:
        assert name.lower() not in lowered.replace("no webhook", "").replace(
            "no callback", ""
        ), f"{name} appears in the notifier; this package sends and does not read"


def test_the_notifier_has_exactly_one_public_verb() -> None:
    # vars(), not getmembers: `from_env` is a classmethod, and inspect.isfunction does not
    # see one. A predicate that quietly misses the constructors would make this test weaker
    # than it reads.
    public = {name for name in vars(TelegramNotifier) if not name.startswith("_")}
    assert public == {"send", "aclose", "from_env"}, (
        f"TelegramNotifier grew a method: {sorted(public)}. An inbound one is an "
        "authorisation path; check before adding."
    )


def test_no_reply_markup_is_ever_attached() -> None:
    """Inline keyboards are how the dropped Approve/Reject flow would come back."""
    tree = ast.parse(Path(fxagent.alerts.telegram.__file__).read_text(encoding="utf-8"))
    constants = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
    }
    assert "reply_markup" not in constants
    assert "inline_keyboard" not in constants
