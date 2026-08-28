"""Send a card. Nothing comes back.

**There are no Approve/Reject buttons, and that is the design.** The original Phase 8 plan had
inline buttons that would place an order on approve. That makes Telegram an execution channel
authorised by a chat ID — a second authorisation path beside the permission grant, weaker than
it, and reachable by anyone who obtains the bot token. The deterministic permission layer gates
execution and nothing else does (CLAUDE.md hard rule 4 and its note on `proceed_recommendation`).

So this module has one public verb, `send`, and no reader of any kind. There is no webhook, no
long poll, no update parsing, and no callback registry. A future need to *acknowledge* a card
is a read of a store the human wrote to by other means, not an instruction arriving over a
messaging API.

**It never raises.** A notifier is a convenience wrapped around a record that has already been
written. Every failure path returns `False` after logging the actual cause — an unset token, a
403 from a chat the bot was removed from, a timeout, a 429. The trader counts the miss and
carries on, because a decision that was journalled and not announced is a decision that
happened.

**Credentials are read from the environment and never logged.** `__repr__` shows whether a
token is configured, never any part of it, and the token does not appear in the failure
messages either — a 401 says the token is wrong, not what it was.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Final

import httpx

__all__ = ["TelegramNotifier", "TelegramSettings"]

logger = logging.getLogger(__name__)

API_BASE: Final = "https://api.telegram.org"

#: Telegram rejects a message body above this. Cards are far shorter; a truncated card is
#: still worth sending, and a 400 for length would lose it entirely.
MAX_MESSAGE_CHARS: Final = 4096

DEFAULT_TIMEOUT: Final = 10.0


@dataclass(frozen=True)
class TelegramSettings:
    """Bot token and destination chat. Both required; neither is ever logged."""

    bot_token: str
    chat_id: str
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> TelegramSettings | None:
        """Build from the environment, or `None` when it is not configured.

        `None` rather than an exception: running without Telegram is a supported mode, not a
        misconfiguration. The trader still journals every decision — which is the part that
        matters — and logs once that it has nowhere to send.
        """
        source = os.environ if env is None else env
        token = (source.get("TELEGRAM_BOT_TOKEN") or "").strip()
        chat = (source.get("TELEGRAM_CHAT_ID") or "").strip()
        if not token or not chat:
            return None
        return cls(bot_token=token, chat_id=chat)


class TelegramNotifier:
    """One outbound method. Satisfies `fxagent.trader.service.Notifier`."""

    def __init__(
        self, settings: TelegramSettings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None

    def __repr__(self) -> str:
        """Never leaks the token — this object ends up in logs and tracebacks."""
        return f"TelegramNotifier(chat_id={self._settings.chat_id!r}, token=<configured>)"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> TelegramNotifier | None:
        settings = TelegramSettings.from_env(env)
        if settings is None:
            logger.info(
                "Telegram is not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID); "
                "decisions will be journalled but not announced"
            )
            return None
        return cls(settings)

    async def send(self, text: str) -> bool:
        """Post one message. Returns whether it was delivered; never raises.

        The body is sent as JSON rather than interpolated into a URL or a shell command: a
        card carries prices, a strategy name and a free-text reason, and any of those can hold
        a character that would otherwise need escaping in three different ways.
        """
        if not text.strip():
            logger.warning("refusing to send an empty message")
            return False

        body = text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 1] + "…"
        payload: dict[str, Any] = {
            "chat_id": self._settings.chat_id,
            "text": body,
            # No parse_mode. A price like `1.1000` or a reason containing an underscore would
            # be a Markdown parse error, and Telegram rejects the whole message for it —
            # losing a card to formatting is a worse trade than plain text.
            "disable_web_page_preview": True,
        }

        client = self._client or httpx.AsyncClient(timeout=self._settings.timeout)
        try:
            response = await client.post(
                f"{API_BASE}/bot{self._settings.bot_token}/sendMessage", json=payload
            )
        except httpx.HTTPError as exc:
            # Named, not swallowed. "Telegram send failed" with no cause is the log line that
            # makes an outage take an hour to diagnose.
            logger.warning("Telegram send failed: %s: %s", type(exc).__name__, exc)
            return False
        finally:
            if self._owns_client:
                await client.aclose()

        if response.status_code == 200:
            return True

        logger.warning(
            "Telegram refused the message: HTTP %s %s",
            response.status_code,
            _describe(response),
        )
        return False

    async def aclose(self) -> None:
        """Close an injected client. A no-op when each send owns its own."""
        if self._client is not None:
            await self._client.aclose()


def _describe(response: httpx.Response) -> str:
    """Telegram's own error text, which is far more useful than the status code alone.

    A 400 says `chat not found` or `message is too long`; a 403 says the bot was blocked or
    removed from the chat. Guarded because an error page is not always JSON, and a notifier
    that raised while reporting a failure would replace a missing card with a traceback.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict):
        return str(payload.get("description") or payload)[:200]
    return str(payload)[:200]
