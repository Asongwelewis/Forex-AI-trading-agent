"""Outbound notification. One direction, deliberately.

Nothing in this package reads. There is no webhook, no update poll and no callback handler,
because a messaging app that can carry an instruction back is an authorisation path — and this
system has exactly one of those, the deterministic permission layer. See
`fxagent/alerts/telegram.py` for why the Approve/Reject buttons in the original plan were
dropped rather than deferred.
"""

from __future__ import annotations

from fxagent.alerts.telegram import TelegramNotifier, TelegramSettings

__all__ = ["TelegramNotifier", "TelegramSettings"]
