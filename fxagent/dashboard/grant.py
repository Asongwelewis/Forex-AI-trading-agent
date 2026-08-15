"""Execution permission, as far as a read-only panel is concerned.

`fxagent.permission` is a stub: the grant state machine has not been built. So the honest
answer to "may this system execute" is **no**, and this module says exactly that rather than
leaving the panel's permission card blank.

Two properties are deliberate and should survive whatever replaces the default reader:

**It fails closed.** `AdvisoryOnly` returns ADVISORY unconditionally, and a reader that raises
is caught by the caller and rendered as ADVISORY with the error attached. There is no path
through this file that reports GRANTED because something was unavailable — an unreadable
permission state is not a permissive one, and a dashboard that showed a phantom grant would be
teaching its reader to trust a display of the one thing in this system that must never be
guessed at.

**It only reads.** There is no grant, revoke or extend here, and no route in the dashboard
reaches anything that could add one. Approving a trade from a web page is a different feature
with a different threat model — CLAUDE.md puts approvals on Telegram with an authenticated chat
id, and an unauthenticated LAN page is not that.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

from fxagent.dashboard.models import GrantSnapshot, GrantState

__all__ = ["AdvisoryOnly", "GrantReader", "read_grant"]

logger = logging.getLogger(__name__)

#: Shown on the permission card while the state machine does not exist. Written out rather than
#: left empty because "ADVISORY" alone reads as a mode that could be changed from this screen.
NOT_BUILT = (
    "The permission state machine has not been built (fxagent.permission is a stub), so no "
    "grant can exist and execution is impossible. This panel never grants anything."
)


class GrantReader(Protocol):
    """Reads the current grant. Async because a real one will read the store."""

    async def current(self, now: datetime) -> GrantSnapshot: ...


class AdvisoryOnly:
    """The only reader this build has. Always ADVISORY, for the reason it states."""

    def __init__(self, reason: str = NOT_BUILT) -> None:
        self._reason = reason

    async def current(self, now: datetime) -> GrantSnapshot:  # noqa: ARG002 - protocol shape
        return GrantSnapshot(
            state=GrantState.ADVISORY,
            reason=self._reason,
            source="fxagent.dashboard.grant.AdvisoryOnly",
        )


async def read_grant(reader: GrantReader, now: datetime) -> GrantSnapshot:
    """Ask `reader` for the grant, degrading to ADVISORY if it cannot answer.

    Broad `except` on purpose. Every failure mode of a future reader — a dropped connection, a
    row that will not parse, a bug — has the same correct answer here, and enumerating the ones
    thought of today would mean the one nobody thought of propagates as a 500 and takes the
    whole panel down with it.
    """
    try:
        return await reader.current(now)
    except Exception:  # noqa: BLE001 - see the docstring; unreadable must mean ADVISORY
        logger.exception("grant reader failed; reporting ADVISORY")
        return GrantSnapshot(
            state=GrantState.ADVISORY,
            reason="The permission state could not be read, so it is reported as ADVISORY.",
            source=type(reader).__name__,
        )
