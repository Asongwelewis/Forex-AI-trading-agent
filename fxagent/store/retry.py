"""Retry with exponential backoff for transient database failures.

The hard part is not the backoff, it is deciding what deserves one. Retrying a genuine error
turns a fast, clear failure into a slow, confusing one, and retrying a write that already
landed can double it.

Only *connection-shaped* failures are retried here: the socket dropped, the pool could not
hand out a connection, the server restarted. Those are the ones Supabase actually produces —
pooler restarts, idle-connection reaps, brief network loss on a home connection.

Everything the database has an opinion about is permanent and raises immediately.
`IntegrityError` means a genuine unique-constraint hit; retrying it just hits it again.
`ProgrammingError` means the SQL is wrong. `DataError` means the value is wrong. A caller
seeing one of those should see it on the first attempt, with the original traceback.

`sleep` and `jitter` are injected so the tests exercise the real schedule without waiting.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy.exc import DBAPIError, DisconnectionError, InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as SQLATimeoutError

__all__ = ["RetryPolicy", "is_transient", "with_retry"]

logger = logging.getLogger(__name__)


#: Exception types that mean "the connection failed", not "the statement was wrong".
#: OperationalError is included but narrowed further by `is_transient` — SQLAlchemy raises it
#: for some permanent server-side conditions too.
_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    DisconnectionError,
    InterfaceError,
    SQLATimeoutError,
    ConnectionError,
    TimeoutError,
    OSError,
)

#: Substrings identifying a dropped or recycled connection inside a generic OperationalError.
_TRANSIENT_MESSAGES = (
    "server closed the connection",
    "connection was closed",
    "connection is closed",
    "connection reset",
    "cannot connect now",
    "the database system is starting up",
    "the database system is shutting down",
    "terminating connection",
    "too many clients",
    "connection refused",
    "no connection to the server",
    "ssl connection has been closed",
)


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to retry and how long to wait between attempts."""

    attempts: int = 4
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 5.0
    multiplier: float = 3.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {self.attempts}")
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")

    def delay_for(self, attempt: int) -> float:
        """Un-jittered delay before `attempt` (1-based). Capped at max_delay_seconds."""
        if attempt < 1:
            raise ValueError(f"attempt is 1-based, got {attempt}")
        raw = self.base_delay_seconds * (self.multiplier ** (attempt - 1))
        return min(raw, self.max_delay_seconds)


def is_transient(exc: BaseException) -> bool:
    """True if `exc` is worth retrying.

    Checked before the type test: `IntegrityError` and `ProgrammingError` are `DBAPIError`
    subclasses, and a `DBAPIError` flagged `connection_invalidated` is transient regardless of
    what its message says.
    """
    if isinstance(exc, DBAPIError) and exc.connection_invalidated:
        return True
    # Before the generic DBAPIError test below: InterfaceError and DisconnectionError are
    # DBAPIError subclasses, so checking that first would classify a dropped connection as a
    # permanent statement error and never retry it.
    if isinstance(exc, DisconnectionError | InterfaceError | SQLATimeoutError):
        return True
    if isinstance(exc, OperationalError):
        text = str(exc).lower()
        return any(marker in text for marker in _TRANSIENT_MESSAGES)
    if isinstance(exc, DBAPIError):
        # IntegrityError, ProgrammingError, DataError and friends. The server understood the
        # statement and rejected it; the same statement will be rejected again.
        return False
    return isinstance(exc, _TRANSIENT_TYPES)


async def with_retry[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    description: str = "database operation",
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> T:
    """Run `operation`, retrying transient failures on an exponential backoff.

    `operation` is a zero-argument coroutine factory rather than a coroutine, because a
    coroutine object cannot be awaited twice — taking one would make retrying impossible.

    Full jitter (a uniform draw over [0, delay]) rather than a fixed wait: when the pooler
    restarts, every worker fails at the same instant, and an unjittered backoff marches them
    all back in lockstep to fail together again.
    """
    active = policy or RetryPolicy()

    for attempt in range(1, active.attempts + 1):
        try:
            return await operation()
        # `Exception`, not `BaseException`: asyncio.CancelledError is a BaseException, and
        # retrying a cancelled operation would ignore the cancellation and keep working.
        except Exception as exc:
            if not is_transient(exc):
                raise
            if attempt == active.attempts:
                logger.error("%s failed after %d attempts", description, active.attempts)
                raise
            delay = jitter(0.0, active.delay_for(attempt))
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.2fs",
                description,
                attempt,
                active.attempts,
                type(exc).__name__,
                delay,
            )
            await sleep(delay)

    # Unreachable: the loop either returns, or raises on its final attempt.
    raise AssertionError(f"retry loop for {description} exited without returning or raising")
