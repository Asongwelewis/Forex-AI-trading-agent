"""Shared repository plumbing.

A repository wraps a session it does not own. That is what lets two of them take part in one
transaction — `write the evaluation, then the trade it produced, or neither` is a single
`Database.begin()` block with an EvaluationRepository and a TradeRepository inside it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["Repository", "require_utc"]


def require_utc(value: datetime, field: str) -> datetime:
    """Reject a naive datetime and normalise an aware one to UTC.

    Postgres would happily accept a naive value and interpret it in the session timezone,
    which is how a point-in-time filter ends up an hour wrong twice a year. The boundary
    rejects instead of guessing, matching `fxagent.adapters.base`.
    """
    if value.tzinfo is None:
        raise ValueError(
            f"{field} must be timezone-aware; naive datetimes are rejected because Postgres "
            "would interpret them in the session timezone"
        )
    return value.astimezone(UTC)


class Repository:
    """Base for the concrete repositories. Holds a session, never a connection."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def session(self) -> AsyncSession:
        return self._session

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
