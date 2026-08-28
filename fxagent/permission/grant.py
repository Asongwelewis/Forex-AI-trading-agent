"""The grant: the only thing that can authorise an order, and it fails closed.

**ADVISORY is not a flag. It is the absence of a grant.** `state` is derived from whether a
grant exists and is still valid, never stored as an independent field that could disagree with
one. There is no `set_state`, no `force`, and no argument anywhere in this module that turns
execution on without a grant object saying so.

**Every "no" is stated with a reason.** `Decision` carries the reason on the refusal path as
well as the approval path, because a system that refuses silently is a system nobody can debug
at 09:00 on a Monday, and because the refusals are data — the same argument that made the
selector's ledger unconditional.

**Unreadable state means ADVISORY.** A corrupt file, a missing file, a file written by a newer
version, a file whose JSON parses but whose fields do not validate: all of them mean no grant.
The alternative — treating an unparseable file as "carry on" — is a system that trades harder
the more broken it is.

**Expiry can never cross the weekend.** `expires_at` is clamped to the next weekly close, which
comes from `regime.sessions` rather than a fixed UTC constant, because the FX week closes at
17:00 New York and that is 21:00 UTC in winter and 22:00 in summer. A grant that survived into
Sunday's open would hold a position through the gap that grants exist to avoid.

**A grant is spent, not just timed.** `max_trades` counts down as orders are authorised, and the
count is persisted with the grant. A process restart cannot refill it — which is the point: the
restart is exactly when a naive implementation forgets and hands out the budget again.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fxagent.adapters.base import UtcDatetime
from fxagent.regime.sessions import next_weekly_close

__all__ = [
    "Decision",
    "GrantManager",
    "GrantState",
    "PermissionGrant",
    "RevocationReason",
]

logger = logging.getLogger(__name__)

#: How far before the weekly close a grant must already have expired. Positions are flat before
#: the weekend, not at the moment of it — the last hour of the week is thin, and a stop filled
#: in it fills badly.
CLOSE_MARGIN_MINUTES: Final = 60


class GrantState(StrEnum):
    """Derived, never stored. See the module docstring."""

    ADVISORY = "ADVISORY"
    GRANTED = "GRANTED"
    REVOKED = "REVOKED"


class RevocationReason(StrEnum):
    """Why a grant ended. Every one of these is logged and persisted with the grant.

    Named rather than free text so that a revocation can be counted, compared across weeks and
    charted. "Three consecutive losses" happening twelve times in a month is a finding; twelve
    distinct sentences saying roughly that are not.
    """

    MANUAL = "MANUAL"
    EXPIRED = "EXPIRED"
    DAILY_LOSS = "DAILY_LOSS"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    CALENDAR_EVENT = "CALENDAR_EVENT"
    CALENDAR_UNAVAILABLE = "CALENDAR_UNAVAILABLE"
    HEARTBEAT_LOST = "HEARTBEAT_LOST"
    TRADING_DISABLED = "TRADING_DISABLED"
    NOT_DEMO = "NOT_DEMO"
    TRADES_EXHAUSTED = "TRADES_EXHAUSTED"
    KILL_SWITCH = "KILL_SWITCH"


class PermissionGrant(BaseModel):
    """One authorisation to trade, bounded four ways: symbols, count, notional and time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    granted_at: UtcDatetime
    expires_at: UtcDatetime
    #: Canonical symbols, no broker suffix. Empty is refused rather than read as "all" — a
    #: grant that means everything should have to say everything.
    allowed_symbols: frozenset[str] = Field(min_length=1)
    max_trades: int = Field(gt=0)
    max_notional: float = Field(gt=0)
    trades_used: int = Field(default=0, ge=0)
    revoked: bool = False
    revocation_reason: RevocationReason | None = None
    revocation_detail: str = ""

    @model_validator(mode="after")
    def _check_grant(self) -> PermissionGrant:
        if self.expires_at <= self.granted_at:
            raise ValueError(
                f"expires_at {self.expires_at.isoformat()} is not after granted_at "
                f"{self.granted_at.isoformat()}; a grant that has already expired is not a grant"
            )
        if self.trades_used > self.max_trades:
            raise ValueError(
                f"trades_used {self.trades_used} exceeds max_trades {self.max_trades}"
            )
        if self.revoked and self.revocation_reason is None:
            raise ValueError("a revoked grant must say why; an unexplained revocation is a bug")
        return self

    @property
    def trades_remaining(self) -> int:
        return max(0, self.max_trades - self.trades_used)

    def is_live_at(self, moment: datetime) -> bool:
        """Whether this grant authorises anything at `moment`. Not a state, a question."""
        return (
            not self.revoked
            and moment < self.expires_at
            and moment >= self.granted_at
            and self.trades_remaining > 0
        )

    def covers(self, symbol: str) -> bool:
        return symbol in self.allowed_symbols

    def revoke(self, reason: RevocationReason, detail: str = "") -> PermissionGrant:
        """A new grant marked revoked. Grants are frozen, so revoking returns rather than mutates.

        Idempotent: revoking an already-revoked grant keeps the *first* reason. The first cause
        is the one worth knowing; a later expiry overwriting "daily loss exceeded" would erase
        the interesting half of the record.
        """
        if self.revoked:
            return self
        return self.model_copy(
            update={
                "revoked": True,
                "revocation_reason": reason,
                "revocation_detail": detail,
            }
        )

    def spend(self, count: int = 1) -> PermissionGrant:
        """Consume `count` trades from the budget. Never goes negative, never refills."""
        if count < 1:
            raise ValueError(f"count must be positive, got {count}")
        used = min(self.max_trades, self.trades_used + count)
        return self.model_copy(update={"trades_used": used})


@dataclass(frozen=True)
class Decision:
    """May this proceed, and why not. The reason is present on both paths."""

    allowed: bool
    reason: str
    state: GrantState

    def __bool__(self) -> bool:
        return self.allowed

    def __str__(self) -> str:
        verdict = "ALLOWED" if self.allowed else "REFUSED"
        return f"{verdict} [{self.state}]: {self.reason}"


def latest_permitted_expiry(moment: datetime) -> datetime:
    """The furthest a grant may run: an hour before the FX week closes.

    Derived from `regime.sessions`, which defines the week in `America/New_York` and converts —
    London is 17:00 New York at 21:00 UTC in winter and 22:00 in summer, and a fixed UTC
    constant here would be an hour wrong for half the year, in the direction of holding a
    position into the gap.
    """
    return next_weekly_close(moment) - timedelta(minutes=CLOSE_MARGIN_MINUTES)


class GrantManager:
    """Holds at most one grant, persists it, and answers whether an order may proceed.

    The state file is the authority across restarts, and it is read on construction rather than
    cached from a previous process. A manager that trusted an in-memory grant after a crash
    would resurrect one the crash was supposed to end.
    """

    def __init__(self, path: Path | str, *, now: Any = None) -> None:
        self._path = Path(path)
        self._now = now or (lambda: datetime.now(UTC))
        self._grant = self._load()

    def __repr__(self) -> str:
        return f"GrantManager(path={str(self._path)!r}, state={self.state()})"

    # -- reading ---------------------------------------------------------------

    @property
    def grant(self) -> PermissionGrant | None:
        return self._grant

    def state(self, moment: datetime | None = None) -> GrantState:
        """Derived from the grant, never stored beside it."""
        at = moment or self._now()
        if self._grant is None:
            return GrantState.ADVISORY
        if self._grant.revoked:
            return GrantState.REVOKED
        if not self._grant.is_live_at(at):
            return GrantState.ADVISORY
        return GrantState.GRANTED

    def check(self, symbol: str, *, moment: datetime | None = None) -> Decision:
        """Whether an order on `symbol` is authorised right now.

        This is the *grant* question only. It says nothing about spread, calendar, exposure or
        whether the terminal will accept an order — those belong to the pre-flight, which calls
        this and then asks the rest. Keeping them apart means neither can be satisfied by
        accident when the other passes.
        """
        at = moment or self._now()
        state = self.state(at)

        if self._grant is None:
            return Decision(False, "no grant exists; the default state is advisory", state)

        grant = self._grant
        if grant.revoked:
            reason = grant.revocation_reason.value if grant.revocation_reason else "unknown"
            detail = f" ({grant.revocation_detail})" if grant.revocation_detail else ""
            return Decision(False, f"the grant was revoked: {reason}{detail}", state)

        if at >= grant.expires_at:
            return Decision(
                False, f"the grant expired at {grant.expires_at.isoformat()}", state
            )

        if at < grant.granted_at:
            # A clock that went backwards, or a grant restored from a file written ahead of us.
            # Refusing is the only safe reading: the alternative is honouring a grant that has
            # not started.
            return Decision(
                False,
                f"the grant does not begin until {grant.granted_at.isoformat()}; "
                "refusing rather than assuming the clock is right",
                state,
            )

        if grant.trades_remaining <= 0:
            return Decision(
                False, f"the grant's {grant.max_trades} trade(s) are all used", state
            )

        if not grant.covers(symbol):
            return Decision(
                False,
                f"the grant covers {sorted(grant.allowed_symbols)}, not {symbol}",
                state,
            )

        return Decision(
            True,
            f"granted until {grant.expires_at.isoformat()}, "
            f"{grant.trades_remaining} trade(s) left",
            state,
        )

    # -- writing ---------------------------------------------------------------

    def issue(
        self,
        *,
        allowed_symbols: frozenset[str] | set[str] | list[str],
        max_trades: int,
        max_notional: float,
        expires_at: datetime,
    ) -> PermissionGrant:
        """Create and persist a grant. The expiry is clamped, never rejected.

        Clamped rather than rejected because the caller asking for Monday is not making an
        error, they are asking for something the weekend rule forbids — and silently giving them
        Friday is both what they meant and what is safe. It is logged so it is not a surprise.
        """
        now = self._now()
        ceiling = latest_permitted_expiry(now)
        effective = min(expires_at.astimezone(UTC), ceiling)
        if effective < expires_at.astimezone(UTC):
            logger.warning(
                "grant expiry clamped from %s to %s: no grant may cross the weekend gap",
                expires_at.isoformat(),
                effective.isoformat(),
            )

        grant = PermissionGrant(
            granted_at=now,
            expires_at=effective,
            allowed_symbols=frozenset(allowed_symbols),
            max_trades=max_trades,
            max_notional=max_notional,
        )
        self._store(grant)
        logger.warning(
            "PERMISSION GRANTED: %s, %d trade(s), max notional %s, until %s",
            sorted(grant.allowed_symbols),
            grant.max_trades,
            grant.max_notional,
            grant.expires_at.isoformat(),
        )
        return grant

    def revoke(self, reason: RevocationReason, detail: str = "") -> Decision:
        """End the grant. Safe to call when there is none, and safe to call twice."""
        if self._grant is None:
            return Decision(False, "there was no grant to revoke", GrantState.ADVISORY)

        already = self._grant.revoked
        self._store(self._grant.revoke(reason, detail))
        if not already:
            logger.warning(
                "PERMISSION REVOKED: %s%s", reason.value, f" - {detail}" if detail else ""
            )
        return Decision(False, f"revoked: {reason.value}", GrantState.REVOKED)

    def spend(self, count: int = 1) -> None:
        """Record that a trade was authorised. Persisted immediately.

        Called *after* the order is accepted, and persisted before the caller continues, so a
        crash between the fill and the next bar cannot hand the budget back.
        """
        if self._grant is None:
            raise RuntimeError("cannot spend from a grant that does not exist")
        self._store(self._grant.spend(count))

    def kill(self, detail: str = "") -> Decision:
        """Revoke immediately. Flattening positions is the caller's job — see `killswitch`."""
        return self.revoke(RevocationReason.KILL_SWITCH, detail)

    # -- persistence -----------------------------------------------------------

    def _store(self, grant: PermissionGrant) -> None:
        self._grant = grant
        payload = grant.model_dump(mode="json")
        payload["allowed_symbols"] = sorted(grant.allowed_symbols)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Written to a temporary file and moved into place. A half-written state file is
        # unreadable, and unreadable means ADVISORY — safe, but it would silently destroy a
        # live grant on a power cut, which on this hardware is a Tuesday.
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self._path)

    def _load(self) -> PermissionGrant | None:
        """Read the stored grant, or `None`. Every failure path is `None`, loudly."""
        if not self._path.exists():
            logger.info("no grant file at %s; starting in ADVISORY", self._path)
            return None

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "grant file at %s is unreadable (%s); failing closed to ADVISORY",
                self._path,
                exc,
            )
            return None

        try:
            return PermissionGrant.model_validate(raw)
        except ValidationError as exc:
            logger.error(
                "grant file at %s does not validate (%s); failing closed to ADVISORY. "
                "A file this process cannot understand is not permission to trade.",
                self._path,
                exc.error_count(),
            )
            return None
