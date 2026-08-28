"""The conditions that end a grant, each one a pure function of observable state.

**Pure, so they are testable and so they cannot lie.** Every trigger takes a snapshot and
returns a reason or `None`. None of them reads a clock, a socket or a file; the caller gathers
the facts and the trigger judges them. That is what lets each one have a test that constructs
the exact condition rather than trying to arrange the world into it.

**They fail closed, and the calendar is the sharp case.** The Forex Factory feed covers the
current week only — `ff_calendar_nextweek.json` is a 404 — so on a Friday the lookahead cannot
see Sunday's open. Absence of a known event is therefore not evidence of no event, and an
unavailable or stale calendar revokes rather than permits. This is the trigger most likely to be
"fixed" by someone who reads the refusal as a bug.

**Every threshold is named and lives here once.** A trigger whose number is written inline in
the caller is a trigger that is one copy-paste from being two different numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from fxagent.permission.grant import RevocationReason

__all__ = [
    "TriggerConfig",
    "TriggerHit",
    "TriggerSnapshot",
    "evaluate_triggers",
]

logger = logging.getLogger(__name__)

#: Hard rule-adjacent: the daily loss limit from `.env`, restated as the default here so the
#: system has a safe number when the variable is unset. It is a fraction of the equity the day
#: *started* with, not of current equity — measuring against a shrinking denominator makes the
#: limit recede exactly as it is being approached.
DEFAULT_DAILY_LOSS_LIMIT: Final = 0.03

#: Three in a row. Not because three is magic, but because a size that does not scale up after a
#: loss (hard rule 9) still leaves a losing streak burning the grant's trade budget, and a human
#: should look before the fourth.
DEFAULT_MAX_CONSECUTIVE_LOSSES: Final = 3

#: Multiple of the symbol-hour spread ceiling. Lane 2 replaces the ceiling itself with a
#: measured p90; until then the caller supplies whatever it has and this compares against it.
DEFAULT_SPREAD_CEILING_MULTIPLE: Final = 1.0

#: How close a high-impact event may be before trading stops. Wide enough to cover the release
#: itself and the first spike after it.
DEFAULT_EVENT_WINDOW: Final = timedelta(minutes=15)

#: Beyond this the broker connection is not slow, it is gone.
DEFAULT_HEARTBEAT_TIMEOUT: Final = timedelta(seconds=60)

#: A calendar older than this is not a calendar, it is a memory of one.
DEFAULT_CALENDAR_STALENESS: Final = timedelta(hours=6)


@dataclass(frozen=True)
class TriggerConfig:
    """Every threshold, in one place."""

    daily_loss_limit: float = DEFAULT_DAILY_LOSS_LIMIT
    max_consecutive_losses: int = DEFAULT_MAX_CONSECUTIVE_LOSSES
    spread_ceiling_multiple: float = DEFAULT_SPREAD_CEILING_MULTIPLE
    event_window: timedelta = DEFAULT_EVENT_WINDOW
    heartbeat_timeout: timedelta = DEFAULT_HEARTBEAT_TIMEOUT
    calendar_staleness: timedelta = DEFAULT_CALENDAR_STALENESS

    def __post_init__(self) -> None:
        if not 0 < self.daily_loss_limit <= 1:
            raise ValueError(
                f"daily_loss_limit must be a fraction in (0, 1], got {self.daily_loss_limit}"
            )
        if self.max_consecutive_losses < 1:
            raise ValueError(
                f"max_consecutive_losses must be at least 1, got {self.max_consecutive_losses}"
            )


@dataclass(frozen=True)
class TriggerSnapshot:
    """Everything the triggers judge. Gathered by the caller; nothing here fetches.

    Optionals mean **unknown**, and unknown is not safe. `account_is_demo=None` is not "probably
    demo"; `calendar_checked_at=None` is not "no events due". Each trigger below treats its own
    unknown as a revocation, which is why they are typed as optional rather than defaulted to
    something comfortable.
    """

    now: datetime
    #: Equity at the start of the trading day, and now. The denominator is the *start*.
    starting_equity: float
    current_equity: float
    consecutive_losses: int = 0
    #: Current spread in points, and the ceiling for this symbol and hour. Both or neither.
    spread_points: float | None = None
    spread_ceiling_points: float | None = None
    #: When a high-impact event affecting either leg of the pair is due, if one is.
    next_high_impact_event: datetime | None = None
    #: When the calendar was last successfully fetched. `None` means never.
    calendar_checked_at: datetime | None = None
    #: Last heard from the broker.
    last_heartbeat: datetime | None = None
    #: From `terminal_info().trade_allowed` and `account_info().trade_mode`.
    trading_allowed: bool | None = None
    account_is_demo: bool | None = None


@dataclass(frozen=True)
class TriggerHit:
    """One condition that fired, and the sentence a human reads in the log."""

    reason: RevocationReason
    detail: str

    def __str__(self) -> str:
        return f"{self.reason.value}: {self.detail}"


def _daily_loss(snapshot: TriggerSnapshot, config: TriggerConfig) -> TriggerHit | None:
    if snapshot.starting_equity <= 0:
        return TriggerHit(
            RevocationReason.DAILY_LOSS,
            "the day's starting equity is not a positive number, so the loss limit cannot be "
            "evaluated; refusing rather than trading against an unknown denominator",
        )
    drawdown = (snapshot.starting_equity - snapshot.current_equity) / snapshot.starting_equity
    if drawdown >= config.daily_loss_limit:
        return TriggerHit(
            RevocationReason.DAILY_LOSS,
            f"down {drawdown:.2%} on the day against a {config.daily_loss_limit:.2%} limit "
            f"({snapshot.starting_equity:.2f} to {snapshot.current_equity:.2f})",
        )
    return None


def _consecutive_losses(snapshot: TriggerSnapshot, config: TriggerConfig) -> TriggerHit | None:
    if snapshot.consecutive_losses >= config.max_consecutive_losses:
        return TriggerHit(
            RevocationReason.CONSECUTIVE_LOSSES,
            f"{snapshot.consecutive_losses} losing trades in a row, limit is "
            f"{config.max_consecutive_losses}",
        )
    return None


def _spread(snapshot: TriggerSnapshot, config: TriggerConfig) -> TriggerHit | None:
    """Both readings or neither. A spread with no ceiling to judge it against is unknown."""
    if snapshot.spread_points is None or snapshot.spread_ceiling_points is None:
        return None
    ceiling = snapshot.spread_ceiling_points * config.spread_ceiling_multiple
    if snapshot.spread_points > ceiling:
        return TriggerHit(
            RevocationReason.SPREAD_TOO_WIDE,
            f"spread is {snapshot.spread_points:.1f} points against a ceiling of "
            f"{ceiling:.1f}; the book has widened and a breakout fill here is not the "
            "fill the backtest measured",
        )
    return None


def _calendar(snapshot: TriggerSnapshot, config: TriggerConfig) -> TriggerHit | None:
    """Fails closed, twice over. Read the module docstring before relaxing either branch."""
    if snapshot.calendar_checked_at is None:
        return TriggerHit(
            RevocationReason.CALENDAR_UNAVAILABLE,
            "the calendar has never been fetched. The feed covers the current week only, so "
            "on a Friday the lookahead cannot see Sunday's open — absence of a known event is "
            "not evidence of no event",
        )

    age = snapshot.now - snapshot.calendar_checked_at
    if age > config.calendar_staleness:
        return TriggerHit(
            RevocationReason.CALENDAR_UNAVAILABLE,
            f"the calendar is {age} old, past the {config.calendar_staleness} limit; a stale "
            "calendar cannot rule out an event that was added since",
        )

    event = snapshot.next_high_impact_event
    if event is not None:
        until = event - snapshot.now
        if -config.event_window <= until <= config.event_window:
            return TriggerHit(
                RevocationReason.CALENDAR_EVENT,
                f"a high-impact event is due at {event.isoformat()}, which is "
                f"{until} away and inside the {config.event_window} window",
            )
    return None


def _heartbeat(snapshot: TriggerSnapshot, config: TriggerConfig) -> TriggerHit | None:
    if snapshot.last_heartbeat is None:
        return TriggerHit(
            RevocationReason.HEARTBEAT_LOST,
            "no broker heartbeat has been recorded; an unknown connection is a lost one",
        )
    age = snapshot.now - snapshot.last_heartbeat
    if age > config.heartbeat_timeout:
        return TriggerHit(
            RevocationReason.HEARTBEAT_LOST,
            f"the broker has been silent for {age}, past the {config.heartbeat_timeout} limit",
        )
    return None


def _terminal(snapshot: TriggerSnapshot, config: TriggerConfig) -> TriggerHit | None:
    """`mt5.initialize()` succeeds with Algo Trading off; only `order_send` fails later.

    So this is checked continuously rather than at startup: the button can be turned off by a
    person, or by the terminal itself after an update, at any point in a session.
    """
    if snapshot.trading_allowed is not True:
        return TriggerHit(
            RevocationReason.TRADING_DISABLED,
            "the terminal does not report trade_allowed. Algo Trading is off, or the terminal "
            "state is unknown — mt5.initialize() succeeds either way and only order_send fails",
        )
    return None


def _demo(snapshot: TriggerSnapshot, config: TriggerConfig) -> TriggerHit | None:
    """Hard rule 1. The one trigger whose failure is not a risk decision but a fatal error."""
    if snapshot.account_is_demo is not True:
        return TriggerHit(
            RevocationReason.NOT_DEMO,
            "the attached account is not confirmed DEMO. This is a fatal condition, not a "
            "warning: a real-money account attached to this process must never trade",
        )
    return None


#: Order matters only for which reason is reported first when several fire at once, and the
#: order is deliberate: the two that mean "this must never have been possible" come first.
_TRIGGERS: Final = (
    _demo,
    _terminal,
    _daily_loss,
    _consecutive_losses,
    _calendar,
    _heartbeat,
    _spread,
)


def evaluate_triggers(
    snapshot: TriggerSnapshot, config: TriggerConfig | None = None
) -> list[TriggerHit]:
    """Every trigger that fired, in the order above. Empty means nothing objects.

    All of them are evaluated rather than short-circuiting on the first. A revocation caused by
    three conditions at once is a different event from one caused by a single condition, and the
    log should say so — the day the spread blew out *and* the heartbeat died is a story about
    the connection, not about the spread.
    """
    settings = config or TriggerConfig()
    hits = [hit for trigger in _TRIGGERS if (hit := trigger(snapshot, settings)) is not None]
    for hit in hits:
        logger.warning("auto-revoke trigger fired: %s", hit)
    return hits


def day_start(moment: datetime) -> datetime:
    """UTC midnight for `moment` — the boundary the daily loss limit resets on.

    UTC rather than the broker's day, and rather than the local one. Exness runs UTC+0 (measured,
    see CLAUDE.md) so today they coincide; naming the choice here means a broker change moves one
    function rather than silently shifting when the limit resets.
    """
    return moment.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
