"""The single check no order path may skip.

Hard rule 1 says re-assert before **every** order, not just at startup, and the reason is
specific: `mt5.initialize()` succeeds with Algo Trading off and only `order_send` fails later.
A startup-only check therefore passes, and the system then trades into a terminal that will not
execute — or, worse, into one that has since been pointed at a different account.

**Everything is asserted here, in one place, immediately before the order.** The grant, the
demo status, the login, the terminal's trade flag, the auto-revoke triggers, the exposure cap,
the notional cap, and the presence of a stop and a target on *this* request. Splitting them
across the call path is how one of them ends up on a branch that some other branch skips.

**A refusal revokes when the cause is a standing condition.** A spread that is too wide right
now is a reason to skip this order; an account that is not demo, or a heartbeat that has died,
is a reason for the grant to end. `check_order` returns both the decision and the trigger hits,
and `enforce` is the wrapper that acts on them — kept separate so the checking is pure and
testable and the acting is one obvious place.

**There is no `force`, no `skip_checks` and no `override`.** If an argument existed to bypass
this, it would eventually be passed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fxagent.adapters.base import OrderRequest
from fxagent.permission.grant import Decision, GrantManager, GrantState, RevocationReason
from fxagent.permission.triggers import (
    TriggerConfig,
    TriggerHit,
    TriggerSnapshot,
    evaluate_triggers,
)
from fxagent.risk.exposure import MAX_TOTAL_RISK

__all__ = ["PreflightResult", "check_order", "enforce"]

logger = logging.getLogger(__name__)

#: Reasons that describe a standing condition rather than a passing one. Hitting any of these
#: ends the grant; the others only refuse the order in front of us.
_STANDING: frozenset[RevocationReason] = frozenset(
    {
        RevocationReason.NOT_DEMO,
        RevocationReason.TRADING_DISABLED,
        RevocationReason.DAILY_LOSS,
        RevocationReason.CONSECUTIVE_LOSSES,
        RevocationReason.HEARTBEAT_LOST,
        RevocationReason.CALENDAR_UNAVAILABLE,
    }
)


@dataclass(frozen=True)
class PreflightResult:
    """The verdict, the reasons, and whether the grant should end because of them."""

    decision: Decision
    hits: tuple[TriggerHit, ...] = ()

    def __bool__(self) -> bool:
        return self.decision.allowed

    @property
    def allowed(self) -> bool:
        return self.decision.allowed

    @property
    def reason(self) -> str:
        return self.decision.reason

    @property
    def revoking_hit(self) -> TriggerHit | None:
        """The first standing condition, if any. A passing condition returns `None`."""
        for hit in self.hits:
            if hit.reason in _STANDING:
                return hit
        return None


def check_order(
    order: OrderRequest,
    *,
    manager: GrantManager,
    snapshot: TriggerSnapshot,
    open_risk_fraction: float,
    order_risk_fraction: float,
    notional: float,
    expected_login: int | None = None,
    actual_login: int | None = None,
    trigger_config: TriggerConfig | None = None,
) -> PreflightResult:
    """Pure. Gathers no facts and changes nothing — see `enforce` for the acting half.

    Ordered so that the cheapest structural refusals come first and the most alarming reasons
    are reported first when several apply. Every branch names what failed.
    """
    hits = tuple(evaluate_triggers(snapshot, trigger_config))

    # 1. Hard rule 3, checked on the request in hand rather than on a policy somewhere.
    #    `OrderRequest` cannot be constructed without these, so this is belt and braces — but
    #    it is the last point before the wire, and that is where belt and braces belong.
    if order.stop_loss is None or order.take_profit is None:  # pragma: no cover - type-forbidden
        return PreflightResult(
            Decision(
                False,
                "the order carries no stop or no target; every order attaches both in the "
                "same request (hard rule 3)",
                manager.state(snapshot.now),
            ),
            hits,
        )

    # 2. Hard rule 1, re-asserted here and not merely at connect time.
    if snapshot.account_is_demo is not True:
        return PreflightResult(
            Decision(
                False,
                "the attached account is not confirmed DEMO; refusing every order",
                manager.state(snapshot.now),
            ),
            hits,
        )

    if expected_login is not None and actual_login != expected_login:
        return PreflightResult(
            Decision(
                False,
                f"the terminal is logged into account {actual_login}, not the pinned "
                f"{expected_login}; refusing rather than trading an account we did not check",
                manager.state(snapshot.now),
            ),
            hits,
        )

    # 3. Any trigger at all stops this order, whether or not it also ends the grant.
    if hits:
        return PreflightResult(
            Decision(False, f"auto-revoke condition: {hits[0]}", manager.state(snapshot.now)),
            hits,
        )

    # 4. The grant itself: exists, live, covers this symbol, has budget left.
    granted = manager.check(order.symbol, moment=snapshot.now)
    if not granted:
        return PreflightResult(granted, hits)

    # 5. Caps. Total open risk is checked *including* this order, because the cap is on the
    #    book after it lands, not on the book before.
    total = open_risk_fraction + order_risk_fraction
    if total > MAX_TOTAL_RISK:
        return PreflightResult(
            Decision(
                False,
                f"this order would take total open risk to {total:.2%}, past the "
                f"{MAX_TOTAL_RISK:.2%} cap (hard rule 8)",
                granted.state,
            ),
            hits,
        )

    grant = manager.grant
    if grant is not None and notional > grant.max_notional:
        return PreflightResult(
            Decision(
                False,
                f"notional {notional:.2f} exceeds the grant's ceiling of {grant.max_notional:.2f}",
                granted.state,
            ),
            hits,
        )

    return PreflightResult(
        Decision(
            True,
            f"cleared: {granted.reason}; open risk {total:.2%} of {MAX_TOTAL_RISK:.2%}",
            GrantState.GRANTED,
        ),
        hits,
    )


def enforce(result: PreflightResult, manager: GrantManager) -> PreflightResult:
    """Act on a checked result: end the grant when the cause is a standing condition.

    Separate from `check_order` so the judgement stays pure and the side effect has exactly one
    home. Returns the result unchanged so it can be used inline.
    """
    hit = result.revoking_hit
    if hit is not None:
        manager.revoke(hit.reason, hit.detail)
    return result


@dataclass
class Preflight:
    """Convenience binding of a manager and its trigger configuration.

    Exists so a caller holds one object rather than threading two through every call site, and
    so there is an obvious place for a future gatherer to live. It adds no policy.
    """

    manager: GrantManager
    trigger_config: TriggerConfig = field(default_factory=TriggerConfig)

    def __call__(self, order: OrderRequest, **facts: Any) -> PreflightResult:
        return enforce(
            check_order(
                order,
                manager=self.manager,
                trigger_config=self.trigger_config,
                **facts,
            ),
            self.manager,
        )
