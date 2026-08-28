"""Permission grant state machine and auto-revoke triggers.

Default state is ADVISORY, in which execution is impossible. Grants expire, and
state is persisted so a restart cannot silently resurrect a revoked grant.
Fails closed: unreadable state means ADVISORY.

Three modules, one responsibility each:

* `grant` — what a grant is, how it is stored, and whether one authorises this symbol now.
* `triggers` — the conditions that end a grant, each a pure function of a snapshot.
* `preflight` — the single check immediately before an order, asserting all of it at once.

**This package authorises. It does not execute.** Nothing here imports an adapter or places an
order; it answers yes or no and says why. The caller that holds a broker connection is the one
that acts on the answer, and `tests/permission/test_permission_cannot_execute.py` keeps the
separation honest.

Two properties worth stating because both are easy to erode:

**No argument turns execution on.** There is no `force`, no `override`, no `skip_checks`, and
ADVISORY is the absence of a grant rather than a flag that could be set. An escape hatch that
existed would eventually be used.

**Unknown is never safe.** An unreadable state file, an unfetched calendar, a heartbeat never
recorded, an account whose demo status is not confirmed — each of those refuses. The failure
mode being avoided is a system that trades harder the more broken it is.
"""

from __future__ import annotations

from fxagent.permission.grant import (
    Decision,
    GrantManager,
    GrantState,
    PermissionGrant,
    RevocationReason,
    latest_permitted_expiry,
)
from fxagent.permission.preflight import Preflight, PreflightResult, check_order, enforce
from fxagent.permission.triggers import (
    TriggerConfig,
    TriggerHit,
    TriggerSnapshot,
    day_start,
    evaluate_triggers,
)

__all__ = [
    "Decision",
    "GrantManager",
    "GrantState",
    "PermissionGrant",
    "Preflight",
    "PreflightResult",
    "RevocationReason",
    "TriggerConfig",
    "TriggerHit",
    "TriggerSnapshot",
    "check_order",
    "day_start",
    "enforce",
    "evaluate_triggers",
    "latest_permitted_expiry",
]
