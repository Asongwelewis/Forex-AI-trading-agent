"""The last check before the wire. Written to answer one question, repeatedly:

*What would have to be true for this to place a trade I did not authorise?*

Each test below closes one answer. The negative cases matter more than the positive one, so
there is exactly one test of the happy path and the rest are refusals.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fxagent.adapters.base import OrderRequest, OrderSide
from fxagent.permission.grant import GrantManager, RevocationReason
from fxagent.permission.preflight import Preflight, check_order, enforce
from fxagent.permission.triggers import TriggerSnapshot
from fxagent.risk.exposure import MAX_TOTAL_RISK

NOW = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
LOGIN = 476187411


def _order(**overrides) -> OrderRequest:
    base = {
        "symbol": "EURUSD",
        "side": OrderSide.BUY,
        "volume": 0.1,
        "entry_price": 1.1000,
        "stop_loss": 1.0980,
        "take_profit": 1.1040,
    }
    base.update(overrides)
    return OrderRequest(**base)


def _snapshot(**overrides) -> TriggerSnapshot:
    base = {
        "now": NOW,
        "starting_equity": 1000.0,
        "current_equity": 1000.0,
        "spread_points": 12.0,
        "spread_ceiling_points": 20.0,
        "calendar_checked_at": NOW - timedelta(minutes=5),
        "last_heartbeat": NOW - timedelta(seconds=5),
        "trading_allowed": True,
        "account_is_demo": True,
    }
    base.update(overrides)
    return TriggerSnapshot(**base)


@pytest.fixture
def granted(tmp_path: Path) -> GrantManager:
    manager = GrantManager(tmp_path / "grant.json", now=lambda: NOW)
    manager.issue(
        allowed_symbols={"EURUSD"},
        max_trades=3,
        max_notional=100_000.0,
        expires_at=NOW + timedelta(hours=6),
    )
    return manager


def _check(manager: GrantManager, **overrides):
    facts = {
        "snapshot": _snapshot(),
        "open_risk_fraction": 0.0,
        "order_risk_fraction": 0.005,
        "notional": 10_000.0,
        "expected_login": LOGIN,
        "actual_login": LOGIN,
    }
    order = overrides.pop("order", _order())
    facts.update(overrides)
    return check_order(order, manager=manager, **facts)


# --- the one path that says yes -------------------------------------------------


def test_a_clean_order_under_a_live_grant_is_cleared(granted: GrantManager) -> None:
    result = _check(granted)
    assert result.allowed, result.reason
    assert result.hits == ()


# --- everything that says no ------------------------------------------------------


def test_no_grant_means_no(tmp_path: Path) -> None:
    manager = GrantManager(tmp_path / "grant.json", now=lambda: NOW)
    result = _check(manager)
    assert not result
    assert "no grant exists" in result.reason


def test_a_revoked_grant_means_no(granted: GrantManager) -> None:
    granted.revoke(RevocationReason.MANUAL, "operator")
    assert not _check(granted)


def test_a_symbol_outside_the_grant_means_no(granted: GrantManager) -> None:
    result = _check(granted, order=_order(symbol="GBPUSD"))
    assert not result
    assert "GBPUSD" in result.reason


def test_a_non_demo_account_means_no_before_anything_else(granted: GrantManager) -> None:
    """Hard rule 1, re-asserted here rather than only at connect time."""
    result = _check(granted, snapshot=_snapshot(account_is_demo=False))
    assert not result
    assert "DEMO" in result.reason


def test_an_unconfirmed_demo_status_means_no(granted: GrantManager) -> None:
    assert not _check(granted, snapshot=_snapshot(account_is_demo=None))


def test_a_terminal_logged_into_the_wrong_account_means_no(granted: GrantManager) -> None:
    """The login is pinned. A terminal that switched accounts is not the one we checked."""
    result = _check(granted, actual_login=99999999)
    assert not result
    assert "99999999" in result.reason


def test_algo_trading_switched_off_means_no(granted: GrantManager) -> None:
    """`mt5.initialize()` succeeded; only `order_send` would fail. So it is checked here."""
    result = _check(granted, snapshot=_snapshot(trading_allowed=False))
    assert not result


def test_any_firing_trigger_stops_the_order(granted: GrantManager) -> None:
    result = _check(granted, snapshot=_snapshot(spread_points=99.0))
    assert not result
    assert "auto-revoke condition" in result.reason


def test_the_total_open_risk_cap_counts_this_order_too(granted: GrantManager) -> None:
    """The cap is on the book *after* the order lands, not the book before it."""
    result = _check(granted, open_risk_fraction=MAX_TOTAL_RISK, order_risk_fraction=0.005)
    assert not result
    assert "total open risk" in result.reason


def test_risk_that_lands_exactly_on_the_cap_is_allowed(granted: GrantManager) -> None:
    result = _check(
        granted, open_risk_fraction=MAX_TOTAL_RISK - 0.005, order_risk_fraction=0.005
    )
    assert result.allowed, result.reason


def test_notional_past_the_grants_ceiling_means_no(granted: GrantManager) -> None:
    result = _check(granted, notional=1_000_000.0)
    assert not result
    assert "notional" in result.reason


def test_an_exhausted_trade_budget_means_no(granted: GrantManager) -> None:
    for _ in range(3):
        granted.spend()
    result = _check(granted)
    assert not result
    assert "all used" in result.reason


# --- there is no way round it -------------------------------------------------------


def test_an_order_cannot_be_built_without_a_stop_or_a_target() -> None:
    """Hard rule 3, enforced by the type rather than by this check.

    The pre-flight asserts it anyway, but the reason it can be belt and braces is that
    `OrderRequest` makes the naked order unconstructible in the first place.
    """
    with pytest.raises(ValueError):
        _order(stop_loss=None)
    with pytest.raises(ValueError):
        _order(take_profit=None)


def test_check_order_has_no_argument_that_bypasses_it() -> None:
    """An escape hatch that existed would eventually be passed."""
    import inspect

    parameters = set(inspect.signature(check_order).parameters)
    for forbidden in ("force", "skip_checks", "override", "bypass", "allow_anyway"):
        assert forbidden not in parameters, f"check_order takes {forbidden!r}"


def test_every_refusal_states_a_reason(granted: GrantManager) -> None:
    """A system that refuses silently is one nobody can debug on a Monday morning."""
    refusals = [
        _check(granted, snapshot=_snapshot(account_is_demo=False)),
        _check(granted, actual_login=1),
        _check(granted, order=_order(symbol="GBPUSD")),
        _check(granted, open_risk_fraction=0.5),
        _check(granted, notional=10**9),
        _check(granted, snapshot=_snapshot(last_heartbeat=None)),
    ]
    for refusal in refusals:
        assert not refusal
        assert len(refusal.reason) > 20, f"unhelpful refusal: {refusal.reason!r}"


# --- acting on the verdict -----------------------------------------------------------


def test_a_standing_condition_ends_the_grant(granted: GrantManager) -> None:
    """A dead heartbeat is not a reason to skip one order; it is a reason to stop."""
    result = enforce(_check(granted, snapshot=_snapshot(last_heartbeat=None)), granted)

    assert result.revoking_hit is not None
    assert granted.grant is not None and granted.grant.revoked
    assert granted.grant.revocation_reason is RevocationReason.HEARTBEAT_LOST


def test_a_passing_condition_does_not_end_the_grant(granted: GrantManager) -> None:
    """A spread that is wide right now will be narrow in ten minutes. Skip the order, keep the
    grant — otherwise one news minute costs the whole session.
    """
    result = enforce(_check(granted, snapshot=_snapshot(spread_points=99.0)), granted)

    assert not result
    assert result.revoking_hit is None
    assert granted.grant is not None and not granted.grant.revoked


def test_a_clean_check_changes_nothing(granted: GrantManager) -> None:
    enforce(_check(granted), granted)
    assert granted.grant is not None and not granted.grant.revoked
    assert granted.grant.trades_used == 0, "checking is not spending"


def test_the_preflight_binding_checks_and_enforces_together(granted: GrantManager) -> None:
    preflight = Preflight(manager=granted)

    result = preflight(
        _order(),
        snapshot=_snapshot(account_is_demo=False),
        open_risk_fraction=0.0,
        order_risk_fraction=0.005,
        notional=10_000.0,
    )

    assert not result
    assert granted.grant is not None and granted.grant.revoked
