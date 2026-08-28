"""The grant: what it permits, when it stops, and what happens when its file is nonsense.

This is safety-critical code, so the tests are written to answer one question the reviewer
should be asking: *what would have to be true for this to authorise a trade I did not intend?*
Each section below closes one answer to that.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fxagent.permission.grant import (
    CLOSE_MARGIN_MINUTES,
    GrantManager,
    GrantState,
    PermissionGrant,
    RevocationReason,
    latest_permitted_expiry,
)
from fxagent.regime.sessions import next_weekly_close

#: A Monday morning, comfortably inside the trading week.
MONDAY = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
SYMBOLS = frozenset({"EURUSD"})


def _manager(tmp_path: Path, *, now: datetime = MONDAY) -> GrantManager:
    clock = {"now": now}
    manager = GrantManager(tmp_path / "grant.json", now=lambda: clock["now"])
    manager._clock = clock  # type: ignore[attr-defined]  # tests advance it
    return manager


def _advance(manager: GrantManager, to: datetime) -> None:
    manager._clock["now"] = to  # type: ignore[attr-defined]


def _issue(manager: GrantManager, **overrides) -> PermissionGrant:
    kwargs = {
        "allowed_symbols": SYMBOLS,
        "max_trades": 3,
        "max_notional": 100_000.0,
        "expires_at": MONDAY + timedelta(hours=6),
    }
    kwargs.update(overrides)
    return manager.issue(**kwargs)


# --- the default is no ---------------------------------------------------------


def test_a_fresh_manager_is_advisory_and_refuses(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    assert manager.state() is GrantState.ADVISORY
    decision = manager.check("EURUSD")
    assert not decision
    assert "no grant exists" in decision.reason


def test_there_is_no_way_to_set_the_state_directly() -> None:
    """ADVISORY is the absence of a grant, not a flag. A setter would be a bypass."""
    public = {name for name in vars(GrantManager) if not name.startswith("_")}
    for forbidden in ("set_state", "force", "override", "enable", "allow"):
        assert forbidden not in public, f"GrantManager exposes {forbidden}"


# --- what a live grant permits ---------------------------------------------------


def test_a_live_grant_permits_its_own_symbol(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    _issue(manager)

    assert manager.state() is GrantState.GRANTED
    assert manager.check("EURUSD")


def test_a_live_grant_refuses_a_symbol_it_does_not_cover(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    _issue(manager)

    decision = manager.check("GBPUSD")
    assert not decision
    assert "GBPUSD" in decision.reason


def test_an_empty_symbol_set_is_refused_not_read_as_all() -> None:
    """A grant that means everything should have to say everything."""
    with pytest.raises(ValueError, match="allowed_symbols"):
        PermissionGrant(
            granted_at=MONDAY,
            expires_at=MONDAY + timedelta(hours=1),
            allowed_symbols=frozenset(),
            max_trades=1,
            max_notional=1000.0,
        )


# --- expiry ------------------------------------------------------------------------


def test_a_grant_expires(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    _issue(manager, expires_at=MONDAY + timedelta(hours=2))

    _advance(manager, MONDAY + timedelta(hours=3))

    assert manager.state() is GrantState.ADVISORY
    assert "expired" in manager.check("EURUSD").reason


def test_expiry_is_clamped_to_before_the_weekly_close(tmp_path: Path) -> None:
    """The Friday cap, derived from New York rather than a fixed UTC hour.

    London is 17:00 New York at 21:00 UTC in winter and 22:00 in summer, so a constant here
    would be an hour wrong for half the year — in the direction of holding into the gap.
    """
    manager = _manager(tmp_path)
    grant = _issue(manager, expires_at=MONDAY + timedelta(days=14))

    ceiling = latest_permitted_expiry(MONDAY)
    assert grant.expires_at == ceiling
    assert grant.expires_at < next_weekly_close(MONDAY)
    assert next_weekly_close(MONDAY) - grant.expires_at == timedelta(
        minutes=CLOSE_MARGIN_MINUTES
    )


def test_a_grant_issued_in_july_is_also_clamped_correctly(tmp_path: Path) -> None:
    """Summer: the same rule, an hour earlier in UTC. This is the case a constant gets wrong."""
    july = datetime(2026, 7, 6, 9, 0, tzinfo=UTC)
    manager = _manager(tmp_path, now=july)
    grant = _issue(manager, expires_at=july + timedelta(days=10))

    assert grant.expires_at == latest_permitted_expiry(july)
    assert grant.expires_at < next_weekly_close(july)


def test_an_expiry_before_the_grant_starts_is_rejected() -> None:
    with pytest.raises(ValueError, match="not after granted_at"):
        PermissionGrant(
            granted_at=MONDAY,
            expires_at=MONDAY - timedelta(hours=1),
            allowed_symbols=SYMBOLS,
            max_trades=1,
            max_notional=1000.0,
        )


def test_a_clock_that_went_backwards_refuses_rather_than_honouring_the_grant(
    tmp_path: Path,
) -> None:
    """A home desktop's clock is not disciplined, and the safe reading of "not yet" is no."""
    manager = _manager(tmp_path)
    _issue(manager)

    _advance(manager, MONDAY - timedelta(hours=2))

    decision = manager.check("EURUSD")
    assert not decision
    assert "does not begin until" in decision.reason


# --- the trade budget ---------------------------------------------------------------


def test_the_trade_budget_counts_down_and_then_refuses(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    _issue(manager, max_trades=2)

    assert manager.check("EURUSD")
    manager.spend()
    assert manager.check("EURUSD")
    manager.spend()

    decision = manager.check("EURUSD")
    assert not decision
    assert "all used" in decision.reason


def test_a_restart_cannot_refill_the_budget(tmp_path: Path) -> None:
    """The restart is exactly when a naive implementation hands the budget back."""
    path = tmp_path / "grant.json"
    first = GrantManager(path, now=lambda: MONDAY)
    first.issue(
        allowed_symbols=SYMBOLS,
        max_trades=1,
        max_notional=100_000.0,
        expires_at=MONDAY + timedelta(hours=6),
    )
    first.spend()

    second = GrantManager(path, now=lambda: MONDAY)

    assert second.grant is not None
    assert second.grant.trades_remaining == 0
    assert not second.check("EURUSD")


def test_spending_never_exceeds_the_maximum() -> None:
    grant = PermissionGrant(
        granted_at=MONDAY,
        expires_at=MONDAY + timedelta(hours=1),
        allowed_symbols=SYMBOLS,
        max_trades=2,
        max_notional=1000.0,
    )
    assert grant.spend(5).trades_used == 2
    assert grant.spend(5).trades_remaining == 0


# --- revocation ------------------------------------------------------------------------


def test_a_revoked_grant_stays_revoked_across_a_restart(tmp_path: Path) -> None:
    """The whole reason state is persisted: a crash must not resurrect a revoked grant."""
    path = tmp_path / "grant.json"
    first = GrantManager(path, now=lambda: MONDAY)
    first.issue(
        allowed_symbols=SYMBOLS,
        max_trades=3,
        max_notional=100_000.0,
        expires_at=MONDAY + timedelta(hours=6),
    )
    first.revoke(RevocationReason.DAILY_LOSS, "down 3.4%")

    second = GrantManager(path, now=lambda: MONDAY)

    assert second.state() is GrantState.REVOKED
    decision = second.check("EURUSD")
    assert not decision
    assert "DAILY_LOSS" in decision.reason
    assert "down 3.4%" in decision.reason


def test_revoking_twice_keeps_the_first_reason(tmp_path: Path) -> None:
    """The first cause is the interesting one; a later expiry must not overwrite it."""
    manager = _manager(tmp_path)
    _issue(manager)

    manager.revoke(RevocationReason.DAILY_LOSS, "the real reason")
    manager.revoke(RevocationReason.EXPIRED, "later noise")

    assert manager.grant is not None
    assert manager.grant.revocation_reason is RevocationReason.DAILY_LOSS
    assert manager.grant.revocation_detail == "the real reason"


def test_revoking_with_no_grant_is_safe(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    decision = manager.revoke(RevocationReason.MANUAL)
    assert not decision


def test_the_kill_switch_revokes_immediately(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    _issue(manager)

    manager.kill("operator pulled it")

    assert manager.state() is GrantState.REVOKED
    assert manager.grant is not None
    assert manager.grant.revocation_reason is RevocationReason.KILL_SWITCH


def test_a_revoked_grant_must_say_why() -> None:
    with pytest.raises(ValueError, match="must say why"):
        PermissionGrant(
            granted_at=MONDAY,
            expires_at=MONDAY + timedelta(hours=1),
            allowed_symbols=SYMBOLS,
            max_trades=1,
            max_notional=1000.0,
            revoked=True,
        )


# --- failing closed -----------------------------------------------------------------------


def test_a_missing_state_file_means_advisory(tmp_path: Path) -> None:
    manager = GrantManager(tmp_path / "nowhere" / "grant.json", now=lambda: MONDAY)
    assert manager.state() is GrantState.ADVISORY
    assert not manager.check("EURUSD")


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json at all",
        "{",
        '{"granted_at": "2026-01-05T09:00:00+00:00"}',  # valid JSON, invalid grant
        '{"allowed_symbols": ["EURUSD"], "max_trades": -1}',
        "[]",
    ],
    ids=["empty", "garbage", "truncated", "incomplete", "invalid-field", "wrong-shape"],
)
def test_an_unreadable_state_file_means_advisory(tmp_path: Path, content: str) -> None:
    """Every failure path is ADVISORY.

    The alternative — treating an unparseable file as "carry on" — is a system that trades
    harder the more broken it is.
    """
    path = tmp_path / "grant.json"
    path.write_text(content, encoding="utf-8")

    manager = GrantManager(path, now=lambda: MONDAY)

    assert manager.state() is GrantState.ADVISORY
    assert not manager.check("EURUSD")


def test_a_state_file_claiming_more_trades_than_its_maximum_is_refused(tmp_path: Path) -> None:
    """A hand-edited file is exactly the shape a bypass attempt takes."""
    path = tmp_path / "grant.json"
    path.write_text(
        json.dumps(
            {
                "granted_at": MONDAY.isoformat(),
                "expires_at": (MONDAY + timedelta(hours=6)).isoformat(),
                "allowed_symbols": ["EURUSD"],
                "max_trades": 1,
                "max_notional": 1000.0,
                "trades_used": 5,
                "revoked": False,
                "revocation_reason": None,
                "revocation_detail": "",
            }
        ),
        encoding="utf-8",
    )

    assert GrantManager(path, now=lambda: MONDAY).state() is GrantState.ADVISORY


def test_the_state_file_is_written_atomically(tmp_path: Path) -> None:
    """A half-written file is unreadable, which is safe but destroys a live grant.

    On this hardware a power cut is a Tuesday, so the write goes to a temporary file and is
    moved into place.
    """
    manager = _manager(tmp_path)
    _issue(manager)

    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"a temporary file was left behind: {leftovers}"
    assert (tmp_path / "grant.json").exists()


def test_a_stored_grant_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "grant.json"
    first = GrantManager(path, now=lambda: MONDAY)
    issued = first.issue(
        allowed_symbols={"EURUSD", "GBPUSD"},
        max_trades=4,
        max_notional=50_000.0,
        expires_at=MONDAY + timedelta(hours=4),
    )

    restored = GrantManager(path, now=lambda: MONDAY).grant

    assert restored == issued
