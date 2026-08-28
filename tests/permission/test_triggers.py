"""Every auto-revoke trigger, one test each, plus the unknown case for every one of them.

The board card says "each one tested". The harder half is the second test in each pair: what
the trigger does when the fact it judges is *missing*. Unknown is not safe, and every optional
field on `TriggerSnapshot` is a place where a comfortable default would quietly re-enable
trading in exactly the conditions the trigger exists to catch.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.permission.grant import RevocationReason
from fxagent.permission.triggers import (
    TriggerConfig,
    TriggerSnapshot,
    day_start,
    evaluate_triggers,
)

NOW = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


def _snapshot(**overrides) -> TriggerSnapshot:
    """A snapshot where nothing is wrong, so each test changes exactly one thing."""
    base = {
        "now": NOW,
        "starting_equity": 1000.0,
        "current_equity": 1000.0,
        "consecutive_losses": 0,
        "spread_points": 12.0,
        "spread_ceiling_points": 20.0,
        "next_high_impact_event": None,
        "calendar_checked_at": NOW - timedelta(minutes=5),
        "last_heartbeat": NOW - timedelta(seconds=5),
        "trading_allowed": True,
        "account_is_demo": True,
    }
    base.update(overrides)
    return TriggerSnapshot(**base)


def _reasons(snapshot: TriggerSnapshot, config: TriggerConfig | None = None):
    return {hit.reason for hit in evaluate_triggers(snapshot, config)}


def test_a_healthy_snapshot_fires_nothing() -> None:
    """Guards every test below: if the baseline tripped, they would all pass vacuously."""
    assert evaluate_triggers(_snapshot()) == []


# --- daily loss ------------------------------------------------------------------


def test_daily_loss_past_the_limit_revokes() -> None:
    assert RevocationReason.DAILY_LOSS in _reasons(_snapshot(current_equity=960.0))


def test_daily_loss_exactly_at_the_limit_revokes() -> None:
    """`>=`, not `>`. A limit you may sit exactly on is a limit with an off-by-one in it."""
    assert RevocationReason.DAILY_LOSS in _reasons(_snapshot(current_equity=970.0))


def test_daily_loss_just_inside_the_limit_does_not() -> None:
    assert RevocationReason.DAILY_LOSS not in _reasons(_snapshot(current_equity=971.0))


def test_the_denominator_is_the_days_start_not_current_equity() -> None:
    """Measuring against a shrinking denominator makes the limit recede as it is approached.

    Down 30 on a 1000 start is 3% and must revoke. Against the current 970 it would be 3.09%
    of *that*, and a naive implementation using current equity computes 0% — always.
    """
    assert RevocationReason.DAILY_LOSS in _reasons(
        _snapshot(starting_equity=1000.0, current_equity=970.0)
    )


def test_an_unknown_starting_equity_revokes_rather_than_dividing_by_it() -> None:
    assert RevocationReason.DAILY_LOSS in _reasons(_snapshot(starting_equity=0.0))


# --- consecutive losses ------------------------------------------------------------


def test_three_consecutive_losses_revokes() -> None:
    assert RevocationReason.CONSECUTIVE_LOSSES in _reasons(_snapshot(consecutive_losses=3))


def test_two_consecutive_losses_does_not() -> None:
    assert RevocationReason.CONSECUTIVE_LOSSES not in _reasons(_snapshot(consecutive_losses=2))


# --- spread ---------------------------------------------------------------------------


def test_a_spread_past_the_ceiling_revokes() -> None:
    assert RevocationReason.SPREAD_TOO_WIDE in _reasons(_snapshot(spread_points=25.0))


def test_a_spread_at_the_ceiling_does_not() -> None:
    assert RevocationReason.SPREAD_TOO_WIDE not in _reasons(_snapshot(spread_points=20.0))


@pytest.mark.parametrize(
    ("spread", "ceiling"), [(None, 20.0), (25.0, None), (None, None)]
)
def test_a_spread_with_no_ceiling_to_judge_it_is_not_a_spread_trigger(
    spread: float | None, ceiling: float | None
) -> None:
    """Deliberately does NOT revoke, unlike the other unknowns.

    A missing spread reading is not evidence the book has widened, and revoking on it would
    make the trigger fire constantly before Lane 2 measures the per-symbol-hour ceilings. The
    order path still refuses — `check_order` requires the whole snapshot to be clean — so the
    unknown is caught there rather than by ending the grant.
    """
    assert RevocationReason.SPREAD_TOO_WIDE not in _reasons(
        _snapshot(spread_points=spread, spread_ceiling_points=ceiling)
    )


# --- the calendar, which fails closed twice ---------------------------------------------


def test_a_high_impact_event_inside_the_window_revokes() -> None:
    assert RevocationReason.CALENDAR_EVENT in _reasons(
        _snapshot(next_high_impact_event=NOW + timedelta(minutes=10))
    )


def test_an_event_just_after_the_release_also_revokes() -> None:
    """The window is symmetric. The spike after the print is the part that fills badly."""
    assert RevocationReason.CALENDAR_EVENT in _reasons(
        _snapshot(next_high_impact_event=NOW - timedelta(minutes=10))
    )


def test_an_event_outside_the_window_does_not() -> None:
    assert RevocationReason.CALENDAR_EVENT not in _reasons(
        _snapshot(next_high_impact_event=NOW + timedelta(hours=3))
    )


def test_a_calendar_never_fetched_revokes() -> None:
    """The sharp one. The feed is current-week-only, so on a Friday the lookahead cannot see
    Sunday's open — absence of a known event is not evidence of no event.
    """
    assert RevocationReason.CALENDAR_UNAVAILABLE in _reasons(
        _snapshot(calendar_checked_at=None)
    )


def test_a_stale_calendar_revokes() -> None:
    assert RevocationReason.CALENDAR_UNAVAILABLE in _reasons(
        _snapshot(calendar_checked_at=NOW - timedelta(hours=12))
    )


def test_no_known_event_with_a_fresh_calendar_is_fine() -> None:
    assert RevocationReason.CALENDAR_UNAVAILABLE not in _reasons(_snapshot())


# --- heartbeat --------------------------------------------------------------------------


def test_a_silent_broker_revokes() -> None:
    assert RevocationReason.HEARTBEAT_LOST in _reasons(
        _snapshot(last_heartbeat=NOW - timedelta(minutes=5))
    )


def test_a_heartbeat_never_recorded_revokes() -> None:
    """An unknown connection is a lost one."""
    assert RevocationReason.HEARTBEAT_LOST in _reasons(_snapshot(last_heartbeat=None))


# --- the terminal and the account -----------------------------------------------------------


def test_algo_trading_switched_off_revokes() -> None:
    """`mt5.initialize()` succeeds with Algo Trading off and only `order_send` fails later.

    So this is checked continuously rather than once at connect: the button can be turned off
    by a person, or by the terminal after an update, mid-session.
    """
    assert RevocationReason.TRADING_DISABLED in _reasons(_snapshot(trading_allowed=False))


def test_an_unknown_terminal_state_revokes() -> None:
    assert RevocationReason.TRADING_DISABLED in _reasons(_snapshot(trading_allowed=None))


def test_a_non_demo_account_revokes() -> None:
    """Hard rule 1. Not a risk decision — a fatal condition."""
    assert RevocationReason.NOT_DEMO in _reasons(_snapshot(account_is_demo=False))


def test_an_unconfirmed_demo_status_revokes() -> None:
    assert RevocationReason.NOT_DEMO in _reasons(_snapshot(account_is_demo=None))


# --- reporting -----------------------------------------------------------------------------


def test_every_firing_trigger_is_reported_not_just_the_first() -> None:
    """The day the spread blew out AND the heartbeat died is a story about the connection.

    Short-circuiting on the first hit would record it as a story about the spread.
    """
    hits = evaluate_triggers(
        _snapshot(
            current_equity=900.0,
            consecutive_losses=5,
            last_heartbeat=None,
            spread_points=99.0,
        )
    )

    reasons = {hit.reason for hit in hits}
    assert {
        RevocationReason.DAILY_LOSS,
        RevocationReason.CONSECUTIVE_LOSSES,
        RevocationReason.HEARTBEAT_LOST,
        RevocationReason.SPREAD_TOO_WIDE,
    } <= reasons


def test_the_gravest_reasons_are_reported_first() -> None:
    """`NOT_DEMO` before a spread complaint: the log's first line should be the alarming one."""
    hits = evaluate_triggers(_snapshot(account_is_demo=False, spread_points=99.0))
    assert hits[0].reason is RevocationReason.NOT_DEMO


def test_every_hit_carries_a_human_readable_detail() -> None:
    for snapshot in (
        _snapshot(current_equity=900.0),
        _snapshot(consecutive_losses=9),
        _snapshot(spread_points=99.0),
        _snapshot(calendar_checked_at=None),
        _snapshot(last_heartbeat=None),
        _snapshot(trading_allowed=False),
        _snapshot(account_is_demo=False),
    ):
        for hit in evaluate_triggers(snapshot):
            assert hit.detail.strip(), f"{hit.reason} fired with no explanation"
            assert len(hit.detail) > 20, f"{hit.reason}: {hit.detail!r} is not an explanation"


# --- configuration -------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [0.0, -0.1, 1.5])
def test_a_nonsensical_daily_loss_limit_is_refused(limit: float) -> None:
    with pytest.raises(ValueError, match="daily_loss_limit"):
        TriggerConfig(daily_loss_limit=limit)


def test_the_day_boundary_is_utc_midnight() -> None:
    """Exness runs UTC+0 (measured), so today this coincides with the broker day.

    Named in one function so a broker change moves one place rather than silently shifting
    when the daily loss limit resets.
    """
    assert day_start(datetime(2026, 1, 5, 23, 59, tzinfo=UTC)) == datetime(
        2026, 1, 5, tzinfo=UTC
    )
