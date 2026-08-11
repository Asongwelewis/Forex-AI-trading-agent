"""Credit accounting. The headline case is that call 801 is refused before it is sent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxagent.adapters.credits import (
    FREE_TIER_DAILY_CREDITS,
    CreditLedger,
    CreditLimitExceeded,
)

START = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


class FakeClock:
    """A clock the test drives, so rollover is exercised without waiting for it."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


def _ledger(clock: FakeClock, **kwargs: int) -> CreditLedger:
    # Per-minute limit lifted by default so daily-limit tests are not throttled by it.
    kwargs.setdefault("per_minute_limit", 10_000)
    return CreditLedger(now=clock, **kwargs)


def test_call_801_is_refused() -> None:
    """The requirement, stated exactly: 800 succeed, the next does not."""
    clock = FakeClock()
    ledger = _ledger(clock)

    for _ in range(FREE_TIER_DAILY_CREDITS):
        ledger.spend()

    assert ledger.spent_today == 800
    assert ledger.remaining_today == 0

    with pytest.raises(CreditLimitExceeded) as caught:
        ledger.spend()

    assert "800" in str(caught.value)
    assert ledger.spent_today == 800, "a refused call must not be counted"


def test_refusal_happens_before_the_request_not_after() -> None:
    """A refusal must leave the budget untouched, or a retry loop drifts the count."""
    clock = FakeClock()
    ledger = _ledger(clock, daily_limit=5)
    ledger.spend(5)

    for _ in range(3):
        with pytest.raises(CreditLimitExceeded):
            ledger.spend()

    assert ledger.spent_today == 5


def test_a_multi_credit_call_that_would_overshoot_is_refused_whole() -> None:
    """Partial spending would leave the caller unsure how much of a batch actually went out."""
    clock = FakeClock()
    ledger = _ledger(clock, daily_limit=10)
    ledger.spend(8)

    with pytest.raises(CreditLimitExceeded):
        ledger.spend(3)

    assert ledger.spent_today == 8
    assert ledger.can_spend(2) is True
    assert ledger.can_spend(3) is False


def test_the_budget_resets_at_utc_midnight() -> None:
    clock = FakeClock(datetime(2026, 3, 10, 23, 59, tzinfo=UTC))
    ledger = _ledger(clock, daily_limit=3)
    ledger.spend(3)

    with pytest.raises(CreditLimitExceeded):
        ledger.spend()

    clock.advance(timedelta(minutes=2))  # crosses midnight

    ledger.spend()
    assert ledger.spent_today == 1


def test_the_budget_does_not_reset_merely_because_hours_passed() -> None:
    """A rolling 24h window would be wrong: the provider resets on the UTC calendar day."""
    clock = FakeClock(datetime(2026, 3, 10, 1, 0, tzinfo=UTC))
    ledger = _ledger(clock, daily_limit=3)
    ledger.spend(3)

    clock.advance(timedelta(hours=20))  # same UTC day, 21:00

    with pytest.raises(CreditLimitExceeded):
        ledger.spend()


def test_retry_after_points_at_midnight_when_the_day_is_spent() -> None:
    clock = FakeClock(datetime(2026, 3, 10, 22, 0, tzinfo=UTC))
    ledger = _ledger(clock, daily_limit=1)
    ledger.spend()

    with pytest.raises(CreditLimitExceeded) as caught:
        ledger.spend()

    assert caught.value.retry_after == timedelta(hours=2)


# -- the per-minute limit, which is what actually throttles a backfill --------


def test_the_per_minute_limit_refuses_the_ninth_call_in_a_minute() -> None:
    clock = FakeClock()
    ledger = CreditLedger(now=clock, daily_limit=800, per_minute_limit=8)

    for _ in range(8):
        ledger.spend()

    with pytest.raises(CreditLimitExceeded, match="per-minute"):
        ledger.spend()


def test_the_per_minute_window_slides_rather_than_resetting_on_the_minute() -> None:
    """A fixed-minute bucket would allow 16 calls across a boundary and earn a 429."""
    clock = FakeClock()
    ledger = CreditLedger(now=clock, daily_limit=800, per_minute_limit=8)

    for _ in range(8):
        ledger.spend()
        clock.advance(timedelta(seconds=1))

    # 8 seconds in: the window still holds all eight.
    with pytest.raises(CreditLimitExceeded):
        ledger.spend()

    # t=60. The window is (t-60, t], so only the spend at t=0 has aged out — the one at t=1 is
    # exactly on the boundary and still counts. Exactly one slot frees.
    clock.advance(timedelta(seconds=52))
    ledger.spend()

    with pytest.raises(CreditLimitExceeded):
        ledger.spend()


def test_per_minute_refusal_reports_a_short_retry_after() -> None:
    clock = FakeClock()
    ledger = CreditLedger(now=clock, daily_limit=800, per_minute_limit=2)
    ledger.spend(2)

    with pytest.raises(CreditLimitExceeded) as caught:
        ledger.spend()

    assert caught.value.retry_after is not None
    assert timedelta(0) < caught.value.retry_after <= timedelta(minutes=1)


def test_daily_spend_survives_the_per_minute_window_expiring() -> None:
    """The two limits are independent; draining the minute window must not refund the day."""
    clock = FakeClock()
    ledger = CreditLedger(now=clock, daily_limit=10, per_minute_limit=2)

    for _ in range(5):
        ledger.spend(2)
        clock.advance(timedelta(minutes=2))

    assert ledger.spent_today == 10
    with pytest.raises(CreditLimitExceeded, match="today"):
        ledger.spend()


@pytest.mark.parametrize("kwargs", [{"daily_limit": 0}, {"per_minute_limit": 0}])
def test_nonsense_limits_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        CreditLedger(**kwargs)  # type: ignore[arg-type]


def test_zero_or_negative_spend_is_rejected() -> None:
    with pytest.raises(ValueError):
        CreditLedger().spend(0)
