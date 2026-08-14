"""Credit accounting for rate-limited data providers.

Twelve Data's free tier has two limits and they fail differently:

* **800 credits per day.** Exhausting it stops the day. A backfill that discovers this at
  credit 801 has already written a partial day into the store, and the gap it leaves is
  indistinguishable from a market holiday until someone reconciles it by hand.
* **8 credits per minute.** This is the one that actually bites mid-backfill. Exceed it and
  the API returns HTTP 429; a naive loop over 200 symbols hits it in the first ten seconds.

Both are checked *before* a request goes out, so the caller is refused locally rather than
throttled remotely. Refusing is the point: a `CreditLimitExceeded` is a clean stop with the
store consistent, where a 429 mid-loop is a partial write plus a retry storm.

The clock is injected so the tests exercise real rollover instead of sleeping through it.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

__all__ = ["CreditLedger", "CreditLimitExceeded"]

logger = logging.getLogger(__name__)

#: Twelve Data free tier, verified against their pricing page.
FREE_TIER_DAILY_CREDITS = 800
FREE_TIER_PER_MINUTE_CREDITS = 8


class CreditLimitExceeded(RuntimeError):
    """A request was refused locally because it would exceed a provider quota."""

    def __init__(self, message: str, *, retry_after: timedelta | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class CreditLedger:
    """Tracks provider credit spend against a daily and a per-minute cap.

    Thread-safe, because the collector may eventually poll several symbols concurrently and a
    lost increment here means an unexplained 429 later.
    """

    def __init__(
        self,
        *,
        daily_limit: int = FREE_TIER_DAILY_CREDITS,
        per_minute_limit: int = FREE_TIER_PER_MINUTE_CREDITS,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if daily_limit < 1:
            raise ValueError(f"daily_limit must be positive, got {daily_limit}")
        if per_minute_limit < 1:
            raise ValueError(f"per_minute_limit must be positive, got {per_minute_limit}")

        self._daily_limit = daily_limit
        self._per_minute_limit = per_minute_limit
        self._now = now or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()

        self._day: date | None = None
        self._spent_today = 0
        #: Timestamps of recent spends, for the sliding per-minute window.
        self._recent: deque[datetime] = deque()

    # -- inspection ------------------------------------------------------------

    @property
    def daily_limit(self) -> int:
        return self._daily_limit

    @property
    def spent_today(self) -> int:
        with self._lock:
            self._roll_day(self._now())
            return self._spent_today

    @property
    def remaining_today(self) -> int:
        return max(0, self._daily_limit - self.spent_today)

    def __repr__(self) -> str:
        return (
            f"CreditLedger(spent_today={self.spent_today}/{self._daily_limit}, "
            f"per_minute_limit={self._per_minute_limit})"
        )

    # -- accounting ------------------------------------------------------------

    def spend(self, credits: int = 1, *, description: str = "request") -> None:
        """Record `credits` against both caps, or refuse before the request is made."""
        if credits < 1:
            raise ValueError(f"credits must be positive, got {credits}")

        with self._lock:
            moment = self._now()
            self._roll_day(moment)
            self._expire_minute_window(moment)

            if self._spent_today + credits > self._daily_limit:
                raise CreditLimitExceeded(
                    f"refusing {description}: it would spend {credits} credit(s) with "
                    f"{self.__remaining_locked()} of {self._daily_limit} left today. Stopping "
                    "cleanly rather than being throttled part-way through a write.",
                    retry_after=self._until_midnight(moment),
                )

            if len(self._recent) + credits > self._per_minute_limit:
                wait = self._until_window_frees(moment, credits)
                raise CreditLimitExceeded(
                    f"refusing {description}: {len(self._recent)} of "
                    f"{self._per_minute_limit} per-minute credits already spent. Retry in "
                    f"{wait.total_seconds():.1f}s.",
                    retry_after=wait,
                )

            self._spent_today += credits
            for _ in range(credits):
                self._recent.append(moment)

        if self.remaining_today <= self._daily_limit // 10:
            logger.warning(
                "provider credits nearly exhausted: %d of %d remaining today",
                self.remaining_today,
                self._daily_limit,
            )

    def can_spend(self, credits: int = 1) -> bool:
        """True if `spend(credits)` would succeed right now. Advisory only — prefer spend()."""
        with self._lock:
            moment = self._now()
            self._roll_day(moment)
            self._expire_minute_window(moment)
            return (
                self._spent_today + credits <= self._daily_limit
                and len(self._recent) + credits <= self._per_minute_limit
            )

    # -- internals -------------------------------------------------------------

    def __remaining_locked(self) -> int:
        return max(0, self._daily_limit - self._spent_today)

    def _roll_day(self, moment: datetime) -> None:
        """Reset the daily count at UTC midnight."""
        today = moment.astimezone(UTC).date()
        if self._day != today:
            if self._day is not None:
                logger.info(
                    "provider credit day rolled over: spent %d on %s", self._spent_today, self._day
                )
            self._day = today
            self._spent_today = 0

    def _expire_minute_window(self, moment: datetime) -> None:
        cutoff = moment - timedelta(minutes=1)
        while self._recent and self._recent[0] <= cutoff:
            self._recent.popleft()

    def _until_window_frees(self, moment: datetime, credits: int) -> timedelta:
        """How long until enough slots leave the sliding window."""
        needed = len(self._recent) + credits - self._per_minute_limit
        if needed <= 0 or len(self._recent) < needed:
            return timedelta(seconds=1)
        freeing = self._recent[needed - 1]
        return max(timedelta(0), (freeing + timedelta(minutes=1)) - moment)

    @staticmethod
    def _until_midnight(moment: datetime) -> timedelta:
        moment_utc = moment.astimezone(UTC)
        tomorrow = datetime.combine(
            moment_utc.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        return tomorrow - moment_utc
