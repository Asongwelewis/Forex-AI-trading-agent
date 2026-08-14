"""Retry classification and backoff schedule. No database, no real waiting."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.exc import (
    DisconnectionError,
    IntegrityError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)

from fxagent.store.retry import RetryPolicy, is_transient, with_retry


def _operational(message: str) -> OperationalError:
    return OperationalError("select 1", {}, Exception(message))


def _integrity() -> IntegrityError:
    return IntegrityError("insert", {}, Exception("duplicate key value violates unique"))


class _Recorder:
    """Captures sleeps instead of performing them."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def _no_jitter(_low: float, high: float) -> float:
    return high


# -- classification -----------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        DisconnectionError("pool disconnect"),
        InterfaceError("iface", {}, Exception("connection already closed")),
        ConnectionError("reset by peer"),
        TimeoutError(),
        OSError("network unreachable"),
        _operational("server closed the connection unexpectedly"),
        _operational("terminating connection due to administrator command"),
        _operational("the database system is starting up"),
        _operational("too many clients already"),
    ],
)
def test_connection_shaped_failures_are_transient(exc: BaseException) -> None:
    assert is_transient(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        _integrity(),
        ProgrammingError("select nope", {}, Exception('column "nope" does not exist')),
        _operational("division by zero"),
        ValueError("not a database problem at all"),
    ],
)
def test_server_side_rejections_are_permanent(exc: BaseException) -> None:
    """Retrying these hits the same wall again, slower and with a worse traceback."""
    assert is_transient(exc) is False


def test_integrity_error_is_never_retried() -> None:
    """A unique-constraint hit is an answer, not a failure — retrying could double a write."""
    assert is_transient(_integrity()) is False


# -- schedule -----------------------------------------------------------------


def test_backoff_grows_geometrically_and_is_capped() -> None:
    policy = RetryPolicy(base_delay_seconds=0.2, multiplier=3.0, max_delay_seconds=5.0)

    assert policy.delay_for(1) == pytest.approx(0.2)
    assert policy.delay_for(2) == pytest.approx(0.6)
    assert policy.delay_for(3) == pytest.approx(1.8)
    assert policy.delay_for(4) == pytest.approx(5.0), "capped, not 5.4"
    assert policy.delay_for(9) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"attempts": 0}, {"base_delay_seconds": 0}, {"base_delay_seconds": -1}, {"multiplier": 0.5}],
)
def test_nonsense_policies_are_rejected_at_construction(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]


# -- behaviour ----------------------------------------------------------------


async def test_succeeds_without_sleeping_when_the_first_attempt_works() -> None:
    recorder = _Recorder()

    async def operation() -> str:
        return "ok"

    result = await with_retry(operation, sleep=recorder.sleep, jitter=_no_jitter)

    assert result == "ok"
    assert recorder.slept == []


async def test_retries_a_transient_failure_then_succeeds() -> None:
    recorder = _Recorder()
    calls = {"n": 0}

    async def operation() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _operational("server closed the connection unexpectedly")
        return "recovered"

    result = await with_retry(
        operation,
        policy=RetryPolicy(attempts=4, base_delay_seconds=0.1, multiplier=2.0),
        sleep=recorder.sleep,
        jitter=_no_jitter,
    )

    assert result == "recovered"
    assert calls["n"] == 3
    assert recorder.slept == pytest.approx([0.1, 0.2])


async def test_permanent_failure_raises_immediately_without_retrying() -> None:
    recorder = _Recorder()
    calls = {"n": 0}

    async def operation() -> str:
        calls["n"] += 1
        raise _integrity()

    with pytest.raises(IntegrityError):
        await with_retry(operation, sleep=recorder.sleep, jitter=_no_jitter)

    assert calls["n"] == 1, "a permanent failure must not be attempted twice"
    assert recorder.slept == []


async def test_gives_up_after_the_configured_attempts_and_reraises_the_last_error() -> None:
    recorder = _Recorder()
    calls = {"n": 0}

    async def operation() -> str:
        calls["n"] += 1
        raise _operational("connection refused")

    with pytest.raises(OperationalError, match="connection refused"):
        await with_retry(
            operation,
            policy=RetryPolicy(attempts=3, base_delay_seconds=0.1),
            sleep=recorder.sleep,
            jitter=_no_jitter,
        )

    assert calls["n"] == 3
    assert len(recorder.slept) == 2, "no sleep after the final attempt"


async def test_jitter_is_applied_over_the_full_delay() -> None:
    """Full jitter: without it, every worker retries in lockstep after a pooler restart."""
    seen: list[tuple[float, float]] = []
    recorder = _Recorder()

    def spy(low: float, high: float) -> float:
        seen.append((low, high))
        return low

    calls = {"n": 0}

    async def operation() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _operational("connection reset by peer")
        return "ok"

    await with_retry(
        operation,
        policy=RetryPolicy(attempts=2, base_delay_seconds=0.5),
        sleep=recorder.sleep,
        jitter=spy,
    )

    assert seen == [(0.0, 0.5)], "jitter draws over [0, delay], not around it"


async def test_cancellation_is_not_retried() -> None:
    """CancelledError is a BaseException; retrying it would ignore the cancellation."""
    recorder = _Recorder()

    async def operation() -> str:
        raise TimeoutError("transient")

    # Sanity: TimeoutError IS retried, so the next assertion is about kind, not luck.
    with pytest.raises(TimeoutError):
        await with_retry(
            operation,
            policy=RetryPolicy(attempts=2, base_delay_seconds=0.01),
            sleep=recorder.sleep,
            jitter=_no_jitter,
        )
    assert len(recorder.slept) == 1

    async def cancelled() -> str:
        raise asyncio.CancelledError

    recorder.slept.clear()
    with pytest.raises(asyncio.CancelledError):
        await with_retry(cancelled, sleep=recorder.sleep, jitter=_no_jitter)
    assert recorder.slept == [], "a cancelled operation must not be retried"
