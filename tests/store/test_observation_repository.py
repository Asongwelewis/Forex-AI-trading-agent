"""The observation repository, and the revision rule the accumulation depends on.

BLS revises the prior two months at every release. The market traded the *first* print, so a
surprise history has to be built from that — which means an upsert that simply overwrote
`value` would destroy the only number we are collecting for, silently, months before anyone
looked. `first_value` is therefore immutable, and these tests are what keep it that way.
"""

from __future__ import annotations

from datetime import date

import pytest

from fxagent.store.repositories.observations import ObservationRepository

from .conftest import requires_postgres

pytestmark = [pytest.mark.db, requires_postgres]


def _row(value: float, *, period: str = "2026-07", start: date = date(2026, 7, 1)):  # noqa: ANN202
    return {
        "source": "bls",
        "series_id": "CES0000000001",
        "reference_period": period,
        "period_start": start,
        "value": value,
    }


@pytest.fixture
async def repo(session) -> ObservationRepository:  # noqa: ANN001
    return ObservationRepository(session)


async def test_a_print_round_trips(repo: ObservationRepository) -> None:
    await repo.upsert_many([_row(158858.0)])
    stored = await repo.series("bls", "CES0000000001")

    assert len(stored) == 1
    assert stored[0].value == 158858.0
    assert stored[0].first_value == 158858.0
    assert stored[0].revisions == 0
    assert stored[0].was_revised is False


async def test_a_revision_updates_value_but_never_the_first_print(
    repo: ObservationRepository,
) -> None:
    """The headline rule. Lose `first_value` and the accumulation is worthless."""
    await repo.upsert_many([_row(158858.0)])
    await repo.upsert_many([_row(159102.0)])

    stored = await repo.series("bls", "CES0000000001")
    assert len(stored) == 1, "a revision must update in place, not append a second row"
    assert stored[0].value == 159102.0
    assert stored[0].first_value == 158858.0
    assert stored[0].revisions == 1
    assert stored[0].revised_at is not None


async def test_refetching_an_unchanged_value_is_not_a_revision(
    repo: ObservationRepository,
) -> None:
    """Re-polling is the normal case and must not inflate the revision count."""
    await repo.upsert_many([_row(158858.0)])
    await repo.upsert_many([_row(158858.0)])
    await repo.upsert_many([_row(158858.0)])

    stored = await repo.series("bls", "CES0000000001")
    assert stored[0].revisions == 0
    assert stored[0].revised_at is None


async def test_repeated_revisions_accumulate(repo: ObservationRepository) -> None:
    for value in (158858.0, 159102.0, 159050.0):
        await repo.upsert_many([_row(value)])

    stored = await repo.series("bls", "CES0000000001")
    assert stored[0].revisions == 2
    assert stored[0].first_value == 158858.0
    assert stored[0].value == 159050.0


async def test_series_are_returned_oldest_first(repo: ObservationRepository) -> None:
    await repo.upsert_many(
        [
            _row(3.0, period="2026-07", start=date(2026, 7, 1)),
            _row(1.0, period="2026-05", start=date(2026, 5, 1)),
            _row(2.0, period="2026-06", start=date(2026, 6, 1)),
        ]
    )
    stored = await repo.series("bls", "CES0000000001")
    assert [o.reference_period for o in stored] == ["2026-05", "2026-06", "2026-07"]


async def test_different_sources_do_not_collide(repo: ObservationRepository) -> None:
    """`source` is in the key: two providers may legitimately publish the same concept."""
    await repo.upsert_many(
        [
            _row(158858.0),
            {
                "source": "eurostat",
                "series_id": "CES0000000001",
                "reference_period": "2026-07",
                "period_start": date(2026, 7, 1),
                "value": 99.0,
            },
        ]
    )
    assert await repo.count() == 2
    assert len(await repo.known_series()) == 2


async def test_a_missing_required_field_is_refused(repo: ObservationRepository) -> None:
    with pytest.raises(ValueError, match="missing required field"):
        await repo.upsert_many([{"source": "bls", "series_id": "X", "value": 1.0}])


async def test_empty_input_is_a_no_op(repo: ObservationRepository) -> None:
    assert await repo.upsert_many([]) == 0
