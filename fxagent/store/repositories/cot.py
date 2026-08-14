"""CFTC positioning, readable only as of a point in time.

Modelled on `EventRepository` and for the same reason: every public read takes `as_of` as its
first positional argument, there is no method that omits it, and all of them compose from
`_visible()`, which selects from the `cot_visible_at()` SQL function rather than from the
`cot_reports` table. The filter lives in the database (hard rule 6), so it holds for an ad-hoc
query too.

The one deliberate exception is `latest_fetch_at`, which reads `fetched_at` — when *we* pulled
the data, not when the CFTC released it. That is cache bookkeeping and is not an input to any
analysis, so gating it would be gating the wrong clock; it is kept clearly separate from every
read that feeds a decision.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Row, func, select
from sqlalchemy.dialects.postgresql import insert

from fxagent.store.repositories.base import Repository, require_utc
from fxagent.store.schema import cot_reports

__all__ = ["CotReportRecord", "CotRepository"]

logger = logging.getLogger(__name__)

_COLUMNS = tuple(column.name for column in cot_reports.c)


@dataclass(frozen=True)
class CotReportRecord:
    """One stored week. Structurally a `CotObservation`, so `CotHistory` accepts it directly."""

    id: int
    currency: str
    contract_code: str
    contract_name: str
    report_date: date
    published_at: datetime
    noncommercial_long: int
    noncommercial_short: int
    net_position: int
    open_interest: int | None
    fetched_at: datetime

    @classmethod
    def from_row(cls, row: Row[Any]) -> CotReportRecord:
        mapping = row._mapping  # noqa: SLF001 - the sanctioned Row -> dict accessor
        return cls(**{name: mapping[name] for name in _COLUMNS})


class CotRepository(Repository):
    """Reads and writes COT positioning. Reads are point-in-time by construction."""

    # -- writes ----------------------------------------------------------------

    async def upsert_many(self, records: Iterable[dict[str, Any]]) -> int:
        """Insert weeks, refreshing the position legs of ones already seen.

        The CFTC does revise prior weeks, so re-fetching must be able to correct a leg. What it
        must never do is move `published_at`: rewriting publication time would let a revision
        retroactively become visible earlier, which is the same hole `EventRepository` closes by
        excluding `publication_time_utc` from its conflict clause. `report_date` and
        `contract_code` are the key and are likewise untouched.

        `net_position` is absent from the update set because it is a generated column — Postgres
        recomputes it from the legs, so a revision cannot leave a stale net behind.
        """
        rows = [self._prepare(record) for record in records]
        if not rows:
            return 0

        statement = insert(cot_reports).values(rows)
        statement = statement.on_conflict_do_update(
            constraint="cot_reports_unique",
            set_={
                "currency": statement.excluded.currency,
                "contract_name": statement.excluded.contract_name,
                "noncommercial_long": statement.excluded.noncommercial_long,
                "noncommercial_short": statement.excluded.noncommercial_short,
                "open_interest": statement.excluded.open_interest,
                "fetched_at": statement.excluded.fetched_at,
            },
        )
        result = await self._session.execute(statement)
        return result.rowcount or 0

    @staticmethod
    def _prepare(record: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(record)
        for required in (
            "currency",
            "contract_code",
            "report_date",
            "published_at",
            "noncommercial_long",
            "noncommercial_short",
        ):
            if prepared.get(required) is None:
                raise ValueError(f"COT report is missing required field {required!r}")
        prepared["published_at"] = require_utc(prepared["published_at"], "published_at")
        prepared.setdefault("contract_name", prepared["currency"])
        prepared.setdefault("fetched_at", datetime.now(UTC))
        # Guard against a caller that computed it themselves: the column is generated, and
        # Postgres rejects an explicit value with a message that names neither this table nor
        # the caller's field.
        prepared.pop("net_position", None)
        return prepared

    # -- reads (all point-in-time) ---------------------------------------------

    def _visible(self, as_of: datetime) -> Any:
        """The only sanctioned source for a read. Selects through the SQL gate function."""
        checked = require_utc(as_of, "as_of")
        return func.cot_visible_at(checked).table_valued(*_COLUMNS)

    async def visible_at(
        self,
        as_of: datetime,
        *,
        currencies: Sequence[str] | None = None,
        limit: int = 2000,
    ) -> list[CotReportRecord]:
        """Every report released at or before `as_of`, oldest reference date first.

        The default limit holds seven currencies at the full three-year window with room to
        spare; `CotHistory` takes the trailing slice it needs from whatever it is given.
        """
        source = self._visible(as_of)
        statement = (
            select(source)
            .order_by(source.c.currency.asc(), source.c.report_date.asc())
            .limit(limit)
        )
        if currencies:
            statement = statement.where(source.c.currency.in_(tuple(c.upper() for c in currencies)))
        result = await self._session.execute(statement)
        return [CotReportRecord.from_row(row) for row in result]

    async def latest_report_date(self, as_of: datetime, currency: str) -> date | None:
        """Newest reference date released at or before `as_of`, for staleness checks."""
        source = self._visible(as_of)
        result = await self._session.execute(
            select(func.max(source.c.report_date))
            .select_from(source)
            .where(source.c.currency == currency.upper())
        )
        return result.scalar_one_or_none()

    async def count_visible(self, as_of: datetime) -> int:
        source = self._visible(as_of)
        result = await self._session.execute(select(func.count()).select_from(source))
        return int(result.scalar_one())

    # -- cache bookkeeping (deliberately not gated; see the module docstring) ---

    async def latest_fetch_at(self) -> datetime | None:
        """When we last pulled from the CFTC, regardless of what was released.

        Drives `should_fetch`. Ungated on purpose — this answers "when did this process last do
        work", which has nothing to do with what a backtest was allowed to know, and gating it
        would make a cold store re-fetch on every cycle.
        """
        result = await self._session.execute(select(func.max(cot_reports.c.fetched_at)))
        return result.scalar_one_or_none()
