"""Raw prints from the primary statistical offices. Numbers only, no interpretation.

Two keyless endpoints, both verified 14 Aug 2026:

    GET https://api.bls.gov/publicAPI/v1/timeseries/data/{series_id}
    GET https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?...

This module exists to start a clock, not to produce a signal. `surprise_score` needs eight or
more historical samples per release before its standard deviation means anything, these releases
are monthly, and that clock only runs while the numbers are being stored. So collection begins
now; the join into calendar events waits until something consumes a surprise score.

Consequently there is deliberately **no**: series-to-event mapping, unit reconciliation,
forecast pairing, or surprise arithmetic. A BLS payrolls series is a level in thousands and the
calendar quotes the month-over-month change; reconciling that is real work with real revision
handling behind it, and guessing at it now would bake an error into the history rather than
leaving the raw numbers to be reinterpreted correctly later.

**These do not go into `events`.** They have no forecast and no publication timestamp — no
primary source publishes one — so they cannot be gated point-in-time and must stay out of
`events_visible_at()` and therefore out of `build_context`. See migration 0010.

**And this module is deliberately not inside `fxagent/fundamentals/`**, where it started.
Importing anything from that package runs its `__init__`, which imports `context.py`, which
imports `fxagent.strategies.base` — so `import fxagent.fundamentals.statistics` transitively
pulled in strategies, regime and indicators. Measured, not assumed. That matters because the
natural home for the poller is the collector, and the collector's whole justification is being
the dumbest process in the system: `tests/collector/test_collector_is_data_only.py` inspects its
imports with AST, which would have seen an innocent-looking line and missed the runtime reach.
Sitting at the top level, this module imports httpx and the standard library and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

__all__ = [
    "BLS_SERIES",
    "EUROSTAT_SERIES",
    "BlsSource",
    "EurostatSeries",
    "EurostatSource",
    "Observation",
]

logger = logging.getLogger(__name__)

BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/{series_id}"
EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"

#: The US releases that move FX. Levels and indices as BLS publishes them — untransformed.
BLS_SERIES: tuple[str, ...] = (
    "CES0000000001",  # All employees, total nonfarm, thousands (payrolls, as a level)
    "CUUR0000SA0",  # CPI-U, all items, index 1982-84=100
)


@dataclass(frozen=True)
class EurostatSeries:
    """One Eurostat dataset plus the dimension filter that makes it a single series."""

    dataset: str
    #: Dimension filters, e.g. {"geo": "EA", "coicop": "CP00"}. Sorted into the series id so the
    #: identifier is stable regardless of dict ordering.
    filters: dict[str, str]

    @property
    def series_id(self) -> str:
        joined = ":".join(f"{k}={v}" for k, v in sorted(self.filters.items()))
        return f"{self.dataset}:{joined}"


EUROSTAT_SERIES: tuple[EurostatSeries, ...] = (
    # HICP, all items, annual rate of change, euro area — the headline euro inflation print.
    EurostatSeries("prc_hicp_manr", {"coicop": "CP00", "geo": "EA"}),
)


@dataclass(frozen=True)
class Observation:
    """One published number for one reference period. Nothing derived, nothing rescaled."""

    source: str
    series_id: str
    #: 'YYYY-MM' or 'YYYY-Qn'. A reference period, not a publication date.
    reference_period: str
    period_start: date
    value: float

    def as_row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "series_id": self.series_id,
            "reference_period": self.reference_period,
            "period_start": self.period_start,
            "value": self.value,
        }


def _period_start(year: int, period: str) -> tuple[str, date] | None:
    """BLS period codes to a canonical label and a first-of-period date.

    M01-M12 are months. M13 is the annual average and Q05 the annual total — both are summaries
    rather than releases, and including them would put a thirteenth 'month' in a monthly series.
    """
    code = period.upper()
    if code.startswith("M"):
        month = int(code[1:])
        if not 1 <= month <= 12:
            return None
        return f"{year}-{month:02d}", date(year, month, 1)
    if code.startswith("Q"):
        quarter = int(code[1:])
        if not 1 <= quarter <= 4:
            return None
        return f"{year}-Q{quarter}", date(year, 3 * (quarter - 1) + 1, 1)
    if code.startswith("S"):  # semiannual
        half = int(code[1:])
        if not 1 <= half <= 2:
            return None
        return f"{year}-S{half}", date(year, 1 if half == 1 else 7, 1)
    return None


def _eurostat_period(label: str) -> tuple[str, date] | None:
    """Eurostat time labels: '2026-07', '2026Q3', '2026'."""
    text = label.strip().upper().replace("M", "-")
    try:
        if "Q" in text:
            year_part, quarter_part = text.split("Q", 1)
            year, quarter = int(year_part.strip("-")), int(quarter_part)
            if not 1 <= quarter <= 4:
                return None
            return f"{year}-Q{quarter}", date(year, 3 * (quarter - 1) + 1, 1)
        if "-" in text:
            year_part, month_part = text.split("-", 1)
            year, month = int(year_part), int(month_part)
            if not 1 <= month <= 12:
                return None
            return f"{year}-{month:02d}", date(year, month, 1)
    except ValueError:
        return None
    # A bare year is an annual figure; skipped rather than stored as January.
    return None


class BlsSource:
    """Fetches BLS series through the keyless v1 API.

    v1 takes no key and is capped at 25 queries a day per IP, which is why the series list is
    short and each is fetched once per run rather than per symbol. v2 also answered without a
    key when tested, but its documented contract requires registration, so v1 is what this
    relies on.
    """

    def __init__(
        self,
        *,
        series: Sequence[str] = BLS_SERIES,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "fx-regime-agent/0.1",
    ) -> None:
        self._series = tuple(series)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._user_agent = user_agent

    @property
    def name(self) -> str:
        return "bls"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> BlsSource:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def fetch(self) -> list[Observation]:
        """Every stored period for every configured series. One failing series loses only itself."""
        collected: list[Observation] = []
        for series_id in self._series:
            try:
                collected.extend(await self._fetch_series(series_id))
            except Exception as exc:  # noqa: BLE001 - one series must not lose the rest
                logger.warning(
                    "BLS series %s failed, continuing without it: %s: %s",
                    series_id,
                    type(exc).__name__,
                    exc,
                )
        return collected

    async def _fetch_series(self, series_id: str) -> list[Observation]:
        response = await self._client.get(
            BLS_URL.format(series_id=series_id), headers={"User-Agent": self._user_agent}
        )
        response.raise_for_status()
        payload = response.json()

        status = payload.get("status")
        if status != "REQUEST_SUCCEEDED":
            # BLS answers 200 with a failure status in the body, including for quota exhaustion.
            raise ValueError(f"BLS returned status {status!r}: {payload.get('message')}")

        observations: list[Observation] = []
        for series in payload.get("Results", {}).get("series", []):
            returned_id = series.get("seriesID", series_id)
            for row in series.get("data", []):
                parsed = self._to_observation(row, returned_id)
                if parsed is not None:
                    observations.append(parsed)
        return observations

    def _to_observation(self, row: dict[str, Any], series_id: str) -> Observation | None:
        try:
            year = int(row["year"])
            value = float(str(row["value"]).replace(",", ""))
        except (KeyError, TypeError, ValueError):
            return None

        period = _period_start(year, str(row.get("period", "")))
        if period is None:
            return None
        label, start = period
        return Observation(
            source="bls",
            series_id=series_id,
            reference_period=label,
            period_start=start,
            value=value,
        )


class EurostatSource:
    """Fetches Eurostat datasets through the keyless dissemination API (JSON-stat 2.0)."""

    def __init__(
        self,
        *,
        series: Sequence[EurostatSeries] = EUROSTAT_SERIES,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "fx-regime-agent/0.1",
        last_periods: int = 36,
    ) -> None:
        self._series = tuple(series)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._user_agent = user_agent
        self._last_periods = last_periods

    @property
    def name(self) -> str:
        return "eurostat"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> EurostatSource:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def fetch(self) -> list[Observation]:
        collected: list[Observation] = []
        for series in self._series:
            try:
                collected.extend(await self._fetch_series(series))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Eurostat series %s failed, continuing without it: %s: %s",
                    series.series_id,
                    type(exc).__name__,
                    exc,
                )
        return collected

    async def _fetch_series(self, series: EurostatSeries) -> list[Observation]:
        params: dict[str, str] = {
            "format": "JSON",
            "lang": "EN",
            "lastTimePeriod": str(self._last_periods),
            **series.filters,
        }
        response = await self._client.get(
            EUROSTAT_URL.format(dataset=series.dataset),
            params=params,
            headers={"User-Agent": self._user_agent},
        )
        response.raise_for_status()
        return self._parse(response.json(), series)

    @staticmethod
    def _parse(payload: dict[str, Any], series: EurostatSeries) -> list[Observation]:
        """JSON-stat 2.0: `value` is a sparse map of flat index -> number.

        The time dimension's `index` maps each period label to that flat position, and periods
        with no observation are simply absent from `value` — hence the `if raw is None: continue`
        rather than a zip, which would silently misalign every period after the first gap.
        """
        try:
            time_index: dict[str, int] = payload["dimension"]["time"]["category"]["index"]
            values: dict[str, Any] = payload["value"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unexpected Eurostat payload shape: {exc}") from exc

        observations: list[Observation] = []
        for label, position in time_index.items():
            raw = values.get(str(position))
            if raw is None:
                continue
            period = _eurostat_period(label)
            if period is None:
                continue
            canonical, start = period
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            observations.append(
                Observation(
                    source="eurostat",
                    series_id=series.series_id,
                    reference_period=canonical,
                    period_start=start,
                    value=value,
                )
            )
        return observations


# -- entry point -------------------------------------------------------------------------------
#
# `python -m fxagent.observations --once`, called daily by
# .github/workflows/observations.yml. Its own workflow rather than a step in the collector:
# the collector's value is being the dumbest process in the system, and this is not bar data.
#
# The store imports live down here rather than at module scope so the source clients stay
# importable on their own — the isolation this module exists to preserve is checked by
# tests/store/test_observations_are_not_in_the_pipeline.py.


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m fxagent.observations",
        description="Poll BLS and Eurostat for raw statistical prints. Computes nothing.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once and exit. The only supported mode; accepted for symmetry with the "
        "collector, whose workflow invokes it the same way.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


async def _run() -> int:
    from fxagent.store import Database, apply_migrations
    from fxagent.store.repositories import ObservationRepository

    database = Database.from_env()
    try:
        async with database.connect() as connection:
            applied = await apply_migrations(connection)
        if applied:
            logger.info("applied %d migration(s)", len(applied))

        collected: list[Observation] = []
        async with BlsSource() as bls:
            collected.extend(await bls.fetch())
        async with EurostatSource() as eurostat:
            collected.extend(await eurostat.fetch())

        if not collected:
            # Both sources fail soft per series, so reaching zero means every one of them
            # failed. That must be loud: this job exists to keep a clock running, and a clock
            # that stopped silently is discovered eight months later when the surprise history
            # everyone assumed was accumulating turns out to be empty.
            logger.error("no observations from any source — the accumulation clock is not running")
            return 1

        async with database.session() as session:
            written = await ObservationRepository(session).upsert_many(
                observation.as_row() for observation in collected
            )
            await session.commit()

        by_source: dict[str, int] = {}
        for observation in collected:
            by_source[observation.source] = by_source.get(observation.source, 0) + 1
        logger.info(
            "stored %d row(s) from %d observation(s): %s",
            written,
            len(collected),
            ", ".join(f"{name}={count}" for name, count in sorted(by_source.items())),
        )
        return 0
    finally:
        await database.dispose()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from dotenv import load_dotenv

    load_dotenv()
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - the top-level boundary; the alert needs a message
        logger.error("observations run failed: %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
