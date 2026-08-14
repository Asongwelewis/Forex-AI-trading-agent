"""Assemble what a strategy and the permission layer need, as of one instant.

Everything here reads through `EventRepository`, whose every method takes `as_of` and selects
through the `events_visible_at()` SQL gate. Nothing in this module touches the `events` table
directly, and nothing filters on `event_time_utc` without `publication_time_utc` also applying
— `in_event_window` enforces both at once, which is why the auto-revoke lookahead can ask
"what is coming in the next 15 minutes" without asking "what do we now know that we did not
know then".

**No policy rates are hardcoded.** `rate_differential` is the only number `carry_divergence`
trades on, and a stale constant here becomes a live position in the wrong direction — the
strategy reads the sign of this value to pick long or short. Rates move on a schedule this
package has no feed for, so they are injected by the caller and a missing one yields None
rather than a default. `MarketContext.neutral()` is the honest fallback: a carry strategy with
no carry number should stand down, not guess zero and then trade the macro tiebreak.

**COT positioning is read through `CotRepository`, which gates on `published_at` in SQL** for
the same reason events are gated on publication time — a COT row references the preceding
Tuesday and is public on Friday, so a read keyed on the reference date back-dates it by three
days. `cot.py` imports `split_symbol` from here, so the import back the other way is
type-checking only; the runtime dependency runs one way and there is no cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from fxagent.store.repositories.cot import CotRepository
from fxagent.store.repositories.events import HIGH_IMPACT, EventRecord, EventRepository
from fxagent.strategies.base import MarketContext

if TYPE_CHECKING:
    # Type-checking only, and necessarily so: `fundamentals.cot` imports `split_symbol` from
    # this module, so a runtime import here would close the loop. Nothing below needs the class
    # itself, only attributes off an instance built by `_positioning`.
    from fxagent.fundamentals.cot import PairPositioning

__all__ = [
    "FundamentalContext",
    "PolicyRate",
    "build_context",
    "rate_differential",
    "split_symbol",
]

logger = logging.getLogger(__name__)

#: How far ahead the calendar blackout looks, and how long it lingers after a release. Mirrors
#: EventRepository.high_impact_near's defaults; restated so a change here is deliberate.
DEFAULT_LOOKAHEAD = timedelta(minutes=15)
DEFAULT_LOOKBEHIND = timedelta(minutes=5)

#: A policy rate older than this is treated as unusable. Central banks meet roughly every six
#: weeks, so a quarter-old rate has had time to move twice without anyone noticing.
DEFAULT_MAX_RATE_AGE = timedelta(days=45)


@dataclass(frozen=True)
class PolicyRate:
    """One currency's annualised policy rate, with the date it was known good.

    `as_of` is not decoration. It is what lets `build_context` refuse a stale rate instead of
    feeding a six-month-old differential into a live carry decision.
    """

    currency: str
    rate_percent: float
    as_of: date
    source: str = "manual"


def split_symbol(symbol: str) -> tuple[str, str]:
    """'EURUSD' -> ('EUR', 'USD'), tolerating a broker suffix or a separator.

    Exness appends 'm' to every symbol and some feeds write 'EUR/USD', so the canonical form is
    reconstructed rather than assumed. Raises on anything that is not six letters after
    cleaning, because a silently mis-split pair inverts the carry sign.
    """
    cleaned = symbol.strip().upper().replace("/", "").replace("_", "").replace("-", "")
    while cleaned and not cleaned[-1].isalpha():
        cleaned = cleaned[:-1]
    # Strip a single lower-case-origin account suffix ('EURUSDM' after upper()).
    if len(cleaned) == 7:
        cleaned = cleaned[:6]
    if len(cleaned) != 6 or not cleaned.isalpha():
        raise ValueError(f"cannot split {symbol!r} into a base and quote currency")
    return cleaned[:3], cleaned[3:]


def rate_differential(
    symbol: str,
    rates: Mapping[str, PolicyRate],
    *,
    as_of: date | None = None,
    max_age: timedelta = DEFAULT_MAX_RATE_AGE,
) -> float | None:
    """Base minus quote, in percentage points, or None if either side is missing or stale.

    Positive means holding the pair long earns carry, matching `MarketContext`'s definition.
    Returning None rather than 0.0 is the point: zero is a real differential that says the two
    currencies pay the same, and `carry_divergence` reads it as "no trade" only by accident of
    the sign check. None forces the caller to decide explicitly.
    """
    base, quote = split_symbol(symbol)
    today = as_of or datetime.now(UTC).date()

    resolved: dict[str, PolicyRate] = {}
    for currency in (base, quote):
        rate = rates.get(currency)
        if rate is None:
            logger.info("no policy rate for %s, carry differential unavailable", currency)
            return None
        age = today - rate.as_of
        if age > max_age:
            logger.warning(
                "policy rate for %s is %d days old (limit %d), treating as unavailable",
                currency,
                age.days,
                max_age.days,
            )
            return None
        resolved[currency] = rate

    return resolved[base].rate_percent - resolved[quote].rate_percent


@dataclass(frozen=True)
class FundamentalContext:
    """Everything fundamentals knows about one symbol at one instant.

    `market` is the object strategies receive. The rest is for the permission layer and the
    dashboard, and is kept out of `MarketContext` because that model is frozen with
    `extra="forbid"` — widening it would reach into every strategy's contract for the benefit
    of one caller.
    """

    as_of: datetime
    symbol: str
    market: MarketContext
    rate_differential: float | None
    imminent_high_impact: tuple[EventRecord, ...] = ()
    recent_events: tuple[EventRecord, ...] = ()
    #: None when no COT repository was supplied or the read failed. Distinct from a positioning
    #: object whose score happens to be 0.0, which means "measured, and mid-range".
    positioning: PairPositioning | None = None
    unavailable: tuple[str, ...] = field(default_factory=tuple)

    @property
    def calendar_blackout(self) -> bool:
        """True when a high-impact release for either leg is inside the blackout window.

        This is the auto-revoke trigger's input. It is deliberately a property over stored
        records rather than a flag computed elsewhere, so the events that caused a revocation
        can be logged alongside the decision.
        """
        return bool(self.imminent_high_impact)

    @property
    def is_degraded(self) -> bool:
        """True when something the context wanted was missing. Worth surfacing, not fatal."""
        return bool(self.unavailable)


async def _positioning(
    cot_repository: CotRepository,
    *,
    as_of: datetime,
    symbol: str,
) -> PairPositioning:
    """Rank both legs of `symbol` against what the CFTC had released at `as_of`.

    Imported inside the function on purpose: `fundamentals.cot` imports `split_symbol` from this
    module, so a module-level import would be a cycle. Confined to the one call site that needs
    it rather than papered over by reshuffling the packages, because `split_symbol` genuinely
    belongs next to `rate_differential` and `CotHistory` genuinely belongs next to the fetcher.
    """
    from fxagent.fundamentals.cot import CONTRACT_CODES, CotHistory

    base, quote = split_symbol(symbol)
    wanted = [leg for leg in (base, quote) if leg in CONTRACT_CODES]
    records = await cot_repository.visible_at(as_of, currencies=wanted) if wanted else []
    return CotHistory(records).pair_positioning_score(symbol)


async def build_context(
    repository: EventRepository,
    *,
    as_of: datetime,
    symbol: str,
    rates: Mapping[str, PolicyRate],
    macro_bias: float = 0.0,
    cot_repository: CotRepository | None = None,
    lookahead: timedelta = DEFAULT_LOOKAHEAD,
    lookbehind: timedelta = DEFAULT_LOOKBEHIND,
    recent_window: timedelta = timedelta(hours=24),
    max_rate_age: timedelta = DEFAULT_MAX_RATE_AGE,
) -> FundamentalContext:
    """Read the store as of `as_of` and assemble the context for `symbol`.

    `macro_bias` is a parameter, not something derived here. Turning a pile of headlines into a
    signed conviction is interpretation, which belongs to the historian agent — and hard rule 4
    keeps that path advisory. Left at 0.0, `carry_divergence` trades on carry and price alone,
    which is the correct behaviour when no agent has spoken.

    `cot_repository` is optional and omitting it yields neutral positioning, which is the same
    behaviour as before this input existed. It is a separate argument rather than a second
    method on `EventRepository` because the two tables have separate gate functions, and one
    repository spanning both would be one `as_of` covering two visibility rules.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware; a naive value shifts the visibility gate")
    checked = as_of.astimezone(UTC)
    base, quote = split_symbol(symbol)
    unavailable: list[str] = []

    differential = rate_differential(symbol, rates, as_of=checked.date(), max_age=max_rate_age)
    if differential is None:
        unavailable.append("rate_differential")

    try:
        imminent = await repository.high_impact_near(
            checked, currencies=(base, quote), lookahead=lookahead, lookbehind=lookbehind
        )
    except Exception as exc:  # noqa: BLE001 - the store being unreachable must not stop analysis
        logger.warning("could not read imminent events: %s: %s", type(exc).__name__, exc)
        imminent = []
        # Recorded rather than swallowed: "no events found" and "could not look" must not read
        # the same to the permission layer, which would otherwise treat an outage as all-clear.
        unavailable.append("imminent_high_impact")

    try:
        recent = await repository.in_event_window(
            checked,
            window_start=checked - recent_window,
            window_end=checked,
            currencies=(base, quote),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read recent events: %s: %s", type(exc).__name__, exc)
        recent = []
        unavailable.append("recent_events")

    positioning: PairPositioning | None = None
    if cot_repository is not None:
        try:
            positioning = await _positioning(cot_repository, as_of=checked, symbol=symbol)
        except Exception as exc:  # noqa: BLE001 - positioning is context, never a precondition
            logger.warning("could not read COT positioning: %s: %s", type(exc).__name__, exc)
            unavailable.append("positioning")
        else:
            # A leg with too little history is degraded, not broken. Recorded so the dashboard
            # can say "EUR ranked, NZD not yet" instead of showing a confident-looking zero.
            unavailable.extend(f"positioning:{leg}" for leg in positioning.unavailable)

    market = MarketContext(
        rate_differential=differential if differential is not None else 0.0,
        macro_bias=macro_bias,
        positioning_score=positioning.score if positioning is not None else 0.0,
    )

    return FundamentalContext(
        as_of=checked,
        symbol=symbol,
        market=market,
        rate_differential=differential,
        imminent_high_impact=tuple(imminent),
        recent_events=tuple(recent),
        positioning=positioning,
        unavailable=tuple(unavailable),
    )


def high_impact_currencies(events: Sequence[EventRecord]) -> tuple[str, ...]:
    """Distinct currencies among high-impact events, for logging a revocation's reason."""
    return tuple(sorted({e.currency for e in events if e.importance in HIGH_IMPACT}))
