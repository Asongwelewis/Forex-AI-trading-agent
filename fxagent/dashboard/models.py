"""Every shape the browser is allowed to receive. One file, so the wire format is readable.

These are view models, not domain models. They exist to be serialised, so they hold plain
numbers and ISO strings rather than `Bar`, `Regime` or `TradeRecord` — the browser cannot
import a Pydantic class and must never be sent one it has to guess the meaning of.

Two conventions the whole file obeys:

**Chart times are Unix seconds.** That is what Lightweight Charts calls a `UTCTimestamp`, and
it is unambiguous in a way `"2026-08-15T08:00"` on a browser in Douala is not. Everything the
*feed* shows a human is an ISO-8601 string with its offset attached, because a human reading a
timestamp needs to see which clock it is on.

**`None` in a `LinePoint` is a hole, not a zero.** An overlay that has not warmed up, or a day
whose Asian session never fully arrived, emits a point with no value. The browser turns those
into whitespace and breaks the line there. Substituting 0.0 would draw a spike to the bottom of
the pane and make a missing measurement look like a measured collapse — the same class of lie
as a classifier reporting 0.0 ADX during warm-up.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from fxagent.regime.sessions import Session

__all__ = [
    "AgentNarration",
    "Analogue",
    "Candle",
    "ChartPayload",
    "Envelope",
    "FeedEntry",
    "FeedPayload",
    "GrantSnapshot",
    "GrantState",
    "LinePoint",
    "Marker",
    "MarkerKind",
    "Overlay",
    "PatternNote",
    "RegimeView",
    "RiskOfficerNote",
    "SeriesOption",
    "SessionBand",
    "Snapshot",
    "TradeLevels",
    "TradeView",
    "VoteView",
]


class _View(BaseModel):
    """Frozen, and forbids unknown fields — a typo in a builder fails here, not in the UI."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- chart -------------------------------------------------------------------------------


class Candle(_View):
    """One bar, keyed on its OPEN time, matching `bars.ts_utc` and every indicator's index."""

    time: int
    open: float
    high: float
    low: float
    close: float


class LinePoint(_View):
    """A point on an overlay. `value is None` means whitespace — see the module docstring."""

    time: int
    value: float | None = None


class Overlay(_View):
    """One computed series drawn over the candles, with the parameters it was computed from.

    `label` carries the period rather than saying "EMA", because an instrument panel that
    cannot tell you *which* EMA you are looking at has shown you a line, not a measurement.
    """

    key: str
    label: str
    colour: str
    points: tuple[LinePoint, ...]
    #: "line" joins points directly; "step" holds a level until it changes, which is what a
    #: session range is — a level that is constant all day and then jumps.
    style: str = "line"
    width: int = 1


class SessionBand(_View):
    """One shaded stretch of one trading session, in Unix seconds.

    Produced by `regime.sessions.session_bounds_utc`, so London is 08:00-17:00 UTC in January
    and 07:00-16:00 in July without anything here knowing why.
    """

    session: Session
    start: int
    end: int


class MarkerKind(StrEnum):
    """Why a marker is on the chart. The panel legend is keyed on this."""

    #: A strategy produced a directional signal and the router let it vote.
    SIGNAL = "SIGNAL"
    #: A strategy produced a directional signal the router had already weighted to zero. Drawn
    #: dimmer, and drawn at all because "it wanted to and was not allowed" is the disagreement
    #: CLAUDE.md asks to be logged, which makes it worth seeing.
    GATED = "GATED"
    TRADE_ENTRY = "TRADE_ENTRY"
    TRADE_EXIT = "TRADE_EXIT"


class Marker(_View):
    """A pin on the price series. `strategy` is set for signal markers and None for trades."""

    time: int
    kind: MarkerKind
    position: str  # aboveBar | belowBar
    shape: str  # arrowUp | arrowDown | circle | square
    colour: str
    text: str
    strategy: str | None = None
    direction: str | None = None


class TradeLevels(_View):
    """One executed trade's three price lines, spanning the life of the position.

    Entry, stop and target together, never entry alone: hard rule 3 says an order without
    server-side protection cannot exist, so a chart that can draw a trade without showing where
    it was protected would be able to show a state the system refuses to create.
    """

    trade_id: int
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    start: int
    end: int
    open: bool
    volume: float
    mode: str
    barrier_touched: str | None = None
    r_multiple: float | None = None


class ChartPayload(_View):
    """Everything the left pane draws."""

    symbol: str
    timeframe: str
    source: str
    price_precision: int
    candles: tuple[Candle, ...]
    overlays: tuple[Overlay, ...]
    session_bands: tuple[SessionBand, ...]
    markers: tuple[Marker, ...]
    trades: tuple[TradeLevels, ...]
    #: Populated when a cap dropped something. Shown in the UI: a truncated chart that does not
    #: say so reads as a complete one.
    notes: tuple[str, ...] = ()


# --- agent panel -------------------------------------------------------------------------


class RegimeView(_View):
    """What the classifier measured at the bar this entry describes."""

    session: str | None
    sessions: tuple[str, ...]
    market_open: bool
    minutes_until_weekly_close: int | None = None
    trend_strength: float | None = None
    volatility_percentile: float | None = None
    is_trending: bool = False
    is_ranging: bool = False


class VoteView(_View):
    """One strategy's vote, its router weight, and why it counted or did not.

    The three states `consensus` distinguishes survive to here intact — silent, gated and flat
    are different facts, and a panel that showed all three as "no" would be throwing away the
    only reason the diagnostics are written.
    """

    strategy: str
    weight: float
    direction: str | None = None
    confidence: float | None = None
    participated: bool = False
    reason: str = ""


class AgentNarration(_View):
    """Prose from one LLM agent. Text only — no number here may reach a decision."""

    agent: str
    text: str
    provider: str | None = None
    model: str | None = None
    generated_at: str | None = None


class Analogue(_View):
    """One historical window the historian retrieved, and how it actually resolved.

    `resolved_at` is displayed rather than hidden because it is the point-in-time claim: a
    window whose outcome resolved after the bar being analysed must never have been retrieved,
    and the panel is where that becomes visible to a human instead of only to a test.
    """

    timestamp: str
    symbol: str
    similarity: float
    outcome: str | None = None
    outcome_r: float | None = None
    resolved_at: str | None = None


class RiskOfficerNote(_View):
    """The risk officer's reading of a plan it did not choose.

    `proceed_recommendation` is advisory, and the UI says so on every card. The deterministic
    permission layer gates execution; this agent's opinion is displayed and logged and does
    nothing else (CLAUDE.md, "The three agents").
    """

    text: str
    proceed_recommendation: str | None = None
    provider: str | None = None
    model: str | None = None
    generated_at: str | None = None


class PatternNote(_View):
    """A detected candle formation and its definition.

    Carries its own banner text rather than relying on the template to remember: two studies
    found no net positive return on EUR/USD after costs, so these are UI context and nothing
    else. The label is part of the data so it cannot be styled away.
    """

    name: str
    definition: str
    bar_time: str | None = None
    label: str = "CONTEXT ONLY — NOT A SIGNAL"


class TradeView(_View):
    """The trade an entry produced, if it produced one."""

    trade_id: int
    direction: str
    volume: float
    entry_price: float
    entry_time: str
    stop_price: float
    target_price: float
    exit_price: float | None = None
    exit_time: str | None = None
    barrier_touched: str | None = None
    r_multiple: float | None = None
    mode: str


class FeedEntry(_View):
    """One evaluation, with everything that was said about it. Newest first in the feed.

    Every evaluation appears, including the ones that fired nothing. The rejections are the
    product: a panel showing only the trades would answer "what happened" and never "why did
    nothing happen", which on most bars is the only question there is.
    """

    evaluation_id: int
    cycle_id: str
    timestamp: str
    symbol: str
    fired: bool
    reason: str
    consensus_score: float
    regime: RegimeView
    votes: tuple[VoteView, ...]
    chartist: AgentNarration | None = None
    historian: AgentNarration | None = None
    analogues: tuple[Analogue, ...] = ()
    risk_officer: RiskOfficerNote | None = None
    patterns: tuple[PatternNote, ...] = ()
    trades: tuple[TradeView, ...] = ()
    #: Blocks that were present and failed validation, named so the panel can say a narration
    #: was discarded rather than silently showing nothing. Hard rule 5: no partial parsing.
    discarded: tuple[str, ...] = ()


# --- permission --------------------------------------------------------------------------


class GrantState(StrEnum):
    """The three states the permission layer can be in, as far as a reader is concerned."""

    #: Execution is impossible. The default, and the only state this build can be in.
    ADVISORY = "ADVISORY"
    GRANTED = "GRANTED"
    #: A grant existed and was withdrawn or expired. Distinct from ADVISORY so the panel can
    #: show that something was revoked rather than never granted.
    REVOKED = "REVOKED"


class GrantSnapshot(_View):
    """Execution permission as displayed. Read-only, like everything else here.

    `expires_at` travels as an instant and the countdown is computed in the browser, once a
    second, from the client's own clock. Pushing a decreasing number over the socket would make
    the countdown a function of network latency, and a permission countdown that lags is worse
    than no countdown.
    """

    state: GrantState = GrantState.ADVISORY
    reason: str = ""
    granted_at: str | None = None
    expires_at: str | None = None
    symbols: tuple[str, ...] = ()
    source: str = ""


# --- envelope ----------------------------------------------------------------------------


class SeriesOption(_View):
    """One (symbol, timeframe, source) the store actually holds bars for."""

    symbol: str
    timeframe: str
    source: str
    bars: int = 0
    latest: str | None = None


class FeedPayload(_View):
    symbol: str
    entries: tuple[FeedEntry, ...]
    grant: GrantSnapshot
    notes: tuple[str, ...] = ()


class Snapshot(_View):
    """One complete view of one (symbol, timeframe). The unit the socket pushes.

    `revision` is a content hash covering the chart and the feed and *not* `generated_at`, so
    a rebuild that found nothing new produces the same revision and nothing is pushed. That is
    what keeps the server-side refresh loop from turning into a broadcast loop.
    """

    symbol: str
    timeframe: str
    revision: str
    generated_at: str
    chart: ChartPayload
    feed: FeedPayload
    options: tuple[SeriesOption, ...] = ()
    #: Set when the store could not be read. The panel shows the last good snapshot greyed out
    #: with this attached, rather than an empty chart that looks like a quiet market.
    error: str | None = None


class Envelope(_View):
    """What actually goes down the socket: a type tag and a body.

    A tagged envelope rather than a bare `Snapshot` because the socket will eventually carry
    more than one kind of message, and adding the tag afterwards means every client has to
    guess from the shape.
    """

    type: str
    snapshot: Snapshot | None = None
    message: str | None = None
