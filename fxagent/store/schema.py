"""SQLAlchemy Core table definitions mirroring the SQL migrations.

The migrations are the source of truth for DDL — this module never creates anything. It exists
so repositories compose typed expressions instead of formatting SQL strings, which is what
keeps raw SQL out of the rest of the codebase.

Mirroring means these definitions can drift from the migrations that actually ran.
`tests/store/test_health_and_schema.py` reflects the migrated database and compares it against
this metadata, so drift fails the suite rather than surfacing as a runtime error in a query
nobody ran yet.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB as _JSONB
from sqlalchemy.dialects.postgresql import UUID

#: `none_as_null=True` matters: without it SQLAlchemy stores Python `None` in a JSONB column as
#: the JSON value `null` rather than SQL NULL. `forward_outcome is null` then reads false, and
#: the windows outcome-consistency constraint rejects a perfectly valid unresolved window.
JSONB = _JSONB(none_as_null=True)

try:  # pragma: no cover - exercised implicitly by every pgvector-backed test
    from pgvector.sqlalchemy import Vector
except ImportError as exc:  # pragma: no cover - dependency is declared, so this is a bad env
    raise ImportError("pgvector is required for the windows table. Run `uv sync`.") from exc

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "bars",
    "cot_reports",
    "service_heartbeats",
    "evaluations",
    "events",
    "metadata",
    "trades",
    "windows",
]

#: Must match `vector(N)` in 0006_windows.sql. Changing it needs a migration and a re-encode.
EMBEDDING_DIMENSIONS = 128

metadata = MetaData()


def _utc_column(name: str, **kwargs: object) -> Column[object]:
    """timestamptz, always. A naive timestamp column is how a session ends up an hour wrong."""
    return Column(name, DateTime(timezone=True), **kwargs)  # type: ignore[arg-type]


bars = Table(
    "bars",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("symbol", Text, nullable=False),
    Column("timeframe", Text, nullable=False),
    _utc_column("ts_utc", nullable=False),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", BigInteger, nullable=False),
    Column("bid_close", Float),
    Column("ask_close", Float),
    Column("source", Text, nullable=False),
    _utc_column("ingested_at", nullable=False),
    UniqueConstraint("symbol", "timeframe", "ts_utc", "source", name="bars_unique"),
    Index("bars_symbol_tf_ts_idx", "symbol", "timeframe", "ts_utc"),
)

events = Table(
    "events",
    metadata,
    Column("id", BigInteger, primary_key=True),
    _utc_column("event_time_utc", nullable=False),
    #: The gate. Every read filters on this, never on event_time_utc.
    _utc_column("publication_time_utc", nullable=False),
    Column("source", Text, nullable=False),
    Column("currency", String(3), nullable=False),
    Column("category", Text),
    Column("importance", Text, nullable=False),
    Column("title", Text, nullable=False),
    Column("body", Text),
    Column("forecast", Float),
    Column("actual", Float),
    Column("previous", Float),
    Column("surprise_score", Float),
    _utc_column("ingested_at", nullable=False),
    UniqueConstraint("source", "currency", "event_time_utc", "title", name="events_unique"),
)

evaluations = Table(
    "evaluations",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("cycle_id", UUID(as_uuid=True), nullable=False),
    _utc_column("ts_utc", nullable=False),
    Column("symbol", Text, nullable=False),
    Column("regime", JSONB, nullable=False),
    Column("votes", JSONB, nullable=False),
    Column("consensus_score", Float, nullable=False),
    Column("fired", Boolean, nullable=False),
    Column("reason", Text, nullable=False),
    _utc_column("created_at", nullable=False),
    UniqueConstraint("cycle_id", "symbol", name="evaluations_unique"),
)

trades = Table(
    "trades",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column(
        "evaluation_id",
        BigInteger,
        ForeignKey("evaluations.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("symbol", Text, nullable=False),
    Column("direction", Text, nullable=False),
    Column("volume", Float, nullable=False),
    Column("entry_price", Float, nullable=False),
    _utc_column("entry_time_utc", nullable=False),
    Column("exit_price", Float),
    _utc_column("exit_time_utc"),
    Column("stop_price", Float, nullable=False),
    Column("target_price", Float, nullable=False),
    Column("barrier_touched", Text),
    _utc_column("label_span_start", nullable=False),
    _utc_column("label_span_end", nullable=False),
    Column("pnl", Float),
    Column("r_multiple", Float),
    Column("mode", Text, nullable=False),
    _utc_column("created_at", nullable=False),
)

windows = Table(
    "windows",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("symbol", Text, nullable=False),
    _utc_column("ts_utc", nullable=False),
    Column("timeframe", Text, nullable=False),
    Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
    Column("normalised_ohlc", JSONB, nullable=False),
    Column("forward_outcome", JSONB),
    #: Point-in-time gate for retrieval, mirroring events.publication_time_utc. A window whose
    #: outcome has not resolved is invisible to search.
    _utc_column("outcome_resolved_at"),
    _utc_column("created_at", nullable=False),
    UniqueConstraint("symbol", "timeframe", "ts_utc", name="windows_unique"),
)


service_heartbeats = Table(
    "service_heartbeats",
    metadata,
    Column("service", Text, primary_key=True),
    _utc_column("started_at_utc", nullable=False),
    _utc_column("last_beat_utc", nullable=False),
    Column("beats", BigInteger, nullable=False),
    Column("detail", JSONB),
)


#: Raw prints from the primary statistical sources, accumulating so that a surprise history
#: exists by the time anything wants one. Deliberately NOT `events`: these carry no forecast and
#: no defensible publication time, so they must stay outside `events_visible_at()` and therefore
#: outside the analysis pipeline. See migration 0010 for the reasoning.
statistical_observations = Table(
    "statistical_observations",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("source", Text, nullable=False),
    Column("series_id", Text, nullable=False),
    #: A reference period ('2026-07'), not a publication date.
    Column("reference_period", Text, nullable=False),
    Column("period_start", Date, nullable=False),
    Column("value", Float, nullable=False),
    #: The original print. Never updated — BLS revises, and the market traded the first number.
    Column("first_value", Float, nullable=False),
    _utc_column("first_seen_at", nullable=False),
    _utc_column("revised_at"),
    Column("revisions", Integer, nullable=False),
    UniqueConstraint("source", "series_id", "reference_period", name="statobs_unique"),
    Index("statobs_series_period_idx", "source", "series_id", "period_start"),
)


#: CFTC Commitments of Traders positioning. The one statistical source with a real publication
#: schedule, so unlike `statistical_observations` it sits behind a point-in-time gate and may be
#: read by the analysis pipeline — through `cot_visible_at()` and nothing else. See migration
#: 0011 for why the two timestamps are three days apart and what that costs if they are confused.
cot_reports = Table(
    "cot_reports",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("currency", String(3), nullable=False),
    Column("contract_code", Text, nullable=False),
    Column("contract_name", Text, nullable=False),
    #: The Tuesday positions were measured. Never a visibility input.
    Column("report_date", Date, nullable=False),
    #: The gate: the following Friday 15:30 US/Eastern, in UTC.
    _utc_column("published_at", nullable=False),
    Column("noncommercial_long", BigInteger, nullable=False),
    Column("noncommercial_short", BigInteger, nullable=False),
    #: Generated in Postgres so the stored net can never disagree with its own legs.
    Column(
        "net_position",
        BigInteger,
        Computed("noncommercial_long - noncommercial_short", persisted=True),
    ),
    Column("open_interest", BigInteger),
    _utc_column("fetched_at", nullable=False),
    UniqueConstraint("contract_code", "report_date", name="cot_reports_unique"),
    Index("cot_reports_currency_date_idx", "currency", "report_date"),
    Index("cot_reports_published_idx", "published_at"),
)
