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
    DateTime,
    Float,
    ForeignKey,
    Index,
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
