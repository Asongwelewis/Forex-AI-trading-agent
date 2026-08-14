-- Encoded price windows for agent 2 (the historian) to retrieve analogues from.
--
-- DIMENSION: 128. The window encoder does not exist yet, so this is a starting choice, not a
-- measurement. Changing it later means a new migration and re-encoding every row — cheap now
-- while the table is empty, expensive once it is not. 128 is small enough that hnsw stays fast
-- and large enough for a multi-bar normalised OHLC window. Revisit when the encoder lands.
--
-- outcome_resolved_at is NOT part of forward_outcome, and is the reason this table can be
-- queried at all without cheating. forward_outcome describes what happened AFTER the window —
-- it is future data by construction. Hard rule 6 requires historical analogues to have
-- resolved before the current bar, so retrieval filters on this column the way events filters
-- on publication_time_utc. Without it, every retrieval hands the historian the future and the
-- backtest becomes meaningless. It is nullable: a window whose outcome has not yet resolved is
-- stored and is simply invisible to retrieval until it has.

create table if not exists windows (
    id                  bigint generated always as identity primary key,
    symbol              text        not null,
    ts_utc              timestamptz not null,
    timeframe           text        not null,
    embedding           vector(128) not null,
    normalised_ohlc     jsonb       not null,
    forward_outcome     jsonb,
    outcome_resolved_at timestamptz,
    created_at          timestamptz not null default now(),

    constraint windows_unique unique (symbol, timeframe, ts_utc),
    constraint windows_timeframe_known
        check (timeframe in ('M1','M5','M15','M30','H1','H4','D1')),
    constraint windows_ohlc_is_object check (jsonb_typeof(normalised_ohlc) = 'object'),

    -- An outcome and its resolution time arrive together or not at all. A forward_outcome with
    -- no resolved_at would be permanently invisible to retrieval, which looks like data loss.
    constraint windows_outcome_consistent check (
        (forward_outcome is null and outcome_resolved_at is null)
        or (forward_outcome is not null and outcome_resolved_at is not null)
    ),
    -- An outcome cannot resolve before the window it describes.
    constraint windows_outcome_after_window
        check (outcome_resolved_at is null or outcome_resolved_at >= ts_utc)
);

-- hnsw, not ivfflat.
--
-- ivfflat clusters the vectors it can see when the index is built. A migration builds it on an
-- EMPTY table, which produces a single degenerate list and recall that stays poor until someone
-- remembers to reindex after backfilling. hnsw builds incrementally and needs no training pass,
-- so it is correct on an empty table and stays correct as rows arrive. It costs more to build
-- and more memory; at this table's scale that is a good trade for not silently returning bad
-- neighbours.
--
-- Cosine distance because the encoder emits normalised windows — shape is the signal, and
-- magnitude has already been divided out.
create index if not exists windows_embedding_idx
    on windows using hnsw (embedding vector_cosine_ops);

-- Retrieval is always filtered to resolved outcomes, so give that predicate an index too.
create index if not exists windows_resolved_idx
    on windows (outcome_resolved_at) where outcome_resolved_at is not null;
create index if not exists windows_symbol_ts_idx on windows (symbol, timeframe, ts_utc desc);
