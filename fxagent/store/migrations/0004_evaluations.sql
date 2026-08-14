-- Every consensus evaluation, including the ones that fired nothing.
--
-- The no-signal rows are the point. Disagreement between the three strategies is the training
-- data for the regime router, and a table holding only the trades that happened cannot answer
-- "what was the router seeing on the days it stayed out".
--
-- regime and votes are jsonb rather than columns because both are still moving. votes holds
-- one entry per strategy including abstentions; regime holds the classifier output.

create table if not exists evaluations (
    id              bigint generated always as identity primary key,
    cycle_id        uuid        not null,
    ts_utc          timestamptz not null,
    symbol          text        not null,
    regime          jsonb       not null,
    votes           jsonb       not null,
    consensus_score double precision not null,
    fired           boolean     not null,
    reason          text        not null default '',
    created_at      timestamptz not null default now(),

    -- One evaluation per symbol per cycle. Makes a retried cycle idempotent instead of
    -- double-counting when the collector reconnects mid-run.
    constraint evaluations_unique unique (cycle_id, symbol),
    constraint evaluations_votes_is_object check (jsonb_typeof(votes) = 'object'),
    constraint evaluations_regime_is_object check (jsonb_typeof(regime) = 'object'),

    -- A fired evaluation must say what fired it; a silent one must say why it stayed out.
    constraint evaluations_reason_present check (length(reason) > 0)
);

create index if not exists evaluations_ts_idx on evaluations (ts_utc desc);
create index if not exists evaluations_symbol_ts_idx on evaluations (symbol, ts_utc desc);
-- Partial index: "show me the days nothing fired" is a routine question and the no-signal
-- rows will outnumber the fired ones by orders of magnitude.
create index if not exists evaluations_fired_idx on evaluations (ts_utc desc) where fired;
