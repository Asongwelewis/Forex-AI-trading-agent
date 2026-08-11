-- Trades, each traceable to the evaluation that produced it.
--
-- label_span_start/end bound the window the outcome label is drawn from. Purged, embargoed
-- walk-forward folds (hard rule 7) need to know which trades' labels overlap a candidate test
-- window, and that is answerable only if the span is stored rather than recomputed later from
-- assumptions about holding period.
--
-- barrier_touched is the triple-barrier outcome: TARGET, STOP, or TIME for a span that expired
-- untouched. Nullable only while the trade is open.

create table if not exists trades (
    id               bigint generated always as identity primary key,
    evaluation_id    bigint      not null references evaluations (id) on delete restrict,
    symbol           text        not null,
    direction        text        not null,
    volume           double precision not null,
    entry_price      double precision not null,
    entry_time_utc   timestamptz not null,
    exit_price       double precision,
    exit_time_utc    timestamptz,
    stop_price       double precision not null,
    target_price     double precision not null,
    barrier_touched  text,
    label_span_start timestamptz not null,
    label_span_end   timestamptz not null,
    pnl              double precision,
    r_multiple       double precision,
    mode             text        not null,
    created_at       timestamptz not null default now(),

    constraint trades_direction_known check (direction in ('LONG', 'SHORT')),
    constraint trades_mode_known check (mode in ('ADVISORY', 'DEMO_AUTO', 'LIVE')),
    constraint trades_barrier_known
        check (barrier_touched is null or barrier_touched in ('TARGET', 'STOP', 'TIME')),
    constraint trades_volume_positive check (volume > 0),
    constraint trades_prices_positive
        check (entry_price > 0 and stop_price > 0 and target_price > 0
               and (exit_price is null or exit_price > 0)),
    constraint trades_label_span_ordered check (label_span_end >= label_span_start),
    constraint trades_exit_after_entry
        check (exit_time_utc is null or exit_time_utc >= entry_time_utc),

    -- A trade is open or fully closed, never half. Without this a crash between writing the
    -- exit price and the barrier leaves a row that reads as closed but cannot be labelled.
    constraint trades_exit_consistent check (
        (exit_time_utc is null and exit_price is null and barrier_touched is null)
        or (exit_time_utc is not null and exit_price is not null and barrier_touched is not null)
    ),

    -- Stop and target on the correct sides of entry — the same invariant OrderRequest enforces
    -- in Pydantic (hard rule 3), restated where the data actually lives.
    constraint trades_protection_sides check (
        (direction = 'LONG' and stop_price < entry_price and target_price > entry_price)
        or (direction = 'SHORT' and stop_price > entry_price and target_price < entry_price)
    )
);

create index if not exists trades_evaluation_idx on trades (evaluation_id);
create index if not exists trades_symbol_entry_idx on trades (symbol, entry_time_utc desc);
-- Purge/embargo asks "which labels overlap this window", which is a range query on the span.
create index if not exists trades_label_span_idx on trades (label_span_start, label_span_end);
-- Open trades are a small hot subset the executor polls constantly.
create index if not exists trades_open_idx on trades (symbol) where exit_time_utc is null;
