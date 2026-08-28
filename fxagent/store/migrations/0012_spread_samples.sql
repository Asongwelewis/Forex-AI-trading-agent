-- Measured Exness bid/ask, sampled during the London open.
--
-- The backtest fills every order at a configured 1-pip spread because no feed in `bars` carries
-- a two-sided quote — `bid_close` and `ask_close` are null on all 13,000 rows. That constant is
-- the largest unmeasured assumption in the result, and `session_breakout` trades the London
-- open, which is exactly when a market-maker's spread is worst.
--
-- This table exists to replace the guess with a distribution. Not an average: the mean spread
-- over a quiet hour tells you nothing about the fill you get on the bar that actually breaks
-- out. The tail is the number.
--
-- Deliberately NOT in `bars`. A spread sample is a point-in-time observation of dealing
-- conditions with no OHLC behind it, and folding it into a bar row would require inventing an
-- interval it does not have.
create table if not exists spread_samples (
    id             bigserial primary key,
    symbol         text        not null,
    -- The broker's own symbol, suffix included ('EURUSDm'), because the suffix identifies the
    -- account type and its spread schedule. Two Exness account types quote the same pair
    -- differently, and a sample that has forgotten which one it came from cannot be compared.
    broker_symbol  text        not null,
    sampled_at     timestamptz not null,
    bid            double precision not null,
    ask            double precision not null,
    -- Reported by the terminal rather than derived, so a disagreement with (ask - bid) / point
    -- is visible instead of silently reconciled. `symbol_info().spread`.
    spread_points  integer     not null,
    point          double precision not null,
    -- Exness spreads float; a fixed-spread account would report false here and its samples
    -- must not be pooled with floating ones.
    spread_float   boolean     not null,
    source         text        not null default 'mt5_exness',
    ingested_at    timestamptz not null default now(),

    constraint spread_samples_ask_not_below_bid check (ask >= bid),
    constraint spread_samples_positive_prices check (bid > 0 and ask > 0),
    -- One sample per symbol per instant. A poller restarted mid-minute must not double-count,
    -- which would pull the measured distribution toward whatever it was doing at restart.
    constraint spread_samples_unique unique (broker_symbol, sampled_at, source)
);

-- The query this table exists to serve: every sample for one symbol inside a session window.
create index if not exists spread_samples_symbol_time_idx
    on spread_samples (symbol, sampled_at);
