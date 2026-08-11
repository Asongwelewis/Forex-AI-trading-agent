-- Price bars, one row per (symbol, timeframe, bar open time, source).
--
-- `source` is part of the identity, not metadata: the same EURUSD H1 bar from OANDA and from
-- MT5 will not agree to the last decimal, and silently overwriting one with the other would
-- make a backtest irreproducible. Keeping both lets a later query pick a source explicitly.
--
-- Prices are double precision rather than numeric. FX quotes carry at most 6 significant
-- decimals, which float64 represents exactly, and the indicator layer is numpy float64 end to
-- end — numeric would force a Decimal round-trip at every read for accuracy we do not gain.
--
-- ts_utc is the bar's OPEN time, matching fxagent.adapters.base.Bar.

create table if not exists bars (
    id              bigint generated always as identity primary key,
    symbol          text        not null,
    timeframe       text        not null,
    ts_utc          timestamptz not null,
    open            double precision not null,
    high            double precision not null,
    low             double precision not null,
    close           double precision not null,
    volume          bigint      not null,
    bid_close       double precision,
    ask_close       double precision,
    source          text        not null,
    ingested_at     timestamptz not null default now(),

    constraint bars_unique unique (symbol, timeframe, ts_utc, source),

    -- The same OHLC invariants Bar enforces in Pydantic. Defence in depth: a bad row must not
    -- be storable even if it arrives by a path that skipped the model.
    constraint bars_timeframe_known check (timeframe in ('M1','M5','M15','M30','H1','H4','D1')),
    constraint bars_positive check (open > 0 and high > 0 and low > 0 and close > 0),
    constraint bars_volume_non_negative check (volume >= 0),
    constraint bars_high_is_highest check (high >= greatest(open, close) and high >= low),
    constraint bars_low_is_lowest check (low <= least(open, close)),
    constraint bars_spread_ordered check (
        bid_close is null or ask_close is null or ask_close >= bid_close
    )
);

-- Range reads are always "this symbol, this timeframe, this time window", newest first.
create index if not exists bars_symbol_tf_ts_idx on bars (symbol, timeframe, ts_utc desc);
