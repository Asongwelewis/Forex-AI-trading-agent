-- Data-freshness check for the daily health workflow.
--
-- Answers one question: over the last N hours, is there an H1 bar for every hour the FX
-- market was actually open? A scheduled collector that quietly stops firing leaves exactly
-- this signature, and nothing else in the system notices — the analyst keeps running happily
-- on whatever was last stored.
--
-- Two things make a naive `count(*) < 24` check useless here:
--
--   * **The market is shut for two days a week.** FX runs Sunday 17:00 to Friday 17:00 in
--     New York. A Saturday check finds zero bars and that is correct. The boundary is derived
--     from `America/New_York` rather than a fixed UTC hour because it moves with US DST —
--     21:00 UTC in winter, 22:00 UTC in summer. Same reason fxagent/regime/sessions.py uses
--     zoneinfo instead of constants.
--   * **`source` is part of a bar's identity** (hard rule 7). An unfiltered count would let
--     the MT5 backfill rows mask a total Twelve Data outage.
--
-- Emits pipe-separated `kind|subject|value|detail` rows for .github/scripts/health-report.sh.
-- Read-only: no writes, no temp tables, safe to run against the live project.
--
-- Parameters (psql -v):
--   symbols    comma-separated, canonical form without the broker suffix
--   source     bars.source to filter on, e.g. twelvedata
--   timeframe  e.g. H1
--   hours      look-back window

\set ON_ERROR_STOP on

-- date_trunc('hour', ...) on a timestamptz truncates in the session timezone. Pin it so the
-- result does not depend on whatever the server or the pooler happens to default to.
set time zone 'UTC';

with bounds as (
    select
        date_trunc('hour', now()) - make_interval(hours => :hours) as lo,
        -- The current hour's bar is still forming and will not be complete until the hour
        -- closes. Demanding it reports a phantom gap on every single run.
        date_trunc('hour', now()) - interval '1 hour' as hi
),
wanted as (
    select trim(both from s) as symbol
    from unnest(string_to_array(:'symbols', ',')) as s
    where trim(both from s) <> ''
),
open_slots as (
    select g.ts
    from bounds b
    cross join generate_series(b.lo, b.hi, interval '1 hour') as g(ts)
    cross join lateral (select g.ts at time zone 'America/New_York' as ny) as n
    where case extract(isodow from n.ny)
              when 6 then false                          -- Saturday: shut all day
              when 5 then extract(hour from n.ny) < 17   -- Friday: shuts at 17:00 New York
              when 7 then extract(hour from n.ny) >= 17  -- Sunday: opens at 17:00 New York
              else true                                  -- Monday to Thursday: open
          end
),
expected as (
    select w.symbol, o.ts
    from wanted w
    cross join open_slots o
),
observed as (
    select
        e.symbol,
        e.ts,
        (b.id is not null) as present
    from expected e
    left join bars b
        on  b.symbol    = e.symbol
        and b.timeframe = :'timeframe'
        and b.source    = :'source'
        and b.ts_utc    = e.ts
),
per_symbol as (
    select
        symbol,
        count(*)::int                            as expected_bars,
        count(*) filter (where not present)::int as missing_bars
    from observed
    group by symbol
)
select kind, subject, value, detail from (
    -- The window itself, so a surprising result can be read without re-deriving the bounds.
    select 1 as ord, 'window' as kind, :'source' as subject,
           to_char((select lo from bounds), 'YYYY-MM-DD"T"HH24:MI"Z"') as value,
           to_char((select hi from bounds), 'YYYY-MM-DD"T"HH24:MI"Z"') as detail

    union all

    -- Total open hours in the window. Zero means the market was shut for the whole window,
    -- which is a legitimate weekend result and not a fault.
    select 2, 'open_hours', :'timeframe',
           (select count(*)::text from open_slots),
           ''

    union all

    select 3, 'symbol', symbol,
           missing_bars::text,
           expected_bars::text
    from per_symbol

    union all

    -- Age of the newest bar actually stored, independent of the window. This is what catches
    -- a collector that died three days ago: the 24h gap count saturates, but the age keeps
    -- climbing and says how long it has been down.
    select 4, 'latest', w.symbol,
           coalesce(to_char(m.newest, 'YYYY-MM-DD"T"HH24:MI"Z"'), 'never'),
           coalesce(floor(extract(epoch from now() - m.newest) / 60)::text, '')
    from wanted w
    left join lateral (
        select max(b.ts_utc) as newest
        from bars b
        where b.symbol    = w.symbol
          and b.timeframe = :'timeframe'
          and b.source    = :'source'
    ) m on true
) rows
order by ord, subject;
