-- Fundamental events (calendar releases, central bank statements, COT reports).
--
-- THE ONE THING THIS TABLE EXISTS TO GET RIGHT: event_time_utc and publication_time_utc are
-- different instants, and every read path filters on PUBLICATION time.
--
-- They diverge constantly and in both directions. A CFTC COT report has a Tuesday reference
-- date and a Friday release — keying on event time leaks three days of the future into every
-- backtest. A calendar entry is published days before the event it describes, but its `actual`
-- field is only populated at release, so a row can be visible long before its outcome is.
-- Filtering on event_time_utc looks correct, passes every smoke test, and quietly inflates
-- backtest results until the strategy meets live data.
--
-- The filter is enforced here in SQL via events_visible_at(), not only in the repository, so
-- an ad-hoc analyst query cannot bypass it by accident.

create table if not exists events (
    id                   bigint generated always as identity primary key,
    event_time_utc       timestamptz not null,
    publication_time_utc timestamptz not null,
    source               text        not null,
    currency             text        not null,
    category             text,
    importance           text        not null,
    title                text        not null,
    body                 text,
    forecast             double precision,
    actual               double precision,
    previous             double precision,
    surprise_score       double precision,
    ingested_at          timestamptz not null default now(),

    -- Forex Factory has no event ID; it dedupes on (date, currency, title). `source` is in the
    -- key so a second provider describing the same release does not collide with the first.
    constraint events_unique unique (source, currency, event_time_utc, title),

    -- Four values, not three. A Pydantic enum missing Holiday rejects the whole feed, and so
    -- would this constraint — better to fail on ingest than to drop rows silently.
    constraint events_importance_known
        check (importance in ('High', 'Medium', 'Low', 'Holiday')),

    -- Forex Factory's `country` field actually holds a currency code.
    constraint events_currency_shape check (currency ~ '^[A-Z]{3}$')
);

-- Every read is "what was visible as of T", so publication_time_utc leads each index.
create index if not exists events_pub_time_idx on events (publication_time_utc);
create index if not exists events_pub_currency_idx
    on events (publication_time_utc, currency);
-- The auto-revoke check is "high-impact event for this currency near now".
create index if not exists events_pub_event_time_idx
    on events (publication_time_utc, event_time_utc);

-- Point-in-time gate, in SQL.
--
-- STABLE, not IMMUTABLE: it reads a table. Marked so the planner may still cache it within a
-- single statement. Every repository read composes from this rather than touching `events`.
create or replace function events_visible_at(as_of timestamptz)
returns setof events
language sql
stable
as $$
    select * from events where publication_time_utc <= as_of
$$;

comment on function events_visible_at(timestamptz) is
    'Events published at or before as_of. The only sanctioned read path — querying the events '
    'table directly re-introduces look-ahead bias (CLAUDE.md hard rule 6).';
