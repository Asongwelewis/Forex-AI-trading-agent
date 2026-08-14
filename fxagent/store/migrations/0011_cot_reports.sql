-- CFTC Commitments of Traders positioning, behind its own point-in-time gate.
--
-- **This is the one statistical source with a defensible publication time**, which is why it
-- gets a gated table rather than joining `statistical_observations`. Migration 0010 keeps BLS
-- and Eurostat prints outside the analysis pipeline because nobody publishes a per-release
-- timestamp for them, so no honest answer to "what was knowable at time T" exists. The COT is
-- different: the CFTC releases it on a fixed weekly schedule — Friday at 15:30 US/Eastern,
-- covering positions as of the preceding Tuesday — and that schedule is the publication time.
-- Computed rather than fetched (the Socrata dataset carries no publication column, verified
-- 14 Aug 2026), but computed from a published rule, not guessed.
--
-- **The three-day gap between the two timestamps is the entire risk.** A row references Tuesday
-- and is public on Friday. Anything filtering on `report_date` sees Tuesday's positioning on
-- Tuesday, which nobody could have. That is a 72-hour look-ahead on a weekly series — large
-- enough to matter and quiet enough to survive review, because the column it reads is real and
-- correctly named. So reads go through `cot_visible_at()`, exactly as events go through
-- `events_visible_at()`, and `CotRepository` offers no ungated alternative.
--
-- **Unlike `events`, there is no second withholding rule here.** A calendar row carries a
-- schedule that is public days ahead and a result that is not, so 0009 nulls the result columns
-- until the release. A COT row carries only the numbers, and they all become public together at
-- one instant. Row-level visibility is therefore sufficient, and adding a column-level rule
-- would be a second definition of the same gate with nothing behind it.
--
-- Holiday weeks are the known imprecision: a US federal holiday pushes the release to the
-- following Monday, and there is no field to detect that from. The error direction is the safe
-- one only if we err *late*, so `fxagent.fundamentals.cot.publication_time` documents the
-- residual exposure rather than pretending it away — see its docstring.

create table if not exists cot_reports (
    id                  bigint generated always as identity primary key,

    -- The currency the contract is a claim on, from the explicit code map in cot.py. Not
    -- derived from the contract name: 'EURO FX' and 'BRITISH POUND' share no naming rule, and
    -- substring matching against a live feed is how AUD ends up reading a gold contract.
    currency            text        not null,
    -- CFTC contract market code, six digits, verbatim ('099741'). The real join key.
    contract_code       text        not null,
    contract_name       text        not null,

    -- The Tuesday the positions were measured. Reference date, NOT a publication date.
    report_date         date        not null,
    -- When the world learned it: the following Friday, 15:30 US/Eastern, in UTC. THE GATE.
    published_at        timestamptz not null,

    noncommercial_long  bigint      not null,
    noncommercial_short bigint      not null,
    -- Generated, not written. The percentile ranks net positioning, and a stored net that some
    -- future writer computed differently from its own legs would be undetectable in the data.
    net_position        bigint generated always as (noncommercial_long - noncommercial_short)
                        stored,
    open_interest       bigint,

    -- When we fetched it. Drives the daily cache check; never a visibility input.
    fetched_at          timestamptz not null default now(),

    constraint cot_reports_unique unique (contract_code, report_date),
    constraint cot_currency_shape check (currency ~ '^[A-Z]{3}$'),
    constraint cot_positions_non_negative
        check (noncommercial_long >= 0 and noncommercial_short >= 0),
    -- Publication must follow the reference date. A row violating this has had one of the two
    -- timestamps filled in from the other, which is the failure this table exists to prevent.
    constraint cot_published_after_reference check (published_at > report_date::timestamptz)
);

-- The read the percentile is built from: one currency, oldest first, gated on publication.
create index if not exists cot_reports_currency_date_idx
    on cot_reports (currency, report_date);

create index if not exists cot_reports_published_idx
    on cot_reports (published_at);

create or replace function cot_visible_at(as_of timestamptz)
returns setof cot_reports
language sql
stable
as $$
    select * from cot_reports where published_at <= as_of
$$;

comment on function cot_visible_at(timestamptz) is
    'COT reports released at or before as_of. The only sanctioned read path — querying '
    'cot_reports directly filters on report_date and back-dates every row by three days '
    '(CLAUDE.md hard rule 6).';
