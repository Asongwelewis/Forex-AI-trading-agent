-- Raw statistical prints from the primary sources, accumulating against a future need.
--
-- `surprise_score` needs eight or more historical (actual - forecast) samples per release
-- before it means anything, and these releases are monthly. That clock only runs while we are
-- collecting, so collection starts now even though nothing consumes the result yet. Joining
-- these to calendar events — a series-to-event map, unit reconciliation, revision handling —
-- is the expensive half and waits until something actually reads a surprise score.
--
-- **This is deliberately not the `events` table.** These are not calendar events: they have no
-- forecast, no importance, no currency-level meaning, and above all no defensible
-- publication_time_utc — none of BLS, Eurostat or the CFTC publishes a per-release timestamp
-- (docs/ADR-003-fundamentals.md). Putting them in `events` would make them visible to
-- `events_visible_at()` and therefore to `build_context`, and a row with a guessed publication
-- time inside the point-in-time gate is worse than no row at all: it looks audited and is not.
-- There is no read path from here into the analysis pipeline, and
-- `tests/store/test_observations_are_not_in_the_pipeline.py` keeps it that way.
--
-- **`first_value` is never updated, and that is the whole point.** BLS revises the prior two
-- months at every release. The market reacted to the original print, so a surprise history has
-- to be built from what was first published, not from today's corrected series. An upsert that
-- simply overwrote `value` would silently destroy the only number we are collecting for, and
-- would do it invisibly, months before anyone looked.

create table if not exists statistical_observations (
    id               bigint generated always as identity primary key,

    -- 'bls' | 'eurostat'. Part of the key: two providers may publish the same concept.
    source           text        not null,
    -- The provider's own identifier, verbatim. 'CES0000000001', or a composite for datasets
    -- that need dimensions to be unique ('prc_hicp_manr:EA:CP00'). Never reinterpreted here.
    series_id        text        not null,

    -- The period the number describes, canonicalised to 'YYYY-MM' or 'YYYY-Qn'. This is a
    -- reference period, NOT a publication date — July's payrolls are published in August.
    reference_period text        not null,
    -- First day of that period, so ranges and ordering are a date comparison rather than
    -- string arithmetic that breaks at a year boundary.
    period_start     date        not null,

    -- Latest known value, revisions included.
    value            double precision not null,
    -- The first value ever seen for this period. Immutable; see the header.
    first_value      double precision not null,

    first_seen_at    timestamptz not null default now(),
    revised_at       timestamptz,
    revisions        integer     not null default 0,

    constraint statobs_unique unique (source, series_id, reference_period),
    constraint statobs_revisions_non_negative check (revisions >= 0)
);

-- The read this table exists to serve: one series, oldest first, to build a history from.
create index if not exists statobs_series_period_idx
    on statistical_observations (source, series_id, period_start);
