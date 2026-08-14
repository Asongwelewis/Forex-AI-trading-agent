-- Close the one hole left in the point-in-time gate: a released *value* visible before release.
--
-- `events_visible_at()` filters rows on publication_time_utc, which is right for the row's
-- existence and wrong for one of its columns. The two facts a calendar event carries become
-- public at different moments:
--
--   * **That the event is scheduled**, with its forecast and the previous print. Public days
--     ahead. This is what lets the permission layer black out the fifteen minutes around a
--     release, and hiding it would disable that entirely.
--   * **The number itself.** Public at event_time_utc, not before. Nobody knows Friday's
--     payrolls on Monday.
--
-- One row carries both, and `upsert_many` is what joins them in time: it inserts the scheduled
-- event with publication_time_utc set to first sighting and deliberately never rewrites that
-- field, then later updates `actual` in place when the number lands. Correct on its own terms —
-- rewriting publication time would let a revision retroactively become visible earlier — but it
-- leaves a row published Monday holding Friday's result, and the gate waves it through.
--
-- A backtest replaying Monday then reads Friday's number. Nothing errors. The equity curve is
-- simply a fiction, which is exactly the failure hard rule 6 exists to prevent, and exactly the
-- kind that survives review because every individual piece looks defensible.
--
-- So the gate now nulls the two result columns until the event has actually happened. `forecast`
-- and `previous` are untouched: both are genuinely published with the schedule.
--
-- This is latent today — no current source populates `actual`, because faireconomy's feed does
-- not carry it (see docs/ADR-003-fundamentals.md). It is fixed before rather than after wiring
-- a source that does, because the alternative is ingesting real results into a store that leaks
-- them and discovering it from a backtest that looked good.

create or replace function events_visible_at(as_of timestamptz)
returns setof events
language sql
stable
as $$
    select
        id,
        event_time_utc,
        publication_time_utc,
        source,
        currency,
        category,
        importance,
        title,
        body,
        -- Published with the schedule; knowable in advance by design.
        forecast,
        -- The result. Withheld until the release moment even though the row is visible.
        case when as_of >= event_time_utc then actual end,
        previous,
        -- Derived from actual, so it leaks exactly as far as actual does.
        case when as_of >= event_time_utc then surprise_score end,
        ingested_at
    from events
    where publication_time_utc <= as_of
$$;

comment on function events_visible_at(timestamptz) is
    'Events published at or before as_of, with actual and surprise_score withheld until '
    'as_of reaches event_time_utc. The only sanctioned read path — querying the events table '
    'directly re-introduces look-ahead bias (CLAUDE.md hard rule 6).';
