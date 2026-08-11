-- Liveness record, so "the collector ran for an hour unattended" is a query rather than a claim.
--
-- One row per service, updated in place. A log file on a container that has since been replaced
-- proves nothing; this survives the process and is readable from anywhere with the DSN.
--
-- `started_at_utc` distinguishes a service that has been up for an hour from one that has
-- crash-looped sixty times a minute apart — both show a recent heartbeat, and only the second
-- has a start time that keeps moving.

create table if not exists service_heartbeats (
    service        text        primary key,
    started_at_utc timestamptz not null,
    last_beat_utc  timestamptz not null,
    beats          bigint      not null default 0,
    detail         jsonb,

    constraint service_heartbeats_ordered check (last_beat_utc >= started_at_utc),
    constraint service_heartbeats_beats_non_negative check (beats >= 0)
);

create index if not exists service_heartbeats_last_beat_idx
    on service_heartbeats (last_beat_utc desc);
