-- Point-in-time refusal ledger for analysis and meta-labelling.
--
-- The evaluations row is deliberately JSONB because the deterministic diagnostics evolve. This
-- function is the typed projection used by analytics: it exposes the fields that are stable
-- enough to query while keeping the raw row available for replay. It takes an as-of instant so
-- a resolved trade outcome cannot leak into a decision made before its label span ended.

create or replace function refusal_ledger_visible_at(as_of timestamptz)
returns table (
    evaluation_id bigint,
    decision_time_utc timestamptz,
    symbol text,
    adx double precision,
    volatility_percentile double precision,
    session text,
    minutes_into_session double precision,
    sleeve text,
    sleeve_confidence double precision,
    spread_percentile double precision,
    calendar_proximity_minutes double precision,
    cot_crowding double precision,
    bias_verdict text,
    barrier_outcome text,
    r_multiple double precision,
    label_span_end timestamptz
)
language sql
stable
as $$
    select
        e.id,
        e.ts_utc,
        e.symbol,
        nullif(e.regime ->> 'trend_strength', '')::double precision,
        nullif(e.regime ->> 'volatility_percentile', '')::double precision,
        e.regime ->> 'session',
        nullif(coalesce(e.regime ->> 'minutes_into_session', e.votes ->> 'minutes_into_session'), '')
            ::double precision,
        e.votes ->> 'selected_sleeve',
        nullif(e.votes ->> 'confidence_before_crowding', '')::double precision,
        nullif(
            coalesce(e.votes ->> 'spread_percentile', e.votes -> 'spread' ->> 'percentile'),
            ''
        )::double precision,
        nullif(
            coalesce(
                e.votes ->> 'calendar_proximity_minutes',
                e.votes -> 'calendar' ->> 'minutes_until_event'
            ),
            ''
        )::double precision,
        nullif(e.votes ->> 'positioning_score', '')::double precision,
        e.votes -> 'bias' ->> 'direction',
        t.barrier_touched,
        t.r_multiple,
        t.label_span_end
    from evaluations as e
    left join lateral (
        select
            trade.barrier_touched,
            trade.r_multiple,
            trade.label_span_end
        from trades as trade
        where trade.evaluation_id = e.id
          -- The outcome is not visible until the entire possible label window has passed.
          and trade.label_span_end <= as_of
        order by trade.id
        limit 1
    ) as t on true
    where e.ts_utc <= as_of
    order by e.ts_utc, e.id
$$;

comment on function refusal_ledger_visible_at(timestamptz) is
    'Point-in-time evaluation features and resolved outcomes. Trade outcomes are joined only '
    'after label_span_end so this function is safe for training and historical analysis.';

-- A named view makes the projection discoverable in database tooling. It intentionally contains
-- no outcomes: callers that need a historical answer must use the as-of function above.
create or replace view refusal_ledger as
select
    e.id as evaluation_id,
    e.ts_utc as decision_time_utc,
    e.symbol,
    nullif(e.regime ->> 'trend_strength', '')::double precision as adx,
    nullif(e.regime ->> 'volatility_percentile', '')::double precision as volatility_percentile,
    e.regime ->> 'session' as session,
    nullif(coalesce(e.regime ->> 'minutes_into_session', e.votes ->> 'minutes_into_session'), '')
        ::double precision as minutes_into_session,
    e.votes ->> 'selected_sleeve' as sleeve,
    nullif(e.votes ->> 'confidence_before_crowding', '')::double precision as sleeve_confidence,
    nullif(
        coalesce(e.votes ->> 'spread_percentile', e.votes -> 'spread' ->> 'percentile'), ''
    )::double precision as spread_percentile,
    nullif(
        coalesce(
            e.votes ->> 'calendar_proximity_minutes',
            e.votes -> 'calendar' ->> 'minutes_until_event'
        ),
        ''
    )::double precision as calendar_proximity_minutes,
    nullif(e.votes ->> 'positioning_score', '')::double precision as cot_crowding,
    e.votes -> 'bias' ->> 'direction' as bias_verdict
from evaluations as e;

comment on view refusal_ledger is
    'Current raw refusal features only. Use refusal_ledger_visible_at(as_of) for any result '
    'that includes realised outcomes or is used in a point-in-time analysis.';
