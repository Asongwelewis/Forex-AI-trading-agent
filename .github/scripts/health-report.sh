#!/usr/bin/env bash
#
# Daily health report. This is how we find out the hourly cron stopped firing.
#
# The main operational risk in this system is silence: GitHub disables schedules on inactive
# repositories, a scheduled trigger is best-effort and can simply not happen, and a collector
# that dies at 03:00 produces no alert of its own because nothing runs to notice. Every other
# failure mode announces itself. This one has to be gone looking for.
#
# So the checks here are deliberately independent of the pipeline they are checking:
#
#   * psql against the store, not fxagent's own engine — an import error in the package must
#     not take the health check down with it.
#   * curl against Twelve Data, not the collector's CreditLedger — that ledger is in-process
#     and dies with the run, so it cannot answer "how much have we spent today".
#   * Telegram over plain HTTP from the workflow, not through any fxagent module.
#
# Checks whose dependencies have not landed yet report UNAVAILABLE and do not fail the run.
# A check that cannot run is not the same as a check that failed, and conflating them trains
# everyone to ignore the alert.
#
# Exit codes: 0 all clear (warnings allowed), 1 something is wrong and needs a human.

set -uo pipefail

SYMBOLS=${HEALTH_SYMBOLS:-EURUSD,GBPUSD,EURGBP}
SOURCE=${HEALTH_SOURCE:-twelvedata}
TIMEFRAME=${HEALTH_TIMEFRAME:-H1}
WINDOW_HOURS=${HEALTH_WINDOW_HOURS:-24}
# Free tier, verified against Twelve Data's pricing page — the same figure as
# FREE_TIER_DAILY_CREDITS in fxagent/adapters/credits.py. Only used when /api_usage does not
# report a plan limit of its own. Must be positive; it is a divisor.
CREDIT_BUDGET=${TWELVEDATA_DAILY_BUDGET:-800}
[[ ${CREDIT_BUDGET} =~ ^[1-9][0-9]*$ ]] || CREDIT_BUDGET=800
CREDIT_WARN_PCT=${TWELVEDATA_WARN_PERCENT:-80}
INACTIVITY_WARN_DAYS=${INACTIVITY_WARN_DAYS:-45}
SUMMARY_FILE=${SUMMARY_FILE:-health-summary.txt}
SQL_FILE=${SQL_FILE:-.github/scripts/bar-gaps.sql}

export PGCONNECT_TIMEOUT=${PGCONNECT_TIMEOUT:-15}

lines=()
worst=0 # 0 ok, 1 warn, 2 fail

note() { lines+=("$1"); }

record() {
  # record <severity 0|1|2> <text>
  [ "$1" -gt "$worst" ] && worst=$1
  note "$2"
}

psql_q() {
  # One-shot query, tuples only, unaligned. Errors go to stderr and yield an empty result;
  # every caller checks for that rather than trusting the string.
  psql "${SUPABASE_DB_URL}" --no-psqlrc --quiet --tuples-only --no-align \
    --field-separator='|' --set=ON_ERROR_STOP=on --command="$1" 2>/tmp/psql-err.txt
}

# =========================================================================================
# 1. Store reachable
# =========================================================================================
if ! reach=$(psql_q "select 1"); then
  record 2 "❌ Store: UNREACHABLE — $(cut -c1-300 </tmp/psql-err.txt | tr '\n' ' ')"
  # Everything downstream needs the database. Report what we have and stop here rather than
  # emitting four more identical connection errors.
  note ""
  note "Every store-backed check was skipped because the connection failed."
  printf '%s\n' "FX agent health — FAILED" "" "${lines[@]}" >"${SUMMARY_FILE}"
  cat "${SUMMARY_FILE}"
  exit 1
fi
[ "${reach}" = "1" ] && note "✅ Store: reachable"

# =========================================================================================
# 2. Bar coverage over the last 24h, excluding hours the FX market was shut
# =========================================================================================
if ! gap_rows=$(psql "${SUPABASE_DB_URL}" --no-psqlrc --quiet --tuples-only --no-align \
  --field-separator='|' \
  --variable=symbols="${SYMBOLS}" \
  --variable=source="${SOURCE}" \
  --variable=timeframe="${TIMEFRAME}" \
  --variable=hours="${WINDOW_HOURS}" \
  --file="${SQL_FILE}" 2>/tmp/psql-err.txt); then
  record 2 "❌ Bar coverage: query failed — $(cut -c1-300 </tmp/psql-err.txt | tr '\n' ' ')"
else
  open_hours=""
  gap_detail=()
  while IFS='|' read -r kind subject value detail; do
    case "${kind}" in
      window)
        note "🕒 Window: ${value} → ${detail} UTC (source=${subject})"
        ;;
      open_hours)
        open_hours=${value}
        ;;
      symbol)
        if [ "${value}" -eq 0 ]; then
          gap_detail+=("   ${subject}: complete (${detail}/${detail})")
        else
          # A gap in a market-open hour is a real hole in the series. Every indicator
          # downstream reads it as a price that did not move.
          record 2 "❌ Bar gaps: ${subject} is missing ${value} of ${detail} expected ${TIMEFRAME} bars"
        fi
        ;;
      latest)
        if [ "${value}" = "never" ]; then
          record 2 "❌ ${subject}: no ${TIMEFRAME} bars stored at all for source=${SOURCE}"
        else
          gap_detail+=("   ${subject}: newest ${value} (${detail} min old)")
        fi
        ;;
    esac
  done <<<"${gap_rows}"

  if [ "${open_hours:-0}" -eq 0 ]; then
    note "😴 Bar coverage: market was shut for the whole window — nothing expected"
  elif [ "${worst}" -lt 2 ]; then
    note "✅ Bar coverage: no gaps across ${open_hours} open hours"
  fi
  for line in "${gap_detail[@]:-}"; do
    [ -n "${line}" ] && note "${line}"
  done
fi

# =========================================================================================
# 3. Collector heartbeat
#
# to_regclass first: the table arrives with the collector's own migration, so on a store that
# predates it a direct query would raise "relation does not exist" and read as an outage.
# =========================================================================================
has_heartbeats=$(psql_q "select to_regclass('public.service_heartbeats') is not null")
if [ "${has_heartbeats}" != "t" ]; then
  note "➖ Heartbeat: UNAVAILABLE — service_heartbeats has not been migrated in yet"
else
  beat=$(psql_q "select service
                      || '|' || floor(extract(epoch from now() - last_beat_utc) / 60)::text
                      || '|' || floor(extract(epoch from now() - started_at_utc) / 60)::text
                 from service_heartbeats where service = 'collector'")
  if [ -z "${beat}" ]; then
    record 2 "❌ Heartbeat: the collector has never recorded one"
  else
    IFS='|' read -r _svc beat_age uptime <<<"${beat}"
    # The hourly cron beats once an hour and scheduled triggers can be delayed 30+ minutes at
    # peak, so the threshold has to tolerate a late run without tolerating a dead one.
    if [ "${beat_age}" -gt 180 ]; then
      record 2 "❌ Heartbeat: last beat ${beat_age} min ago — the hourly cron has stopped firing"
    elif [ "${beat_age}" -gt 100 ]; then
      record 1 "⚠️ Heartbeat: last beat ${beat_age} min ago — a run was delayed or skipped"
    else
      note "✅ Heartbeat: ${beat_age} min ago"
    fi
    # A start time that keeps moving means crash-looping, which a recent beat alone hides.
    if [ "${uptime}" -lt 5 ]; then
      record 1 "⚠️ Heartbeat: process started ${uptime} min ago — check for a restart loop"
    fi
  fi
fi

# =========================================================================================
# 4. Twelve Data credit budget
#
# Their own /api_usage endpoint, because the in-process CreditLedger cannot survive a
# short-lived run. Field names are read defensively: an unrecognised response shape is
# reported as UNAVAILABLE rather than silently scored as zero usage.
# =========================================================================================
if [ -z "${TWELVEDATA_API_KEY:-}" ]; then
  note "➖ Credits: UNAVAILABLE — TWELVEDATA_API_KEY is not set"
else
  usage_json=$(curl --silent --show-error --max-time 20 --retry 2 \
    "https://api.twelvedata.com/api_usage?apikey=${TWELVEDATA_API_KEY}" 2>/dev/null)
  used=$(printf '%s' "${usage_json}" | jq -r '.current_usage // empty' 2>/dev/null)

  if ! [[ ${used} =~ ^[0-9]+$ ]]; then
    # Never echo usage_json — it is a response to a URL carrying the API key, and some
    # error payloads reflect the request back.
    record 1 "⚠️ Credits: UNAVAILABLE — /api_usage returned no numeric current_usage field"
  else
    reported_limit=$(printf '%s' "${usage_json}" | jq -r '.plan_limit // empty' 2>/dev/null)
    budget=${CREDIT_BUDGET}
    [[ ${reported_limit} =~ ^[1-9][0-9]*$ ]] && budget=${reported_limit}
    pct=$((used * 100 / budget))
    if [ "${pct}" -ge 100 ]; then
      record 2 "❌ Credits: ${used}/${budget} used (${pct}%) — the quota is exhausted, collection has stopped"
    elif [ "${pct}" -ge "${CREDIT_WARN_PCT}" ]; then
      record 1 "⚠️ Credits: ${used}/${budget} used (${pct}%)"
    else
      note "✅ Credits: ${used}/${budget} used (${pct}%)"
    fi
  fi
fi

# =========================================================================================
# 5. Repository inactivity
#
# GitHub disables scheduled workflows on a repository with no commit activity for 60 days.
# Sources disagree on whether private repositories are included, so this warns either way —
# the cost of being wrong is the entire pipeline stopping with no notification at all.
# A workflow *run* does not reset the clock; only a commit does.
# =========================================================================================
if [[ ${DAYS_SINCE_LAST_COMMIT:-} =~ ^[0-9]+$ ]]; then
  if [ "${DAYS_SINCE_LAST_COMMIT}" -ge "${INACTIVITY_WARN_DAYS}" ]; then
    record 1 "⚠️ Inactivity: ${DAYS_SINCE_LAST_COMMIT} days since the last commit. GitHub disables schedules at 60 — push something to reset it."
  else
    note "✅ Activity: last commit ${DAYS_SINCE_LAST_COMMIT} days ago"
  fi
else
  note "➖ Activity: UNAVAILABLE — commit date was not supplied"
fi

# =========================================================================================
# Report
# =========================================================================================
case "${worst}" in
  0) headline="FX agent health — ALL CLEAR" ;;
  1) headline="FX agent health — DEGRADED" ;;
  *) headline="FX agent health — FAILED" ;;
esac

printf '%s\n' "${headline}" "" "${lines[@]}" >"${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"

[ "${worst}" -ge 2 ] && exit 1
exit 0
