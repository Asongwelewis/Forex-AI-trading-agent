#!/usr/bin/env bash
#
# Preflight: fail the run when a required secret is empty, and say which one.
#
# GitHub substitutes an unset secret with the empty string rather than erroring, so without
# this the job gets all the way to a connection attempt and dies with something like
# "database URL has no scheme" — a message that describes the symptom and not the cause.
# Twenty minutes of the wrong debugging follow. Name the missing secret up front instead.
#
# No secret value is ever printed, compared against a literal, or written to a file. The only
# facts this emits about a value are whether it is empty and, for the database URL, which
# scheme it starts with.
#
# Usage (names are space-separated, and are read from this process's own environment):
#
#   REQUIRED_SECRETS="SUPABASE_DB_URL TELEGRAM_BOT_TOKEN" \
#   OPTIONAL_SECRETS="GEMINI_API_KEY" \
#     bash .github/scripts/check-secrets.sh

set -uo pipefail

required=${REQUIRED_SECRETS:-}
optional=${OPTIONAL_SECRETS:-}

missing=()
present=()
degraded=()

is_blank() {
  # Indirect expansion: $1 is the *name* of the variable to inspect, never its value.
  local value="${!1-}"
  [ -z "${value//[[:space:]]/}" ]
}

for name in $required; do
  if is_blank "$name"; then
    missing+=("$name")
  else
    present+=("$name")
  fi
done

for name in $optional; do
  if is_blank "$name"; then
    degraded+=("$name")
  else
    present+=("$name")
  fi
done

echo "Secret preflight"
echo "----------------"
for name in "${present[@]:-}"; do
  [ -n "$name" ] && echo "  set        $name"
done
for name in "${degraded[@]:-}"; do
  [ -n "$name" ] && echo "  optional   $name (unset — the feature it powers degrades, the run continues)"
done
for name in "${missing[@]:-}"; do
  [ -n "$name" ] && echo "  MISSING    $name"
done
echo

# ---------------------------------------------------------------------------------------
# Shape check on the database URL. This is trap #1 in fxagent/store/config.py: SUPABASE_URL
# is the PostgREST endpoint (https://<ref>.supabase.co) and no Postgres driver can dial it,
# but it looks enough like a connection string to get pasted into the wrong secret. Catching
# it here costs one string comparison; missing it costs a confusing failure every hour.
# ---------------------------------------------------------------------------------------
if [ -n "${SUPABASE_DB_URL:-}" ]; then
  case "${SUPABASE_DB_URL}" in
    postgresql://* | postgres://* | postgresql+asyncpg://*) ;;
    http://* | https://*)
      echo "::error title=SUPABASE_DB_URL is not a database URL::It starts with http(s):// which means the PostgREST endpoint (SUPABASE_URL) was pasted in by mistake. The connection string lives under Supabase > Settings > Database > Connection string > URI, and starts with postgresql://. Prefer the Session pooler on port 5432 — GitHub-hosted runners have no IPv6, and Supabase's direct host is IPv6-only."
      missing+=("SUPABASE_DB_URL")
      ;;
    *)
      echo "::error title=SUPABASE_DB_URL has an unrecognised scheme::Expected a postgresql:// URL. See .env.example under STORE."
      missing+=("SUPABASE_DB_URL")
      ;;
  esac
fi

if [ "${#missing[@]}" -gt 0 ]; then
  list=$(printf '%s, ' "${missing[@]}")
  list=${list%, }
  echo "::error title=Required secrets are missing::${list}. Set them under Settings > Secrets and variables > Actions, then re-run. Nothing was executed."
  exit 1
fi

echo "All required secrets are present."
