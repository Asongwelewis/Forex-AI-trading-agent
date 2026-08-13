# Forex AI trading agent
An AI bot that analyzes the market within the charts and also suggests when i should make a trade and when i should not make a trade and possible trades on it's own when given permission

## Running it

There is nothing to deploy. Every entrypoint is short-lived and idempotent, and GitHub Actions
cron invokes them on a timer — see [ADR-002](docs/ADR-002-scheduling.md) for why, and for the
platform constraints the workflows are built around.

| Workflow | Schedule | Does |
|---|---|---|
| [`collect-and-analyse.yml`](.github/workflows/collect-and-analyse.yml) | `7 * * * *` | collector → analyst → resolver, one job. Alerts to Telegram on failure. |
| [`health.yml`](.github/workflows/health.yml) | `23 6 * * *` | Bar gaps, collector heartbeat, Twelve Data credits, repo inactivity. Telegram summary every run. |

Both also take a manual `workflow_dispatch`, which is the only way to run them before this
lands on `main` — GitHub schedules a workflow from the default branch only.

### Required repository secrets

Settings → Secrets and variables → Actions. A run refuses to start when any of these is empty
and names the one that is missing.

| Secret | Needed for |
|---|---|
| `SUPABASE_DB_URL` | Everything. Use the **Session pooler** (port 5432) — runners have no IPv6. |
| `TWELVEDATA_API_KEY` | Bar collection, and the credit budget check. |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Recommendations, failure alerts, health summaries. |

`GEMINI_API_KEY`, `GROQ_API_KEY` and `NVIDIA_API_KEY` are optional: with none of them set, the
deterministic template explanation is used and the pass completes normally.

### Local development

```bash
uv run pytest
uv run ruff check fxagent tests
docker compose -f docker-compose.test.yml up -d   # pgvector, for the store tests
```
