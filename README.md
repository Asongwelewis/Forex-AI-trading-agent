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

## The dashboard

The one long-lived process, and the only thing here with a container. It is a **read-only
window** on the store: our bars with our own overlays on the left, one entry per consensus
evaluation — including the ones that fired nothing — on the right. See
[ADR-004](docs/ADR-004-dashboard.md).

```bash
uv sync --extra dashboard
uv run python -m fxagent.dashboard            # http://localhost:8080

docker compose -f docker-compose.dashboard.yml up -d --build
```

It needs `SUPABASE_DB_URL` and nothing else. `FX_DASHBOARD_PORT` (8080) and
`FX_DASHBOARD_REFRESH_SECONDS` (15) are the only other knobs.

| Route | |
|---|---|
| `/` | the panel |
| `/ws?symbol=EURUSD&timeframe=H1` | a snapshot on connect, then one per change |
| `/api/snapshot?symbol=EURUSD` | the same object, for curl |
| `/api/options` | the (symbol, timeframe, source) triples that have bars |
| `/api/health` | store reachable, chart library present, who is watching what |

**It binds `0.0.0.0` and has no authentication.** That is deliberate and only safe because
every route is a GET and nothing in `fxagent/dashboard/` can write — asserted by
`tests/dashboard/test_app.py::test_no_route_can_change_anything`, which is the actual security
boundary. On a public host, put it behind a firewall or an authenticating proxy. Do not add a
write route without adding authentication in the same change.

The charting library is vendored (Lightweight Charts v4.2.3, Apache 2.0) and checked against a
pinned SHA-256. To refresh it after changing the pin:

```bash
uv run python scripts/vendor_lightweight_charts.py          # fetch
uv run python scripts/vendor_lightweight_charts.py --check  # verify what is committed
```

### Local development

```bash
uv run pytest
uv run ruff check fxagent tests
docker compose -f docker-compose.test.yml up -d   # pgvector, for the store tests
```
