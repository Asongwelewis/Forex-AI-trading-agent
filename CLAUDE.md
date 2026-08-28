# FX Regime Agent

A forex market-analysis agent. Three uncorrelated strategies sit behind a session-aware regime
router, which selects **one** of them per bar, sizes the trade deterministically, and executes on
a **demo account only**. LLM agents narrate the reasoning but never touch the decision path.

Phased build steps live on the Notion board; the current, authoritative card set is
`docs/BOARD.md` in this repo. Architecture decisions live in `docs/ADR-*.md`.

## Two standing gates — check these before proposing work

> **GATE A — no code path places an order until `fxagent/permission/` is complete.**
> The grant state machine, the auto-revoke triggers and the per-order pre-flight are Lane 3 of
> `docs/BOARD.md`. `MT5LocalAdapter.place_order` works and is deliberately called by nothing.
>
> **GATE B — no new strategy or feature work until costs are measured on Exness bars.**
> The only result this project has is 217 trades over 2024–25 with an expectancy interval of
> **[-0.15, +0.15]R**, and it assumed a flat 1-pip spread on TwelveData bars while fills would
> occur on Exness. That is not "no edge" — it is "no measurement." Lane 2 of `docs/BOARD.md`.

**The agent runs only while the MT5 terminal is open on the Windows desktop.** This is an
accepted constraint, not a defect: it rules out a paid always-on host. It is also what makes MT5
both the feed and the venue — see `docs/ADR-005-single-process.md`.

## Hard rules — YOU MUST follow these

1. **Demo accounts only.** Startup asserts `account_info().trade_mode` is DEMO and pins the login.
   A real-money account attached to this process is a fatal error, not a warning. Re-assert
   before every order, not just at startup.
2. **Never commit secrets.** `.env`, `*.key`, Supabase service keys and API tokens are never
   staged, never logged, never pasted into a commit message.
3. **Every order carries server-side SL and TP**, attached in the same request at placement time.
   Never a naked order with a stop set afterwards.
4. **The LLM never decides anything.** Indicators, signals, position sizes, stops and order types
   are deterministic Python. Agents explain, score against fixed rules, and narrate. An agent
   that can move a stop is an agent that can empty an account.
5. **All LLM output is Pydantic-validated.** Any response failing validation is discarded
   entirely and the system holds its previous state. No partial parsing, no regex repair, no
   retry-until-it-parses. An agent must never output a number that wasn't in its input.
6. **Point-in-time correctness everywhere.** Events filter on publication time, never event time.
   Historical analogues must have resolved before the current bar. Enforce in SQL, not in
   prompts. Every retrieval path needs a look-ahead test.
7. **Backtests model spread, swap and slippage**, and use purged + embargoed walk-forward folds.
   A backtest on mid-price, or with unpurged overlapping labels, is invalid and must not be
   reported as a result.
8. **Risk caps are absolute:** ≤0.5% of equity risked per trade, ≤2% total open risk. Size in
   risk units derived from stop distance, never fixed lots. Round volume down to lot step.
9. **No martingale, ever.** Size modulation scales *down* in unfavourable conditions and never
   up after a loss. "Recovery trading" is not a feature of this system.
10. **vibe-trading-ai is reference only, never imported.** It lives in a separate sandbox at
    `../vibe-sandbox`. Learning from their design is encouraged; coupling to their release cycle
    is not.

## Architecture — one local process, plus two things that outlive it

| Runs | Where | Job |
|---|---|---|
| `fxagent.trader` | Local Windows, MT5 open | The whole pipeline: collect → classify → route → select → size → journal → advise or execute |
| `fxagent.resolve` | Same process, or standalone | Closes out past decisions against bars that have since arrived |
| GitHub Actions cron | GitHub-hosted | Only what must run with the desktop off: calendar, COT, statistical observations, health |
| `fxagent.dashboard` | Optional container | Read-only window on the store. Cannot write; asserted by test. |

There is **no always-on service and no VPS.** The Oracle ARM three-service split was designed
for a deployment that was never bought and has been retired — see `docs/ADR-002-scheduling.md`
for the cron decision and `docs/ADR-005-single-process.md` for the collapse to one local process.

The tradeoff is explicit: a missed collection window is gone forever, and the desktop being off
is a missed window. Uptime is recorded so those gaps are visible rather than silently absent.

## Stack

- Python 3.12 (`uv`, never pip into system Python). Package is `fxagent/`, **not** `src/`.
- **Data and execution are the same venue:** MetaTrader 5 + Exness demo (hedging, symbol suffix
  `m`). One book, so the backtest and the live run agree about what a bar is.
- **Twelve Data** (`fxagent/adapters/twelvedata.py`) is a gap-filler and cross-check only. It is
  never the source a fill is modelled against — it quotes a different book, and it carries no
  bid/ask. Its credit budget lives in `adapters/credits.py`.
- **Store:** Supabase Postgres + pgvector. Not SQLite — the dashboard and the trader are separate
  processes and Actions runs on a different machine entirely.
- Indicators are hand-written in `fxagent/indicators/`. No TA library (`pandas-ta` was dropped
  after its archival warning).
- pandas `>=2.3.2,<3` (**not** 3.x — see the pin comment in `pyproject.toml`), numpy, Pydantic v2,
  SQLAlchemy async + asyncpg, pytest, ruff (line length 100). No APScheduler, no vectorbt.
- **Chart UI:** TradingView Lightweight Charts (Apache 2.0), vendored and SHA-pinned. Advanced
  Charts is company-licence only and must not be used.
- LLM routing via LiteLLM, an optional extra. Hard daily call cap of 50, enforced in code.
  `fxagent.agents` imports without LiteLLM installed and degrades to deterministic templates.

## Project structure

```
fxagent/
  costs.py       Spread, slippage, swap. TOP LEVEL so backtest and resolver cannot fork.
  spreadwatch.py Local Exness bid/ask sampler. Measurement only, never imported by analysis.
  observations.py Raw BLS/Eurostat prints. Deliberately NOT in fundamentals/ — import graph.
  adapters/      BrokerAdapter protocol, MT5LocalAdapter, TwelveDataAdapter, MockAdapter
  indicators/    EMA, ATR, ADX, rolling z-score, rolling percentile — hand-written
  patterns/      Candle formation detection. CONTEXT ONLY — must not reach selection.py
  strategies/    session_breakout, range_reversion, carry_divergence
  regime/        sessions (zoneinfo), classifier, router, bias, selection
  risk/          position sizing, exposure caps, symbol specs
  permission/    grant state machine, auto-revoke triggers  ← GATE A: incomplete
  fundamentals/  Forex Factory calendar, central bank RSS, CFTC COT
  memory/        window encoding spec only — no encoder, no index, no retrieval yet
  agents/        chartist (Groq), historian (Gemini), risk_officer (NVIDIA NIM)
  stats/         returns, performance estimates with intervals, bootstrap resampling
  store/         Supabase repositories and migrations
  collector/     data-only ingest service; its import graph is asserted by test
  backtest/      replay through the live pipeline, purged walk-forward, bootstrap report
  dashboard/     FastAPI + Lightweight Charts, read-only
tests/           mirrors fxagent/
```

## The three agents

| Agent | Provider | Reads | Never |
|---|---|---|---|
| Chartist | Groq | Structured core output | Pixels. Raw price series. |
| Historian | Gemini Flash | pgvector analogues + past trades | Anything resolving after the current bar |
| Risk officer | NVIDIA NIM · `nvidia/nemotron-3-ultra-550b-a55b` | The computed execution plan | Choose size, stop, or target |

`proceed_recommendation` from the risk officer is **advisory only**. It is displayed and logged.
It does not gate execution — the deterministic permission layer does, and nothing else.

The historian currently narrates an empty `analogues` list, because `fxagent/memory/` is a
128-dimension encoding spec with no encoder, no index and no retrieval query. Do not describe
retrieval as working until it is.

## Commands

```bash
uv run pytest                          # full suite
uv run pytest -m "not db"              # no-container subset
uv run ruff check fxagent tests        # lint
uv run ruff format fxagent tests       # format

uv run --extra mt5 python -m fxagent.trader --dry-run     # one full cycle, no side effects
uv run --extra mt5 python -m fxagent.spreadwatch --symbols EURUSD,GBPUSD
uv run python -m fxagent.collector --symbols EURUSD --timeframes H1
uv run python -m fxagent.backtest --symbol EUR/USD --from 2024-01-01 --to 2025-12-31
uv run python -m fxagent.dashboard                        # http://localhost:8080
```

## Git workflow

Two long-lived branches with distinct meanings:

- `develop` — phase work landed. Tests green, ruff clean, code complete.
- `main` — **verified against a live broker.** Ran for real and behaved.

Phase 3 earned this split: 104 tests passed against a fake MT5 module while the offset detection
was entirely unverified.

- One branch per phase: `phase/NN-short-name` → merged into `develop`.
- `develop` → `main` only after a live smoke test or paper run confirms behaviour.
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`.
- Before any merge: full test suite, then diff summary, then wait for explicit approval.
- Merge with `--no-ff` so phase boundaries stay visible.

## Domain facts — verified, not assumed

**Exness server time is UTC+0.** Measured 08 Aug 2026 by comparing MT5 Market Watch (20:49)
against a local UTC+1 clock (21:49). This is unusual — most brokers run GMT+2/+3.

**MT5 does not return UTC.** `copy_rates_*` reports the broker clock encoded as a Unix timestamp.
Converting naively produces timestamps correctly labelled UTC and silently wrong.

**`mt5.initialize()` succeeds with Algo Trading off.** Only `order_send` fails later. A separate
`terminal_info().trade_allowed` check is required on the connect path.

**Account:** login 476187411, `Exness-MT5Trial9`, hedging, leverage 1:200, suffix `m`.
Crypto CFDs (BTCUSDm) quote 24/7 — use them for the offset heartbeat.

**Sessions use `zoneinfo`, not fixed UTC constants.** London is 08:00–17:00 UTC in winter and
07:00–16:00 in summer. Define boundaries in `Europe/London`, `America/New_York` and `Asia/Tokyo`,
then convert at evaluation time. Forex runs Sunday 21:00 UTC to Friday 21:00 UTC.

| Strategy | Timeframe | Regime gate |
|---|---|---|
| `session_breakout` | H1 | London open, ADX > 25 |
| `range_reversion` | Intraday, any | ADX < 20, outside the overlap |
| `carry_divergence` | D1 | Macro trend, rate differential |

**The router selects one sleeve. Strategies do not vote.** The old rule — "≥2 of 3 agree AND the
router permits" — was unsatisfiable by construction and is gone. `session_breakout` gates on
ADX > 25 and `range_reversion` on ADX < 20, so the second vote could never arrive; over 12,341
decisions in the 2024–25 replay the router gave positive weight to two strategies on **zero**
bars and the system took **zero** trades. The full autopsy is the module docstring of
`fxagent/regime/selection.py`. Do not reintroduce an agreement count.

What replaces the lost redundancy: confirmation moved *inside* each strategy, where the evidence
families are genuinely independent, and `regime/bias` filters an intraday signal that opposes the
daily view (it never originates one).

**The rejection ledger is non-negotiable.** Every strategy in the router's slate gets a line on
every bar, including the silent and gated ones, with the reason recorded on the rejection path
exactly as on the firing path. That ledger is what found the failure above — the trade log could
not have. It is training data, not noise.

## Success metrics

Win rate is **not** a success metric and must never gate anything. `session_breakout` targets 2R,
so its breakeven win rate is 33% and 40% is strongly profitable. Judge on **expectancy in R**,
profit factor and max drawdown — and return `INSUFFICIENT_DATA` below 100 trades per strategy.

## Known traps

- **Look-ahead bias.** In LLM prompts (never include dates in backtest mode), in COT data
  (Tuesday reference, Friday publication — key on publication), and in cross-validation folds
  (purge and embargo).
- **Overfitting.** Tune the *rules*, not the parameters. If out-of-sample is far worse than
  in-sample, that is the honest answer.
- **Demo fills are optimistic.** Exness is a market-maker CFD broker; spreads widen on news.
- **Weekend gaps.** Grants expire and force flat before Friday close.
- **`symbol_info().filling_mode` is a bitmask**, not an enum value.
- **Candlestick patterns have weak evidence** — two studies found no net positive return on
  EUR/USD after costs. They are UI context, never signal inputs.
- **Feed and venue must be the same book.** A backtest on one source and fills on another is a
  different experiment wearing the same name. `bars.source` is part of a bar's identity; a replay
  and a live run reading different sources are not comparable and must not be reported as if
  they were.
- **A green suite over disconnected parts stays green.** The test suite covers components. It
  cannot see that a workflow names a module which does not exist, or that nothing calls the
  code path being protected. Assembly is not tested by unit tests — wire an integration test
  through the real entrypoint.
- **The Forex Factory feed is current-week only.** `ff_calendar_nextweek.json` is a 404, so on a
  Friday the lookahead cannot see Sunday's open. Absence of an event is not evidence of no
  event: fail **closed** and refuse.

## Style

- Small pure functions over classes with mutable state. No module-level singletons — they break
  when services split.
- Type hints everywhere. `from __future__ import annotations` at the top of every module.
- Tests for every strategy and every risk calculation. Untested risk code is a defect. Every
  strategy needs a test proving it does *not* fire when its gate fails.
- No print statements in `fxagent/` — use `logging`.
- When uncertain about a financial convention, ask rather than guessing.