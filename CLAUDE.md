# FX Regime Agent

A forex market-analysis agent. Three uncorrelated strategies sit behind a session-aware regime
router, produce scored suggestions across 10–12 pairs, and execute automatically on a **demo
account only**. Three LLM agents narrate the reasoning but never touch the decision path.

Phased build steps and copy-paste prompts live in the Notion board. Architecture and requirements
live on the Notion main page.

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

## Architecture — three services

| Service | Runs | Job |
|---|---|---|
| `collector` | Oracle ARM, always on | Pulls bars and events → Supabase. No indicators, no LLM. |
| `analyst` | Oracle ARM, always on | Deterministic core → agents → dashboard + Telegram |
| `executor` | Local Windows + MT5 | Places orders on the Exness demo |

A power cut at home pauses execution only. The data record stays complete, because analysis can
always be re-run over stored data while a missed collection window is gone forever.

## Stack

- Python 3.12 (`uv`, never pip into system Python). Package is `fxagent/`, **not** `src/`.
- **Data:** OANDA v20 REST (practice) — primary. No terminal, runs on ARM Linux.
- **Execution:** MetaTrader 5 + Exness demo (hedging, symbol suffix `m`).
- **Store:** Supabase Postgres + pgvector. Not SQLite — three services can't share a file.
- Indicators are hand-written in `fxagent/indicators/`. No TA library (`pandas-ta` was dropped
  after its archival warning).
- pandas 3.0.5, numpy, Pydantic v2, APScheduler, vectorbt, pytest, ruff (line length 100).
- **Chart UI:** TradingView Lightweight Charts (Apache 2.0). Advanced Charts is company-licence
  only and must not be used.
- Docker Compose, three services, deployed to Oracle Always Free (ARM — verify ARM wheels).
- LLM routing via LiteLLM. Hard daily call cap of 50, enforced in code.

## Project structure

```
fxagent/
  adapters/      BrokerAdapter protocol, OandaAdapter, MT5LocalAdapter, MockAdapter
  indicators/    EMA, ATR, ADX, rolling z-score, rolling percentile — hand-written
  patterns/      Candle formation detection. CONTEXT ONLY — must not reach consensus.py
  strategies/    session_breakout, range_reversion, carry_divergence
  regime/        sessions (zoneinfo), classifier, router, consensus
  risk/          position sizing, exposure caps
  permission/    grant state machine, auto-revoke triggers
  fundamentals/  Forex Factory calendar, central bank RSS, CFTC COT
  memory/        window encoding, pgvector index, point-in-time retrieval
  agents/        chartist (Groq), historian (Gemini), risk_officer (NVIDIA NIM)
  store/         Supabase repositories and migrations
  collector/     standalone always-on data service
  backtest/      purged walk-forward harness
  dashboard/     FastAPI + Lightweight Charts
tests/           mirrors fxagent/
```

## The three agents

| Agent | Provider | Reads | Never |
|---|---|---|---|
| Chartist | Groq | Structured core output | Pixels. Raw price series. |
| Historian | Gemini Flash | pgvector analogues + past trades | Anything resolving after the current bar |
| Risk officer | NVIDIA NIM · `deepseek-ai/deepseek-v4-pro` | The computed execution plan | Choose size, stop, or target |

`proceed_recommendation` from the risk officer is **advisory only**. It is displayed and logged.
It does not gate execution — the deterministic permission layer does, and nothing else.

## Commands

```bash
uv run pytest                          # full suite
uv run ruff check fxagent tests        # lint
uv run ruff format fxagent tests       # format
uv run python -m fxagent.main --dry-run
uv run python -m fxagent.backtest --symbol EUR_USD --from 2024-01-01 --to 2025-12-31
docker compose up                      # all three services locally
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
| `session_breakout` | Intraday | London open, ADX > 25 |
| `range_reversion` | Intraday | ADX < 20, outside the overlap |
| `carry_divergence` | Multi-day | Macro trend, rate differential |

A signal fires only when **≥2 of 3 agree AND the router permits that strategy now.** Log every
disagreement — it is training data, not noise.

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

## Style

- Small pure functions over classes with mutable state. No module-level singletons — they break
  when services split.
- Type hints everywhere. `from __future__ import annotations` at the top of every module.
- Tests for every strategy and every risk calculation. Untested risk code is a defect. Every
  strategy needs a test proving it does *not* fire when its gate fails.
- No print statements in `fxagent/` — use `logging`.
- When uncertain about a financial convention, ask rather than guessing.