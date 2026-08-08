# FX Regime Agent

A forex market-analysis agent that runs three uncorrelated strategies behind a session-aware
regime router, produces scored buy/sell suggestions, and executes only under an explicit,
expiring permission grant. Built **on top of** `vibe-trading-ai` (MIT), not forked from it.

Detailed phased build steps and prompts: `docs/BUILD_PLAN.md` — read it when starting a new phase.

## Hard rules — YOU MUST follow these

1. **Demo accounts only.** Never connect to, place orders on, or configure a live-funded
   trading account. If a task seems to require live credentials, stop and ask.
2. **Never commit secrets.** `.env`, `.env.local`, `*.key`, and `~/.vibe-trading/.env` are
   never staged, never printed to logs, never pasted into a commit message or PR body.
3. **Every order carries server-side SL and TP** attached at placement time, in the same
   request. Never place a naked order and set the stop afterwards.
4. **The LLM never computes indicators or math.** Indicators, signals, and position sizes are
   deterministic Python. The LLM only synthesizes, explains, and scores against fixed rules.
5. **All LLM output is Pydantic-validated.** Any response failing schema validation is rejected
   entirely and the system holds its previous state (no new position). No partial parsing,
   no regex repair, no retry-until-it-parses loops.
6. **Never edit vendor code.** `vibe-trading-ai` is a dependency. Extensions go in our own
   package or in `~/.vibe-trading/skills/`. If upstream needs a fix, open a PR upstream.
7. **Backtests must model spread, swap/rollover, and slippage.** A backtest on mid-price is
   invalid and must not be reported as a result.
8. **Risk caps are absolute:** ≤0.5% of equity risked per trade, ≤2% total open risk. Size
   positions in risk units derived from stop distance, never in fixed lots.

## Stack

- Python 3.12 (`uv` for env management, NOT pip directly into system Python). Not 3.11 —
  `pandas-ta` dropped it. Not 3.13+ — several deps still lag.
- `vibe-trading-ai` — broker connectors, backtest engines, data loaders, mandate/kill-switch
- MetaTrader 5 + Exness demo — execution and price data
- pandas, pandas-ta, numpy — deterministic strategy layer
- Pydantic v2 — every boundary contract
- SQLite — signal journal
- pytest — tests
- LLM: Google Gemini free tier (primary), Groq (fallback). No paid API keys in this project.

## Project structure

Our package is `fxagent/`, **not** `src/`. `vibe-trading-ai` installs unnamespaced top-level
`src`, `cli`, and `backtest` packages into site-packages, so `src/` would collide with it the
moment the two share an environment. Never reintroduce a top-level `src/` here.

- `fxagent/adapters/` — `BrokerAdapter` protocol + `MT5LocalAdapter`, `MetaApiAdapter`
- `fxagent/strategies/` — the three strategies, each a `Strategy` subclass returning a `Signal`
- `fxagent/regime/` — session clock, volatility/trend classification, strategy weighting
- `fxagent/risk/` — position sizing, exposure caps
- `fxagent/permission/` — grant state machine, auto-revoke triggers
- `fxagent/journal/` — SQLite signal + outcome logging
- `fxagent/llm/` — provider gateway, prompt templates, Pydantic response schemas
- `tests/` — mirrors `fxagent/` structure
- `docs/` — BUILD_PLAN.md, PHASE1_FINDINGS.md, and design notes

## Commands

```bash
uv run pytest                      # full test suite
uv run pytest tests/strategies -v  # scoped
uv run ruff check fxagent tests    # lint
uv run ruff format fxagent tests   # format
uv run python -m fxagent.main --dry-run  # one analysis cycle, no orders

# vibe-trading-ai lives in an isolated sandbox at ../vibe-sandbox, never in this venv.
s:\vibe-sandbox\.venv\Scripts\vibe-trading.exe provider doctor   # diagnose LLM provider
s:\vibe-sandbox\.venv\Scripts\vibe-trading.exe connector check   # verify broker connection
```

## Git workflow

- `develop` is the integration branch. Every phase branch is cut from `develop` and merged
  back into `develop`. **Phase branches never merge into `main`.**
- `main` receives exactly one merge from `develop`, at the end of the build. Until then it
  is deliberately behind and is not expected to run.
- Never commit directly to `main`. Both `develop` and `main` stay green.
- One branch per phase: `phase/NN-short-name` (e.g. `phase/04-strategies`), cut from `develop`.
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`.
- Before any merge: run the full test suite, then show me the diff summary and wait for my
  approval. Do not merge on your own initiative.
- Merge with `--no-ff` so phase boundaries stay visible in history.

## Domain facts

Local timezone is **UTC+1** (Yaoundé, WAT). Forex runs Sunday 21:00 UTC to Friday 21:00 UTC.

| Session | UTC | Local |
|---|---|---|
| Tokyo | 00:00–09:00 | 01:00–10:00 |
| London | 08:00–17:00 | 09:00–18:00 |
| New York | 13:00–22:00 | 14:00–23:00 |
| London/NY overlap | 13:00–17:00 | 14:00–18:00 |

**All internal timestamps are UTC.** Convert to local only at display boundaries.

The three strategies and the regimes they need:

| Strategy | Timeframe | Regime gate |
|---|---|---|
| `session_breakout` | Intraday | London open, ADX > 25 |
| `range_reversion` | Intraday | ADX < 20, outside London/NY overlap |
| `carry_divergence` | Multi-day | Macro trend, rate differential |

A signal fires only when **≥2 of 3 agree AND the regime router permits that strategy today.**
Log every disagreement — it is training data, not noise.

## Known traps

- **Look-ahead bias in LLM backtests.** The model already knows what happened in historical
  windows. Never include dates in prompts during backtests; validate on post-cutoff data only.
- **Demo fills are optimistic.** Exness is a market-maker CFD broker; spreads widen sharply on
  news. Assume live execution is meaningfully worse than demo.
- **Weekend gaps.** FX gaps at Sunday open. Any permission grant must expire and force-flat
  before Friday close.
- **MetaTrader5 Python package is Windows-only** and needs the terminal running. Keep the
  `BrokerAdapter` interface clean so a cloud adapter can replace it without touching strategies.

## Style

- Prefer small, pure functions over classes with mutable state.
- Type hints everywhere. `from __future__ import annotations` at the top of every module.
- Tests for every strategy and every risk calculation. Untested risk code is a defect.
- No print statements in `fxagent/` — use the `logging` module.
- When you are uncertain about a financial convention, ask me rather than guessing.
