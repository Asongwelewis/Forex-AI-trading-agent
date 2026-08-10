# FX Regime Agent — Phased Build Plan

Companion to `CLAUDE.md`. This file is **not** loaded every session — point Claude at it
explicitly when you start a phase: `Read docs/BUILD_PLAN.md and execute Phase 4.`

---

## Honest timing expectation

You asked to finish in a day. Here is what that realistically means:

| Phases | Content | Realistic |
|---|---|---|
| 0–3 | Setup, install, broker connection | **Day 1 morning** — very achievable |
| 4–6 | Strategies, regime router, risk + permissions | **Day 1 afternoon/evening** — achievable if MT5 connects cleanly |
| 7–8 | Backtest validation, journal, alerts | **Day 2** — will spill, and should |
| 9 | LLM synthesis layer | **Day 3+** — deliberately last |

Phases 0–6 in one day is a real target. If you compress Phase 7 to fit the day, you will ship
a system whose edge is unmeasured, which is the one failure mode that matters. Let it spill.

---

## Pre-flight: gather these before you start

Claude Code cannot get these for you. Have them in a scratch file (not in the repo).

**Accounts and credentials**
- [ ] Exness demo account: login number, password, server name (e.g. `Exness-MT5Trial9`)
- [ ] Google AI Studio API key — free, no card: https://aistudio.google.com/apikey
- [ ] Groq API key (fallback) — free, no card: https://console.groq.com/keys
- [ ] ~~Finnhub API key~~ — NOT NEEDED. Dropped in the Phase 5 teardown. `/calendar/economic`
      is premium and returns HTTP 403 on a free key (verified 2026-08-10), and no code ever
      called Finnhub. The calendar comes from Forex Factory's keyless weekly feed — see Phase 6.
- [ ] Telegram bot token + your chat ID (Phase 8 — skip for now if short on time)
- [ ] GitHub account with a new **private** empty repo created

**Installed on your Windows 11 machine**
- [ ] Python 3.11 or 3.12 (NOT 3.13 — some deps lag)
- [ ] Git
- [ ] MetaTrader 5 terminal, logged into the Exness demo, **Algo Trading enabled** (toolbar button)
- [ ] Claude Code
- [ ] `uv`: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Decisions to make now**
- [ ] Which 3 pairs to start with. Recommended: `EURUSD`, `GBPUSD`, `EURGBP` — the first two for
      breakout, the third for range reversion.
- [ ] Starting demo balance. Use something realistic (e.g. $1,000), not $100,000. Large fake
      balances produce risk habits that don't survive contact with a real account.

---

## Phase 0 — Repo scaffold and memory files

**Goal:** an empty, correctly-configured repo that Claude Code understands.
**Time:** 20 min

```
Create a new Python project in this directory called fx-regime-agent.

Requirements:
- Use uv. Create pyproject.toml targeting Python 3.11+.
- Dependencies: pandas, pandas-ta, numpy, pydantic>=2, python-dotenv, apscheduler,
  pytest, pytest-cov, ruff. Add MetaTrader5 as a Windows-only optional dependency.
- Create the directory structure exactly as described in CLAUDE.md under "Project structure",
  with __init__.py in each package and a placeholder test in tests/.
- Create a .gitignore covering Python, .env, .venv, *.db, __pycache__, .pytest_cache,
  and any file matching *.key or *credentials*.
- Create .env.example with every variable named but no values filled in.
- Configure ruff in pyproject.toml: line length 100, target py311.
- git init, then make one commit: "chore: scaffold project structure".

Do NOT create any strategy logic yet. Structure only.
```

**Verify:** `uv run pytest` passes (trivially). `git log` shows one commit. `cat .gitignore`
contains `.env`.

**Then:** place `CLAUDE.md` in the repo root and this file at `docs/BUILD_PLAN.md`, commit as
`docs: add project memory and build plan`. Push to your private GitHub repo.

---

## Phase 1 — Install and validate vibe-trading-ai

**Goal:** prove the upstream project runs on your machine before you build on it.
**Time:** 45 min (budget more if pip resolution fights you)

```
Install vibe-trading-ai into a SEPARATE uv environment at ../vibe-sandbox — I want to
evaluate it in isolation before depending on it. Steps:

1. Create ../vibe-sandbox with uv, Python 3.11.
2. pip install vibe-trading-ai in that environment.
3. Run `vibe-trading init` to bootstrap its .env, then tell me exactly which file path it
   wrote to and which variables it expects.
4. Run `vibe-trading provider doctor` and show me the redacted output.
5. Report back: the installed version, whether the install succeeded cleanly, and any
   dependency conflicts or warnings.

Do not put any API keys in yet. Do not modify anything in our own repo during this phase.
```

Then, once it installs, configure it yourself (keys are yours to paste, not Claude's to handle):

```
Tell me the exact lines I need to add to ~/.vibe-trading/.env to configure Google Gemini as
the LLM provider and Finnhub as a data source. Show me the variable names only — I will paste
the values myself. Then tell me the command to verify the provider works.
```

**Verify:**
```bash
vibe-trading run -p "What is the current EURUSD price trend over the last 20 daily bars?"
```
If that returns a coherent answer, the LLM and data layers work.

**Critical check — do this before Phase 3:**
```
Investigate how vibe-trading-ai's MetaTrader 5 connector works. Specifically I need to know:
1. Does it use the MetaTrader5 Python package (Windows-only, requires running terminal), or
   does it connect some other way?
2. What configuration does it need in .env?
3. Does it support Exness demo servers specifically?

Read the installed package source in ../vibe-sandbox to answer. Report findings as a short
summary — do not modify anything.
```

This answer determines whether cloud hosting is possible later. Write it down.

---

## Phase 2 — The BrokerAdapter interface

**Goal:** the abstraction that makes everything else portable.
**Time:** 30 min

```
Create a new branch: phase/02-broker-adapter

Implement fxagent/adapters/base.py containing a BrokerAdapter Protocol (typing.Protocol) with
these methods, all fully type-hinted, all returning Pydantic models defined in the same module:

- get_bars(symbol: str, timeframe: str, count: int) -> BarSeries
- get_tick(symbol: str) -> Tick  (must include bid, ask, and spread in points)
- get_account() -> AccountState  (balance, equity, margin, currency)
- get_positions() -> list[Position]
- place_order(order: OrderRequest) -> OrderResult
- close_position(ticket: int) -> OrderResult

IMPORTANT: OrderRequest MUST require stop_loss and take_profit as non-optional fields. Make it
structurally impossible to construct an order without them. Add a Pydantic validator that
rejects a stop_loss on the wrong side of the entry price.

Also create fxagent/adapters/mock.py — a MockAdapter implementing the protocol against synthetic
data, so strategies can be tested without MT5 running.

Write tests covering: the SL-side validator rejects bad input, MockAdapter satisfies the
protocol, and OrderRequest cannot be built without SL/TP.

Run pytest and ruff. Show me the diff summary, then wait for my approval before committing.
```

**Verify:** try to construct an `OrderRequest` without a stop loss in a Python REPL. It must fail.

---

## Phase 3 — MT5 adapter and live connection smoke test

**Goal:** real bars from your real Exness demo.
**Time:** 45 min

```
Create branch phase/03-mt5-adapter.

Implement fxagent/adapters/mt5_local.py — MT5LocalAdapter implementing BrokerAdapter using the
MetaTrader5 Python package.

Requirements:
- Read credentials from environment variables only (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER).
  Never hardcode, never log them.
- initialize() must fail loudly with an actionable message if the terminal isn't running or
  Algo Trading is disabled.
- Convert MT5's numpy structured arrays to our Pydantic models. Coerce numpy scalars to
  Python native types — numpy int64 does not serialize to JSON.
- All returned timestamps must be timezone-aware UTC.
- place_order must attach SL and TP in the same order_send call, never as a follow-up.
- Add a context manager so the connection always shuts down cleanly.

Then create scripts/smoke_test.py that connects, prints account balance, fetches 100 H1 bars
for EURUSD, prints the current spread, and exits. It must NOT place any orders.

Run ruff and the tests. Then tell me the command to run the smoke test — I'll run it myself
with the terminal open.
```

**Verify:** run the smoke test. You should see your demo balance and a spread figure. If the
spread looks absurd (>5 pips on EURUSD in London hours), check you're on the right symbol name —
Exness sometimes suffixes symbols (`EURUSDm`, `EURUSDz`).

**Merge prompt (use this pattern for every phase):**
```
Run the full test suite and ruff. If everything passes, show me:
1. A one-paragraph summary of what this branch changed
2. The list of files added or modified with line counts
3. Anything you're uncertain about or that needs revisiting

Then wait. Do not merge until I say "approved".
```
After you approve:
```
Merge phase/03-mt5-adapter into main with --no-ff, delete the branch, and push.
```

---

## Phase 4 — The three strategies

**Goal:** deterministic signal generation, fully tested.
**Time:** 90 min

```
Create branch phase/04-strategies.

Implement fxagent/strategies/. First create base.py with:
- A Signal Pydantic model: symbol, direction (LONG/SHORT/FLAT), confidence (0-1),
  entry_price, stop_loss, take_profit, strategy_name, timestamp (UTC), and a
  reasoning: dict field for structured diagnostics.
- A Strategy abstract base class with: name property, required_bars property, and
  generate(bars: BarSeries, context: MarketContext) -> Signal | None

Then implement three strategies as separate modules:

1. session_breakout.py — Mark the Asian session range (00:00-07:00 UTC high/low). On the
   first H1 close beyond that range after 08:00 UTC, signal in the breakout direction. Stop at
   the opposite side of the range or 1.5x ATR(14), whichever is tighter. Target 2x risk.
   Return None outside 08:00-12:00 UTC.

2. range_reversion.py — Compute a 20-period z-score of close vs its rolling mean. Signal SHORT
   when z > 2, LONG when z < -2. Stop at 1.5x ATR beyond the extreme, target at the mean.
   Return None if ADX(14) >= 20.

3. carry_divergence.py — Takes an interest rate differential and a macro bias score from
   MarketContext (do NOT fetch these here — they're injected). Signal in the direction of the
   positive carry only when the 50-period EMA slope agrees. Daily timeframe. Stop at 2x
   ATR(14) daily.

Every strategy must be pure: same inputs produce same outputs, no I/O, no clock reads
(timestamps come from the bars).

Write thorough tests using MockAdapter with hand-constructed bar sequences that exercise:
each signal firing, each returning None when its gate fails, and stop-loss placement being on
the correct side. Aim for meaningful assertions, not coverage theatre.

Run pytest and ruff, then report before committing.
```

**Verify:** every strategy has at least one test proving it *doesn't* fire when its regime gate
fails. That negative test matters more than the positive one.

---

## Phase 5 — Regime router

**Goal:** the part that is genuinely yours.
**Time:** 60 min

```
Create branch phase/05-regime-router.

Implement fxagent/regime/:

1. sessions.py — pure functions mapping a UTC datetime to the active session(s):
   TOKYO (00-09), LONDON (08-17), NEW_YORK (13-22), OVERLAP (13-17). Handle the Friday
   21:00 UTC close and Sunday 21:00 UTC open. Include is_market_open() and
   minutes_until_close().

2. classifier.py — RegimeClassifier producing a Regime model with:
   - session: Session
   - trend_strength: float (ADX 14)
   - volatility_percentile: float (current ATR vs trailing 100-period distribution)
   - is_ranging: bool (ADX < 20)
   - is_trending: bool (ADX > 25)

3. router.py — RegimeRouter.weights(regime) -> dict[str, float] mapping strategy name to a
   weight in [0,1]. Rules:
   - session_breakout: 1.0 during LONDON before 12:00 UTC and trending; else 0.0
   - range_reversion: 1.0 when is_ranging and NOT in OVERLAP; else 0.0
   - carry_divergence: always 0.5 (it's a slow strategy, always partially on)
   Make these rules data-driven from a config dataclass, not hardcoded in the function body,
   so I can tune them without editing logic.

4. consensus.py — combine weighted signals. A trade fires only when the summed weight of
   agreeing non-zero-weighted strategies is >= 1.0 AND at least 2 strategies agree on
   direction. Return a ConsensusSignal or None, and ALWAYS return a diagnostics dict
   recording each strategy's vote and weight — including when the result is None.

Write tests for session boundary edge cases (exactly 08:00, Friday 20:59 vs 21:01, Sunday
open) and for consensus rejecting a 1-vs-1 disagreement.

Run pytest and ruff, report before committing.
```

**Verify:** ask Claude to print the router's weights for a few specific timestamps and sanity
check them against the session table in `CLAUDE.md`.

---

## Phase 6 — Risk sizing and the permission state machine

**Goal:** the safety layer. This is the phase not to rush.
**Time:** 75 min

```
Create branch phase/06-risk-and-permission.

Part A — fxagent/risk/sizing.py:
- position_size(account_equity, risk_fraction, entry, stop_loss, symbol_spec) -> volume
  computed from stop distance in risk units, NOT fixed lots. Must handle JPY pairs and
  account-currency conversion correctly.
- Enforce: risk_fraction defaults to 0.005 and is hard-capped at 0.005.
- total_open_risk(positions) and a check that rejects a new trade pushing total risk above 2%.
- Round volume down to the symbol's lot step, never up. Reject if below minimum lot.

Part B — fxagent/permission/grant.py:
A PermissionGrant Pydantic model with: allowed_symbols, max_trades, max_notional,
expires_at (UTC), granted_at, and a revoked flag with revocation reason.

A GrantManager state machine with states ADVISORY / GRANTED / REVOKED and these rules:
- Default state is ADVISORY. Execution is impossible in ADVISORY.
- A grant auto-expires at expires_at, and expires_at may never be later than the next
  Friday 20:00 UTC (before weekend gap risk).
- Auto-revoke triggers, each producing a logged reason: daily loss exceeds 3% of starting
  equity, spread exceeds 3x its 20-period median, high-impact calendar event within 15
  minutes, three consecutive losses, broker heartbeat lost for >60s.
Part C — fxagent/calendar/: the economic calendar behind the "high-impact event within 15
minutes" auto-revoke. Write it ourselves; vibe-trading's Finnhub loader is US-equity
dailies only.

Finnhub's /calendar/economic is PREMIUM — verified 2026-08-10, it returns HTTP 403
{"error":"You don't have access to this resource."} on our free key, while /quote,
/calendar/earnings, /calendar/ipo and /country all return 200 on the same key. Do not
retry it. Use Forex Factory's weekly JSON feed:

  https://nfs.faireconomy.media/ff_calendar_thisweek.json

Verified live: HTTP 200, a JSON array of 74 events, fields exactly
{title, country, date, impact, forecast, previous}. Five things about it that will
otherwise bite:

1. `date` is ISO 8601 carrying a -04:00 offset (US Eastern), NOT UTC. Parse with
   datetime.fromisoformat and .astimezone(UTC) immediately. Never strip the offset — it
   shifts with US DST, and a 15-minute revoke window computed an hour out is worse than
   no window at all.
2. `country` holds a CURRENCY code (USD, EUR, JPY, GBP, AUD, NZD, CAD, CHF, CNY), not a
   country. Match it against both legs of the pair being traded.
3. `impact` has four values, not three: High, Medium, Low, and Holiday. A Pydantic enum
   missing Holiday will reject the whole feed.
4. There is no event ID. Deduplicate on (date, country, title).
5. A User-Agent header is REQUIRED. Without one the host returns HTTP 429. The feed is
   Cloudflare-fronted and sends no-cache headers, so cache to disk ourselves and back off
   on 429 rather than polling.

The feed covers the CURRENT WEEK ONLY; ff_calendar_nextweek.json is 404. So on a Friday
the lookahead cannot see Sunday's open. Fail CLOSED: if the calendar is unavailable,
stale, or the week has run out, treat the window as unsafe and refuse execution rather
than assuming no event is due.

- A kill_switch() method that revokes immediately and attempts to flatten all positions.
- Persist grant state to disk so a process restart cannot silently resurrect a revoked grant.
  Fail closed: if state can't be read, assume ADVISORY.

Write tests for EVERY auto-revoke trigger, for expiry, for the Friday cap, and for the
fail-closed path when the state file is missing or corrupt.

This is safety-critical code. Prefer clarity over cleverness. Run pytest and ruff, report.
```

**Verify:** read the tests yourself, not just the code. Ask: "what would have to be true for
this to place a trade I didn't authorize?" If Claude can't answer convincingly, iterate.

---

## Phase 7 — Backtest harness

**Goal:** find out whether any of this actually works.
**Time:** 90 min. Do not skip.

```
Create branch phase/07-backtest.

Build fxagent/backtest/ that replays historical MT5 bars through the full pipeline —
strategies, regime router, consensus, risk sizing — and records results.

Cost model requirements (IMPORTANT — a backtest without these is invalid):
- Fill at ask for longs, bid for shorts, using actual historical spread where available and
  a configurable fixed spread where not.
- Apply swap/rollover for positions held past 21:00 UTC, triple on Wednesdays.
- Apply a configurable slippage in points, defaulting to 0.5 pips.

Metrics: total return, max drawdown, Sharpe, profit factor, win rate, average R multiple,
trade count, and a per-strategy breakdown so I can see which of the three is carrying.

Implement walk-forward validation: split the period into N folds, and report out-of-sample
metrics separately from in-sample. Report BOTH. Never report a single in-sample number.

Add a CLI: uv run python -m fxagent.backtest --symbol EURUSD --from 2024-01-01 --to 2025-12-31
Output a summary table to stdout and write full trade-by-trade results to CSV.

Run it on EURUSD H1 and show me the out-of-sample results.
```

**Verify:** if out-of-sample results are dramatically worse than in-sample, that's the honest
answer and it's telling you the parameters are overfit. Tune the *rules*, not the parameters, and
re-run. Resist the urge to search parameter space until the number looks good.

---

## Phase 8 — Journal, dashboard, Telegram

**Goal:** see what the agent is thinking, and approve trades from your phone.
**Time:** 90 min

```
Create branch phase/08-journal-and-alerts.

1. fxagent/journal/ — SQLite store logging every consensus evaluation (including the ones that
   produced no signal), the full diagnostics dict, whether a trade was taken, and the outcome
   once closed. Use a migration-friendly schema. Add a repository class, not raw SQL scattered
   through the codebase.

2. A Streamlit dashboard at fxagent/dashboard/app.py showing: current regime and session, each
   strategy's latest vote, open positions, current grant state with countdown to expiry,
   equity curve from the journal, and a table of recent signals with outcomes.

3. fxagent/alerts/telegram.py — send a signal card to Telegram with inline Approve/Reject buttons.
   On approve, place the order through the adapter; on reject, log the rejection with reason.
   Read TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env. Verify the sender chat ID matches
   the configured one and ignore messages from anyone else.

Tests for the journal repository and for the Telegram callback handler rejecting an
unauthorized chat ID.
```

**Verify:** send yourself a test signal card and press both buttons. Confirm the rejection path
logs correctly.

---

## Phase 9 — LLM synthesis layer

**Goal:** the last 20%. Deliberately last.
**Time:** 60 min

```
Create branch phase/09-llm-synthesis.

Build fxagent/llm/:

1. gateway.py — a provider-agnostic client. Primary: Google Gemini free tier. Fallback: Groq.
   On rate limit or error, fall through automatically and log which provider served the call.
   Read keys from env. Never log prompt content containing account details.

2. schemas.py — Pydantic models for LLM responses. The macro brief returns:
   bias (BULLISH/BEARISH/NEUTRAL), confidence (0-1), key_drivers (list of strings),
   risk_events (list), and reasoning (string, max 500 chars).

3. macro_agent.py — takes structured input (rate differentials, recent calendar surprises,
   central bank statement summaries) and returns a validated macro brief that feeds
   carry_divergence's MarketContext.

IMPORTANT constraints:
- The prompt must ask the model to SCORE STRUCTURED DATA AGAINST STATED RULES. Never ask
  "should I buy". Never send raw price series and ask for analysis.
- Never include the current date in backtest-mode prompts (look-ahead bias).
- Any response failing Pydantic validation is discarded entirely; the caller receives None and
  carry_divergence falls back to NEUTRAL bias. No retry loops, no partial parsing.
- Cache responses keyed on input hash. The macro brief runs at most 3x per day.

Write tests using a stubbed LLM client covering: valid response parses, malformed JSON returns
None, schema violation returns None, and the fallback provider is used when primary fails.
```

---

## Phase 10 — Paper run

```
Create branch phase/10-runner.

Build fxagent/main.py — an APScheduler loop that wakes on a schedule keyed to UTC session times,
runs one full analysis cycle per configured symbol, writes to the journal, and either sends a
Telegram card (ADVISORY) or places an order (GRANTED). Add --dry-run to skip all side effects.

Add graceful shutdown: on SIGINT, cancel pending work, close the broker connection cleanly,
and persist grant state.
```

Then run it in ADVISORY mode for **at least two weeks** before you consider granting any
execution permission, even on demo. You need a journal with real entries before the numbers mean
anything.

---

## Appendix A — .env.example

`.env.example` in the repo root is the source of truth; this is a summary. EXECUTION and DATA
are separated because they belong to different services — the executor needs the MT5 block and
nothing else, and the collector needs the DATA block and nothing else.

```bash
# ===== EXECUTION — MetaTrader 5 / Exness demo. Execution venue ONLY, not a data source. =====
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
MT5_SYMBOLS=EURUSD,GBPUSD,EURGBP
MT5_SYMBOL_SUFFIX=
MT5_SERVER_UTC_OFFSET_HOURS=
MT5_TERMINAL_PATH=
MT5_DEVIATION_POINTS=20

# ===== DATA — Forex Factory calendar needs no key, but DOES need a User-Agent (else 429). =====
CALENDAR_USER_AGENT=fx-regime-agent/0.1

# ===== LLM =====
GEMINI_API_KEY=
GROQ_API_KEY=
LLM_PRIMARY_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=groq

# ===== ALERTS =====
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# ===== RISK =====
RISK_PER_TRADE=0.005
MAX_TOTAL_RISK=0.02
DAILY_LOSS_LIMIT=0.03

# ===== RUNTIME =====
LOG_LEVEL=INFO
JOURNAL_DB_PATH=./data/journal.db
TRADING_MODE=advisory
```

---

## Appendix B — Reusable Claude Code prompts

**Start of every session:**
```
Read CLAUDE.md. We're working on Phase N — read docs/BUILD_PLAN.md for that phase's spec.
Before writing code, tell me your plan and wait for approval.
```

**Before merging:**
```
Run the full test suite and ruff check. Then give me: a one-paragraph change summary, the
file list with line counts, and anything you're uncertain about. Wait for my approval.
```

**When something breaks:**
```
Don't fix it yet. First: reproduce it, then tell me the root cause in one paragraph and
propose two options with trade-offs. I'll pick one.
```

**Weekly hygiene:**
```
Review CLAUDE.md against the current codebase. Flag anything that's now stale or wrong.
Propose edits but don't apply them — show me the diff first.
```

**Learning from upstream:**
```
Look at how vibe-trading-ai implements [X] in ../vibe-sandbox. Summarize their approach and
tell me whether ours should change. Do not copy code — MIT allows it, but I want to
understand the design before adopting it.
```

---

## Appendix C — Troubleshooting

| Symptom | Likely cause |
|---|---|
| `mt5.initialize()` returns False | Terminal not running, or Algo Trading button off |
| Symbol not found | Exness suffixes symbols — check Market Watch for the exact name |
| Spread looks enormous | Weekend, or you're outside session hours |
| Orders rejected with "invalid stops" | SL too close to price — check broker's minimum stop level |
| Numpy serialization errors | MT5 returns int64/float64; coerce to native Python types |
| Everything worked, then died overnight | PC slept. Disable sleep, or accept it until you move to cloud |

---

## What comes after

Once Phase 10 has run for a few weeks with a populated journal:

1. Compare each strategy's live-paper expectancy against its backtest. Divergence tells you
   what the backtest was missing.
2. Only then consider a permission grant — smallest possible size, one symbol, one session.
3. Revisit cloud hosting once you know whether the MT5 Windows dependency can be replaced
   (the Phase 1 investigation answers this).
