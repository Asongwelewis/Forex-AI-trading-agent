# FX Regime Agent — Kanban rewrite

> **This file mirrors the Notion board** (`Build Tasks`, under the *FX Regime Agent* page).
> Notion is the working copy; this is the version-controlled record of the same card set.
>
> **Status as of 2026-08-28.** Lane 0 is complete except 0.4. Lane 1 is complete except 1.3
> (needs the terminal). Lane 3 is complete. Card numbering on the board uses `L<lane>.<card>`
> prefixes in the title, and the `No.` column for execution order.

Paste-ready for the Notion board. Every card has **Goal**, **Done when**, **Why**.
Cards are ordered by dependency: nothing in a later lane is startable until its lane's
prerequisites are green.

**Actions on existing cards** are marked `CLOSE`, `REWRITE` or `NEW` at the top of each card.

---

## The one-paragraph strategy

MT5 must be open on the desktop for execution. Therefore MT5 is **both the feed and the
venue** — one local process, one source of truth for what a bar is, free bid/ask, free
history, no credit budget, no source/venue mismatch. That is the *simple*. The *unusual*
is three things almost no retail bot has: a meta-labeller trained on triple-barrier labels
with purged folds, a cost-aware gate that vetoes on spread percentile at signal time, and
a refusal ledger treated as a dataset. The gate in front of all of it: **no new alpha work
until the edge is re-measured on Exness bars with measured spreads.**

---

# Lane 0 — Truth and unblocking

*Nothing else is safe to start until the repo describes itself accurately and the workflow
stops referring to modules that do not exist.*

### 0.1 — Land the working tree
`NEW`

**Goal.** Commit the 24 staged files: the consensus→selection replacement, `regime/bias`,
`costs`, `spreadwatch`, migration 0012.

**Done when.** `git status` is clean on `develop`, full suite green, ruff clean, and the
selection change is one commit with the autopsy in the message.

**Why.** A 1,788-line unlanded change containing an architectural replacement is the single
largest source of "which version am I reviewing" confusion. It is finished work sitting in
a state where nobody else can build on it.

---

### 0.2 — Rewrite CLAUDE.md against the actual codebase
`REWRITE` — supersedes any "keep memory files current" card

**Goal.** Every factual claim in CLAUDE.md is true of `develop` today.

**Done when.** Each of these is corrected, and a test or a doc line names the source of truth:

- Data source: Twelve Data today, **MT5/Exness after Lane 1** — not OANDA v20.
- Deployment: GitHub Actions cron + one optional dashboard container — not three always-on
  Oracle ARM services. The three-service table is aspiration, not architecture.
- `regime/consensus` is deleted. It is `regime/selection` — one sleeve, no agreement rule.
- Delete "a signal fires only when ≥2 of 3 agree." It was unsatisfiable by construction and
  has a written autopsy in `selection.py`. Replace with the router-selects-one-sleeve rule.
- pandas is pinned `>=2.3.2,<3`, not 3.0.5.
- The NIM model is `nvidia/nemotron-3-ultra-550b-a55b`, not `deepseek-v4-pro`.
- `fxagent/permission/` is empty. Say so, or Lane 3 fixes it and this line goes.
- Structure block: add `costs.py`, `spreadwatch.py`, `observations.py`, `stats/`.

**Why.** This file is loaded into every session and overrides default behaviour. Six false
statements in it is six ways for the next change to be built on a wrong premise. It is the
cheapest high-leverage fix on the board.

---

### 0.3 — Stop the workflow calling modules that do not exist
`REWRITE` — supersedes the Actions-cron card

**Goal.** `collect-and-analyse.yml` refers only to modules that exist.

**Done when.** Either the `analyse` and `resolve` steps are removed until Lane 1 builds
them, or they are pointed at the real entrypoints. A CI check asserts every `module:` input
in every workflow is importable.

**Why.** `fxagent.analyst` and `fxagent.resolve` do not exist. Two of the three stages in
the only production workflow would fail on every scheduled run. The suite is green and the
pipeline is broken — exactly the failure mode the test suite cannot see.

---

### 0.4 — Get `main` current so cron can fire
`NEW`

**Goal.** `develop` merged to `main` with `--no-ff` once Lane 0 is green.

**Done when.** `main` is level with `develop`, one `workflow_dispatch` run of the health
workflow completes green, and the schedule has fired at least once unattended.

**Why.** GitHub schedules workflows from the default branch only. `develop` is 8 commits
ahead. Nothing has ever run on a timer. Until this lands the system has no operating history
at all.

---

# Lane 1 — Collapse to one local process

*The architecture the accepted constraint implies. Largest simplification available, and it
removes the biggest silent bias in the backtest.*

### 1.1 — ADR-005: MT5 is the feed and the venue
`NEW`

**Goal.** Write the decision down before the code moves.

**Done when.** `docs/ADR-005-single-process.md` records: MT5 as sole bar and quote source
for the trading path; Supabase demoted to journal and history store; Twelve Data retained
only as a gap-filler and cross-check, never as the source a fill is modelled against; GitHub
Actions reduced to what must run with the desktop off (calendar, COT, health); and the
explicit cost of the choice — the agent only analyses and trades while the desktop is on,
and a missed window is gone.

**Why.** Every subsequent card in this lane is a consequence of this decision. Writing it
down is what stops it being relitigated in three weeks, and the cost paragraph is what makes
the tradeoff honest rather than hidden.

---

### 1.2 — MT5 collector: bars *and* two-sided quotes
`REWRITE` — supersedes the Twelve Data collector card

**Goal.** `MT5LocalAdapter` becomes a `DataSource`, writing bars with `bid_close` and
`ask_close` populated on every row.

**Done when.** The collector runs against MT5 with `source="mt5_exness"`, every written row
carries a non-null bid and ask, `test_collector_is_data_only` still passes against the new
import graph, and re-ingesting a window is still a no-op on `bars_unique`.

**Why.** `bid_close`/`ask_close` are null on all 12,456 existing rows, which is why the
2024–25 replay had to assume a flat 1-pip spread. Fills are the largest unmeasured
assumption in the only result the project has. Only the venue's own feed can fix it.

---

### 1.3 — Backfill Exness history, H1 and M15
`REWRITE` — supersedes/extends `scripts/backfill_mt5.py`

**Goal.** Two to three years of Exness H1 and M15 for the traded symbols, in `bars`, tagged
`mt5_exness`.

**Done when.** The `--source mt5_exness` gap report is clean over the range for every symbol
and both timeframes, and the row count is reconciled against MT5's own bar count per symbol.

**Why.** The backtest currently measures TwelveData's book while the fills would happen on
Exness's. Different candles, different highs, different session prints. Until the replay runs
on the venue's own bars, every number measures a system nobody will run. M15 is included here
because Lane 4 needs it and history is cheap to pull once.

---

### 1.4 — `fxagent.trader`: the runner that does not exist
`REWRITE` — supersedes BUILD_PLAN Phase 10 "APScheduler loop"

**Goal.** One long-lived local process: wake on bar close → collect → classify → route →
select → size → journal → (advise | execute) → resolve.

**Done when.**

- `uv run --extra mt5 python -m fxagent.trader --dry-run` completes a full cycle on one
  symbol and writes one `evaluations` row, *including* when nothing fired.
- Every decision, fired or refused, lands in the ledger with full diagnostics.
- `--advisory` (default) sends a Telegram card and places nothing. There is no flag that
  places an order until Lane 3 is green.
- Graceful shutdown: SIGINT finishes the bar in flight, persists state, closes MT5 cleanly.
- One integration test drives a full cycle end to end against `MockAdapter`.

**Why.** This is the missing 5% that makes the other 95% a system rather than a library.
Nineteen thousand lines of tested components and no code path from a stored bar to a
recommendation. Build the ugly vertical slice; thicken it after.

---

### 1.5 — `fxagent.resolve`: close out past decisions
`NEW`

**Goal.** Walk unresolved journal entries and resolve them against bars that have since
arrived, using `backtest.barriers` and `costs` — the same modules the backtest uses.

**Done when.** A resolver run marks TARGET/STOP/TIME per open entry, charges spread,
slippage and swap through `fxagent.costs`, records the R multiple, and `test_costs_are_shared`
proves it charges the same costs the backtest does.

**Why.** Without this the journal accumulates decisions and never learns their outcomes,
which means no live-vs-backtest comparison and no training set for Lane 4. Sharing `costs.py`
is what keeps the paper run and the backtest comparable — the whole reason that module sits
at the top level.

---

### 1.6 — Telegram: cards out, no buttons
`REWRITE` — supersedes BUILD_PLAN Phase 8 "approve/reject inline buttons"

**Goal.** One outbound notifier: the signal card, the refusal summary, the failure alert.

**Done when.** `fxagent/alerts/telegram.py` sends a formatted card; the chat ID is verified;
nothing inbound is parsed; no code path in this module can place an order.

**Why.** Approve/Reject buttons make Telegram an execution channel, and an execution channel
authenticated by a chat ID is not one. The permission grant in Lane 3 is the authorisation
mechanism; a chat button would be a second one that bypasses it.

---

# Lane 2 — Measure the real cost

*Gate: no edge claim, in either direction, until this lane is green.*

### 2.1 — Run spreadwatch for ten sessions
`NEW`

**Goal.** A real distribution of Exness spread, per symbol, per hour — especially the London
open minute.

**Done when.** `spread_samples` holds ≥10 trading days across the traded symbols, and a short
report gives p50/p75/p90/p99 spread per symbol per hour, plus the London-open minute in
isolation.

**Why.** The mean spread over a quiet hour is not the number that matters — `session_breakout`
fires exactly when a market-maker widens, so the fill that decides the trade is drawn from the
tail. `spreadwatch.py` exists to measure this and has never been run.

---

### 2.2 — Measure slippage and swap from the terminal
`NEW`

**Goal.** Replace the config defaults in `CostConfig` with measured Exness numbers.

**Done when.** Swap long/short points per symbol are read from `symbol_info()` and stored;
realised slippage is measured from ≥30 demo market orders at varying times; both are wired
into `CostConfig` defaults with the measurement date recorded beside them.

**Why.** Slippage defaults to 0.5 pips and swap is whatever the config says. On a strategy
whose expectancy interval is ±0.15R, half a pip of error is a material fraction of the entire
signal. Guessed costs make a backtest an opinion.

---

### 2.3 — Re-run 2024–25 on Exness bars with measured spread
`REWRITE` — supersedes card 16 (the backtest card)

**Goal.** The honest number, finally.

**Done when.** Replay runs on `source=mt5_exness` bars, draws spread from the measured
distribution rather than a constant, and the report gives out-of-sample expectancy **with its
interval**, profit factor, both drawdown modes, the ambiguity rate, and a per-sleeve breakdown.
The result is committed to `docs/RESULTS-<date>.md` whatever it says.

**Why.** The current number is 217 trades, expectancy interval [-0.15, +0.15]R, on a flat
1-pip assumption over the wrong venue's bars. That is not "no edge" — it is "no measurement."
This card produces the first measurement the project has ever had.

---

### 2.4 — Source divergence report
`NEW`

**Goal.** Quantify how much TwelveData and Exness disagree.

**Done when.** A short report over the overlapping range gives per-bar OHLC deltas in pips,
the share of bars where the high/low differ enough to flip a barrier touch, and a verdict on
whether TwelveData remains usable as a gap-filler.

**Why.** You are about to have two years of history from two books. Knowing the size of the
disagreement tells you whether the old result was merely imprecise or actively misleading —
and whether backfill gaps can safely be plugged from the other source.

---

# Lane 3 — The safety layer

**COMPLETE — commit `e5e7cb8`, 92 tests.** Shipped as one card on the board rather than the
four below, because the three modules only make sense together. The four cards are kept here
as the specification they were built to.

*What remains is not code: nothing in this lane has been run against a real terminal. That is
Lane 5 card 5.3, and it is what GATE A now waits on.*

### 3.1 — `permission/grant.py` — the state machine
`REWRITE` — supersedes BUILD_PLAN Phase 6 Part B

**Goal.** ADVISORY / GRANTED / REVOKED, persisted, fail-closed.

**Done when.** `PermissionGrant` carries allowed_symbols, max_trades, max_notional,
granted_at, expires_at, revoked + reason. `expires_at` can never exceed the next Friday
20:00 UTC. State persists across restart. An unreadable or corrupt state file yields
ADVISORY, tested. Execution is structurally impossible in ADVISORY — not a branch, a type.

**Why.** This is the only thing standing between a working `place_order` and an unauthorised
trade, and it does not exist. The Friday cap is what stops a grant surviving into a weekend
gap.

---

### 3.2 — Auto-revoke triggers, every one tested
`REWRITE` — supersedes the auto-revoke half of Phase 6 Part B

**Goal.** Seven triggers, each producing a logged reason.

**Done when.** Each of these has its own test: daily loss > 3% of starting equity; spread
above its ceiling *(now computable from Lane 2's real distribution — use the symbol-hour p90,
not a multiple of the median)*; high-impact calendar event within 15 minutes; three
consecutive losses; broker heartbeat lost > 60s; MT5 `trade_allowed` gone false; account
`trade_mode` not DEMO. Calendar unavailable or stale ⇒ **unsafe**, refuse.

**Why.** Fail-closed on the calendar matters more than it looks: the Forex Factory feed is
current-week-only, so on a Friday the lookahead cannot see Sunday's open. Assuming "no event
found" means "no event" is exactly backwards.

---

### 3.3 — Kill switch and weekend flat
`NEW`

**Goal.** One call that revokes and flattens; one schedule that guarantees flat before the
weekend.

**Done when.** `kill_switch()` revokes, attempts to close every position, and reports what it
could not close. A Friday cutoff forces flat regardless of grant state, tested against the
Friday 21:00 UTC boundary.

**Why.** Weekend gap risk on a leveraged demo teaches the wrong habits and on a real account
ends them. The flatten path must be tested when it *fails*, not only when it succeeds.

---

### 3.4 — Re-assert everything at the order, not at startup
`NEW`

**Goal.** A single pre-flight the order path cannot skip.

**Done when.** Immediately before every `order_send`: account is DEMO and the login matches
the pinned one; `terminal_info().trade_allowed` is true; the grant is GRANTED, unexpired and
covers this symbol; total open risk plus this trade ≤ 2%; current spread is under its ceiling;
no high-impact event inside the window; SL and TP are both attached to *this* request. Any
failure refuses and journals the reason. A test asserts no order can be constructed that
skips it.

**Why.** Hard rule 1 says re-assert before every order, not just at startup. `mt5.initialize()`
succeeds with Algo Trading off and only `order_send` fails later — a startup-only check passes
and then trades into a terminal that will not execute.

---

# Lane 4 — Find an actual edge

*Gate: Lane 2 green. This is where "advanced and unusual" lives.*

### 4.1 — The refusal ledger as a dataset
`NEW`

**Goal.** Turn `evaluations` into a queryable feature table.

**Done when.** One point-in-time SQL view gives, per decision: regime features (ADX, vol
percentile, session, minutes-into-session), the sleeve and its confidence, spread percentile
at that instant, calendar proximity, COT crowding, the bias verdict — and the realised barrier
outcome where one exists. A look-ahead test proves nothing in the row post-dates the decision.

**Why.** This ledger already caught one structural failure that the trade log could not see.
It is the most differentiated asset in the repo and it is currently write-only. It is also the
training set for 4.2 — and the point-in-time test is what makes it trustworthy rather than a
leak with a schema.

---

### 4.2 — Meta-labeller: should we take this signal?
`NEW`

**Goal.** A model that sizes or vetoes the sleeve's signal. It never originates one.

**Done when.** A gradient-boosted or logistic model trained on 4.1's features against the
triple-barrier labels, evaluated **only** through `purged_walk_forward` with label-span purging
and embargo. Reported out-of-sample with intervals alongside the unfiltered baseline. Ships
behind a flag, default off, and cannot change direction, stop or target — only *take / skip /
downsize*.

**Why.** You have already built the hard, boring half of this: triple-barrier labels with
spans, purged folds, and an exhaustive decision record. This is one step from done and it is
the honest answer to "unusual" — almost no retail system meta-labels, and none of them have a
refusal ledger to train on. It also respects hard rule 4: the model gates a deterministic plan,
it does not construct one.

---

### 4.3 — Cost-aware gate
`NEW`

**Goal.** Spread percentile at signal time becomes a first-class veto.

**Done when.** A signal arriving when the current spread is above its symbol-hour p90 is
refused with a journaled reason. The threshold comes from Lane 2's measured distribution. The
replay applies the identical rule so backtest and live agree.

**Why.** Breakout strategies die on cost, not on direction — they fire precisely when the
market-maker widens. Retail systems ignore this entirely, which is exactly why it is worth
doing. Cheap to build, plausibly the largest single improvement to expectancy available.

---

### 4.4 — Widen the sample so validation is reachable
`NEW`

**Goal.** Enough observations to ever conclude anything.

**Done when.** M15 is collected and `session_breakout` is evaluated on it as well as H1; the
symbol set is extended to 8–10 majors and crosses; `carry_divergence` is finally run on a D1
replay so it is measured at all; trade count per sleeve per year is reported.

**Why.** Two years produced 217 trades, `carry_divergence` was excluded from every replay ever
run, and the two intraday sleeves are gated mutually exclusively so at most one can trade at
any instant. At ~2 trades/week, `INSUFFICIENT_DATA` never clears. The design currently cannot
generate the evidence needed to validate it.

---

### 4.5 — Kill or keep: the verdict card
`NEW`

**Goal.** An explicit decision point with a pre-committed rule.

**Done when.** After 4.2–4.4, each sleeve gets a verdict written down: **keep** (out-of-sample
expectancy interval strictly above zero), **keep gated** (positive only under the meta-label
and cost gate), or **kill**. A killed sleeve is deleted, not disabled.

**Why.** The rule must be written before the number is seen. Without a pre-committed threshold,
"the interval straddles zero" becomes "it's nearly positive," and that is how every overfit
system survives its own evaluation.

---

# Lane 5 — Run it

### 5.1 — Four weeks in ADVISORY
`REWRITE` — supersedes card 22 (the paper-run card) and BUILD_PLAN Phase 10's "two weeks"

**Goal.** The trader runs whenever the desktop is on, places nothing, journals everything.

**Done when.** ≥4 weeks of entries; a daily Telegram summary of decisions and refusals; the
resolver has closed out every entry old enough to resolve; uptime is recorded so gaps from the
desktop being off are visible rather than silently missing.

**Why.** Four weeks not two, because at ~2 trades/week two weeks is four trades. The uptime
record matters: the desktop-only constraint means missing windows are structural, and a journal
that hides them will overstate how much the agent actually saw.

---

### 5.2 — Live-vs-backtest divergence report
`NEW`

**Goal.** Compare paper expectancy against backtest expectancy, per sleeve.

**Done when.** A report gives realised vs modelled spread, realised vs modelled slippage,
signal count vs the replay's expectation over the same window, and the R-multiple distributions
side by side. Every divergence is named and attributed.

**Why.** This is how you find out what the backtest was missing, and it is only possible because
`costs.py` is shared between the two paths. Divergence here is information; unexplained
divergence is a defect.

---

### 5.3 — The first grant
`NEW`

**Goal.** The smallest possible authorised trade.

**Done when.** One symbol, one session, minimum lot, a grant expiring the same day, on the
Exness demo, with 3.4's pre-flight in the path. The order carries server-side SL and TP in the
same request. It fills, and the resolver closes it out correctly.

**Why.** Everything until now is unverified against a live broker. Phase 3 earned the
`develop`/`main` split precisely because 104 green tests sat on top of entirely unverified
offset detection. This card is what moves `main` honestly.

---

### 5.4 — Weekly review ritual
`NEW`

**Goal.** A standing 30-minute review that keeps the honest numbers in front of you.

**Done when.** Every Sunday: expectancy interval per sleeve, trade count against the 100-trade
floor, refusal reason histogram, cost divergence, and any auto-revoke that fired.
`INSUFFICIENT_DATA` is stated first when it applies.

**Why.** Win rate is not a success metric and must never gate anything — `session_breakout`
targets 2R, so 40% is strongly profitable. A ritual that puts expectancy and the interval at
the top is what stops the win rate becoming the number you steer by.

---

# Lane 6 — Trim

*Cards that mean "stop", not "build". Each one is scope you are carrying that Lane 2 has not
justified.*

### 6.1 — Freeze fundamentals, observations and COT
`REWRITE` — supersedes cards for Phase 9 fundamentals and 9.6 COT

**Goal.** Keep collecting, stop building.

**Done when.** The daily collection workflows keep running (the clock they start cannot be
restarted later), and no new feature work lands in `fundamentals/`, `observations.py` or the
COT path until a Lane 4 result shows a feature there carries signal.

**Why.** ~2,300 lines built downstream of an unmeasured edge. The collection is worth
continuing because history cannot be backfilled; the feature work is not, until something
consumes it.

---

### 6.2 — Dashboard to maintenance
`REWRITE` — supersedes the card for Phase 10 dashboard

**Goal.** No new dashboard features.

**Done when.** It renders the ledger and the equity curve and nothing else changes. The Vercel
deployment is dropped unless it is actually being used — it is a degraded polling variant of a
local page.

**Why.** 2,600 lines on a read-only viewer for a system that has not yet placed a trade. It
works; leave it working.

---

### 6.3 — Three agents to one
`NEW`

**Goal.** Cut the narration layer to the chartist.

**Done when.** The historian is retired until `memory/` retrieval actually exists (it is a
128-dim spec with no encoder, no index and no query — the analogues it narrates are never
populated). The risk officer is retired: it is advisory-only by design and the deterministic
template already says what it says. LLM budget drops from 50 calls/day to ~15.

**Why.** Two of three agents narrate things that either do not exist or do not affect anything.
Hard rule 4 already guarantees they cannot decide — which means removing them costs nothing but
noise. Re-add the historian the day retrieval is real.

---

### 6.4 — Decide the fate of `memory/`
`NEW`

**Goal.** Build it or delete it.

**Done when.** Either an encoder plus a pgvector index plus a point-in-time retrieval query
exist with a look-ahead test, or `fxagent/memory/`, migration 0006 and the windows repository
are removed and the intent is recorded in an ADR.

**Why.** A spec and a table with no implementation is a standing invitation to assume the
feature exists. It has already produced one agent that narrates an empty list.

---

# Suggested board columns

`Blocked` · `Ready` · `In progress` · `In review` · `Done`

With two gate markers as board-level callouts, because they are the two places the project can
go wrong quietly:

> **GATE A — no order until a human has watched a live smoke test (5.3).**
> The permission layer exists now; what is missing is that nothing has called it for real.
>
> **GATE B — no new alpha work until Lane 2 is green.**

---

# Order of play

| Week | Lane |
|---|---|
| 1 | Lane 0 entirely, then 1.1–1.3. Start 2.1 running on day one — it needs ten days of wall-clock. |
| 2 | 1.4–1.6. The vertical slice. 2.2 alongside. |
| 3 | 2.3, 2.4. The first honest number. Then Lane 3 begins. |
| 4 | Lane 3 complete, 5.1 begins. |
| 5–8 | Lane 4 while 5.1 accumulates. 6.1–6.4 as they come up. |
| 9 | 4.5, then 5.2, then 5.3. |


---

# Addendum — the MetaTrader 5 MCP (added 2026-08-28)

### 6.1 — Set up the MT5 MCP, measurement only
`NEW` — board card L6.1, decision in `docs/ADR-006-mt5-mcp.md`

**Goal.** Read swap points, filling modes and symbol specs off the live terminal from a Claude
Code session, without giving anything the ability to trade through it.

**Done when.** Terminal build is 6060+, AI-initiated trading is set to **prohibited** (not
"require confirmation"), Claude Code is connected, and `tests/test_mcp_is_not_a_trading_path.py`
still passes.

**Why.** MetaQuotes shipped native MCP in build 6060 on 23 Jul 2026. It exposes trading
operations, which rules it out of the trading path three times over: it is an LLM holding an
order ticket (hard rule 4), it is a second authorisation path beside `permission/`, and it is
not replayable so nothing it does can appear in an expectancy interval.

It is still worth having, because Lane 2 is currently blocked on facts only the terminal knows
and every one of them costs a round trip through a human today. That is a real acceleration of
the one lane that gates everything else.
