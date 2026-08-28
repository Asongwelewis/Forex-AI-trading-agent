# ADR-005: MT5 is the feed and the venue, in one local process

**Status:** accepted (2026-08-28)

**Supersedes:** the analyst and resolver halves of [ADR-002](ADR-002-scheduling.md). The cron
runtime survives for the jobs that must run with the desktop off; the trading path leaves it.

## Decision

**MetaTrader 5 is the sole source of bars and quotes for the trading path, and the sole
execution venue.** One long-lived local process — `fxagent.trader` — collects, decides, journals
and (under a grant) executes. Supabase is demoted from "the thing three services share" to a
journal and a history store. Twelve Data is retained as a gap-filler and a cross-check, and is
never the source a fill is modelled against.

GitHub Actions keeps only what must run while the desktop is off: the calendar, COT, statistical
observations, and health.

## What forced it

The user accepted, for cost reasons, that the agent analyses and trades only while the Exness MT5
terminal is open on their Windows desktop. That is not a limitation to engineer around. It is the
binding constraint, and taking it seriously deletes most of the architecture:

Once MT5 must be running anyway, keeping a separate data feed buys nothing and costs a great deal.

## What it fixes

**The feed was not the venue, and that invalidated the only result the project had.** Bars came
from Twelve Data; fills would happen on Exness. Different books, different candles, different
highs — so a barrier touch in the backtest is not necessarily a barrier touch in the account.
`backtest/replay.py` already hardcoded `DEFAULT_SOURCE = "mt5_exness"` while the collector wrote
TwelveData rows: the two halves of the system disagreed in writing about what a bar is.

**`bid_close` and `ask_close` were null on all 12,456 stored rows**, because Twelve Data does not
publish a two-sided quote. So every one of the 217 trades in the 2024–25 replay filled at a
configured flat 1-pip spread. On a result whose expectancy interval is `[-0.15, +0.15]R`, half a
pip of unmodelled spread moves the distribution by more than the point estimate is worth. MT5
gives bid and ask for free, on the book that will actually fill the order.

**The credit budget disappears.** `adapters/credits.py` exists because Twelve Data's free tier is
metered and 10–12 pairs of H1 across 744 runs a month strains it. MT5's history is unmetered.

**The Actions minute budget stops being a design constraint.** The hourly pass was consuming
750–1,500 of the 2,000 free minutes and the trimming comment in the workflow was load-bearing.
The trading path leaving Actions returns that headroom to the jobs that genuinely need a machine
in the cloud.

**The ARM wheel question goes away** for the trading path, along with the Docker Compose
three-service topology that was written for a host nobody bought.

## What it costs, stated plainly

**The agent sees nothing while the desktop is off.** A missed collection window is gone forever —
that asymmetry was the original argument for splitting the collector onto an always-on box, and
this decision knowingly gives it up. Two mitigations, neither of which pretends to be a fix:

1. **Uptime is recorded.** `service_heartbeats` already carries this shape. A journal that
   silently omits the hours the machine was asleep will overstate how much the agent saw, and
   every metric derived from it will be quietly conditioned on "the desktop happened to be on".
   Gaps must be visible in the record.
2. **Twelve Data stays wired as a gap-filler**, tagged with its own `source`, so a hole in the
   Exness series can be *seen* and optionally *plugged* — but never silently, and never for a bar
   a fill is modelled against. `bars.source` is part of a bar's identity precisely so these two
   series cannot merge by accident.

**Backfill is bounded by what the terminal will serve.** MT5 returns what the broker keeps, which
for Exness demo is generous on H1 and finite. The backfill script reports what it actually got
rather than assuming the requested range arrived.

**This is a home machine.** It sleeps, it reboots for updates, and its clock is not disciplined.
The trader must therefore be restart-safe by construction: state persisted, grants fail-closed on
an unreadable file, and no decision that depends on the process having been alive for the
previous bar.

## What does not change

**`bars.source` remains part of the bar's identity.** Two feeds writing the same symbol and
timeframe are two series, not one, and merging them would fork the history silently. This is the
mechanism that makes retaining Twelve Data safe.

**The collector stays the dumbest thing in the system.** `MT5LocalAdapter` satisfies the narrow
`DataSource` protocol — `source` and `bars_ending_at` — so the collector can be handed an
execution-capable adapter without being able to reach the parts of it that trade.
`tests/collector/test_collector_is_data_only.py` inspects the import graph with AST and must keep
passing against the new wiring.

**Costs stay in one module.** `fxagent/costs.py` is imported as a peer by both the backtest and
the live resolver. The whole point of this ADR is that the two become comparable; a forked cost
model would give that back immediately.

**Hard rule 1 is untouched, and gets stricter.** Bringing execution and analysis into one process
puts a working `place_order` in the same address space as the decision that would call it. The
permission layer is what stands between them, and until it exists there is no flag, no argument
and no config key on `fxagent.trader` that places an order.

## Consequences for the workflows

`collect-and-analyse.yml` named `fxagent.analyst` and `fxagent.resolve` as its second and third
stages. Neither module has ever existed.

**This would not have shown up as a failure.** `.github/actions/stage` checks for a `__main__`
submodule and, when it is absent, emits a `::warning` annotation and exits 0 — a deliberate
choice, made so that a cron waiting on unlanded phases does not sit permanently red, because a
permanently red cron is indistinguishable from a broken one. That reasoning is sound. Its
consequence here is not: the hourly pass would have reported success while two of its three
stages did nothing, indefinitely, behind an annotation nobody reads. Nobody saw it either way,
because schedules fire only from the default branch and this has lived on `develop`.

The escape hatch is right for a phase that is *about* to land and wrong as a permanent state. So
under this ADR the two stages do not move to Actions and do not stay as pending skips — they move
to the desktop and leave the workflow. What remains is the collector, which is exactly the job
that benefits from a machine that is always on.

A test now asserts that every module named by a workflow either exists or is listed as a known
pending stage, so "silently skipped forever" requires writing the module's name down as pending
rather than merely forgetting it.
