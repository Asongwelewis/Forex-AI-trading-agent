# ADR-006: The MetaTrader 5 MCP server is a measurement tool, not a trading path

**Status:** accepted (2026-08-28)

## Decision

Adopt MetaQuotes' built-in MCP support **for development and measurement only**, on the
developer's machine, with AI-initiated trading **disabled in the terminal's own settings**.

`fxagent/` neither imports nor shells out to it. No production code path reaches an MCP tool,
and `tests/test_mcp_is_not_a_trading_path.py` asserts that. The MCP is something *Claude Code
uses while working on this repo*, in the same category as a REPL — not something the agent uses
while trading.

## What it actually is

MetaQuotes shipped native MCP support in the MetaTrader 5 terminal: beta build 6030 on
2026-07-16, general release in build 6060 on 2026-07-23/24, with build 6090 on 2026-07-30
expanding the method set (adding indicators to a chart, listing available indicators and their
parameters). The terminal exposes market data, charts, account state and **trading operations**
over MCP, and can be driven by external agents including Claude Code. Access is configured
through an MQL5.community sign-in.

The critical line from the release notes: *"AI operations are governed by configurable security
settings. You can explicitly allow or prohibit AI-initiated trading operations, or require
manual confirmation."*

So this is a read-**write** interface with a settings-level guard, not a read-only data feed.

There are also several community MT5 MCP servers on PyPI and GitHub. Everything below applies to
those identically, and more so: they take the account password as a launch argument.

## Why it is not allowed near the trading path

**It is an LLM holding an order ticket.** Hard rule 4 exists because an agent that can move a
stop is an agent that can empty an account. An MCP tool that places trades is precisely that,
whatever the prompt around it says. This is not a probabilistic worry about model quality; it is
that the decision would no longer be deterministic Python, and the whole architecture is built
on it being deterministic Python.

**It would be a second authorisation path.** `fxagent/permission/` exists so that exactly one
deterministic code path can authorise an order, with a grant that expires, a trade budget that
does not refill on restart, and seven auto-revoke triggers. An order placed through MCP is
gated instead by a checkbox in the terminal's settings dialogue. That is the same objection that
removed the Telegram Approve/Reject buttons in ADR-005's neighbourhood, and it is stronger here,
because the MCP path can also *modify* an existing position.

**It is not replayable, so nothing it does is measurable.** `backtest/replay.py` replays
`trader/cycle.run_cycle` bar by bar. An MCP-initiated trade has no counterpart in the replay, so
it cannot appear in an expectancy interval, cannot be purged into a walk-forward fold, and
cannot be attributed to a sleeve. GATE B says no new alpha work until costs are measured on
Exness bars; a trading path that is unmeasurable by construction does not clear that bar, it
steps around it.

**It adds no data capability we lack.** The `MetaTrader5` Python package already provides bars,
ticks with bid/ask, the per-bar spread field, account state, positions and `order_send`, from the
same terminal. The MCP is a different transport to the same place. It is convenience, not reach.

## Why it is still worth having

The measurement work in Lane 2 of `docs/BOARD.md` is currently blocked on facts that only the
live terminal knows, and that a Claude Code session has no way to read:

* `symbol_info().swap_long` / `swap_short` per symbol, in points — card 2.2. `CostConfig`
  currently defaults both to zero, and the docstring says why: a wrong swap number is worse than
  an absent one.
* `symbol_info().filling_mode`, which is a bitmask rather than an enum value and has already
  caused confusion.
* `symbol_info().point`, `digits`, `volume_min`, `volume_step`, `trade_contract_size` per symbol
  — the `SymbolSpec` values that sizing depends on and that are currently derived from the
  currency legs rather than read from the broker.
* Whether the per-bar `spread` field is actually populated on Exness history, and at what
  coverage. `adapters/mt5_local._quote_from_rate` treats zero as absent for exactly this reason,
  and nobody has yet checked which case is the common one.
* Confirmation that the server clock is still UTC+0. It was measured once, on 2026-08-08, by
  eye.

Each of those is a one-line question that currently costs a round trip: "please run this and
paste the output." With the MCP configured, the session reads it directly, writes the measured
number into the repo, and the value arrives with a provenance note instead of a guess.

That is a real acceleration of the one lane that gates everything else. It is worth the setup.

## The configuration, and the one setting that matters

In the terminal: **Tools → Options → AI Assistant**, sign in with the MQL5.community account,
and set AI-initiated trading operations to **prohibited**. Not "require confirmation" —
prohibited. A confirmation dialogue is a thing that gets clicked through at 2am, and there is no
reason for this project's MCP access to have write capability at all, since every order it will
ever place goes through `fxagent/permission/` on a path the MCP is not part of.

Then connect Claude Code to the terminal per MetaQuotes' instructions
(`https://www.mql5.com/en/articles/21905`). Verify the build is 6060 or later — MCP support does
not exist below it — with **Help → About**.

Two things this ADR does not authorise, and both are easy to drift into:

1. **Do not install a community MT5 MCP server that takes the account password as an argument.**
   Hard rule 2. The credentials live in `.env` and reach exactly one process.
2. **Do not point the MCP at a real-money account, ever**, including for read-only measurement.
   Hard rule 1 pins the login for a reason, and an MCP session has no `_verify_account` in front
   of it.

## Consequences

The measurement cards in Lane 2 stop being blocked on manual paste-backs, which is the point.

`fxagent/` is unchanged. There is no dependency, no import and no new extra — the MCP is a
property of the developer's machine, not of the package, and a checkout of this repo on a
machine without it behaves identically.

If MetaQuotes later ships a genuinely read-only MCP profile, this ADR should be revisited: the
objection above is entirely to the write capability and the second authorisation path, and a
read-only interface would carry neither. It would still not become a trading path, because the
determinism objection stands on its own.
