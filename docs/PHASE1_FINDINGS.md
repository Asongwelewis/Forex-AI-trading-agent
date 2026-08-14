# Phase 1 findings — vibe-trading-ai evaluation

Investigated 2026-08-08 against **vibe-trading-ai 0.1.12** (MIT, `HKUDS/Vibe-Trading`),
installed into an isolated environment at `../vibe-sandbox` (Python 3.12).

Nothing here was copied into our codebase. The sandbox exists so we can read upstream
source without depending on it.

---

## Install

Clean: 183 packages, exit 0, no dependency conflicts. The single warning was a uv
hardlink fallback (cache on `C:`, target on `S:`) — cosmetic.

## The `src` collision — why our package is `fxagent/`

`vibe-trading-ai` installs **unnamespaced top-level packages** into site-packages:
`src/`, `cli/`, and `backtest/`.

Our project originally used `src/` too, and Phase 7 planned a `src/backtest/`. Both would
collide outright in a shared environment; `import src.adapters` would resolve by `sys.path`
ordering, which favours cwd today and breaks the moment anything is installed or deployed.

Renamed to `fxagent/` before Phase 2 wrote a single import. **Do not reintroduce `src/`.**

A third collision exists but is handled: `caio` (a transitive dep) ships a top-level `tests`
package. Our pytest config uses `importmode = "importlib"`, which resolves test modules by
path rather than `sys.path` order, so ours can never be shadowed.

## pandas is pinned below 3

`vibe-trading-ai` requires `pandas<3.0.0`; `pandas-ta` requires `pandas>=2.3.2`. We pin
`pandas>=2.3.2,<3` and resolve to **2.3.3**, matching the sandbox exactly. Phase 4 indicator
code must be written against pandas 2.x semantics — copy-on-write is not mandatory there.

## LLM provider configuration

The `vibe-trading init` wizard is interactive and its provider catalogue lists **neither
Gemini nor Groq**. Both are nevertheless supported, defined in `cli/_legacy.py` as
OpenAI-compatible endpoints. Config lives at `~/.vibe-trading/.env`:

| Variable | Gemini | Groq |
|---|---|---|
| `LANGCHAIN_PROVIDER` | `gemini` | `groq` |
| `LANGCHAIN_MODEL_NAME` | `gemini-3.5-flash` | `meta-llama/llama-4-maverick-17b-128e-instruct` |
| key env | `GEMINI_API_KEY` | `GROQ_API_KEY` (prefix `gsk_`) |
| base env | `GEMINI_BASE_URL` | `GROQ_BASE_URL` |
| base URL | `https://generativelanguage.googleapis.com/v1beta/openai/` | `https://api.groq.com/openai/v1` |

There is only **one** `LANGCHAIN_PROVIDER` — upstream has no primary/fallback chain. The
Gemini-primary/Groq-fallback behaviour in CLAUDE.md must be built in our own Phase 9 gateway.

`vibe-trading provider doctor` reports redacted diagnostics and is the fastest way to confirm
a key is being picked up.

## MT5 connector — determines whether cloud hosting is ever possible

**It is not, with this connector.**

1. **Mechanism:** the Windows-only `MetaTrader5` package, imported in
   `src/trading/connectors/mt5/_client.py`, requiring a running logged-in terminal. The API
   is process-global, so every operation runs inside a locked
   initialize → verify → work → shutdown session. No REST or cloud transport exists anywhere
   in the package — `MetaApiAdapter` in our structure is ours to write from scratch.
2. **Config:** not `.env` but a separate `~/.vibe-trading/mt5.json` — `login`, `password`,
   `server`, `terminal_path`, `profile`, `symbol_suffix`, `deviation_points`,
   `max_order_volume`, `max_order_notional_usd`.
3. **Exness demo:** supported by name. Upstream uses `"Exness-MT5Trial8"` as its server
   example and documents `symbol_suffix: "m"` → `EURUSDm` — the suffix trap in Appendix C of
   the build plan is already solved there.

### Worth adopting in Phase 6

Upstream's identity guard re-reads `account_info().trade_mode` from the terminal on **every
session** and pins the configured login, hard-rejecting a real-money account attached to a
paper profile (and vice versa; contest accounts are rejected everywhere). That enforces
CLAUDE.md hard rule 1 structurally rather than by convention, and is stronger than anything
currently specified in our plan.

## Finnhub is not what the build plan assumes

The pre-flight checklist lists Finnhub as the economic-calendar source. Upstream's
`backtest/loaders/finnhub_loader.py` is **US-equity daily candles only** (`AAPL.US` → `AAPL`).
It provides neither FX bars nor the calendar. The high-impact-event auto-revoke trigger in
Phase 6 needs a direct Finnhub calendar call that we write ourselves.
