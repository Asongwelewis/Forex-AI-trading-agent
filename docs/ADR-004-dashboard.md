# ADR-004: The dashboard is a read-only window with one server-side poll

**Status:** accepted (Phase 10)

## Decision

A FastAPI service serving a vanilla front end and a self-hosted copy of TradingView's
Lightweight Charts. Two panes: our bars with our own overlays on the left, one entry per
consensus evaluation on the right. Browsers receive updates over a WebSocket. Every HTTP route
is a GET. It runs in its own container and binds `0.0.0.0`.

## Read-only, and mechanically so

This is the only process in the system meant to be reachable from another machine, and it has
no authentication. The mitigation is not a login — it is that there is nothing behind it to
attack. No route mutates, no repository method that writes is reachable from
`fxagent/dashboard/`, and `Database.session()` is used rather than `begin()` so there is not
even an open transaction to commit.

`tests/dashboard/test_app.py::test_no_route_can_change_anything` walks the route table and
asserts every method is in `{GET, HEAD, OPTIONS}`. That test is the actual security boundary;
this paragraph is a description of it. A single convenient POST would erase the property
permanently and silently, and a sentence in a README does not catch that.

Approving a trade from a phone is a real feature and it is **not** this one. It belongs on
Telegram, where the sender's chat id is checked against a configured value, per CLAUDE.md's
Phase 8 note. An unauthenticated LAN page is not an approval channel.

## WebSocket push, and one poll that is honest about itself

The browser opens one socket, receives a snapshot immediately, and receives another only when
the content hash of that view changes. There is no interval in the client.

The server does poll. Nothing notifies this process when a row lands: the writers are
short-lived GitHub Actions jobs, so there is no in-process event to hook, and Postgres
`LISTEN` is not usable through Supabase's transaction pooler — consecutive statements land on
different backends and no connection is held open to listen on.

So one background task rebuilds each *subscribed* view every `FX_DASHBOARD_REFRESH_SECONDS`
(15 by default) and pushes only when `revision` moves. Ten browsers on one symbol cost one
rebuild. A quiet hour costs zero messages.
`tests/dashboard/test_live.py::test_an_unchanged_view_pushes_nothing` is what stops that
degrading into a poll with the direction reversed.

`revision` is a SHA-256 over the chart and feed payloads and deliberately **excludes**
`generated_at` — a hash covering the build time changes on every rebuild by definition, and the
comparison it feeds would never be equal.

## Every number on the chart is computed by the code the strategies read

The EMA and the Bollinger bands come from `fxagent.indicators`. The Asian range comes from
`session_breakout.asian_range`, which was extracted from a private method for exactly this
reason. The session shading comes from `regime.sessions.session_bounds_utc`.

The alternative — drawing any of it in JavaScript — was available and is worse than not drawing
it at all. A front end with its own idea of "the London session" would agree with the system
until somebody tuned one of them, and the picture is the thing that gets believed. London
shades 08:00–17:00 UTC in January and 07:00–16:00 in July because `zoneinfo` says so, and
`test_london_shading_follows_daylight_saving` is the assertion that keeps it that way.

Bollinger bands were added to `fxagent/indicators/volatility.py` for this phase. They use the
**population** deviation (ddof=0), which is Bollinger's convention and differs from
`rolling_zscore`'s sample deviation (ddof=1). The two are within 3% at period 20, which is
precisely why the difference is asserted rather than assumed.

## Three panel sections have no producer yet

The agent narrations, the retrieved analogues and the candle formations are read from reserved
keys inside the `evaluations.votes` JSONB document — `agents` and `patterns`, laid out in
`fxagent/dashboard/contract.py`. Nothing writes them today.

Two alternatives were rejected. A migration adding columns for an agent layer that does not
exist would be designing storage for an unbuilt producer. Leaving the sections out entirely
would mean the panel had to be rebuilt when the agents land, and the contract they must satisfy
would go unstated until then.

So the panel reads the keys, renders nothing when they are absent, and **says** that the
producer is absent rather than showing an empty section that looks like a quiet one. A block
that is present and fails validation is discarded whole, named in `discarded`, and reported on
screen — hard rule 5 applied on the read side, because a half-rendered narration is more
misleading than a missing one.

Grant state is the same shape: `fxagent.permission` is a stub, so `AdvisoryOnly` reports
ADVISORY with the reason spelled out, and a reader that raises degrades to ADVISORY. There is
no path through `grant.py` that reports GRANTED because something was unavailable.

## The charting library is vendored and pinned

Lightweight Charts v4.2.3, Apache 2.0, committed under
`fxagent/dashboard/static/vendor/` and verified against a SHA-256 in
`fxagent/dashboard/vendored.py`. No CDN: a dashboard that cannot render without reaching the
internet goes blank exactly when the network is what you are debugging.

v4 rather than v5 because v5 moved markers to a `createSeriesMarkers` plugin and changed series
creation; the front end is written against v4, and a major-version move is a change to make
deliberately rather than something a `latest` tag does underneath it.

The vendor directory is marked `binary` in `.gitattributes`. Without that, `* text=auto` would
rewrite the file's line endings on a Windows checkout and the pinned hash would fail on a fresh
clone — a failure that looks like tampering and is not.

## Consequences

**`fastapi` and `uvicorn` are an optional extra, not a base dependency.** The hourly Actions job
installs with `--no-dev` and no extras and never serves a request; it should not carry a web
server and its C extensions. `chart.py`, `feed.py` and `contract.py` import without FastAPI
installed, which is what lets the builders — the part with the logic worth testing — be tested
without one.

**The dashboard being down is not an incident.** It writes nothing, schedules nothing and is
depended on by nothing. A restart costs a page reload. That is why the container's healthcheck
tests liveness only: a database blink should leave the last snapshot on screen with a note, not
restart the process and take the page you were reading with it.

**Adding a write route would invalidate the deployment posture**, not just this ADR. If one is
ever genuinely needed, the authentication has to arrive in the same change, and the port should
stop being open to the LAN.
