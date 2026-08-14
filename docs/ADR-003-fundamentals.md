# ADR-003: Calendar and central bank sources

**Status:** accepted (Phase 9)

## Decision

Take the economic calendar from **faireconomy's keyless weekly JSON**, and central bank
communications from the **four banks' own RSS feeds**. Do not use the MQL5-backed News API.

## The calendar API that was specified, and why it is not used

The phase card asked for "Forex Factory's free News API (MQL5-backed, GET endpoints, no key)".
That service exists, but three of those four descriptors are wrong, and the fourth is the one
that rules it out.

It is **jblanked.com**, not Forex Factory. It is advertised in a Forex Factory forum thread,
which is where the association comes from, but it is a third party aggregating MQL5, Forex
Factory and FxStreet. Its endpoints are real and MQL5-backed:

```
GET /news/api/mql5/calendar/{today,week,range}/
GET /news/api/forex-factory/calendar/{today,week,range}/
GET /news/api/fxstreet/calendar/{today,week,range}/
```

**It requires a key**, contrary to the card. Verified rather than assumed — both
`/news/api/mql5/calendar/week/` and `/news/api/forex-factory/calendar/week/` return **HTTP 401**
unauthenticated. Authentication is `Authorization: Api-Key <key>`.

**Its free tier is one request per day.** That is the disqualifying fact. The pipeline runs
hourly, so a free key buys 1 of the 24 fetches we need, and the other 23 fail. No amount of
caching fixes a calendar that cannot be refreshed on the day its numbers land.

So the requirement that survives is "no key", and only faireconomy satisfies it.

## What was verified, 14 Aug 2026

```
https://nfs.faireconomy.media/ff_calendar_thisweek.json     200   73 events
https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json 000   does not resolve
```

The `cdn-` host appears in several write-ups and is dead; `nfs.` is live. Three properties of
the live response shape the implementation:

**A User-Agent is mandatory.** Without one the host answers **429**, not 401 — which reads as
rate limiting and sends you tuning poll intervals. Measured both ways against the same URL.
`.env.example` already carried `CALENDAR_USER_AGENT` for this reason.

**It rate-limits real repeats too.** A second fetch seconds later returned 429 even with a
User-Agent. Hourly polling is far inside that, but the retry path backs off and honours
`Retry-After` rather than looping.

**There is no `actual` field.** Not "usually empty" — the union of keys across all 73 events is
exactly `country, date, forecast, impact, previous, title`, and 36 of those events were already
in the past. This is the finding with consequences: **this source can never produce a surprise
score.** `surprise_score()` is implemented and tested anyway, because it is the contract a
result-bearing source plugs into, and because writing it later against a live feed means
debugging the arithmetic and the integration at the same time.

**Only the current week exists.** `lastweek`, `nextweek`, `thismonth` and `thisyear` all 404.
No backfill, and no lookahead past Sunday — which is what
`EventRepository.latest_publication_time` is for: when the feed falls behind, the permission
layer must fail closed rather than read an empty calendar as safety.

`country` holds a currency code (`USD`, `JPY`), not a country. The store's
`events_currency_shape` constraint already assumed this.

## Central bank feeds

Four verified, and two are not the URL you would guess:

| Bank | Currency | Feed | Note |
|---|---|---|---|
| Fed | USD | `federalreserve.gov/feeds/press_monetary.xml` | |
| ECB | EUR | `ecb.europa.eu/rss/press.html` | `.html` path, serves `application/rss+xml` |
| BoE | GBP | `bankofengland.co.uk/rss/news` | the long-published `boeapps/rss/feeds.aspx` now 404s |
| BoJ | JPY | `boj.or.jp/en/rss/whatsnew.xml` | not `/rss/whatsnew_en.xml`, which 404s |

All four are RSS 2.0 carrying `pubDate` in RFC-2822, across four different offsets (GMT, +0200,
+0100, +0900). That field is why this source is point-in-time safe without any inference: a
press release states when it became public, so `publication_time_utc` is read from the feed
rather than stamped at fetch time, and a backfill after an outage lands items with their true
original timestamps instead of all appearing to arrive at once.

## Publication time for the calendar, which has none

The calendar feed carries no publication metadata, so the only defensible stamp is **the fetch
time**: we learned of this event now. Using the event's own date would claim we knew Friday's
payrolls number on Friday when the row was first seen on Monday — look-ahead, and permanent,
because `upsert_many` never rewrites `publication_time_utc` on conflict.

The consequence is worth stating plainly: **an event's first sighting fixes its visibility
forever.** That is the correct direction to be wrong in. It means a backfill cannot retroactively
make events visible earlier than we knew them, and it means the calendar's usefulness for
backtests grows only as the store accumulates its own history.

## A hazard for whoever adds a source with results

`upsert_many` updates `actual` on an existing row while leaving `publication_time_utc` at its
original value. For a source that publishes a scheduled event first and its result later, that
combination leaks: a row published Monday, updated Friday with the number, is visible from
Monday **with Friday's actual attached**.

This is latent today only because no current source carries `actual`. Before wiring one, the
`actual` and `surprise_score` columns need gating on `event_time_utc` at read time — the row may
be visible early because the *schedule* was public, but the *value* was not. That is a change to
`events_visible_at()`, not to this package.

## Rate differentials are injected, never defaulted

`carry_divergence` reads the sign of `rate_differential` to choose long or short. A hardcoded
table here goes stale on a schedule this package has no feed for, and a stale constant becomes a
live position in the wrong direction — so there is no default table. A missing or
older-than-45-days rate yields `None`, and the caller decides. Fabricating plausible policy
rates would have been the worst available option: it would look like data and behave like a coin
flip.

## When to revisit

- A paid jblanked plan, or any source carrying `actual`, makes surprise scoring reachable — read
  the hazard section above first.
- A feed offering more than the current week would make calendar history available for backtests
  without waiting for the store to accumulate it.
- If faireconomy's shape changes, `tests/fundamentals/test_calendar.py -m network` is the test
  that says so.
