# ADR-002: GitHub Actions cron as the runtime

**Status:** accepted (Phase 8)

## Decision

Run the collector, analyst and resolver as **scheduled GitHub Actions jobs**. No VPS, no
always-on container, no in-process scheduler. `.github/workflows/collect-and-analyse.yml`
runs the hourly pass; `.github/workflows/health.yml` checks daily that it is still happening.

## The reframe

The collector and analyst were being designed as daemons because the earlier plan deployed
them to a box. They are not daemons. They wake, work for a minute, and exit. Everything about
the always-on shape — a container to keep alive, an APScheduler loop, a restart policy, a
server whose disk fills up — was solving a problem created by the deployment, not by the work.

Once every entrypoint is `--once` and idempotent (hard rule 8), the entire hosting requirement
collapses to "something that invokes a command on a timer". GitHub already does that, for free,
next to the code, with logs and manual re-runs included.

## Cost

Private repositories get 2,000 free Linux minutes a month; public repositories have no minute
cap. The hourly workflow runs the three stages in one job, at 1–2 minutes per run with the
dependency cache warm.

Running 24/7 is ~744 runs a month, so 750–1,500 minutes — inside the allowance, but not
comfortably, and the daily health job sits on top. FX only trades ~22 days a month, so the
cron can be trimmed to market hours for roughly a 30% saving; the two-line replacement is
written out in a comment in the workflow. Left running 24/7 for now because a collector that
polls through the weekend costs Twelve Data credits and nothing else, and because the gap
between "scheduled but idle" and "not scheduled" is where silent failure hides.

## The four platform constraints this is built around

**1. Scheduled triggers are best-effort.** A run can be delayed 30+ minutes at peak load, and
can be dropped entirely. Two consequences: the cron fires at :07 rather than :00, off the
top-of-hour rush; and no alert timestamp may ever be treated as an execution trigger. This
system does not place orders, so a late recommendation is an inconvenience rather than a bad
fill — the constraint is survivable precisely because of hard rule 1.

**2. Five minutes is the minimum interval.** Irrelevant for H1. A hard floor if M5 is ever
wanted, and the reason M5 would need a different runtime rather than a different cron line.

**3. Inactivity disables schedules.** GitHub disables scheduled workflows on a repository with
60 days of no commit activity. Sources disagree on whether private repositories are included,
so the health workflow warns at 45 days either way — the cost of guessing wrong is the whole
pipeline stopping with no notification. A workflow *run* does not reset the clock; only a
commit does.

**4. Only the default branch is scheduled.** A `schedule:` trigger on `develop` never fires.
Cron does not start until this file reaches `main`, and several people report it not activating
until the first manual run. Hence `workflow_dispatch` on both workflows — it is how the
pipeline is tested before it is scheduled, and how it is kicked into life afterwards.

## Consequences

**A missed hour needs no catch-up path.** The collector re-fetches an overlapping window every
poll and upserts on `bars_unique`, so the next run closes the hole. This is the property that
makes a best-effort scheduler acceptable, and it is worth protecting: any future entrypoint
that cannot be run twice for the same bar breaks the runtime, not just its own tests.

**Concurrency is capped at one run.** `cancel-in-progress: false` — killing a run mid-write to
chase a newer one is strictly worse than letting it finish, and GitHub discards stale queued
runs on its own, which is the behaviour we want.

**Secrets live in GitHub Secrets and are checked before anything executes.** An unset secret is
substituted as the empty string rather than raising, so without a preflight the job dies deep
in a connection attempt with a message describing the symptom. `.github/scripts/check-secrets.sh`
names the missing secret instead, and never prints a value.

**Silence is the failure mode that matters.** Everything else in this system announces itself;
a cron that stops firing does not. So the failure path is part of the deliverable: the hourly
workflow alerts to Telegram on failure, and the daily health workflow reports to Telegram on
*every* run, pass or fail — a check that only speaks up when something is wrong is itself a
thing that can silently stop.

**The health checks do not import `fxagent`.** They use `psql` against the store, `curl`
against Twelve Data's `/api_usage`, and `curl` against Telegram. A monitor that shares a
failure mode with the thing it monitors is not a monitor: an import error in the package would
otherwise take down both the pipeline and the alarm that reports on it.

**Supabase must be reached through the Session pooler.** GitHub-hosted runners have no IPv6
and Supabase's direct database host is IPv6-only. On a laptop the direct string works, which
is what makes this an easy mistake to ship. See ADR-001.

## What this replaces

The Docker/VPS deployment plan, in the form it actually existed in this repo:

- `apscheduler`, a dependency of the always-on shape that no module ever imported.
- The Oracle Always Free ARM framing in ADR-001 and `.env.example`. The IPv4-only advice it
  gave is still correct — for a different host, for the same reason.

`docker-compose.test.yml` stays. It is the pgvector container the store tests run against, not
a deployment artifact.

## When to revisit

- A timeframe below M5, where the interval floor bites.
- A strategy that needs sub-minute reaction, which this system is explicitly not.
- Consistently exceeding the free minute allowance after trimming to market hours.
