# ADR-001: SQLAlchemy + asyncpg for the Supabase store

**Status:** accepted (Phase 5.5)

## Decision

Talk to Supabase Postgres directly with **SQLAlchemy 2.x Core over asyncpg**. Do not use
`supabase-py`.

## Why not supabase-py

`supabase-py` is a client for the Supabase *platform* — PostgREST for queries, GoTrue for auth,
all over HTTPS. It is built for browser and edge clients that must not hold a database
credential. None of our three services are that; all are trusted server-side processes.

Five things it cannot do that this project needs:

1. **Be tested against a plain Postgres container.** This is decisive. `supabase-py` needs
   PostgREST running to answer a query, so testing means standing up the whole Supabase stack,
   or testing against the live project. Both are worse than a `pgvector/pgvector:pg16`
   container. SQLAlchemy speaks the wire protocol, so the test container and production behave
   identically.
2. **Transactions.** PostgREST has no multi-statement transaction. Writing an evaluation and
   the trade it produced atomically — either both land or neither does — is not expressible.
3. **Connection pooling.** PostgREST is stateless HTTP. There is no pool to size, and no
   `pool_pre_ping` to survive a dropped idle connection on a home internet link.
4. **Point-in-time filters in SQL** (hard rule 6). The events gate is a SQL function; over
   REST it would be a hand-rolled RPC wrapper, or a filter reimplemented in every caller.
5. **pgvector.** Nearest-neighbour search with an hnsw index and a `WHERE` clause on outcome
   resolution is a query, not a REST filter.

The cost is that `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are not enough — a Postgres DSN is
needed as well, as `SUPABASE_DB_URL`. Supabase > Settings > Database has it.

## Consequences

**Use the Session pooler (port 5432 on `*.pooler.supabase.com`) unless the host has IPv6.**
Supabase's direct database host is IPv6-only. Oracle Always Free ARM instances do not reliably
have IPv6, so the direct string will fail there while working on a developer laptop.

**Port 6543 is the Transaction pooler and behaves differently.** It is pgbouncer in transaction
mode: prepared statements do not survive between statements. `DatabaseConfig` detects the port
and disables asyncpg's statement cache automatically, so both work, but the session pooler is
preferred.

**Migrations are numbered SQL files, not Alembic.** The DDL that runs in tests is byte-identical
to what would be pasted into Supabase's SQL editor. Alembic's autogenerate does not understand
hnsw indexes, partial indexes or the check constraints this schema leans on, so the files would
be hand-written either way — leaving Alembic as a version table and machinery around it.

**The service key is never used by this layer.** It authenticates to PostgREST, not to Postgres.
It stays out of the store entirely, which removes one way to leak it.

## Revisit if

The dashboard grows a browser-side query path. That client *should* use `supabase-py` with the
anon key and Row Level Security, precisely because it must not hold a database credential. That
is a different layer, not a reversal of this decision.
