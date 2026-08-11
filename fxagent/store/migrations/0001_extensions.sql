-- Extensions the schema depends on.
--
-- pgvector is preinstalled but not enabled on Supabase; enabling is idempotent and safe to
-- re-run. On a bare postgres container the image must already ship the extension binaries
-- (pgvector/pgvector:pg16), otherwise this fails loudly here rather than at first query.

create extension if not exists vector;
