-- Refuse to store a trade in LIVE mode. Hard rule 1, at the layer that cannot be refactored
-- around.
--
-- The enum in `trades_mode_known` (0005) deliberately still permits 'LIVE'. Keeping the value
-- means a future v2 needs no type change and no data migration of existing rows — it needs
-- exactly one line: `alter table trades drop constraint trades_no_live_mode;`
--
-- That line is the point. Going live then becomes something a person writes on purpose, in a
-- reviewed migration, with a commit message explaining themselves. Compare the alternative,
-- where the only guard is a Python `if` that a plausible-looking refactor deletes without
-- anyone noticing until real money moves.
--
-- Separate from trades_mode_known rather than folded into it, so the diff that lifts the ban
-- touches only the ban and the enum stays intact.

alter table trades
    add constraint trades_no_live_mode check (mode <> 'LIVE');

comment on constraint trades_no_live_mode on trades is
    'CLAUDE.md hard rule 1: demo accounts only. Dropping this constraint is how v2 enables '
    'live trading — it must be a deliberate, reviewed migration, never an app-layer change.';
