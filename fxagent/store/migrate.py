"""Migration runner: applies the numbered SQL files in `migrations/` exactly once, in order.

Plain SQL files rather than Alembic, deliberately:

* The DDL that runs in tests is byte-identical to the DDL pasted into Supabase's SQL editor.
  An ORM-generated migration is a translation, and translations of `create index ... using
  hnsw (embedding vector_cosine_ops)` are where pgvector setups quietly lose their index.
* Alembic's autogenerate does not understand pgvector index types, partial indexes or the
  check constraints this schema leans on, so the files would be hand-written regardless —
  at which point Alembic is a version table and a lot of machinery around it.

Applied versions are recorded in `schema_migrations`, and each file runs inside its own
transaction. Postgres has transactional DDL, so a failure mid-file leaves nothing behind.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

__all__ = ["Migration", "MigrationError", "apply_migrations", "load_migrations"]

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_FILENAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")

_CREATE_VERSION_TABLE = """
create table if not exists schema_migrations (
    version     text        primary key,
    name        text        not null,
    checksum    text        not null,
    applied_at  timestamptz not null default now()
)
"""


class MigrationError(RuntimeError):
    """A migration could not be loaded or applied."""


@dataclass(frozen=True)
class Migration:
    """One versioned SQL file."""

    version: str
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        """SHA-256 of the file body, so an edited applied migration is detectable."""
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return f"Migration(version={self.version!r}, name={self.name!r})"


def load_migrations(directory: Path | None = None) -> list[Migration]:
    """Read and sort every migration file. Rejects anything misnamed rather than skipping it."""
    source = directory or MIGRATIONS_DIR
    if not source.is_dir():
        raise MigrationError(f"migrations directory not found: {source}")

    migrations: list[Migration] = []
    seen: dict[str, str] = {}
    for path in sorted(source.glob("*.sql")):
        match = _FILENAME.match(path.name)
        if match is None:
            raise MigrationError(
                f"migration {path.name!r} is misnamed; expected NNNN_lower_snake_name.sql"
            )
        version, name = match.group(1), match.group(2)
        if version in seen:
            raise MigrationError(
                f"duplicate migration version {version}: {seen[version]!r} and {path.name!r}"
            )
        seen[version] = path.name
        migrations.append(
            Migration(version=version, name=name, sql=path.read_text(encoding="utf-8"))
        )

    if not migrations:
        raise MigrationError(f"no migration files found in {source}")
    return migrations


async def apply_migrations(
    connection: AsyncConnection,
    *,
    directory: Path | None = None,
    verify_checksums: bool = True,
) -> list[Migration]:
    """Apply every migration not yet recorded. Returns the ones actually applied.

    Idempotent: calling it twice applies nothing the second time, which is what lets the
    collector and analyst both run it on startup without coordinating.
    """
    migrations = load_migrations(directory)

    await connection.execute(text(_CREATE_VERSION_TABLE))
    await connection.commit()

    rows = await connection.execute(text("select version, checksum from schema_migrations"))
    applied = {row.version: row.checksum for row in rows}

    if verify_checksums:
        _verify(migrations, applied)

    freshly_applied: list[Migration] = []
    for migration in migrations:
        if migration.version in applied:
            continue
        logger.info("applying migration %s_%s", migration.version, migration.name)
        try:
            # Postgres DDL is transactional: a failure here rolls the whole file back.
            await _execute_script(connection, migration.sql)
            await connection.execute(
                text(
                    "insert into schema_migrations (version, name, checksum) "
                    "values (:version, :name, :checksum)"
                ),
                {
                    "version": migration.version,
                    "name": migration.name,
                    "checksum": migration.checksum,
                },
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            logger.error("migration %s_%s failed", migration.version, migration.name)
            raise
        freshly_applied.append(migration)

    if not freshly_applied:
        logger.debug("schema already up to date (%d migrations)", len(migrations))
    return freshly_applied


async def _execute_script(connection: AsyncConnection, sql: str) -> None:
    """Run a multi-statement SQL file.

    asyncpg sends every statement as a *prepared* statement, and Postgres rejects a prepared
    statement containing more than one command ("cannot insert multiple commands into a
    prepared statement"). Every migration here is several commands — a table plus its indexes.

    Splitting on `;` is not an option: 0003 defines a function whose body is dollar-quoted, and
    a naive split would cut it in half. asyncpg's own `execute()` uses the simple query
    protocol, which accepts a whole script, so the driver connection is used directly. It is
    the same physical connection, so this still runs inside the transaction SQLAlchemy opened
    and `connection.commit()` below still governs it.
    """
    raw = await connection.get_raw_connection()
    driver = raw.driver_connection
    if driver is None:  # pragma: no cover - only on a non-asyncpg driver
        raise MigrationError("could not reach the underlying asyncpg connection")
    await driver.execute(sql)


def _verify(migrations: list[Migration], applied: dict[str, str]) -> None:
    """Refuse to run if an already-applied migration has been edited since.

    Editing an applied migration means the database and the files disagree, and every
    environment that already ran it silently keeps the old schema. Fixing it forward with a new
    file is the only safe move, so this fails loudly rather than papering over the difference.
    """
    for migration in migrations:
        recorded = applied.get(migration.version)
        if recorded is not None and recorded != migration.checksum:
            raise MigrationError(
                f"migration {migration.version}_{migration.name} has changed since it was "
                "applied. Databases that already ran it still hold the old schema. Add a new "
                "migration with the correction instead of editing this one."
            )
