"""Migration runner: ordering, idempotency, and refusal to run an edited migration."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from fxagent.store.engine import Database
from fxagent.store.migrate import MigrationError, apply_migrations, load_migrations

from .conftest import requires_postgres

pytestmark = pytest.mark.db


# -- loading (no database) ----------------------------------------------------


def test_migrations_load_in_version_order() -> None:
    migrations = load_migrations()
    versions = [m.version for m in migrations]

    assert versions == sorted(versions)
    assert versions[0] == "0001"
    assert len(set(versions)) == len(versions), "versions must be unique"


def test_extensions_migration_runs_first() -> None:
    """`vector(128)` in 0006 is a syntax error unless the extension already exists."""
    assert load_migrations()[0].name == "extensions"


def test_misnamed_migration_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "not-a-migration.sql").write_text("select 1;", encoding="utf-8")
    with pytest.raises(MigrationError, match="misnamed"):
        load_migrations(tmp_path)


def test_duplicate_version_is_rejected(tmp_path: Path) -> None:
    """Two files claiming 0002 means one silently never runs."""
    (tmp_path / "0002_alpha.sql").write_text("select 1;", encoding="utf-8")
    (tmp_path / "0002_beta.sql").write_text("select 2;", encoding="utf-8")
    with pytest.raises(MigrationError, match="duplicate migration version"):
        load_migrations(tmp_path)


def test_empty_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="no migration files"):
        load_migrations(tmp_path)


def test_checksum_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "0001_thing.sql"
    path.write_text("select 1;", encoding="utf-8")
    before = load_migrations(tmp_path)[0].checksum
    path.write_text("select 2;", encoding="utf-8")
    assert load_migrations(tmp_path)[0].checksum != before


# -- applying (database) ------------------------------------------------------


@requires_postgres
async def test_applying_twice_is_a_no_op(database: Database) -> None:
    """The collector and analyst both migrate on startup without coordinating."""
    async with database.connect() as connection:
        first = await apply_migrations(connection)
        second = await apply_migrations(connection)

    assert second == [], f"second run re-applied {[m.version for m in second]}"
    assert first == [] or all(m.version for m in first)


@requires_postgres
async def test_every_migration_is_recorded(database: Database) -> None:
    async with database.session() as session:
        rows = await session.execute(text("select version from schema_migrations"))
        applied = {row.version for row in rows}

    assert applied == {m.version for m in load_migrations()}


@requires_postgres
async def test_editing_an_applied_migration_is_refused(database: Database, tmp_path: Path) -> None:
    """A database that already ran the old text keeps the old schema; fix forward instead."""
    for migration in load_migrations():
        body = migration.sql if migration.version != "0002" else migration.sql + "\n-- edited\n"
        (tmp_path / f"{migration.version}_{migration.name}.sql").write_text(body, encoding="utf-8")

    async with database.connect() as connection:
        with pytest.raises(MigrationError, match="has changed since it was applied"):
            await apply_migrations(connection, directory=tmp_path)


@requires_postgres
async def test_pgvector_extension_is_installed(database: Database) -> None:
    async with database.session() as session:
        result = await session.execute(
            text("select exists (select 1 from pg_extension where extname = 'vector')")
        )
        assert result.scalar_one() is True


@requires_postgres
async def test_hnsw_index_exists_on_the_embedding_column(database: Database) -> None:
    """ivfflat on an empty table gives poor recall; the migration must build hnsw."""
    async with database.session() as session:
        result = await session.execute(
            text("select indexdef from pg_indexes where indexname = 'windows_embedding_idx'")
        )
        definition = result.scalar_one()

    assert "hnsw" in definition.lower()
    assert "vector_cosine_ops" in definition.lower()


@requires_postgres
async def test_events_visible_at_function_exists(database: Database) -> None:
    async with database.session() as session:
        result = await session.execute(
            text("select exists (select 1 from pg_proc where proname = 'events_visible_at')")
        )
        assert result.scalar_one() is True
