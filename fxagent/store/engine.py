"""Async engine and session construction, with pooling sized for three small services.

`Database` owns one engine per process. An engine is a pool, not a connection — creating one
per request exhausts Supabase's connection allowance quickly, which on the free tier is low
enough to matter.

Sessions are handed out through `session()` / `begin()` context managers so a caller cannot
leak one. Nothing outside this module constructs a session directly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker as _async_sessionmaker
from sqlalchemy.pool import NullPool

from fxagent.store.config import DatabaseConfig
from fxagent.store.retry import RetryPolicy, with_retry

__all__ = ["Database"]

logger = logging.getLogger(__name__)


def _prepared_statement_name() -> str:
    """Unique prepared-statement names, required behind pgbouncer.

    In transaction pooling mode consecutive statements can land on different backends, so
    asyncpg's default sequential names (`__asyncpg_stmt_1__`) collide across sessions sharing
    a backend and fail with "prepared statement already exists".
    """
    return f"__asyncpg_{uuid4().hex}__"


class Database:
    """Owns the engine and hands out sessions. One instance per process."""

    def __init__(self, config: DatabaseConfig, *, retry: RetryPolicy | None = None) -> None:
        self._config = config
        self._retry = retry or RetryPolicy()
        self._engine = _build_engine(config)
        self._sessionmaker = _async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,  # attributes stay readable after commit
            autoflush=False,
        )
        logger.info("database engine created for %s", config.safe_url)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **overrides: Any) -> Database:
        return cls(DatabaseConfig.from_env(env, **overrides))

    @property
    def config(self) -> DatabaseConfig:
        return self._config

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry

    def __repr__(self) -> str:
        """Redacted — `config`'s repr masks the password."""
        return f"Database({self._config!r})"

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A session with no transaction opened. Caller commits."""
        async with self._sessionmaker() as session:
            yield session

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[AsyncSession]:
        """A session inside a transaction, committed on clean exit and rolled back on error.

        This is what makes "write the evaluation and its trade, or neither" expressible — the
        reason the store speaks Postgres directly rather than going over PostgREST.
        """
        async with self._sessionmaker() as session, session.begin():
            yield session

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """A raw Core connection, for the migration runner."""
        async with self._engine.connect() as connection:
            yield connection

    async def run(self, operation: Any, *, description: str = "database operation") -> Any:
        """Run a zero-argument coroutine factory under the configured retry policy."""
        return await with_retry(operation, policy=self._retry, description=description)

    async def dispose(self) -> None:
        """Close every pooled connection. Call on shutdown."""
        await self._engine.dispose()
        logger.info("database engine disposed")


def _build_engine(config: DatabaseConfig) -> AsyncEngine:
    kwargs: dict[str, Any] = {
        "echo": False,
        # Verifies a connection before handing it out. Costs one round trip and removes the
        # single most common failure on a home connection: a pooled socket that died while
        # idle, surfacing as an error on the next unrelated query.
        "pool_pre_ping": True,
        "connect_args": dict(config.connect_args),
    }

    if config.uses_transaction_pooler:
        # pgbouncer is already the pool. A second pool in front of it holds server-side slots
        # open for no benefit, so hand connections straight through.
        kwargs["poolclass"] = NullPool
        kwargs["connect_args"]["prepared_statement_name_func"] = _prepared_statement_name
    else:
        kwargs["pool_size"] = config.pool_size
        kwargs["max_overflow"] = config.max_overflow
        kwargs["pool_timeout"] = config.pool_timeout_seconds
        # Supabase reaps idle connections; recycle before it does so the reap is never observed.
        kwargs["pool_recycle"] = config.pool_recycle_seconds

    return create_async_engine(config.url, **kwargs)
