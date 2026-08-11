"""Durable LangGraph checkpointer backed by Postgres (psycopg async).

The orchestrator's primary SQLAlchemy engine uses asyncpg, but LangGraph's
official Postgres saver connects over psycopg. This module bridges the two: it
derives a psycopg DSN from ``DATABASE_URL`` and manages an ``AsyncPostgresSaver``
inside a connection pool that lives for the process lifetime.

When the database is unreachable (offline tests, mock mode) the engine degrades
to an in-memory ``MemorySaver`` so the graph still runs — with the caveat that
checkpoints and HITL interrupts are then process-local.
"""

import asyncio
from functools import lru_cache

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = structlog.get_logger(__name__)


def _psycopg_dsn(database_url: str) -> str:
    """Convert a SQLAlchemy async DSN to a psycopg async connection string."""
    for prefix in ("postgresql+asyncpg://", "postgres+asyncpg://"):
        if database_url.startswith(prefix):
            return "postgresql://" + database_url.split("://", 1)[1]
    if database_url.startswith(("postgresql://", "postgres://")):
        return database_url
    return database_url


class CheckpointEngine:
    """Holds a process-lifetime checkpointer, defaulting to memory on failure."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = _psycopg_dsn(dsn) if dsn else None
        self._saver: BaseCheckpointSaver | None = None
        self._pool = None
        self._mode: str = "uninitialized"
        self._lock = asyncio.Lock()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def saver(self) -> BaseCheckpointSaver:
        """Return the current saver, falling back to memory if never initialised."""
        if self._saver is None:
            return MemorySaver()
        return self._saver

    async def setup(self) -> None:
        """Connect to Postgres (once). Safe to call repeatedly and from tests."""
        if self._mode != "uninitialized":
            return
        async with self._lock:
            if self._mode != "uninitialized":
                return
            if self._dsn is None:
                self._mode = "memory"
                self._saver = MemorySaver()
                return
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from psycopg_pool import AsyncConnectionPool

                pool = AsyncConnectionPool(
                    conninfo=self._dsn,
                    open=False,
                    timeout=3.0,
                    kwargs={"autocommit": True},
                )
                await pool.open()
                await pool.wait()
                saver = AsyncPostgresSaver(pool)
                await saver.setup()
                self._pool = pool
                self._saver = saver
                self._mode = "postgres"
                logger.info("checkpoint_engine_initialized", mode="postgres")
            except Exception as exc:
                logger.warning(
                    "checkpoint_engine_fallback_memory",
                    error=str(exc)[:300],
                )
                self._mode = "memory"
                self._saver = MemorySaver()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._mode = "closed"


@lru_cache
def get_checkpoint_engine() -> CheckpointEngine:
    from orchestrator.config import settings

    return CheckpointEngine(dsn=settings.DATABASE_URL)
