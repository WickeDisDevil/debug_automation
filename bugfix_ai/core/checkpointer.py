"""Async checkpointer factory for the LangGraph state machine.

Why a checkpointer matters for this product:
  The graph has three `interrupt_before` points (consent_gate,
  pre_execute_review, hitl_checkpoint). At each interrupt the run
  pauses indefinitely while we wait for a human decision via the API.
  Without a checkpointer, "indefinitely" would mean "until the worker
  process dies" — restarts would lose every in-flight session. With
  one, state lives in Postgres / SQLite and a fresh worker can resume
  any thread by id. The checkpointer also persists `capture_events`
  for Lane A (silent observer), so a worker restart mid-capture does
  not lose the trace built up so far.

Why two backends:
  * postgres — production. AsyncPostgresSaver wraps psycopg's async
    pool so we get connection pooling for free.
  * sqlite — local dev and tests. AsyncSqliteSaver uses aiosqlite,
    so no synchronous thread-crossing happens; the same async
    event-loop owns the file handle.

Lifecycle pattern:
  This module exposes ONE public API: `open_checkpointer()`, an async
  context manager. It yields a `BaseCheckpointSaver` that has been
  `setup()`-ed (creating tables / running migrations) on the current
  event loop. FastAPI's lifespan hook in `core.app_factory` is the
  expected caller — it builds the compiled graph once and stashes it
  on `app.state` for the lifetime of the worker.

DSN translation:
  LangGraph's PostgresSaver wants a libpq-style DSN, not SQLAlchemy's
  "+asyncpg" variant — `_to_psycopg_url` strips the dialect tag so
  one settings value (`postgres_url`) can feed both consumers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    """Yield a fully initialized async checkpointer.

    Use as:
        async with open_checkpointer() as cp:
            graph = build_graph(cp)
            ...

    The context manager handles connection setup AND graceful close.
    """
    settings = get_settings()

    if settings.checkpointer_type == "sqlite":
        async with AsyncSqliteSaver.from_conn_string(settings.sqlite_db_path) as cp:
            await cp.setup()
            log.info("checkpointer.ready", backend="sqlite", path=settings.sqlite_db_path)
            yield cp
        return

    # Postgres path. AsyncPostgresSaver wraps psycopg's async pool.
    pg_url = _to_psycopg_url(settings.postgres_url)
    async with AsyncPostgresSaver.from_conn_string(pg_url) as cp:
        await cp.setup()
        log.info("checkpointer.ready", backend="postgres")
        yield cp


def _to_psycopg_url(sqlalchemy_url: str) -> str:
    """LangGraph's PostgresSaver wants a libpq-style DSN, not SQLAlchemy's.

    SQLAlchemy URL: postgresql+asyncpg://user:pw@host:5432/db
    psycopg DSN:    postgresql://user:pw@host:5432/db
    """
    return sqlalchemy_url.replace("+asyncpg", "").replace("+psycopg", "")
