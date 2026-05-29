"""Async PostgreSQL client — structured (rules) leg of hybrid retrieval + bookkeeping.

Why Postgres for this and not Qdrant alone:
  Hybrid retrieval needs TWO independent signals to fuse via RRF:
    * semantic — Qdrant vector ANN (handles fuzzy / paraphrased matches)
    * structured — exact predicates over normalized columns (this module)
  Without the structured leg, retrieval drifts whenever embedding
  similarity disagrees with what's actually a known mechanical match
  (same error_type + same service is a near-certain hit, regardless of
  what the embedding thinks).

Schema (managed by alembic in db/migrations):
  fixes
    Canonical record of a resolved bug. Source of truth.
    Columns:
      id PK uuid, alert_number int, repository text, service text,
      environment text, error_type text, severity text,
      root_cause text, fix_summary text, steps_json jsonb,
      dev_narrative text, resolved_at timestamptz,
      time_to_resolve_min int, ruler_score float, created_at timestamptz
    Read by:    query_by_rules, get_by_id (fix_store.py)
    Written by: store_fix (fix_store.py)

  error_type_baselines
    Aggregated baseline resolution time per error_type (and optionally
    per service). Used by the time-saved metric so we don't fabricate
    comparisons when no baseline exists.
    Columns: error_type PK, avg_resolution_minutes, sample_count, updated_at

  ingestion_seen
    Idempotency table for alert ingestion. Prevents two concurrent
    poll passes (or a poll + a webhook) from starting two graph runs
    for the same alert.
    Columns: provider, alert_number PK, last_state, etag, last_seen_at

Connection management:
  * `_engine()` and `_session_factory()` are `lru_cache`d to give one
    pool per process. pool_size=10, max_overflow=5, pool_pre_ping=True.
  * `session()` is the public async-context manager — every public
    function uses it. Don't reach into the engine directly.

Scoring formula in query_by_rules:
  Score is a weighted blend of (exact-error-type, exact-service,
  exact-env, recency, ruler quality). Exposed inline rather than as a
  Python post-process so Postgres can ORDER BY in-DB and return only
  `:limit` rows.

Safety notes:
  * `stack_pattern` is matched as a substring via `position()`, NOT a
    regex. Accepting arbitrary regex from upstream would expose ReDoS.
  * All inputs are bound parameters — no string interpolation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _engine() -> AsyncEngine:
    s = get_settings()
    return create_async_engine(
        s.postgres_url,
        pool_size=10,
        max_overflow=5,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache(maxsize=1)
def _session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(_engine(), expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    sf = _session_factory()
    async with sf() as s:
        yield s


# ── Rule-based retrieval ────────────────────────────────────────────────────


async def query_by_rules(
    *,
    error_type: str,
    service: str,
    environment: str | None = None,
    stack_pattern: str | None = None,
    limit: int = 20,
    max_age_days: int | None = 365,
) -> list[dict[str, Any]]:
    """Structured retrieval: matches on error_type/service/env, scored by recency.

    `stack_pattern` is matched as a substring (Postgres `position`); we don't
    accept arbitrary regex from upstream because that can DoS the DB.
    """
    sql = """
        SELECT
            id::text AS fix_id,
            alert_number,
            service,
            environment,
            error_type,
            severity,
            root_cause,
            fix_summary,
            ruler_score,
            EXTRACT(EPOCH FROM (now() - resolved_at)) / 86400.0 AS age_days,
            -- Score combines exactness + recency + ruler quality
            (
                (CASE WHEN error_type = :error_type THEN 1.0 ELSE 0.0 END) * 0.5
              + (CASE WHEN service    = :service    THEN 1.0 ELSE 0.0 END) * 0.2
              + (CASE WHEN environment = :environment THEN 1.0 ELSE 0.0 END) * 0.1
              + GREATEST(0.0, 1.0 - EXTRACT(EPOCH FROM (now() - resolved_at)) / (180.0 * 86400.0)) * 0.1
              + COALESCE(ruler_score, 0.5) * 0.1
            ) AS score
        FROM fixes
        WHERE
            (error_type = :error_type OR service = :service)
            AND (:max_age_days IS NULL OR resolved_at >= now() - (:max_age_days || ' days')::interval)
            AND (:stack_pattern IS NULL OR position(:stack_pattern in coalesce(root_cause,'')) > 0
                                       OR position(:stack_pattern in coalesce(fix_summary,'')) > 0)
        ORDER BY score DESC
        LIMIT :limit
    """
    async with session() as s:
        rows = (
            await s.execute(
                text(sql),
                {
                    "error_type": error_type,
                    "service": service,
                    "environment": environment or "",
                    "stack_pattern": stack_pattern,
                    "max_age_days": max_age_days,
                    "limit": limit,
                },
            )
        ).mappings().all()
    return [dict(r) for r in rows]


# ── Baselines for time_saved metric ─────────────────────────────────────────


async def get_avg_resolution_time(error_type: str, service: str | None = None) -> float:
    """Get baseline resolution time, controlling for service when possible.

    Returns 0.0 if no baseline exists (caller treats as 'no comparison'
    rather than fabricating a metric).
    """
    sql = """
        SELECT avg_resolution_minutes
        FROM error_type_baselines
        WHERE error_type = :et AND (service = :svc OR service IS NULL)
        ORDER BY (service IS NOT NULL) DESC
        LIMIT 1
    """
    async with session() as s:
        row = (
            await s.execute(text(sql), {"et": error_type, "svc": service})
        ).first()
    return float(row[0]) if row else 0.0


# ── Idempotency for ingestion ───────────────────────────────────────────────


async def has_seen_alert(provider: str, alert_number: int) -> bool:
    sql = """
        SELECT 1 FROM ingestion_seen
        WHERE provider = :p AND alert_number = :n
        LIMIT 1
    """
    async with session() as s:
        row = (await s.execute(text(sql), {"p": provider, "n": alert_number})).first()
    return row is not None


async def mark_alert_seen(provider: str, alert_number: int, state: str, etag: str = "") -> None:
    sql = """
        INSERT INTO ingestion_seen (provider, alert_number, last_state, etag, last_seen_at)
        VALUES (:p, :n, :s, :e, now())
        ON CONFLICT (provider, alert_number)
        DO UPDATE SET last_state = excluded.last_state, etag = excluded.etag, last_seen_at = now()
    """
    async with session() as s:
        await s.execute(text(sql), {"p": provider, "n": alert_number, "s": state, "e": etag})
        await s.commit()
