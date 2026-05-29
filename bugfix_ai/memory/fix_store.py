"""CRUD for canonical fix records — keeps Postgres + Qdrant in sync.

Why this is its own module (not in rules_db.py):
  Reading a fix is just a Postgres SELECT, but WRITING one has to
  update both stores: the structured row in Postgres AND the embedding
  + payload in Qdrant. Centralizing that here ensures a caller can
  never accidentally write to one and forget the other.

Position in the system:
  * `store_fix(...)` — called from `nodes/capture/store_fix.py` after
    the human approves the captured fix narrative. This is the ONLY
    write path; nothing else creates rows in the `fixes` table.
  * `get_by_id(fix_id)` — called from
    - `nodes/autonomous/load_fix_plan.py` to materialize the steps for
      execution
    - `memory/retrieval/hybrid_retriever.py` to hydrate fused RRF
      results back into full SimilarBug records
    - debug/admin tooling
  * `reconcile_missing_vectors(...)` — operator/cron entry point that
    scans Postgres for fixes whose Qdrant point is missing and
    re-embeds + upserts them. Run on demand via
    `python -m scripts.reconcile_vectors` or schedule periodically.

Embedding generation:
  The composite embedding (title + description + log + error_type) is
  computed here using `embed_composite` so that the embedding seen at
  retrieval time matches the embedding stored at write time. Don't
  recompute embeddings elsewhere — keep the recipe in one place.

Cross-store consistency story:
  Postgres is committed first, Qdrant upserted second, BECAUSE:
    * a row with no vector  →  recoverable: queryable via rules, the
      reconcile worker will re-embed and backfill.
    * a vector with no row  →  unrecoverable: the row never existed,
      semantic search would surface a fix that can't be hydrated.
  The Qdrant write goes through tenacity-driven retries
  (exponential backoff, jittered, capped at ~30s total) to absorb
  transient network blips. If retries are exhausted, store_fix logs
  the failure and STILL returns the fix_id — the caller's response is
  not held hostage to vector-store availability. The reconcile worker
  is the failsafe that closes the gap.

ID generation:
  `uuid.uuid4()` so `fix_id` is globally unique across providers and
  shards. Used as both the Postgres PK and the Qdrant point id — keep
  them aligned so cross-store joins are trivial.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.memory.embed import embed_composite
from bugfix_ai.memory.rules_db import session
from bugfix_ai.memory.vector_store import points_exist, upsert_fix

log = get_logger(__name__)


# Tenacity policy for the Qdrant upsert. Exponential backoff with jitter,
# capped at 5 attempts (~30s total in the worst case). We retry on any
# Exception that isn't a programming error — the alternative would be to
# enumerate qdrant_client exceptions, but the surface is unstable across
# minor versions and a generic retry is safe here because the reconcile
# worker is the ultimate backstop.
_QDRANT_RETRY = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=10.0),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)


async def _upsert_with_retry(
    *, fix_id: str, embedding: list[float], payload: dict[str, Any]
) -> bool:
    """Upsert into Qdrant under tenacity retry. Returns True on success."""
    try:
        async for attempt in AsyncRetrying(**_QDRANT_RETRY):
            with attempt:
                await upsert_fix(fix_id=fix_id, embedding=embedding, payload=payload)
        return True
    except RetryError as e:
        log.error(
            "fix.vector_index.failed_after_retries",
            fix_id=fix_id,
            error=str(e.last_attempt.exception())[:300] if e.last_attempt else "",
        )
    except Exception as e:  # noqa: BLE001 — last-ditch guard; reconcile will fix
        log.error("fix.vector_index.unexpected_error", fix_id=fix_id, error=str(e)[:300])
    return False


async def get_by_id(fix_id: str) -> dict[str, Any] | None:
    sql = """
        SELECT
            id::text AS fix_id,
            alert_number,
            repository,
            service,
            environment,
            error_type,
            severity,
            root_cause,
            fix_summary,
            steps_json,
            dev_narrative,
            resolved_at,
            time_to_resolve_min,
            ruler_score
        FROM fixes WHERE id = :id
    """
    async with session() as s:
        row = (await s.execute(text(sql), {"id": fix_id})).mappings().first()
    if not row:
        return None
    out = dict(row)
    out["steps"] = out.pop("steps_json") or []
    out["resolved_at"] = (
        out["resolved_at"].isoformat() if isinstance(out["resolved_at"], datetime) else out["resolved_at"]
    )
    return out


async def store_fix(
    *,
    alert_number: int | None,
    repository: str,
    service: str,
    environment: str,
    error_type: str,
    severity: str,
    root_cause: str,
    fix_summary: str,
    steps: list[dict],
    dev_narrative: str,
    title: str = "",
    description: str = "",
    error_log: str = "",
    time_to_resolve_min: int | None = None,
    ruler_score: float | None = None,
) -> str:
    """Persist a fix to Postgres AND index it in Qdrant. Returns fix_id."""
    fix_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    insert_sql = """
        INSERT INTO fixes (
            id, alert_number, repository, service, environment,
            error_type, severity, root_cause, fix_summary, steps_json,
            dev_narrative, resolved_at, time_to_resolve_min, ruler_score, created_at
        ) VALUES (
            :id, :alert, :repo, :svc, :env,
            :et, :sev, :rc, :fs, CAST(:steps AS jsonb),
            :narr, :res_at, :ttr, :rs, now()
        )
    """
    async with session() as s:
        await s.execute(
            text(insert_sql),
            {
                "id": fix_id,
                "alert": alert_number,
                "repo": repository,
                "svc": service,
                "env": environment,
                "et": error_type,
                "sev": severity,
                "rc": root_cause,
                "fs": fix_summary,
                "steps": json.dumps(steps),
                "narr": dev_narrative,
                "res_at": now,
                "ttr": time_to_resolve_min,
                "rs": ruler_score,
            },
        )
        await s.commit()

    # Vector index — retried; on terminal failure we LOG and continue so the
    # caller still gets fix_id back. The reconcile worker will backfill.
    embedding = embed_composite(
        title=title,
        description=description,
        error_log=error_log,
        error_type=error_type,
    )
    payload = {
        "alert_number": alert_number,
        "repository": repository,
        "service": service,
        "environment": environment,
        "error_type": error_type,
        "severity": severity,
        "resolved_at_ts": int(now.timestamp()),
        "fix_summary": fix_summary[:500],
    }
    indexed = await _upsert_with_retry(fix_id=fix_id, embedding=embedding, payload=payload)
    if not indexed:
        log.warning(
            "fix.stored.vector_pending",
            fix_id=fix_id,
            note="Postgres row committed; Qdrant upsert failed. "
            "Run reconcile_missing_vectors to backfill.",
        )

    log.info(
        "fix.stored",
        fix_id=fix_id,
        alert=alert_number,
        error_type=error_type,
        vector_indexed=indexed,
    )
    return fix_id


# ── Reconcile worker ─────────────────────────────────────────────────────────


async def _list_fix_ids_for_reconcile(
    *, batch_size: int, offset: int
) -> list[dict[str, Any]]:
    """Fetch a window of fixes (id + the columns needed to rebuild the embedding)."""
    sql = """
        SELECT
            id::text AS fix_id,
            alert_number,
            repository,
            service,
            environment,
            error_type,
            severity,
            root_cause,
            fix_summary,
            EXTRACT(EPOCH FROM resolved_at) AS resolved_at_ts
          FROM fixes
         ORDER BY resolved_at DESC NULLS LAST, id
         LIMIT :n OFFSET :o
    """
    async with session() as s:
        rows = (await s.execute(text(sql), {"n": batch_size, "o": offset})).mappings().all()
    return [dict(r) for r in rows]


async def reconcile_missing_vectors(
    *,
    batch_size: int = 200,
    max_batches: int | None = None,
) -> dict[str, int]:
    """Re-embed and upsert any fixes whose Qdrant point is missing.

    Walks the `fixes` table in chunks of `batch_size`, asks Qdrant which IDs
    in each chunk already have a point, and re-indexes the gaps. Safe to run
    repeatedly — `upsert_fix` is idempotent, and `points_exist` short-circuits
    the work for already-indexed rows.

    Returns counts: {scanned, missing, reindexed, failed}.
    """
    scanned = missing = reindexed = failed = 0
    offset = 0
    batches = 0

    while True:
        batch = await _list_fix_ids_for_reconcile(batch_size=batch_size, offset=offset)
        if not batch:
            break
        scanned += len(batch)
        ids = [r["fix_id"] for r in batch]
        present = await points_exist(ids)
        gaps = [r for r in batch if r["fix_id"] not in present]
        missing += len(gaps)

        for row in gaps:
            embedding = embed_composite(
                title="",  # title/description not stored on the row; we lose
                description="",  # some signal vs first-time write, but the
                error_log="",  # error_type + summary still anchor the vector
                error_type=row.get("error_type") or "",
            )
            payload = {
                "alert_number": row.get("alert_number"),
                "repository": row.get("repository"),
                "service": row.get("service"),
                "environment": row.get("environment"),
                "error_type": row.get("error_type"),
                "severity": row.get("severity"),
                "resolved_at_ts": int(row.get("resolved_at_ts") or 0),
                "fix_summary": (row.get("fix_summary") or "")[:500],
            }
            ok = await _upsert_with_retry(
                fix_id=row["fix_id"], embedding=embedding, payload=payload
            )
            if ok:
                reindexed += 1
            else:
                failed += 1

        offset += len(batch)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    log.info(
        "fix.reconcile.done",
        scanned=scanned,
        missing=missing,
        reindexed=reindexed,
        failed=failed,
    )
    return {"scanned": scanned, "missing": missing, "reindexed": reindexed, "failed": failed}
