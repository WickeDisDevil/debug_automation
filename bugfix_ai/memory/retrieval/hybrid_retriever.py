"""Orchestrates the two retrieval legs and RRF-fuses them into one ranked list.

Position in the graph:
  This is the workhorse called by `nodes/assist/semantic_retrieve.py`
  (and indirectly by anything that needs "find prior fixes that look
  like this bug"). The two upstream legs run CONCURRENTLY:

      ┌──────────────────────────┐
      │ embed_composite(query)   │
      └──────────┬───────────────┘
                 │
        ┌────────┴─────────┐
        ▼                  ▼
  similarity_search    query_by_rules
   (Qdrant ANN +        (Postgres scoring
    payload filter)      formula)
        │                  │
        └────────┬─────────┘
                 ▼
         fuse_results (RRF + staleness decay)
                 ▼
         hydrate via get_by_id
                 ▼
         list[SimilarBug]

Why both legs:
  Each one alone has a known failure mode:
    * Semantic-only — drifts on near-paraphrases of unrelated fixes
      (e.g. two different OOM scenarios with similar wording).
    * Rules-only — misses any fix where the human used different
      vocabulary for the same root cause.
  RRF is the cheapest way to combine them without weight tuning per
  error type.

candidate_multiplier:
  We over-fetch from each leg by 4x before fusion so RRF has enough
  overlap to score reliably. With a 2x multiplier (the original) the
  two lists were often disjoint, which makes RRF degenerate to "first
  list wins". Keep this >= 4.

Staleness decay:
  Applied INSIDE `fuse_results` — see rrf.py. A fix that's >180 days
  old gets its RRF score halved per half-life. This keeps stale
  patterns from dominating (libraries change, services get rewritten,
  what fixed last year may be wrong now).

Hydration:
  Both legs return only `fix_id` + score. We re-fetch the full record
  from Postgres so downstream code can show summary, root_cause,
  steps, time-to-resolve, etc. Fixes that no longer exist in the
  store (deleted? GC'd?) are silently skipped — the list shrinks but
  the call doesn't fail.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import SimilarBug
from bugfix_ai.memory.embed import embed_composite
from bugfix_ai.memory.fix_store import get_by_id
from bugfix_ai.memory.retrieval.rrf import fuse_results
from bugfix_ai.memory.rules_db import query_by_rules
from bugfix_ai.memory.vector_store import similarity_search

log = get_logger(__name__)


async def retrieve_similar_bugs(
    *,
    title: str,
    description: str,
    error_logs: str,
    error_type: str,
    service: str,
    environment: str,
    stack_pattern: str | None = None,
    top_n: int = 5,
    candidate_multiplier: int = 4,
) -> list[SimilarBug]:
    """Run both retrieval paths concurrently, fuse, hydrate, return top N.

    Production fixes:
      - Composite embedding (title + description + error log).
      - Pre-filtered semantic search by service/environment.
      - candidate_multiplier=4 (was 2x in original) so RRF has enough overlap.
      - Staleness decay applied via RRF helper.
      - Concurrent semantic + rules calls.
    """
    query_vec = embed_composite(
        title=title,
        description=description,
        error_log=error_logs,
        error_type=error_type,
    )

    semantic_task = similarity_search(
        embedding=query_vec,
        top_k=top_n * candidate_multiplier,
        service=service,
        environment=environment,
    )
    rules_task = query_by_rules(
        error_type=error_type,
        service=service,
        environment=environment,
        stack_pattern=stack_pattern,
        limit=top_n * candidate_multiplier,
    )
    semantic_raw, rules_raw = await asyncio.gather(semantic_task, rules_task)

    # Normalize age_days into both result lists for the staleness decay
    now_ts = datetime.now(timezone.utc).timestamp()
    for r in semantic_raw:
        ts = r.get("payload", {}).get("resolved_at_ts")
        if ts:
            r["age_days"] = max(0.0, (now_ts - float(ts)) / 86400.0)

    fused = fuse_results(
        semantic_results=semantic_raw,
        rules_results=rules_raw,
        top_n=top_n,
    )
    log.info(
        "hybrid_retrieval.fused",
        n_semantic=len(semantic_raw),
        n_rules=len(rules_raw),
        n_fused=len(fused),
        top_score=fused[0].rrf_score if fused else 0.0,
    )

    # Hydrate
    similar: list[SimilarBug] = []
    for r in fused:
        record = await get_by_id(r.fix_id)
        if not record:
            continue
        similar.append(
            SimilarBug(
                fix_id=r.fix_id,
                alert_number=record.get("alert_number") or 0,
                rrf_score=r.rrf_score,
                semantic_score=r.semantic_score,
                rules_score=r.rules_score,
                summary=record.get("fix_summary", ""),
                root_cause=record.get("root_cause", ""),
                steps=record.get("steps", []),
                resolved_at=record.get("resolved_at", ""),
                age_days=int(r.age_days),
                time_to_resolve_min=int(record.get("time_to_resolve_min") or 0),
            )
        )
    return similar
