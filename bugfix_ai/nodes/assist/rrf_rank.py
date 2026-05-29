"""Assist mode — fuse semantic + rule recall, decay by age, hydrate the top N.

Position in the graph (assist mode):
  rule_filter → rrf_rank → present_similar → consent_gate

What "RRF" buys us:
  Reciprocal Rank Fusion (`memory.retrieval.rrf.fuse_results`)
  combines two ranked lists into one without needing the per-list
  scores to be on comparable scales. Vectors return cosine similarity
  (~0..1); the rules pass returns its own composite score; trying to
  add or weight those raw values would be brittle. RRF only looks at
  RANK position, so the fusion is robust to score-distribution
  differences.

Recency decay:
  Before fusing, we annotate each semantic candidate with `age_days`
  derived from the `resolved_at_ts` payload field. The RRF stage
  applies an exponential decay (180-day half-life) so a 2-year-old
  fix doesn't outrank a 2-week-old one of equal text similarity.

Hydration:
  The fused list contains only `fix_id`s plus the fused score. We
  fetch the full record from Postgres (`get_by_id`) for each top
  result, then build a `SimilarBug` entry — that's what the consent
  gate UI will render.

Cleanup:
  The transient `_semantic_raw` / `_rules_raw` keys are explicitly
  reset to `[]` so they don't bloat the checkpoint payloads stored
  by the saver. Checkpoints are written on every state mutation,
  so cleaning up large transient lists matters at scale.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bugfix_ai.core.state import BugFixState, SimilarBug
from bugfix_ai.memory.fix_store import get_by_id
from bugfix_ai.memory.retrieval.rrf import fuse_results
from bugfix_ai.observability.decision_logger import log_decision


async def rrf_rank_node(state: BugFixState) -> dict:
    semantic = state.get("_semantic_raw", []) or []  # type: ignore[arg-type]
    rules = state.get("_rules_raw", []) or []  # type: ignore[arg-type]

    # Normalize age into the semantic results from the payload's resolved_at_ts
    now_ts = datetime.now(timezone.utc).timestamp()
    for r in semantic:
        ts = (r.get("payload") or {}).get("resolved_at_ts")
        if ts:
            r["age_days"] = max(0.0, (now_ts - float(ts)) / 86400.0)

    fused = fuse_results(
        semantic_results=semantic,
        rules_results=rules,
        top_n=5,
    )

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

    return {
        "similar_bugs": similar,
        # Clear the staging keys so they don't bloat checkpoint payloads
        "_semantic_raw": [],
        "_rules_raw": [],
        "obs_log": [
            log_decision(
                "rrf_rank",
                f"Fused into {len(similar)} candidates.",
                {"top_rrf": similar[0]["rrf_score"] if similar else 0.0},
            )
        ],
    }
