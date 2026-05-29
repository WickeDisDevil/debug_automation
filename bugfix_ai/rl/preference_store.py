"""Pairwise preference store for DPO/GRPO training.

Two ways preferences enter the store:
  1. Implicit: when a dev REJECTS a presented similar-bug (chooses a different
     candidate or 'manual'), we record (chosen=top_clicked, rejected=top_RRF)
     pairs for the assist-mode policy.
  2. Explicit: RULER's ranking gives us pairwise preferences across completed
     trajectories for the capture/autonomous policy.

We store these in Postgres so they survive restarts and can be exported as a
HuggingFace `datasets` shard at training time.

Schema (created by alembic migration `001_initial_schema.py`):
  preference_pairs(
    pair_id text PK, created_at timestamptz, source text,
    error_type text NULL, run_id text NULL,
    chosen_payload text, rejected_payload text, margin double precision NULL
  )
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.memory.rules_db import session

log = get_logger(__name__)


_ALLOWED_SOURCES = {"user_choice", "ruler", "outcome"}


# ── Public API ──────────────────────────────────────────────────────────────


async def record_preference(
    *,
    source: str,
    chosen_payload: str,
    rejected_payload: str,
    error_type: str | None = None,
    run_id: str | None = None,
    margin: float | None = None,
) -> str:
    """Insert one preference pair. Returns the new pair_id."""
    if source not in _ALLOWED_SOURCES:
        raise ValueError(f"unknown preference source: {source!r}")

    pair_id = uuid.uuid4().hex
    sql = """
        INSERT INTO preference_pairs
            (pair_id, created_at, source, error_type, run_id,
             chosen_payload, rejected_payload, margin)
        VALUES
            (:pair_id, :created_at, :source, :error_type, :run_id,
             :chosen_payload, :rejected_payload, :margin)
    """
    async with session() as s:
        await s.execute(
            text(sql),
            {
                "pair_id": pair_id,
                "created_at": datetime.now(timezone.utc),
                "source": source,
                "error_type": error_type,
                "run_id": run_id,
                "chosen_payload": chosen_payload,
                "rejected_payload": rejected_payload,
                "margin": margin,
            },
        )
        await s.commit()
    log.info("preference.recorded", pair_id=pair_id, source=source, error_type=error_type)
    return pair_id


async def list_pairs(
    *, source: str | None = None, error_type: str | None = None, limit: int = 1000
) -> list[dict]:
    """Read preference pairs back out (for offline export / inspection)."""
    sql = """
        SELECT pair_id, created_at, source, error_type, run_id,
               chosen_payload, rejected_payload, margin
        FROM preference_pairs
        WHERE (:source IS NULL OR source = :source)
          AND (:error_type IS NULL OR error_type = :error_type)
        ORDER BY created_at DESC
        LIMIT :limit
    """
    async with session() as s:
        rows = (
            await s.execute(
                text(sql),
                {"source": source, "error_type": error_type, "limit": limit},
            )
        ).mappings().all()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        ts = d.get("created_at")
        if hasattr(ts, "isoformat"):
            d["created_at"] = ts.isoformat()
        out.append(d)
    return out


# ── Helpers that turn raw signals into pairs ────────────────────────────────


def _payload_for_fix(record: dict) -> str:
    """Compact JSON-ish text representation of a candidate fix for a pair row."""
    return json.dumps(
        {
            "fix_id": record.get("fix_id"),
            "root_cause": record.get("root_cause"),
            "fix_summary": record.get("fix_summary"),
            "steps": record.get("steps"),
        },
        ensure_ascii=False,
    )


async def record_user_choice_pair(
    *,
    chosen_record: dict,
    rejected_record: dict,
    error_type: str | None,
    run_id: str | None,
) -> str | None:
    """Record a preference pair from a dev choosing one candidate over another."""
    if not chosen_record or not rejected_record:
        return None
    if chosen_record.get("fix_id") == rejected_record.get("fix_id"):
        return None
    return await record_preference(
        source="user_choice",
        chosen_payload=_payload_for_fix(chosen_record),
        rejected_payload=_payload_for_fix(rejected_record),
        error_type=error_type,
        run_id=run_id,
    )


async def record_ruler_pairs(
    *,
    ranked_records: list[dict],
    scores: dict[str, float],
    error_type: str | None,
) -> int:
    """Convert a ranked list (best→worst) into adjacent-pair preferences.

    Adjacent-only keeps the signal high-confidence (small-margin pairs are
    most informative for DPO) and avoids quadratic explosion. Returns the
    number of pairs inserted.
    """
    inserted = 0
    for i in range(len(ranked_records) - 1):
        chosen = ranked_records[i]
        rejected = ranked_records[i + 1]
        margin = float(
            scores.get(chosen.get("fix_id") or "", 0.0)
            - scores.get(rejected.get("fix_id") or "", 0.0)
        )
        if margin <= 0.0:
            continue
        await record_preference(
            source="ruler",
            chosen_payload=_payload_for_fix(chosen),
            rejected_payload=_payload_for_fix(rejected),
            error_type=error_type,
            margin=margin,
        )
        inserted += 1
    return inserted
