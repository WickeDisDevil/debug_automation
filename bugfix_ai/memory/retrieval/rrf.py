"""Reciprocal Rank Fusion + a recency decay multiplier.

Pure math. No I/O. Trivially unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RankedResult:
    fix_id: str
    semantic_rank: int | None = None
    rules_rank: int | None = None
    rrf_score: float = 0.0
    semantic_score: float = 0.0
    rules_score: float = 0.0
    age_days: float = 0.0
    payload: dict = field(default_factory=dict)


def rrf_score(rank: int, k: int = 60) -> float:
    """Standard RRF: 1 / (k + rank). Rank is 0-indexed."""
    return 1.0 / (k + rank)


def staleness_multiplier(age_days: float, half_life_days: float = 180.0) -> float:
    """Exponential decay so 6-month-old fixes are weighted ~half of fresh ones.

    Returns a value in (0, 1]. Newer = closer to 1, older = closer to 0.
    """
    if age_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life_days)


def fuse_results(
    *,
    semantic_results: list[dict],
    rules_results: list[dict],
    k: int = 60,
    top_n: int = 5,
    apply_staleness: bool = True,
    half_life_days: float = 180.0,
) -> list[RankedResult]:
    """Fuse semantic and rules result lists into a single ranked list.

    Each input dict shape:
      {fix_id: str, score: float, age_days?: float, payload?: dict}

    The recency decay is applied to the FUSED RRF score, not to the inputs,
    so it's a single tunable knob rather than something that drifts between
    the two signals.
    """
    by_id: dict[str, RankedResult] = {}

    for rank, item in enumerate(semantic_results):
        fid = item["fix_id"]
        r = by_id.setdefault(fid, RankedResult(fix_id=fid))
        r.semantic_rank = rank
        r.semantic_score = float(item.get("score", 0.0))
        r.rrf_score += rrf_score(rank, k)
        if "age_days" in item:
            r.age_days = max(r.age_days, float(item["age_days"]))
        if "payload" in item:
            r.payload.update(item["payload"])

    for rank, item in enumerate(rules_results):
        fid = item["fix_id"]
        r = by_id.setdefault(fid, RankedResult(fix_id=fid))
        r.rules_rank = rank
        r.rules_score = float(item.get("score", 0.0))
        r.rrf_score += rrf_score(rank, k)
        if "age_days" in item:
            r.age_days = max(r.age_days, float(item["age_days"]))

    if apply_staleness:
        for r in by_id.values():
            r.rrf_score *= staleness_multiplier(r.age_days, half_life_days)

    return sorted(by_id.values(), key=lambda r: r.rrf_score, reverse=True)[:top_n]
