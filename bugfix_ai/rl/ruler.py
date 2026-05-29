"""RULER — LLM-as-judge that ranks completed trajectories.

We use this to:
  * Generate preference pairs for offline DPO/GRPO training.
  * Provide a *secondary* automated reward signal alongside the dev's
    explicit hitl_decision (which is the primary, ground-truth signal).

Design notes:
  * The judge sees REDACTED logs only — same redactor as production calls.
  * We bound input size so a runaway log doesn't blow the context window.
  * The judge returns a *ranking* + per-trajectory score in [0, 1], with a
    one-paragraph reasoning we keep for audit.
  * We refuse to score sets of fewer than 2 trajectories (no comparison to make).
"""

from __future__ import annotations

from dataclasses import dataclass

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.model_constants import RulerRanking
from bugfix_ai.core.llm_client import LLMError, chat_structured

log = get_logger(__name__)


_MAX_LOG_CHARS = 1500
_MAX_TRAJECTORIES = 8


@dataclass
class TrajectoryForJudging:
    fix_id: str
    error_type: str
    error_log_redacted: str
    fix_summary: str
    root_cause: str
    steps_summary: str          # short bulleted summary, NOT raw step JSON
    autonomous_success: bool
    time_to_resolve_min: int


_JUDGE_SYSTEM = (
    "You are an expert SRE evaluating bug-fix trajectories. You will be given "
    "several candidate fixes for the same (or very similar) error. Rank them "
    "from BEST to WORST based on:\n"
    "  1. Correctness — does the fix address the root cause, not just symptoms?\n"
    "  2. Safety — fewer irreversible operations is better.\n"
    "  3. Efficiency — shorter time-to-resolve and fewer steps is better.\n"
    "  4. Generality — does the explanation hold up beyond this single instance?\n\n"
    "Score each candidate in [0.0, 1.0] (1.0 = excellent, 0.0 = harmful).\n"
    "Use ONLY the information provided. Do not invent details.\n"
)


def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n] + "…"


def _format_trajectory(idx: int, t: TrajectoryForJudging) -> str:
    return (
        f"Candidate #{idx + 1} (fix_id={t.fix_id}):\n"
        f"  error_type: {t.error_type}\n"
        f"  autonomous_success: {t.autonomous_success}\n"
        f"  time_to_resolve_min: {t.time_to_resolve_min}\n"
        f"  root_cause: {_truncate(t.root_cause, 400)}\n"
        f"  fix_summary: {_truncate(t.fix_summary, 400)}\n"
        f"  steps:\n{_truncate(t.steps_summary, 800)}\n"
        f"  redacted_error_log:\n{_truncate(t.error_log_redacted, _MAX_LOG_CHARS)}\n"
    )


async def judge_trajectories(
    trajectories: list[TrajectoryForJudging],
) -> RulerRanking | None:
    """Rank trajectories. Returns None if input is too small or LLM fails."""
    if len(trajectories) < 2:
        log.info("ruler.skipped", reason="need >= 2 trajectories", count=len(trajectories))
        return None

    if len(trajectories) > _MAX_TRAJECTORIES:
        log.info(
            "ruler.truncated",
            given=len(trajectories),
            keeping=_MAX_TRAJECTORIES,
        )
        trajectories = trajectories[:_MAX_TRAJECTORIES]

    user_msg = "Trajectories to rank:\n\n" + "\n".join(
        _format_trajectory(i, t) for i, t in enumerate(trajectories)
    ) + (
        "\n\nReturn the ranking (best first) by fix_id, plus per-fix scores in [0,1] "
        "and a brief reasoning paragraph."
    )

    try:
        ranking = await chat_structured(
            system=_JUDGE_SYSTEM,
            user=user_msg,
            schema=RulerRanking,
            temperature=0.0,
            max_tokens=1024,
        )
    except LLMError as e:
        log.warning("ruler.failed", error=str(e)[:300])
        return None

    # Light validation — drop ranking entries that don't reference real fix_ids.
    known = {t.fix_id for t in trajectories}
    cleaned_ranking = [fid for fid in ranking.ranking if fid in known]
    cleaned_scores = [s for s in ranking.scores if s.fix_id in known]
    if not cleaned_ranking:
        log.warning("ruler.invalid_ranking", raw=ranking.ranking[:200])
        return None

    return RulerRanking(
        ranking=cleaned_ranking,
        scores=cleaned_scores,
        reasoning=ranking.reasoning,
    )
