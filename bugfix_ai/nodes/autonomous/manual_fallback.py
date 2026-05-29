"""Terminal node for autonomous failures — hand back to the human cleanly.

Reached when:
  * The dev chose `manual` at the consent gate.
  * The dev chose `manual` at any HITL checkpoint.
  * The dev rejected an adapted command at `pre_execute_review`.
  * Redo limit was exceeded.

This node does NOT throw; it produces a clean state for the UI/API to surface
and marks autonomous_success = False so downstream metrics record the outcome.
"""

from __future__ import annotations

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import BugFixState
from bugfix_ai.observability.decision_logger import log_decision

log = get_logger(__name__)


async def manual_fallback_node(state: BugFixState) -> dict:
    idx = state.get("current_step_idx", 0)
    plan = state.get("fix_plan") or []
    results = state.get("execution_results") or []
    completed = sum(1 for r in results if r.get("success"))

    log.info(
        "manual_fallback.entered",
        step_idx=idx,
        plan_len=len(plan),
        steps_completed=completed,
    )

    return {
        "autonomous_success": False,
        "obs_log": [
            log_decision(
                "manual_fallback",
                (
                    f"Autonomous mode handed off to dev at step {idx}/{max(0, len(plan) - 1)} "
                    f"({completed} steps succeeded before handoff)."
                ),
                {
                    "step_idx": idx,
                    "plan_len": len(plan),
                    "steps_completed": completed,
                    "selected_fix_id": state.get("selected_fix_id"),
                },
            )
        ],
    }
