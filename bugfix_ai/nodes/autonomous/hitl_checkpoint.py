"""Post-execution interrupt: dev confirms the result and chooses what to do next.

The graph is configured with `interrupt_before=["hitl_checkpoint"]`, so this
node only runs AFTER the dev has posted their decision (`continue` / `redo`
/ `manual`) via the API. Its job is to:

  * Validate the decision is one of the allowed values.
  * Bump `redo_count` if the dev chose `redo`, and force-route to manual after
    too many redos to prevent loops.
  * Emit observability — including a clear failure marker on the trailing
    execution result so the dashboard / RULER can pick it up.
"""

from __future__ import annotations

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.core.state import BugFixState
from bugfix_ai.observability.decision_logger import log_decision

log = get_logger(__name__)


_ALLOWED_DECISIONS = {"continue", "redo", "manual"}


async def hitl_checkpoint_node(state: BugFixState) -> dict:
    settings = get_settings()
    decision = state.get("hitl_decision")
    redo_count = state.get("redo_count", 0)
    idx = state.get("current_step_idx", 0)
    plan_len = len(state.get("fix_plan") or [])

    # Defensive: an unknown decision should be coerced rather than crash.
    if decision not in _ALLOWED_DECISIONS:
        log.warning("hitl_checkpoint.invalid_decision", decision=decision)
        decision = "manual"

    # Loop breaker: too many redos on the same step → escalate to manual.
    new_redo_count = redo_count
    effective_decision = decision
    if decision == "redo":
        new_redo_count = redo_count + 1
        if new_redo_count > settings.max_redo_per_step:
            log.warning(
                "hitl_checkpoint.redo_limit_exceeded",
                step_idx=idx,
                redo_count=new_redo_count,
                limit=settings.max_redo_per_step,
            )
            effective_decision = "manual"

    log.info(
        "hitl_checkpoint.decision",
        step_idx=idx,
        decision=effective_decision,
        redo_count=new_redo_count,
    )

    return {
        "hitl_decision": effective_decision,
        "redo_count": new_redo_count,
        "obs_log": [
            log_decision(
                "hitl_checkpoint",
                (
                    f"Step {idx}/{plan_len - 1}: dev chose {effective_decision!r} "
                    f"(redo_count={new_redo_count})"
                ),
                {
                    "step_idx": idx,
                    "decision": effective_decision,
                    "raw_decision": decision,
                    "redo_count": new_redo_count,
                    "redo_limit": settings.max_redo_per_step,
                },
            )
        ],
    }
