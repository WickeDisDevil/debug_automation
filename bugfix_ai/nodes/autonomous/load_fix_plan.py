"""Autonomous mode — first node: load the selected fix's steps into state.

Position in the graph (autonomous mode):
  consent_gate (HITL) → load_fix_plan → pre_execute_review (HITL) → execute_step → ...

What "the selected fix" means:
  Normally `consent_gate` puts a `selected_fix_id` into state — the
  fix the dev clicked. But autonomous mode can also be entered
  directly via classify (when the model is highly confident this is
  a known mechanical fix), in which case selected_fix_id may be
  unset. The fallback in that case: take the top RRF candidate from
  `similar_bugs` rather than fail. The `selected_fix_id` is also
  echoed back into state so downstream nodes have a stable reference.

Failure modes:
  * No selected_fix_id AND no similar_bugs → emits an empty plan and
    a decision-log entry. The next node will see an empty plan and
    the executor loop will end on its first iteration.
  * fix_id resolves to a missing record (deleted? never persisted?) →
    same outcome: empty plan, recorded decision.

Why we don't escalate to manual_fallback here:
  Routing decisions live in `core.graph._route_after_*`. This node
  does pure data loading. The `pre_execute_review` node will be
  invoked next and will see an empty plan — which the user can then
  resolve via the HITL decision (e.g., "manual").
"""

from __future__ import annotations

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import BugFixState
from bugfix_ai.memory.fix_store import get_by_id
from bugfix_ai.observability.decision_logger import log_decision

log = get_logger(__name__)


async def load_fix_plan_node(state: BugFixState) -> dict:
    fix_id = state.get("selected_fix_id")
    if not fix_id:
        # Autonomous mode entered without a selected fix — fall back to top RRF result
        similar = state.get("similar_bugs") or []
        if similar:
            fix_id = similar[0]["fix_id"]
        else:
            log.warning("load_fix_plan.no_selection")
            return {
                "fix_plan": [],
                "obs_log": [log_decision("load_fix_plan", "No fix selected, no plan to load.")],
            }

    record = await get_by_id(fix_id)
    if not record:
        return {
            "fix_plan": [],
            "obs_log": [log_decision("load_fix_plan", f"Fix {fix_id} not found in store.")],
        }

    steps = record.get("steps") or []
    log.info("load_fix_plan.loaded", fix_id=fix_id, steps=len(steps))
    return {
        "fix_plan": steps,
        "current_step_idx": 0,
        "selected_fix_id": fix_id,
        "obs_log": [
            log_decision(
                "load_fix_plan",
                f"Loaded {len(steps)} steps from fix {fix_id}.",
                {"fix_id": fix_id, "step_count": len(steps)},
            )
        ],
    }
