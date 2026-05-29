"""Assist mode — second HITL interrupt: developer's decision to replay or not.

Position in the graph (assist mode):
  present_similar → consent_gate (HITL) → {load_fix_plan | manual_fallback}

Why this is an interrupt:
  Listed in `interrupt_before`, so the graph PAUSES here. The session
  API expects the dev to POST a decision payload that sets:
    * `hitl_decision`   — "autonomous" (replay) or "manual" (do it myself)
    * `selected_fix_id` — which similar bug the dev wants to replay
                           (only meaningful when decision == "autonomous")
  Then the run resumes via `ainvoke(None)`.

What this node does on resume:
  Validates the decision, pulls the chosen fix_id into top-level
  state, and records the choice into `obs_log`. The conditional edge
  `_route_after_consent` in `core.graph` reads `hitl_decision` to pick
  the next node — so this is the only place in assist mode where the
  human's intent translates into routing.

Defaults:
  Missing or unrecognized decision falls back to "manual" — the
  conservative path. We never proceed to autonomous replay without
  an explicit "autonomous" choice.
"""

from __future__ import annotations

from bugfix_ai.core.state import BugFixState
from bugfix_ai.observability.decision_logger import log_decision


async def consent_gate_node(state: BugFixState) -> dict:
    decision = state.get("hitl_decision") or "manual"
    selected = state.get("selected_fix_id")
    return {
        "obs_log": [
            log_decision(
                "consent_gate",
                f"Dev chose {decision!r} for fix_id={selected!r}.",
                {"decision": decision, "selected_fix_id": selected},
            )
        ]
    }
