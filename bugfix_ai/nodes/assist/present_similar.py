"""Assist mode — record that we're presenting candidates to the human.

Position in the graph (assist mode):
  rrf_rank → present_similar → consent_gate (HITL)

Why this is a node at all:
  No transformation happens here — `similar_bugs` is already the
  list the API will return. The node exists purely to emit a
  decision-log entry so the audit trail records "we offered N
  candidates with top RRF score X" at this point in the run. That
  matters for explainability: when a dev later asks "why did the
  system suggest those?", the obs_log has the answer at the right
  graph step rather than buried inside rrf_rank's payload.

Why immediately followed by an HITL interrupt:
  The next node, `consent_gate`, is in `interrupt_before` so the
  graph pauses while the dev picks one of the candidates (or none).
  Splitting "present" from "decide" keeps the UI integration clean
  and lets us add UI-only side-effects to this node later without
  touching the gate's logic.
"""

from __future__ import annotations

from bugfix_ai.core.state import BugFixState
from bugfix_ai.observability.decision_logger import log_decision


async def present_similar_node(state: BugFixState) -> dict:
    similar = state.get("similar_bugs") or []
    return {
        "obs_log": [
            log_decision(
                "present_similar",
                f"Presenting {len(similar)} candidates to dev for review.",
                {"top_score": similar[0]["rrf_score"] if similar else 0.0},
            )
        ]
    }
