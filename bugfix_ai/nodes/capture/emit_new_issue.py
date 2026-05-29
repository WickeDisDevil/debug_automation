"""Capture-mode entry node — silent observer pattern (Lane A, first draft).

Position in the graph (capture mode, post-refactor):

    classify ──▶ emit_new_issue ──▶ END
                       │
                       ▼ (out of band)
              recorder.append_event()  ← shell / editor / git hooks
                       │
                       ▼ (when engineer marks the bug resolved)
              finalize_capture_node()  → extract_steps → store_fix

Why this is NO LONGER an interrupt:
  The previous design paused the graph and asked the engineer to type
  out a narrative of the fix they performed. That gated every capture
  run on a chunk of human writing — engineers either skipped it or
  scribbled something useless, and the long-term memory grew slowly
  because the act of contributing was distinct from the act of fixing.

  Lane A's new contract:
    * Outwardly, the system says one thing — "New issue" — and then
      gets out of the way.
    * In the background, the recorder collects events as they happen
      (commands run, tools invoked, files touched). The engineer is
      not asked to narrate, summarise, or do anything extra.
    * When the bug is resolved (a marker event, an editor hook, or
      `POST /sessions/{tid}/capture/resolve`), the finalize coordinator
      stitches the trace into `fix_steps` / `root_cause` / `fix_summary`
      and persists it through the same `store_fix` path that Lane B / C
      consume on retrieval.

  The result: every fix the engineer performs becomes a memory entry,
  with zero engineer cost. Lane B (assist) and Lane C (autonomous) get
  fed automatically.

What the node does, concretely:
  * Stamps `capture_active=True` and `capture_started_ts` so the
    recorder + finalize coordinator can detect "we are inside a capture
    run" without a side-channel.
  * Initialises `capture_events=[]` so the API endpoint can append
    without checking for None.
  * Emits a single, well-known obs_log decision — "New issue" — that
    the API surfaces verbatim to the caller.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import BugFixState
from bugfix_ai.observability.decision_logger import log_decision

log = get_logger(__name__)


# Surfaced verbatim by the session API so the engineer / dashboard sees
# the exact phrase the architecture doc and client diagram promise.
NEW_ISSUE_SIGNAL = "New issue"


async def emit_new_issue_node(state: BugFixState) -> dict:
    """Mark the ticket as a 'New issue' and arm the background recorder."""
    started = datetime.now(timezone.utc).isoformat()
    log.info(
        "capture.new_issue.armed",
        ticket=state.get("ticket_id"),
        started=started,
    )
    return {
        "capture_active": True,
        "capture_started_ts": started,
        "capture_events": [],
        "capture_resolved": False,
        "capture_resolution_note": "",
        "obs_log": [
            log_decision(
                "emit_new_issue",
                NEW_ISSUE_SIGNAL,
                {
                    "ticket_id": state.get("ticket_id"),
                    "recorder": "armed",
                    "lane": "A",
                    "mode": "silent_observer",
                },
            )
        ],
    }
