"""Persist a finalized capture-mode fix to Postgres + Qdrant.

Position in the (refactored) capture flow:
  emit_new_issue ─▶ END
                   └── (out of band, on resolve) extract_steps ─▶ store_fix

What changed vs. the original:
  * The required input used to be `dev_narrative` (free-form engineer
    typing). Lane A no longer asks for that — `dev_narrative` is now
    optional and stored verbatim only if the engineer chose to add a
    note. The persisted `dev_narrative` column is filled from, in
    priority order: `state.dev_narrative` → `capture_resolution_note`
    → a synthetic "<silent capture: N events>" placeholder.
  * The skip path now also fires when the capture was abandoned (no
    events recorded AND no narrative) so we never write an empty
    placeholder row to the long-term memory.

Storage policy is unchanged:
  * `memory.fix_store.store_fix()` writes the structured row to
    Postgres first and the embedding to Qdrant second. If Qdrant fails
    the Postgres row stays committed and the reconcile worker
    backfills the vector. See `memory/fix_store.py` for the full
    consistency story.

What gets returned:
  * `fix_id` (so the caller can deep-link to the saved record) and a
    single obs_log entry. The capture buffer is cleared by the
    finalize coordinator AFTER this node succeeds — store_fix itself
    is purely about persistence.
"""

from __future__ import annotations

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import BugFixState
from bugfix_ai.memory.fix_store import store_fix
from bugfix_ai.observability.decision_logger import log_decision
from bugfix_ai.observability.mlflow_tracker import log_final_metrics

log = get_logger(__name__)


def _persisted_narrative(state: BugFixState) -> str:
    """What gets written into the `dev_narrative` column.

    Lane A no longer demands narration, but downstream tooling (admin
    UI, retrieval previews) expects the column to be human-readable.
    We use a graceful fallback chain so the column is never NULL even
    when the engineer typed nothing.
    """
    legacy = (state.get("dev_narrative") or "").strip()
    if legacy:
        return legacy
    note = (state.get("capture_resolution_note") or "").strip()
    if note:
        return f"[silent capture] {note}"
    n = len(state.get("capture_events") or [])
    return f"[silent capture: {n} event(s) recorded]"


async def store_fix_node(state: BugFixState) -> dict:
    if not state.get("fix_steps"):
        log.info("store_fix.skip", reason="no_steps")
        await log_final_metrics(state)
        return {
            "obs_log": [log_decision("store_fix", "Skipped: no steps to persist.")],
        }

    alert_source = state.get("alert_source") or {}
    fix_id = await store_fix(
        alert_number=alert_source.get("alert_number"),
        repository=alert_source.get("repository", ""),
        service=state.get("service", "unknown"),
        environment=state.get("environment", "code"),
        error_type=state.get("error_type", "unknown"),
        severity=state.get("severity", "medium"),
        root_cause=state.get("root_cause", ""),
        fix_summary=state.get("fix_summary", ""),
        steps=state.get("fix_steps", []),
        dev_narrative=_persisted_narrative(state),
        title=state.get("ticket_title", ""),
        description=state.get("ticket_description", ""),
        error_log=state.get("error_logs_redacted", ""),
    )

    await log_final_metrics(state)
    return {
        "fix_id": fix_id,
        "obs_log": [
            log_decision(
                "store_fix",
                f"Persisted fix {fix_id} from silent capture trace.",
                {
                    "fix_id": fix_id,
                    "events": len(state.get("capture_events") or []),
                    "steps": len(state.get("fix_steps") or []),
                },
            )
        ],
    }
