"""Graph entry node: normalize input, fetch logs, redact, start MLflow run.

Position in the graph:
  intake → classify → {capture | assist | autonomous}

Responsibilities, in order:
  1. Stamp the run_id (UUID4) and run_start_ts (UTC ISO-8601) so every
     subsequent log line and decision can be tied to this graph run.
  2. Pull the raw error logs. If the caller supplied them in
     `error_logs_raw` we trust that; otherwise we delegate to
     `integrations.logs.fetcher.fetch_logs_for_ticket`, which speaks
     whichever backend `settings.log_source` selected (file / s3 /
     elk / splunk / none).
  3. Run the log preprocessor — extracts stack frames, tallies log
     levels, hints at the error type. Cheap, deterministic, no LLM.
  4. Redact PII / secrets BEFORE anything LLM-bound is constructed.
     `error_logs_redacted` is what every downstream prompt uses.
  5. Open an MLflow run. We use an explicit run_id (not thread-local)
     because LangGraph nodes can resume on different workers.
  6. Initialize the autonomous-mode bookkeeping fields
     (`current_step_idx`, `redo_count`, `execution_results`) and the
     empty-list fields the downstream nodes expect to be present.
  7. Append a single `obs_log` entry summarizing the intake decision.

Idempotency:
  Re-entering this node would re-fetch the logs and start a new MLflow
  run. The graph's edge wiring guarantees we only run intake once per
  thread; if you find yourself wanting to "rewind to intake", create a
  new thread instead.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import BugFixState
from bugfix_ai.integrations.logs.fetcher import fetch_logs_for_ticket
from bugfix_ai.integrations.logs.preprocessor import preprocess_logs_full
from bugfix_ai.integrations.logs.redactor import redact_text
from bugfix_ai.observability.decision_logger import log_decision
from bugfix_ai.observability.mlflow_tracker import start_run

log = get_logger(__name__)


async def intake_node(state: BugFixState) -> dict:
    run_id = state.get("run_id") or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    raw_logs = state.get("error_logs_raw") or fetch_logs_for_ticket(
        ticket_id=state.get("ticket_id", ""),
        service=state.get("service", "unknown"),
        created_ts=state.get("ticket_created_ts", now),
    )

    pp = preprocess_logs_full(raw_logs)
    redacted = redact_text(pp.clean_text)

    mlflow_run_id = start_run(
        run_name=state.get("ticket_id", run_id),
        tags={
            "service": state.get("service", "unknown"),
            "alert_provider": (state.get("alert_source") or {}).get("provider", "manual"),
        },
    )

    log.info(
        "intake.complete",
        ticket=state.get("ticket_id"),
        log_len=len(raw_logs),
        clean_len=len(pp.clean_text),
        error_type_hint=pp.error_type_hint,
    )

    return {
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "error_logs_raw": raw_logs,
        "error_logs_clean": pp.clean_text,
        "error_logs_redacted": redacted,
        "stack_pattern": pp.stack_frames[0] if pp.stack_frames else "",
        "run_start_ts": now,
        "current_step_idx": 0,
        "redo_count": 0,
        "execution_results": [],
        "similar_bugs": [],
        "fix_steps": [],
        "fix_plan": [],
        "obs_log": [
            log_decision(
                "intake",
                f"Parsed input ({len(raw_logs)} bytes raw, {len(pp.clean_text)} clean)",
                {"error_type_hint": pp.error_type_hint, "log_levels": pp.log_level_counts},
            )
        ],
    }
