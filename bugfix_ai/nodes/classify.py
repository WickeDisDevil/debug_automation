"""Classify node: pick the graph mode and extract structured triage fields.

Position in the graph:
  intake → classify → {capture | assist | autonomous}

What this node decides:
  * `mode` — drives the conditional edge in `core.graph._route_after_classify`.
       capture     — the ticket is already resolved; we want to record
                     the human's fix narrative for future re-use.
       autonomous  — high-confidence match to a known fix pattern that
                     is mechanical / safely reversible.
       assist      — everything else; surface similar past fixes and
                     ask the dev to pick one.
  * `error_type`, `severity`, `service`, `stack_pattern` — populated
    so downstream nodes (retrieval, classification analytics) don't
    each have to re-derive them.

Inputs the prompt sees:
  * Ticket id / title / description (truncated to 2000 chars)
  * The REDACTED log slice (truncated to 4000 chars). Always redacted —
    intake wrote `error_logs_redacted` for exactly this reason.
  * Optional upstream hint when the alert source already gave us a
    strong signal (e.g. CodeQL rule id mapped to error_type). The
    prompt is instructed to override the hint on disagreement, so the
    model isn't strait-jacketed into a wrong answer.

Validation contract:
  Output is parsed against `ClassifyResponse` (Literal-typed mode and
  severity fields) — see `config.model_constants`. Hallucinated values
  are caught by the parse-retry loop in `chat_structured`.
"""

from __future__ import annotations

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.model_constants import ClassifyResponse
from bugfix_ai.config.settings import get_settings
from bugfix_ai.core.llm_client import chat_structured
from bugfix_ai.core.state import BugFixState
from bugfix_ai.observability.decision_logger import log_decision

log = get_logger(__name__)

_SYSTEM = """You are an expert software engineer triaging bug tickets.

Extract structured fields from the ticket description and the redacted error log.

Set 'suggested_mode' as follows:
  - 'capture'    if this ticket has been resolved/closed and we should save the fix
  - 'autonomous' if you are very confident this matches a previously solved bug
                 AND the fix is mechanical (e.g. restart, clear cache, env var change)
  - 'assist'     in all other cases (default)

Set 'confidence' between 0.0 and 1.0 based on how clear the signals are.
"""


async def classify_node(state: BugFixState) -> dict:
    settings = get_settings()

    user = (
        f"Ticket: {state.get('ticket_id', '')}\n"
        f"Title: {state.get('ticket_title', '')}\n"
        f"Description:\n{state.get('ticket_description', '')[:2000]}\n\n"
        f"Redacted log:\n{state.get('error_logs_redacted', '')[:4000]}\n"
    )

    # If upstream (e.g. CodeQL parser) already gave us a strong error_type/severity,
    # bias the classifier by mentioning it but let it override on disagreement.
    hint = ""
    if state.get("error_type"):
        hint = f"\nUpstream hint: error_type='{state['error_type']}' severity='{state.get('severity','')}'"

    result = await chat_structured(
        system=_SYSTEM + hint,
        user=user,
        schema=ClassifyResponse,
        temperature=0.0,
        max_tokens=512,
    )

    log.info(
        "classify.done",
        mode=result.suggested_mode,
        error_type=result.error_type,
        confidence=result.confidence,
    )

    return {
        "error_type": result.error_type,
        "service": state.get("service") or result.service,
        "severity": result.severity,
        "stack_pattern": result.stack_pattern or state.get("stack_pattern", ""),
        "mode": result.suggested_mode,
        "classify_confidence": result.confidence,
        "obs_log": [
            log_decision(
                "classify",
                f"mode={result.suggested_mode} error_type={result.error_type} conf={result.confidence:.2f}",
                {"hint_used": bool(hint)},
                llm_model=settings.gpt_oss_model,
            )
        ],
    }
