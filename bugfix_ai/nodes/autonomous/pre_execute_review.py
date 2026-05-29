"""Adapt the saved command to the current environment, then pause for human review.

This node is the FIRST half of the safety sandwich around autonomous execution.
The graph is configured with `interrupt_before=["pre_execute_review"]`, but the
node itself runs FIRST to do the LLM-based command adaptation, THEN the graph
pauses on the next interrupt cycle so the human can approve / edit / reject the
adapted command before `execute_step` actually runs it.

Design notes:
  * The original architecture did the LLM adaptation INSIDE `execute_step`,
    which meant the human only ever saw the raw saved command — they couldn't
    review what would actually run. We split it so the adaptation is visible
    at the interrupt.
  * We *also* run the static safety check here so the UI can surface a "this
    will be rejected by the executor" warning even before the human approves.
  * If adaptation fails (LLM transport error, validation error), we fall back
    to the saved command unchanged and tag the result so the UI can display it.
"""

from __future__ import annotations

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.model_constants import AdaptedCommandResponse
from bugfix_ai.core.llm_client import LLMError, chat_structured
from bugfix_ai.core.state import BugFixState, StepExecutionResult
from bugfix_ai.integrations.terminal.safe_executor import evaluate_safety
from bugfix_ai.observability.decision_logger import log_decision

log = get_logger(__name__)


_ADAPT_SYSTEM = (
    "You are a careful DevOps assistant adapting a saved shell command to the "
    "current environment.\n"
    "Rules:\n"
    "  - Preserve the command's INTENT exactly. Do not change verbs (get vs delete, "
    "    apply vs rollback, etc.).\n"
    "  - Only substitute identifiers that clearly differ between environments "
    "    (namespace, service name, pod name, branch name, env-specific paths).\n"
    "  - If nothing needs to change, return the command unchanged and set "
    "    `unchanged: true`.\n"
    "  - NEVER add new verbs, NEVER add `--force`, NEVER add `rm`, NEVER add pipes.\n"
    "  - Output ONLY the adapted command in the `adapted_command` field."
)


def _build_user_prompt(state: BugFixState, saved_command: str, step_description: str) -> str:
    service = state.get("service") or "(unknown)"
    environment = state.get("environment") or "(unknown)"
    error_type = state.get("error_type") or "(unknown)"
    return (
        f"Step description: {step_description}\n"
        f"Saved command (from a previous fix):\n  {saved_command}\n\n"
        f"Current environment context:\n"
        f"  service: {service}\n"
        f"  environment: {environment}\n"
        f"  error_type: {error_type}\n\n"
        "Adapt the saved command to this environment. If no change is needed, "
        "return it unchanged and set `unchanged: true`."
    )


async def pre_execute_review_node(state: BugFixState) -> dict:
    plan = state.get("fix_plan") or []
    idx = state.get("current_step_idx", 0)

    if idx >= len(plan):
        log.warning("pre_execute_review.idx_out_of_range", idx=idx, plan_len=len(plan))
        return {
            "obs_log": [
                log_decision(
                    "pre_execute_review",
                    "Step index out of range; nothing to review.",
                    {"current_step_idx": idx, "plan_len": len(plan)},
                )
            ]
        }

    step = plan[idx]
    saved_command = step.get("command")
    description = step.get("description", "")
    is_reversible = bool(step.get("is_reversible", True))

    # Manual / non-terminal steps: nothing to adapt, just stage a result and
    # let the human confirm at the interrupt.
    if step.get("tool") != "terminal" or not saved_command:
        staged: StepExecutionResult = {
            "step_idx": idx,
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "success": False,
            "duration_sec": 0.0,
            "adapted_command": saved_command or "",
            "safety_verdict": "manual",
        }
        return {
            "execution_results": [*(state.get("execution_results") or []), staged],
            "obs_log": [
                log_decision(
                    "pre_execute_review",
                    "Non-terminal step; awaiting manual confirmation.",
                    {"step_idx": idx, "tool": step.get("tool")},
                )
            ],
        }

    # LLM adaptation
    adapted_command = saved_command
    unchanged = True
    reasoning = ""
    try:
        adapted = await chat_structured(
            system=_ADAPT_SYSTEM,
            user=_build_user_prompt(state, saved_command, description),
            schema=AdaptedCommandResponse,
            temperature=0.0,
            max_tokens=512,
        )
        adapted_command = adapted.adapted_command
        unchanged = adapted.unchanged
        reasoning = adapted.reasoning
    except LLMError as e:
        log.warning("pre_execute_review.adapt_failed", error=str(e)[:300])
        # Fall back to saved command — safer than fabricating an adaptation.
        adapted_command = saved_command
        unchanged = True
        reasoning = f"adaptation failed, using saved command: {e}"

    # Static safety pre-check (informational — execute_step also re-checks)
    allowed, rejection_reason = evaluate_safety(
        adapted_command, is_reversible_hint=is_reversible
    )
    safety_verdict = "allowed" if allowed else "rejected"

    staged_result: StepExecutionResult = {
        "step_idx": idx,
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "success": False,
        "duration_sec": 0.0,
        "adapted_command": adapted_command,
        "safety_verdict": safety_verdict,
    }

    log.info(
        "pre_execute_review.staged",
        step_idx=idx,
        unchanged=unchanged,
        safety_verdict=safety_verdict,
    )

    return {
        # Replace the trailing staged result if we already staged one for this
        # step (re-entry on `redo`); otherwise append.
        "execution_results": _upsert_staged(state.get("execution_results") or [], staged_result),
        "obs_log": [
            log_decision(
                "pre_execute_review",
                (
                    f"Adapted command for step {idx}; safety={safety_verdict}; "
                    f"unchanged={unchanged}"
                ),
                {
                    "step_idx": idx,
                    "saved_command": saved_command[:200],
                    "adapted_command": adapted_command[:200],
                    "unchanged": unchanged,
                    "safety_verdict": safety_verdict,
                    "rejection_reason": rejection_reason or None,
                    "adaptation_reasoning": reasoning[:300] if reasoning else None,
                },
            )
        ],
    }


def _upsert_staged(
    existing: list[StepExecutionResult], staged: StepExecutionResult
) -> list[StepExecutionResult]:
    """Replace the last entry if it's for the same step_idx and not yet executed."""
    if existing:
        last = existing[-1]
        if (
            last.get("step_idx") == staged.get("step_idx")
            and last.get("safety_verdict") in ("allowed", "rejected", "manual")
            and last.get("exit_code", 0) == 0
            and last.get("stdout", "") == ""
            and last.get("stderr", "") == ""
        ):
            return [*existing[:-1], staged]
    return [*existing, staged]
