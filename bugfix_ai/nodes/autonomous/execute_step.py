"""Execute the (already-adapted, already-human-approved) command for the current step.

`pre_execute_review_node` did the LLM adaptation and staged a result with the
adapted command. The graph then paused at the `pre_execute_review` interrupt
for the human to approve. By the time we reach this node, the adapted command
is sitting on the trailing entry of `execution_results` and the human has said
"go".

This node:
  * Pulls the adapted command from the staged trailing result.
  * Runs it through `safe_executor.execute_command` (which re-checks safety;
    we never trust a single layer).
  * Replaces the staged trailing result with the real execution outcome.
  * Never adapts the command itself — that's `pre_execute_review`'s job. This
    keeps the human-reviewed string and the executed string identical.
"""

from __future__ import annotations

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.core.state import BugFixState, StepExecutionResult
from bugfix_ai.integrations.terminal.safe_executor import execute_command
from bugfix_ai.observability.decision_logger import log_decision

log = get_logger(__name__)


async def execute_step_node(state: BugFixState) -> dict:
    plan = state.get("fix_plan") or []
    idx = state.get("current_step_idx", 0)
    results = list(state.get("execution_results") or [])

    if idx >= len(plan):
        log.warning("execute_step.idx_out_of_range", idx=idx, plan_len=len(plan))
        return {
            "obs_log": [
                log_decision(
                    "execute_step",
                    "Step index out of range; nothing to execute.",
                    {"current_step_idx": idx, "plan_len": len(plan)},
                )
            ]
        }

    step = plan[idx]
    is_reversible = bool(step.get("is_reversible", True))

    # Find the staged result (must exist — pre_execute_review created it)
    staged = None
    if results and results[-1].get("step_idx") == idx:
        staged = results[-1]

    if staged is None:
        log.warning("execute_step.no_staged_result", step_idx=idx)
        return {
            "obs_log": [
                log_decision(
                    "execute_step",
                    "No staged adapted command found; pre_execute_review skipped?",
                    {"step_idx": idx},
                )
            ]
        }

    adapted_command = staged.get("adapted_command") or ""

    # Manual / non-terminal step: nothing to actually run.
    if step.get("tool") != "terminal" or not adapted_command:
        executed: StepExecutionResult = {
            **staged,
            "stdout": "[manual step — confirmed by operator]",
            "exit_code": 0,
            "success": True,
            "safety_verdict": "manual",
        }
        return {
            "execution_results": [*results[:-1], executed],
            "obs_log": [
                log_decision(
                    "execute_step",
                    f"Step {idx} marked complete (manual / non-terminal tool).",
                    {"step_idx": idx, "tool": step.get("tool")},
                )
            ],
        }

    # Run it. dry_run defaults to settings.terminal_dry_run_default unless the
    # caller (or upstream config) explicitly disabled it.
    settings = get_settings()
    exec_result = await execute_command(
        adapted_command,
        timeout_sec=settings.terminal_timeout_seconds,
        is_reversible_hint=is_reversible,
    )

    executed_result: StepExecutionResult = {
        "step_idx": idx,
        "stdout": exec_result.stdout,
        "stderr": exec_result.stderr,
        "exit_code": exec_result.exit_code,
        "success": exec_result.success,
        "duration_sec": exec_result.duration_sec,
        "adapted_command": adapted_command,
        "safety_verdict": exec_result.safety_verdict,
    }

    log.info(
        "execute_step.done",
        step_idx=idx,
        success=exec_result.success,
        exit_code=exec_result.exit_code,
        verdict=exec_result.safety_verdict,
    )

    return {
        "execution_results": [*results[:-1], executed_result],
        "obs_log": [
            log_decision(
                "execute_step",
                (
                    f"Executed step {idx}: success={exec_result.success}, "
                    f"verdict={exec_result.safety_verdict}, "
                    f"exit={exec_result.exit_code}"
                ),
                {
                    "step_idx": idx,
                    "success": exec_result.success,
                    "exit_code": exec_result.exit_code,
                    "duration_sec": exec_result.duration_sec,
                    "safety_verdict": exec_result.safety_verdict,
                    "rejection_reason": exec_result.rejection_reason or None,
                },
            )
        ],
    }
