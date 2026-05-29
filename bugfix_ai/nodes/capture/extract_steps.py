"""Synthesize a structured fix plan from the silently-recorded capture trace.

Position in the (refactored) capture flow:
  emit_new_issue ──▶ END
       └── (out of band) recorder accumulates events
                          └── finalize_capture ─▶ extract_steps ─▶ store_fix

What changed vs. the original design:
  Before: this node read `dev_narrative` — a free-text blob the engineer
  was forced to type — and asked the LLM to invent a step list from it.
  Now: the engineer types nothing. The recorder has been collecting
  every command, tool invocation, file touched, and the eventual
  resolution marker. This node turns that trace into a clean, replayable
  step list the way a code review would: keep the load-bearing actions,
  drop the noise, and write a one-line "what / why" for each step.

Two-tier synthesis:
  1. Deterministic baseline. We always project the trace into a list of
     FixStep dicts using simple rules (one captured "command" event ⇒
     one FixStep, file_touched events grouped under the preceding
     command, resolution marker terminates the trace). This guarantees
     a non-empty plan even when the LLM is unavailable.
  2. LLM refinement (best effort). When the local model is reachable we
     hand it the deterministic plan + the raw trace and ask it to
     collapse trivial dupes, infer `is_reversible`, and write a short
     `root_cause` / `fix_summary`. If anything goes wrong (timeout,
     parse failure, model down) we keep the deterministic plan and
     leave `root_cause` / `fix_summary` empty — `store_fix` is happy
     with that.

Why we still gate on a non-trivial trace:
  A capture run that recorded zero or one events is almost certainly an
  abandoned fix attempt. Persisting it would pollute the long-term
  memory with noise that surfaces later as bad "similar bug" matches.
  We mirror the previous threshold (refuse to synthesize from nothing)
  and let `store_fix` short-circuit on the empty plan.

Backward compatibility:
  If `capture_events` is empty but `dev_narrative` is set, we fall back
  to the legacy narrative-driven synthesis so any in-flight runs from
  before the refactor still terminate cleanly. New runs never set the
  narrative, so this branch atrophies naturally.
"""

from __future__ import annotations

from typing import Any

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.model_constants import FixStepsResponse
from bugfix_ai.config.settings import get_settings
from bugfix_ai.core.llm_client import chat_structured
from bugfix_ai.core.state import BugFixState, CaptureEvent, FixStep
from bugfix_ai.observability.decision_logger import log_decision

log = get_logger(__name__)


_SYSTEM = """You convert observed bug-fix traces into structured, reproducible procedures.

You will receive:
  * The captured trace — a sequence of commands, tool invocations, and file touches
    that the engineer performed (this is the SOURCE OF TRUTH; do not invent steps
    that are not represented in the trace).
  * A deterministic baseline plan derived from the trace.

Refine the baseline:
  - Collapse trivially repeated commands.
  - Each step should be a SINGLE atomic, executable action.
  - 'command' must be the exact shell / git / code command from the trace.
    Use null only when 'tool' is 'manual'.
  - 'is_reversible' is true ONLY if the step can be cleanly undone.
    Restarts/reloads/git revert are reversible. Drops/deletes/migrations are not.
  - 'root_cause' must explain WHY the bug occurred, inferred from the trace.
  - 'fix_summary' is one sentence — what the engineer did, in plain English.

Do NOT invent steps that are not present in the trace.
"""


# ── Deterministic baseline ──────────────────────────────────────────────────


_NOISY_KINDS = {"note"}  # kept in trace for context, dropped from the plan


def _coalesce_files(carry: list[str], more: list[str] | None) -> list[str]:
    if not more:
        return carry
    seen = set(carry)
    for f in more:
        if f and f not in seen:
            carry.append(f)
            seen.add(f)
    return carry


def _trace_to_baseline(events: list[CaptureEvent]) -> list[FixStep]:
    """Project a recorded trace into a baseline FixStep list.

    Rule of thumb: every `command` event becomes a step; following
    `file_touched` events attach to the preceding command (so the plan
    doesn't grow a step per saved file). `resolution` terminates the
    plan — anything after is treated as cleanup and dropped.
    """
    plan: list[FixStep] = []
    idx = 0
    pending_files: list[str] = []

    for ev in events:
        kind = ev.get("kind") or "note"
        if kind == "resolution":
            break
        if kind in _NOISY_KINDS:
            continue
        if kind == "file_touched":
            _coalesce_files(pending_files, ev.get("files"))
            if plan:
                last = dict(plan[-1])
                # Stash files into the description so we don't lose them.
                desc = last.get("description") or ""
                file_str = ", ".join(pending_files)
                if file_str and "files: " not in desc:
                    last["description"] = f"{desc} (files: {file_str})".strip()
                plan[-1] = last  # type: ignore[index]
            continue

        cmd = ev.get("command")
        tool = ev.get("tool") or ("terminal" if cmd else "manual")
        exit_code = ev.get("exit_code")
        success = exit_code is None or exit_code == 0
        descr = (ev.get("note") or cmd or f"{tool} action").strip()
        # Heuristic reversibility: shell-y restarts/reloads/git ops are reversible;
        # delete/drop/rm/migrate are not. Caller (LLM) can override.
        reversible = True
        lower = (cmd or "").lower()
        if any(tok in lower for tok in (" rm ", "rm -", "drop ", "delete ", "migrate ")):
            reversible = False

        step: FixStep = {
            "step_idx": idx,
            "description": descr[:300],
            "tool": tool,
            "command": cmd,
            "expected_outcome": "succeeded" if success else f"exit_code={exit_code}",
            "is_reversible": reversible,
        }
        if pending_files:
            file_str = ", ".join(pending_files)
            step["description"] = f"{step['description']} (files: {file_str})"
            pending_files = []
        plan.append(step)
        idx += 1

    # Any file touches after the last command become a trailing manual step
    # so we don't silently drop a "edited X then resolved" trace.
    if pending_files:
        plan.append(
            {
                "step_idx": idx,
                "description": f"Edited files: {', '.join(pending_files)}",
                "tool": "manual",
                "command": None,
                "expected_outcome": "engineer-confirmed",
                "is_reversible": False,
            }
        )

    return plan


def _baseline_summary(events: list[CaptureEvent], plan: list[FixStep]) -> tuple[str, str]:
    """Best-effort, model-free root_cause / fix_summary.

    These are deliberately blunt — the LLM tier replaces them when
    available. We never want to ship an empty summary downstream because
    consumers (assist-mode previews, MLflow tags) display them verbatim.
    """
    n = len(plan)
    resolution_note = ""
    for ev in events:
        if ev.get("kind") == "resolution" and ev.get("note"):
            resolution_note = (ev["note"] or "").strip()
            break

    summary = (
        f"Captured {n} action(s) silently while the engineer fixed the bug."
        if not resolution_note
        else f"{resolution_note} (captured {n} action(s))."
    )
    root_cause = (
        "Root cause not stated in the trace; inferred plan above reflects the "
        "actions the engineer took to resolve it."
    )
    return root_cause, summary


# ── Node ────────────────────────────────────────────────────────────────────


def _trim_event_for_prompt(ev: CaptureEvent) -> dict[str, Any]:
    """Strip large stdout/stderr blobs before handing the trace to the LLM."""
    keep = {
        "timestamp": ev.get("timestamp"),
        "kind": ev.get("kind"),
        "tool": ev.get("tool"),
        "command": (ev.get("command") or "")[:300],
        "files": (ev.get("files") or [])[:10],
        "exit_code": ev.get("exit_code"),
        "note": (ev.get("note") or "")[:400],
    }
    return {k: v for k, v in keep.items() if v not in (None, "", [])}


async def extract_steps_node(state: BugFixState) -> dict:
    settings = get_settings()
    events = list(state.get("capture_events") or [])
    narrative = (state.get("dev_narrative") or "").strip()

    # Empty trace AND no legacy narrative → abandoned capture; skip persistence.
    if not events and len(narrative) < 20:
        return {
            "fix_steps": [],
            "root_cause": "",
            "fix_summary": "",
            "obs_log": [
                log_decision(
                    "extract_steps",
                    "Skipped: no capture events and no fallback narrative.",
                    {"events": 0, "narrative_len": len(narrative)},
                )
            ],
        }

    # Deterministic baseline first — guarantees a non-empty plan.
    baseline = _trace_to_baseline(events)
    base_root_cause, base_summary = _baseline_summary(events, baseline)

    # Best-effort LLM refinement. On any failure we keep the baseline.
    refined_steps: list[dict] = [dict(s) for s in baseline]
    root_cause = base_root_cause
    fix_summary = base_summary
    llm_used = False

    if events:
        try:
            user = (
                f"Ticket: {state.get('ticket_id','')}\n\n"
                f"Original error log (redacted):\n"
                f"{state.get('error_logs_redacted','')[:1500]}\n\n"
                f"Captured trace (silent observer):\n"
                f"{[_trim_event_for_prompt(e) for e in events][:60]}\n\n"
                f"Baseline plan derived from the trace:\n{baseline}\n\n"
                f"Optional engineer note (may be empty):\n{narrative[:1500]}\n"
            )
            result = await chat_structured(
                system=_SYSTEM,
                user=user,
                schema=FixStepsResponse,
                temperature=0.0,
                max_tokens=2048,
            )
            refined_steps = [s.model_dump() for s in result.steps] or refined_steps
            root_cause = result.root_cause or root_cause
            fix_summary = result.fix_summary or fix_summary
            llm_used = True
        except Exception as e:  # noqa: BLE001 — refinement is best-effort
            log.warning(
                "extract_steps.refinement_failed",
                error=str(e)[:200],
                kept="deterministic_baseline",
            )
    elif narrative:
        # Legacy narrative-only fallback — old in-flight runs from before
        # the refactor. New runs never reach this branch.
        try:
            user = (
                f"Ticket: {state.get('ticket_id','')}\n\n"
                f"Original error log (redacted):\n"
                f"{state.get('error_logs_redacted','')[:2000]}\n\n"
                f"Developer's fix narrative (legacy capture mode):\n{narrative}\n"
            )
            result = await chat_structured(
                system=_SYSTEM,
                user=user,
                schema=FixStepsResponse,
                temperature=0.0,
                max_tokens=2048,
            )
            refined_steps = [s.model_dump() for s in result.steps]
            root_cause = result.root_cause
            fix_summary = result.fix_summary
            llm_used = True
        except Exception as e:  # noqa: BLE001
            log.warning("extract_steps.legacy_narrative_failed", error=str(e)[:200])

    log.info(
        "extract_steps.done",
        events=len(events),
        baseline_steps=len(baseline),
        final_steps=len(refined_steps),
        llm_used=llm_used,
        root_cause_len=len(root_cause),
    )

    return {
        "fix_steps": refined_steps,
        "root_cause": root_cause,
        "fix_summary": fix_summary,
        "obs_log": [
            log_decision(
                "extract_steps",
                f"Synthesized {len(refined_steps)} step(s) from "
                f"{len(events)} captured event(s) "
                f"({'LLM-refined' if llm_used else 'baseline-only'}).",
                {
                    "events": len(events),
                    "baseline_steps": len(baseline),
                    "final_steps": len(refined_steps),
                    "llm_used": llm_used,
                },
                llm_model=settings.gpt_oss_model if llm_used else None,
            )
        ],
    }
