"""Coordinator that turns a silent capture trace into a persisted fix.

The graph itself does NOT walk this path — `emit_new_issue` terminates
the capture sub-graph synchronously so the engineer is never blocked.
Finalization is triggered out-of-band, when the engineer (or an editor
hook) marks the bug resolved via:

    POST /sessions/{thread_id}/capture/resolve

The session router calls `finalize_capture(graph, thread_id, note)` and
this module:

  1. Asks the recorder for the live trace and merges it into the
     checkpointed state, so any events that were appended in a different
     process / event loop are reflected in `state.capture_events`.
  2. Marks the trace as resolved (so the recorder won't accept further
     events for this thread_id).
  3. Calls `extract_steps_node` to synthesize a structured plan.
  4. Calls `store_fix_node` to persist into Postgres + Qdrant.
  5. Writes the resulting state delta back through the graph's
     checkpointer so the next `aget_state` reflects the persisted fix.
  6. Clears the recorder buffer to release memory.

Why a coordinator instead of stitching this into the LangGraph itself:
  Adding a second entry-point to the graph (one for "intake → classify"
  and another for "finalize from outside") would require duplicating the
  graph wiring or adding a dummy router node whose only job is to detect
  resolution. A small async function is clearer, easier to test, and
  keeps the LangGraph edges describing the *automatic* part of the run.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph.state import CompiledStateGraph

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import BugFixState
from bugfix_ai.nodes.capture import recorder
from bugfix_ai.nodes.capture.extract_steps import extract_steps_node
from bugfix_ai.nodes.capture.store_fix import store_fix_node
from bugfix_ai.observability.decision_logger import log_decision

log = get_logger(__name__)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _merge_obs(*deltas: dict) -> list:
    out: list = []
    for d in deltas:
        for entry in d.get("obs_log") or []:
            out.append(entry)
    return out


async def finalize_capture(
    graph: CompiledStateGraph,
    thread_id: str,
    *,
    resolution_note: str = "",
) -> dict[str, Any]:
    """Synthesize + persist the capture trace for `thread_id`.

    Returns a dict with `fix_id` (None if persistence was skipped) plus
    a snapshot of the merged state for the caller to forward to the API
    response.
    """
    snap = await graph.aget_state(_config(thread_id))
    if snap is None:
        raise LookupError(f"no session for thread_id={thread_id!r}")

    state: BugFixState = dict(snap.values or {})  # type: ignore[assignment]

    if not state.get("capture_active"):
        # Either the run wasn't a capture-mode run, or it has already
        # been finalized. Either way, nothing to do.
        log.info(
            "capture.finalize.noop",
            thread_id=thread_id,
            reason="not_a_capture_run" if not state.get("capture_active") else "already_done",
        )
        return {"fix_id": state.get("fix_id"), "skipped": True}

    # 1. Merge live recorder buffer with checkpointed events. The recorder
    #    is the source of truth for events appended since the last state
    #    write; the checkpointed list survives process restarts.
    live_events = await recorder.snapshot(thread_id)
    seen_ts = {e.get("timestamp") for e in (state.get("capture_events") or [])}
    merged = list(state.get("capture_events") or []) + [
        e for e in live_events if e.get("timestamp") not in seen_ts
    ]

    # 2. Make sure the recorder is flagged resolved before we synthesise.
    if not recorder.is_resolved(thread_id):
        await recorder.mark_resolved(thread_id, resolution_note)
        # Re-snapshot so the resolution marker is in `merged` too.
        live_events = await recorder.snapshot(thread_id)
        seen_ts = {e.get("timestamp") for e in merged}
        merged.extend(e for e in live_events if e.get("timestamp") not in seen_ts)

    state["capture_events"] = merged
    state["capture_resolved"] = True
    if resolution_note:
        state["capture_resolution_note"] = resolution_note

    # 3. Synthesize a structured plan from the trace.
    extract_delta = await extract_steps_node(state)
    state.update({k: v for k, v in extract_delta.items() if k != "obs_log"})

    # 4. Persist (no-ops cleanly when extract produced an empty plan).
    store_delta = await store_fix_node(state)
    state.update({k: v for k, v in store_delta.items() if k != "obs_log"})

    fix_id = store_delta.get("fix_id") or state.get("fix_id")

    # 5. Write the merged delta back through the checkpointer so future
    #    `aget_state` calls see the synthesised plan + fix_id.
    delta = {
        "capture_events": merged,
        "capture_resolved": True,
        "capture_resolution_note": resolution_note or state.get("capture_resolution_note", ""),
        "capture_active": False,
        "fix_steps": state.get("fix_steps") or [],
        "root_cause": state.get("root_cause", ""),
        "fix_summary": state.get("fix_summary", ""),
        "fix_id": fix_id,
        "obs_log": _merge_obs(extract_delta, store_delta)
        + [
            log_decision(
                "finalize_capture",
                f"Capture finalized: {len(merged)} event(s), "
                f"{len(state.get('fix_steps') or [])} step(s), "
                f"fix_id={fix_id}.",
                {"events": len(merged), "fix_id": fix_id},
            )
        ],
    }
    await graph.aupdate_state(_config(thread_id), delta)

    # 6. Release recorder memory now that the trace lives in Postgres.
    await recorder.clear(thread_id)

    log.info(
        "capture.finalize.done",
        thread_id=thread_id,
        events=len(merged),
        steps=len(state.get("fix_steps") or []),
        fix_id=fix_id,
    )

    return {
        "fix_id": fix_id,
        "skipped": False,
        "events": len(merged),
        "steps": len(state.get("fix_steps") or []),
    }
