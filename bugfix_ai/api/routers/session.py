"""Session router — start a new bug-fix run, post HITL decisions, fetch state.

LangGraph thread model:
  * Each bug-fix run is identified by a `thread_id` (== ticket_id).
  * `config = {"configurable": {"thread_id": thread_id}}` is what makes the
    checkpointer key off this run; we pass the same config to every call so
    state persists across the HITL interrupts.

Endpoints:
  POST /sessions                 — start a new run from a manual ticket payload.
  GET  /sessions/{thread_id}     — fetch the latest state snapshot.
  POST /sessions/{thread_id}/decision — resume from an interrupt with a decision.

Capture-mode (silent observer) extensions:
  POST /sessions/{thread_id}/capture/event   — append one observed action.
  POST /sessions/{thread_id}/capture/resolve — finalize, synthesize, persist.

The capture endpoints are how the engineer's editor / shell / git hooks
feed the background recorder. The engineer is never asked to narrate.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import BugFixState, CaptureEvent
from bugfix_ai.nodes.capture import recorder
from bugfix_ai.nodes.capture.emit_new_issue import NEW_ISSUE_SIGNAL
from bugfix_ai.nodes.capture.finalize import finalize_capture

router = APIRouter(prefix="/sessions", tags=["sessions"])
log = get_logger(__name__)


# ── Request / response models ───────────────────────────────────────────────


class StartSessionRequest(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=10_000)
    error_logs: str = Field(default="", max_length=200_000)
    service: str = Field(default="unknown", max_length=128)
    environment: str = Field(default="development", max_length=64)


class DecisionRequest(BaseModel):
    """One unified payload covering all four interrupts."""

    decision: str = Field(min_length=1, max_length=64)
    selected_fix_id: str | None = None
    dev_narrative: str | None = None
    edited_command: str | None = None  # for pre_execute_review approval-with-edits


class SessionStateResponse(BaseModel):
    thread_id: str
    next_node: list[str]
    state: dict[str, Any]
    is_complete: bool


# ── Helpers ─────────────────────────────────────────────────────────────────


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _graph(request: Request):
    g = getattr(request.app.state, "graph", None)
    if g is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="graph not initialized",
        )
    return g


async def _snapshot(graph, thread_id: str) -> SessionStateResponse:
    snap = await graph.aget_state(_config(thread_id))
    if snap is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such session")
    nxt = list(snap.next or [])
    return SessionStateResponse(
        thread_id=thread_id,
        next_node=nxt,
        state=dict(snap.values or {}),
        is_complete=len(nxt) == 0,
    )


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post("", status_code=status.HTTP_201_CREATED)
async def start_session(req: StartSessionRequest, request: Request) -> SessionStateResponse:
    """Start a new run. Streams up to the first interrupt, then returns state."""
    from datetime import datetime, timezone

    graph = _graph(request)
    initial: BugFixState = {  # type: ignore[typeddict-item]
        "ticket_id": req.ticket_id,
        "ticket_title": req.title,
        "ticket_description": req.description,
        "error_logs_raw": req.error_logs,
        "service": req.service,
        "environment": req.environment,
        "ticket_created_ts": datetime.now(timezone.utc).isoformat(),
    }
    log.info("session.start", thread_id=req.ticket_id)
    await graph.ainvoke(initial, config=_config(req.ticket_id))
    return await _snapshot(graph, req.ticket_id)


@router.get("/{thread_id}")
async def get_session(thread_id: str, request: Request) -> SessionStateResponse:
    return await _snapshot(_graph(request), thread_id)


@router.post("/{thread_id}/decision")
async def post_decision(
    thread_id: str, req: DecisionRequest, request: Request
) -> SessionStateResponse:
    """Resume a paused run with the dev's decision.

    The state-update payload depends on which interrupt we're at, but the
    graph's reducer merges by key so we can always send the same shape.
    """
    graph = _graph(request)
    snap = await graph.aget_state(_config(thread_id))
    if snap is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such session")
    if not snap.next:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="session is already complete; nothing to resume",
        )

    update: dict[str, Any] = {"hitl_decision": req.decision}
    if req.selected_fix_id is not None:
        update["selected_fix_id"] = req.selected_fix_id
    if req.dev_narrative is not None:
        update["dev_narrative"] = req.dev_narrative

    # If the dev edited the adapted command at pre_execute_review, replace the
    # last staged result so execute_step picks up the human's version.
    if req.edited_command is not None:
        existing_results = list((snap.values or {}).get("execution_results") or [])
        if existing_results:
            last = dict(existing_results[-1])
            last["adapted_command"] = req.edited_command
            existing_results[-1] = last  # type: ignore[assignment]
            update["execution_results"] = existing_results

    await graph.aupdate_state(_config(thread_id), update)
    await graph.ainvoke(None, config=_config(thread_id))
    return await _snapshot(graph, thread_id)


# ── Capture-mode endpoints (silent observer) ────────────────────────────────


class CaptureEventRequest(BaseModel):
    """One observed action streamed in from a hook on the engineer's machine."""

    kind: str = Field(default="note", max_length=32)
    tool: str | None = Field(default=None, max_length=64)
    command: str | None = Field(default=None, max_length=4_000)
    files: list[str] = Field(default_factory=list)
    stdout: str | None = Field(default=None, max_length=20_000)
    stderr: str | None = Field(default=None, max_length=20_000)
    exit_code: int | None = None
    note: str | None = Field(default=None, max_length=4_000)
    timestamp: str | None = None


class CaptureEventAck(BaseModel):
    thread_id: str
    accepted: bool
    buffered_events: int
    signal: str = NEW_ISSUE_SIGNAL


class CaptureResolveRequest(BaseModel):
    note: str = Field(default="", max_length=4_000)


class CaptureResolveResponse(BaseModel):
    thread_id: str
    fix_id: str | None
    events: int
    steps: int
    skipped: bool


@router.post("/{thread_id}/capture/event", status_code=status.HTTP_202_ACCEPTED)
async def capture_event(
    thread_id: str, req: CaptureEventRequest, request: Request
) -> CaptureEventAck:
    """Append one observed action to the recorder for `thread_id`.

    No-op (still 202) if the session is not currently in capture mode —
    we accept the event so the hook on the engineer's machine doesn't
    have to reason about lane state, but we drop it on the floor and
    return `accepted=false`.
    """
    graph = _graph(request)
    snap = await graph.aget_state(_config(thread_id))
    if snap is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such session")
    state = dict(snap.values or {})
    if not state.get("capture_active"):
        return CaptureEventAck(
            thread_id=thread_id,
            accepted=False,
            buffered_events=len(state.get("capture_events") or []),
        )

    event: CaptureEvent = {
        "kind": req.kind,
        "tool": req.tool,
        "command": req.command,
        "files": req.files,
        "stdout": req.stdout,
        "stderr": req.stderr,
        "exit_code": req.exit_code,
        "note": req.note,
        "timestamp": req.timestamp or "",
    }
    recorded = await recorder.record_event(thread_id, event)

    # Mirror into checkpointed state so a process restart preserves the
    # trace. Append-only via state update (we re-emit the merged list).
    merged = list(state.get("capture_events") or [])
    merged.append(recorded)
    await graph.aupdate_state(_config(thread_id), {"capture_events": merged})

    return CaptureEventAck(
        thread_id=thread_id,
        accepted=True,
        buffered_events=len(merged),
    )


@router.post("/{thread_id}/capture/resolve")
async def capture_resolve(
    thread_id: str, req: CaptureResolveRequest, request: Request
) -> CaptureResolveResponse:
    """Engineer (or an editor / git hook) declares the bug resolved.

    Triggers the finalize coordinator: synthesize a structured plan from
    the recorded trace and persist it through `store_fix`. Returns the
    `fix_id` of the newly persisted record (or null if the trace was
    too thin to be worth keeping).
    """
    graph = _graph(request)
    try:
        result = await finalize_capture(graph, thread_id, resolution_note=req.note)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such session")
    return CaptureResolveResponse(
        thread_id=thread_id,
        fix_id=result.get("fix_id"),
        events=int(result.get("events", 0)),
        steps=int(result.get("steps", 0)),
        skipped=bool(result.get("skipped", False)),
    )
