"""LangGraph state schema — the central contract every node reads / writes.

Why a flat TypedDict instead of a Pydantic model:
  LangGraph's checkpointer (Postgres / SQLite) serializes state with
  msgpack. Pydantic models re-validate on every state mutation; that's
  too expensive in a graph that mutates state inside many small nodes.
  TypedDict is a static-typing hint with zero runtime cost — perfect
  here. We pay validation cost only at the *boundaries* (LLM outputs go
  through Pydantic in `core.llm_client.chat_structured`).

Why one big top-level dict instead of nested sub-states:
  LangGraph reducers (the `Annotated[..., add]` pattern below) operate
  on top-level keys. Nested reducer support exists but is fiddly. The
  pragmatic compromise: keep the schema flat, but document logical
  groupings in the BugFixState docstring and stick to that grouping in
  every node.

Append-only conventions:
  `obs_log` is the only field with a reducer — `add` (list concat).
  Every other field is replaced wholesale on write. Nodes that need
  cumulative behavior (e.g., `execution_results`) read the existing
  list and re-emit it with one more element appended; this is more
  verbose but lets us swap entries in place during the redo loop.

Lifecycle pointer:
  Field groups map directly to graph phases:
    intake → classify → {capture | assist | autonomous}
  See `core.graph.build_graph()` for the wiring.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict


# ── Sub-types ────────────────────────────────────────────────────────────────


class FixStep(TypedDict, total=False):
    step_idx: int
    description: str
    tool: str  # "terminal" | "github" | "code_edit" | "manual"
    command: str | None
    expected_outcome: str
    is_reversible: bool


class SimilarBug(TypedDict, total=False):
    fix_id: str
    alert_number: int
    rrf_score: float
    semantic_score: float
    rules_score: float
    summary: str
    root_cause: str
    steps: list[FixStep]
    resolved_at: str
    age_days: int
    time_to_resolve_min: int


class ObservationEntry(TypedDict, total=False):
    node: str
    timestamp: str
    decision: str
    scores: dict | None
    llm_model: str | None


class CaptureEvent(TypedDict, total=False):
    """One observed action recorded silently while the engineer fixes a bug.

    Lane A (capture mode) does NOT prompt the engineer to narrate — it
    flags the ticket as "New issue" and records what they do behind the
    scenes. Each emitted event is one of these dicts: a tool the engineer
    invoked, a command they ran, files they touched, or the eventual
    resolution marker.

    Sources that feed events:
      * `POST /sessions/{thread_id}/capture/event` — driven by an IDE /
        shell hook that the engineer has installed once and forgotten.
      * `integrations.terminal.safe_executor` mirrors any commands the
        engineer runs through the assistant.
      * Editor / git hooks (file save, commit) emit `kind="file_touched"`.
    """

    timestamp: str
    kind: str          # "command" | "tool" | "file_touched" | "note" | "resolution"
    tool: str | None   # "terminal" | "git" | "editor" | "browser" | ...
    command: str | None
    files: list[str]   # absolute paths touched by this event
    stdout: str | None
    stderr: str | None
    exit_code: int | None
    note: str | None   # free-form, used for "resolution" / "manual checkpoint"


class StepExecutionResult(TypedDict, total=False):
    step_idx: int
    stdout: str
    stderr: str
    exit_code: int
    success: bool
    duration_sec: float
    adapted_command: str
    safety_verdict: str  # "allowed" | "rejected" | "dry_run"


class AlertSource(TypedDict, total=False):
    """Origin information for an ingested alert."""

    provider: Literal["github_code_scanning", "manual", "framework"]
    alert_number: int
    rule_id: str
    rule_severity: str
    repository: str
    ref: str
    html_url: str
    file_path: str
    start_line: int
    end_line: int


# ── Master state ─────────────────────────────────────────────────────────────


class BugFixState(TypedDict, total=False):
    """Flat graph state. All nodes read/write here.

    Field groups (by convention, enforce via code review):
      - identity: ticket_id, run_id, mlflow_run_id
      - input:    title, description, error_logs_*, alert_source
      - classification: error_type, severity, mode, classify_confidence
      - capture:  dev_narrative, fix_steps, root_cause, fix_summary, fix_id
      - assist:   similar_bugs, selected_fix_id
      - autonomous: fix_plan, current_step_idx, execution_results,
                    hitl_decision, redo_count, autonomous_success
      - shared:   obs_log (append-only), timing
    """

    # Identity
    ticket_id: str
    run_id: str               # internal UUID, always set in intake
    mlflow_run_id: str        # set when MLflow run starts

    # Input
    ticket_title: str
    ticket_description: str
    error_logs_raw: str
    error_logs_clean: str
    error_logs_redacted: str  # after PII/secret scrubbing — what we send to LLMs
    service: str
    environment: str
    alert_source: AlertSource

    # Classification
    error_type: str
    severity: str
    stack_pattern: str
    classify_confidence: float
    mode: Literal["capture", "assist", "autonomous"]

    # Capture
    #
    # Lane A no longer asks the engineer to narrate. The graph emits a
    # synchronous "New issue" signal and finishes; a background recorder
    # then accumulates `capture_events` (commands, tools, files touched)
    # until the engineer (or an editor hook) marks the bug resolved via
    # POST /sessions/{tid}/capture/resolve. At that point the finalize
    # coordinator synthesises `fix_steps`, `root_cause`, `fix_summary`
    # from the trace and persists the row through `store_fix`.
    #
    # `dev_narrative` is retained ONLY because the synthesizer can use
    # it as a free-form addendum if the engineer chose to type one — it
    # is no longer required, and is empty for the zero-cost path.
    capture_active: bool
    capture_started_ts: str
    capture_events: list[CaptureEvent]
    capture_resolved: bool
    capture_resolution_note: str
    dev_narrative: str
    fix_steps: list[FixStep]
    root_cause: str
    fix_summary: str
    fix_id: str | None

    # Assist
    similar_bugs: list[SimilarBug]
    selected_fix_id: str | None

    # Autonomous
    fix_plan: list[FixStep]
    current_step_idx: int
    execution_results: list[StepExecutionResult]
    hitl_decision: Literal["continue", "redo", "manual", "autonomous"] | None
    redo_count: int
    autonomous_success: bool

    # Shared (append-only via reducer)
    obs_log: Annotated[list[ObservationEntry], add]

    # Timing
    run_start_ts: str
    run_end_ts: str | None
    ticket_created_ts: str
    ticket_closed_ts: str | None
