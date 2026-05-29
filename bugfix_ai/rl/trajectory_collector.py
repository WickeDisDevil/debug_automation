"""Collect completed trajectories from finished runs and persist them as JSONL.

A *trajectory* is everything a future SFT/DPO/GRPO trainer will need from one
run: input, intermediate decisions, output, terminal reward.

We pull from two sources:
  * The final BugFixState (in-memory at end-of-run): cheap, lossless.
  * The Postgres `fixes` table (for older runs): rebuild from row + obs_log.

Output format is JSONL — one trajectory per line, easy to shard, easy to load
into HuggingFace `datasets` for SFT or into TRL for DPO/GRPO.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.core.state import BugFixState
from bugfix_ai.memory.fix_store import get_by_id

log = get_logger(__name__)


# ── Trajectory schema ───────────────────────────────────────────────────────


@dataclass
class Trajectory:
    """One bug-fix trajectory ready for training."""

    run_id: str
    fix_id: str | None
    error_type: str
    severity: str
    mode: str

    # INPUT (what the model saw)
    ticket_title: str
    ticket_description: str
    error_log_redacted: str

    # CHAIN (intermediate decisions)
    classification_confidence: float
    selected_fix_id: str | None
    similar_bugs_top3: list[dict]      # ids + rrf_scores
    decision_log: list[dict]            # obs_log slice

    # OUTPUT (what the model produced)
    fix_steps: list[dict]
    root_cause: str
    fix_summary: str

    # TERMINAL REWARD
    autonomous_success: bool
    time_to_resolve_min: int
    redo_count: int                    # how many times the dev rejected a step

    def to_jsonl_line(self) -> str:
        return json.dumps(asdict(self), default=str, ensure_ascii=False)


# ── Build from live state ───────────────────────────────────────────────────


def _time_to_resolve_min(state: BugFixState) -> int:
    """Derive minutes between run_start_ts and run_end_ts. 0 if unknown."""
    from datetime import datetime

    start = state.get("run_start_ts")
    end = state.get("run_end_ts")
    if not start or not end:
        return 0
    try:
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        return max(0, int((e - s).total_seconds() // 60))
    except (ValueError, TypeError):
        return 0


def trajectory_from_state(state: BugFixState) -> Trajectory:
    similar = state.get("similar_bugs") or []
    return Trajectory(
        run_id=state.get("run_id", ""),
        fix_id=state.get("fix_id") or state.get("selected_fix_id"),
        error_type=state.get("error_type", "unknown"),
        severity=state.get("severity", "unknown"),
        mode=state.get("mode", "unknown"),
        ticket_title=state.get("ticket_title", ""),
        ticket_description=state.get("ticket_description", ""),
        error_log_redacted=state.get("error_logs_redacted", ""),
        classification_confidence=float(state.get("classify_confidence", 0.0)),
        selected_fix_id=state.get("selected_fix_id"),
        similar_bugs_top3=[
            {
                "fix_id": s.get("fix_id"),
                "rrf_score": s.get("rrf_score"),
                "semantic_score": s.get("semantic_score"),
                "rules_score": s.get("rules_score"),
            }
            for s in similar[:3]
        ],
        decision_log=list(state.get("obs_log") or []),
        fix_steps=list(state.get("fix_steps") or state.get("fix_plan") or []),
        root_cause=state.get("root_cause", ""),
        fix_summary=state.get("fix_summary", ""),
        autonomous_success=bool(state.get("autonomous_success", False)),
        time_to_resolve_min=_time_to_resolve_min(state),
        redo_count=int(state.get("redo_count", 0)),
    )


# ── Persistence ─────────────────────────────────────────────────────────────


async def append_trajectory(trajectory: Trajectory, *, output_path: str | None = None) -> Path:
    """Append a single trajectory line to the JSONL file. Returns the path used."""
    settings = get_settings()
    out = Path(output_path or settings.trajectory_jsonl_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    line = trajectory.to_jsonl_line()

    # File append must not interleave partial writes from concurrent runs.
    # On POSIX, single write() to O_APPEND is atomic up to PIPE_BUF; we keep
    # lines short enough that this holds. Use asyncio.to_thread so we don't
    # block the event loop on disk I/O.
    def _do_append() -> None:
        with out.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    await asyncio.to_thread(_do_append)
    log.info("trajectory.appended", run_id=trajectory.run_id, path=str(out))
    return out


# ── Backfill from Postgres ──────────────────────────────────────────────────


async def trajectory_from_fix_record(fix_id: str) -> Trajectory | None:
    """Rebuild a (lossy) trajectory from a stored fix row. For backfill."""
    record = await get_by_id(fix_id)
    if not record:
        return None

    return Trajectory(
        run_id=record.get("run_id") or "",
        fix_id=fix_id,
        error_type=record.get("error_type") or "unknown",
        severity=record.get("severity") or "unknown",
        mode=record.get("mode") or "capture",
        ticket_title=record.get("ticket_title") or "",
        ticket_description=record.get("ticket_description") or "",
        error_log_redacted=record.get("error_logs_redacted") or "",
        classification_confidence=float(record.get("classify_confidence") or 0.0),
        selected_fix_id=None,
        similar_bugs_top3=[],
        decision_log=[],
        fix_steps=list(record.get("steps") or []),
        root_cause=record.get("root_cause") or "",
        fix_summary=record.get("fix_summary") or "",
        autonomous_success=bool(record.get("autonomous_success") or False),
        time_to_resolve_min=int(record.get("time_to_resolve_min") or 0),
        redo_count=int(record.get("redo_count") or 0),
    )


async def backfill_trajectories(fix_ids: list[str], *, output_path: str | None = None) -> int:
    """Append a trajectory for every fix_id, returning how many were written."""
    written = 0
    for fid in fix_ids:
        traj = await trajectory_from_fix_record(fid)
        if traj is None:
            log.warning("trajectory.backfill_missing", fix_id=fid)
            continue
        await append_trajectory(traj, output_path=output_path)
        written += 1
    return written


# ── Helpers used by the API ─────────────────────────────────────────────────


def collect_and_serialize(state: BugFixState) -> dict[str, Any]:
    """Convenience for in-process consumers (e.g. an API endpoint)."""
    return asdict(trajectory_from_state(state))
