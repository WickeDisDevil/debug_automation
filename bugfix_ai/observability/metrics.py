"""Business-value metrics — what we report to MLflow at run termination.

Two metrics:
  * time_saved_min — `baseline_resolution_time(error_type, service) - actual`.
    Baseline comes from `error_type_baselines` (rules_db). When no
    baseline exists we return 0.0 and log it, rather than fabricate a
    "you saved 0 minutes" metric that would skew dashboards downward.
  * lines_saved — git log+diff line count for commits whose message
    references this ticket_id. ONLY meaningful for autonomous mode
    that succeeded; skipped otherwise. Opt-in via $GIT_REPO_PATH.

Async-only API (the previous sync bridge has been removed):
  Both `calculate_time_saved` and `calculate_lines_saved` are async.
  This keeps DB calls inside the same event loop as the rest of the
  graph and avoids the "engine bound to a different loop" failure
  mode that the old `calculate_time_saved_blocking` could hit if it
  was ever invoked from inside a running loop. Callers MUST `await`.

  The git invocation in `calculate_lines_saved` is CPU/IO-bound and
  blocking (`subprocess.run`), so it is dispatched via
  `asyncio.to_thread` so the event loop is never stalled.

`calculate_lines_saved` safety:
  * 5-commit cap so a stale/loose grep can't dominate the metric.
  * 10-second timeout on each git invocation.
  * Returns 0 on FileNotFoundError so missing git in the runtime
    environment doesn't crash MLflow finalization.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from datetime import datetime

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import BugFixState
from bugfix_ai.memory.rules_db import get_avg_resolution_time

log = get_logger(__name__)


async def calculate_time_saved(state: BugFixState) -> float:
    """Async metric: baseline resolution time minus actual, in minutes.

    Returns 0.0 (with a debug log) if either timestamp is missing/malformed
    or no baseline exists for the (error_type, service) pair. Caller treats
    0.0 as "no comparison available" rather than "we saved zero minutes".
    """
    if not state.get("ticket_closed_ts") or not state.get("ticket_created_ts"):
        return 0.0
    try:
        created = datetime.fromisoformat(state["ticket_created_ts"].replace("Z", "+00:00"))
        closed = datetime.fromisoformat(state["ticket_closed_ts"].replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    actual_min = (closed - created).total_seconds() / 60.0

    baseline = await get_avg_resolution_time(
        error_type=state.get("error_type", ""),
        service=state.get("service"),
    )
    if baseline <= 0:
        log.debug("metrics.no_baseline", error_type=state.get("error_type"))
        return 0.0
    return round(baseline - actual_min, 1)


async def calculate_lines_saved(state: BugFixState) -> int:
    """Async metric: lines changed in commits referencing this ticket_id.

    Disabled by default in non-prod; returns 0 if GIT_REPO_PATH is unset.
    Runs the blocking git calls in a worker thread so the event loop is
    not stalled.
    """
    if state.get("mode") != "autonomous" or not state.get("autonomous_success"):
        return 0
    repo = os.environ.get("GIT_REPO_PATH")
    if not repo or not os.path.isdir(repo):
        return 0
    ticket_id = state.get("ticket_id", "")
    if not ticket_id:
        return 0
    return await asyncio.to_thread(_count_changed_lines, repo, ticket_id)


def _count_changed_lines(repo: str, ticket_id: str) -> int:
    """Sync helper executed in a worker thread by `calculate_lines_saved`."""
    try:
        result = subprocess.run(
            ["git", "-C", repo, "log", "--oneline", "--all", f"--grep={ticket_id}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        commits = [line.split()[0] for line in result.stdout.strip().splitlines() if line.strip()]
        if not commits:
            return 0
        total = 0
        for sha in commits[:5]:
            diff = subprocess.run(
                ["git", "-C", repo, "diff", "--stat", f"{sha}~1", sha],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for n in re.findall(r"(\d+)\s+(?:insertion|deletion)", diff.stdout):
                total += int(n)
        return total
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0
