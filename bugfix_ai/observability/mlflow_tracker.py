"""MLflow tracking with EXPLICIT run_id (not thread-local).

Original design used `mlflow.start_run()` / `end_run()` which is thread-local.
Under async + concurrent tickets that's a guaranteed race condition. We use
the lower-level MlflowClient API which takes run_id on every call, and we
keep the run_id on BugFixState so it survives across nodes and checkpoint
rehydrations.
"""

from __future__ import annotations

from functools import lru_cache

import mlflow
from mlflow.tracking import MlflowClient

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.core.state import BugFixState

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _client() -> MlflowClient:
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment_name)
    return MlflowClient()


@lru_cache(maxsize=1)
def _experiment_id() -> str:
    s = get_settings()
    exp = _client().get_experiment_by_name(s.mlflow_experiment_name)
    if exp:
        return exp.experiment_id
    return _client().create_experiment(s.mlflow_experiment_name)


def start_run(run_name: str, tags: dict | None = None) -> str:
    """Create a new MLflow run, return its run_id. Caller stores in state."""
    s = get_settings()
    full_tags = {"environment": s.environment, **(tags or {})}
    run = _client().create_run(
        experiment_id=_experiment_id(),
        run_name=run_name,
        tags=full_tags,
    )
    return run.info.run_id


def log_param(run_id: str, key: str, value) -> None:
    _client().log_param(run_id, key, str(value))


def log_metric(run_id: str, key: str, value: float) -> None:
    _client().log_metric(run_id, key, float(value))


def end_run(run_id: str, status: str = "FINISHED") -> None:
    _client().set_terminated(run_id, status=status)


async def log_final_metrics(state: BugFixState) -> None:
    """Called at the terminal node of any branch.

    Async because the metric calculators do Postgres lookups; running them
    in the same event loop as the rest of the graph keeps the cached
    SQLAlchemy AsyncEngine bound to the right loop.

    Per-call MLflow operations (log_param/log_metric/end_run) are sync and
    cheap; we leave them inline. If MLflow ever becomes the bottleneck,
    wrap them in `asyncio.to_thread`.
    """
    from bugfix_ai.observability.metrics import (
        calculate_lines_saved,
        calculate_time_saved,
    )

    run_id = state.get("mlflow_run_id")
    if not run_id:
        log.warning("mlflow.no_run_id")
        return

    for k, v in {
        "ticket_id": state.get("ticket_id", ""),
        "mode": state.get("mode", ""),
        "error_type": state.get("error_type", ""),
        "service": state.get("service", ""),
        "severity": state.get("severity", ""),
    }.items():
        log_param(run_id, k, v)

    time_saved = await calculate_time_saved(state)
    lines_saved = await calculate_lines_saved(state)
    log_metric(run_id, "time_saved_min", time_saved)
    log_metric(run_id, "lines_saved", lines_saved)
    log_metric(run_id, "autonomous_success", int(state.get("autonomous_success", False)))
    log_metric(run_id, "hitl_redo_count", state.get("redo_count", 0))
    log_metric(run_id, "classify_confidence", state.get("classify_confidence", 0.0))
    log_metric(run_id, "similar_bugs_found", len(state.get("similar_bugs", [])))
    log_metric(run_id, "steps_executed", len(state.get("execution_results", [])))

    similar = state.get("similar_bugs") or []
    if similar:
        log_metric(run_id, "retrieval_top_rrf_score", similar[0].get("rrf_score", 0.0))

    end_run(run_id)
