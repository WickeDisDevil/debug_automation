"""Build ObservationEntry dicts — the audit trail every node appends to.

Why a dedicated builder instead of inlining the dict literal:
  Every node in the graph appends one or more entries to `obs_log` to
  record what decision it made and why. Going through this helper:
    * keeps the entry shape (node, timestamp, decision, scores,
      llm_model) uniform, so downstream tooling (the API's GET
      /session/{id}/log endpoint, MLflow tagging, RL trajectory
      collection) can rely on it.
    * stamps a UTC timestamp at decision time, not at log-shipping
      time — important when nodes run on different workers.
    * gives a single chokepoint to add new fields later (e.g. cost,
      token usage) without grepping every node.

Flow:
  node returns -> {"obs_log": [log_decision(...)]}
              -> graph reducer appends to state["obs_log"]
              -> trajectory_collector serializes the full list
              -> persisted in trajectories.jsonl + visible via API

Note:
  This module is deliberately I/O-free and synchronous. Anything more
  involved (MLflow metrics, file persistence) lives elsewhere so this
  call stays cheap enough to invoke many times per node.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bugfix_ai.core.state import ObservationEntry


def log_decision(
    node: str,
    decision: str,
    scores: dict | None = None,
    llm_model: str | None = None,
) -> ObservationEntry:
    return ObservationEntry(
        node=node,
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        scores=scores,
        llm_model=llm_model,
    )
