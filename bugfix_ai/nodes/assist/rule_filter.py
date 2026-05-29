"""Assist mode — structured (rule-based) recall over the fix store.

Position in the graph (assist mode):
  semantic_retrieve → rule_filter → rrf_rank → present_similar → consent_gate

Why a separate retrieval pass beside vector similarity:
  Vectors are good at "semantically similar text" but bad at "exact
  match on error_type='OOMKilled' and service='auth-svc'". The rule
  pass uses Postgres-side WHERE clauses on the structured columns
  (error_type, service, environment, stack_pattern) — the kind of
  query SQL is built for. Combining the two via RRF in the next node
  gives us the best of both: lexical rigor + semantic flexibility.

Score semantics:
  `query_by_rules()` returns a `score` per row that already includes
  the staleness decay (180-day half-life). RRF only cares about
  ranks, not absolute scores, but we surface `top_score` in the
  decision log so operators can see whether the top match was strong.

Hidden-state convention:
  `_rules_raw` (leading underscore) signals "transient between nodes;
  consumed by rrf_rank and discarded". Same convention as
  `_semantic_raw` from the prior node.
"""

from __future__ import annotations

from bugfix_ai.core.state import BugFixState
from bugfix_ai.memory.rules_db import query_by_rules
from bugfix_ai.observability.decision_logger import log_decision

_RULES_KEY = "_rules_raw"


async def rule_filter_node(state: BugFixState) -> dict:
    results = await query_by_rules(
        error_type=state.get("error_type", ""),
        service=state.get("service", ""),
        environment=state.get("environment"),
        stack_pattern=state.get("stack_pattern") or None,
        limit=20,
    )
    return {
        _RULES_KEY: results,
        "obs_log": [
            log_decision(
                "rule_filter",
                f"Got {len(results)} candidates from Postgres rules.",
                {"top_score": results[0]["score"] if results else 0.0},
            )
        ],
    }
