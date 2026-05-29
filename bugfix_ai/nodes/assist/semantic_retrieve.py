"""Assist mode — vector-similarity recall over the fix store.

Position in the graph (assist mode):
  classify → semantic_retrieve → rule_filter → rrf_rank → present_similar → consent_gate (HITL)

Why semantic recall is its OWN node (rather than fused inline):
  We want each retrieval path independently observable. Splitting the
  semantic and rule-based passes lets us see, per-run, "did vectors
  find anything? did the rules?" — a single fused score hides those
  two failure modes. The fusion happens in `rrf_rank_node`.

Embedding composition:
  `embed_composite()` concatenates title + description + redacted log
  + error_type into one composite document and embeds it with
  BAAI/bge-small-en-v1.5. We use the redacted log (intake's output)
  so secrets never enter the embedding model — Important because
  embeddings are persisted in Qdrant payloads and could otherwise
  leak via vector search debugging tools.

Filter pre-application:
  `similarity_search()` accepts service / environment as Qdrant
  payload filters; this is *pre*-filtering at the index level, not
  post-filtering in Python. That keeps the candidate list small even
  for very large stores.

Hidden state convention:
  Intermediate results live under the `_semantic_raw` key — the
  leading underscore is a convention meaning "transient between
  nodes, not part of the public schema". `rrf_rank_node` reads it,
  the fusion happens, and we move on. The key is not declared on
  `BugFixState` because LangGraph's reducer accepts arbitrary keys
  but we don't want it to read like a normal field.
"""

from __future__ import annotations

from bugfix_ai.core.state import BugFixState
from bugfix_ai.memory.embed import embed_composite
from bugfix_ai.memory.vector_store import similarity_search
from bugfix_ai.observability.decision_logger import log_decision

# We stash intermediate results under a private key on state to avoid bloating
# the public schema. They're consumed by rrf_rank_node and discarded.
_SEMANTIC_KEY = "_semantic_raw"


async def semantic_retrieve_node(state: BugFixState) -> dict:
    embedding = embed_composite(
        title=state.get("ticket_title", ""),
        description=state.get("ticket_description", ""),
        error_log=state.get("error_logs_redacted", ""),
        error_type=state.get("error_type", ""),
    )
    results = await similarity_search(
        embedding=embedding,
        top_k=20,
        service=state.get("service"),
        environment=state.get("environment"),
    )
    return {
        _SEMANTIC_KEY: results,
        "obs_log": [
            log_decision(
                "semantic_retrieve",
                f"Got {len(results)} candidates from Qdrant.",
                {"top_score": results[0]["score"] if results else 0.0},
            )
        ],
    }
