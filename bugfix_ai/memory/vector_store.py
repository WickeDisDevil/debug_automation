"""Qdrant vector store wrapper.

Production fixes:
  * Async client (qdrant-client supports `AsyncQdrantClient`).
  * Pre-filtering by service/environment via Qdrant payload filters
    BEFORE the ANN search — this is what makes hybrid retrieval not blow up
    on cross-tenant noise.
  * Idempotent collection creation (safe to call on every startup).
  * Single client instance per process.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _client() -> AsyncQdrantClient:
    s = get_settings()
    return AsyncQdrantClient(
        host=s.qdrant_host,
        port=s.qdrant_port,
        api_key=s.qdrant_api_key,
        prefer_grpc=False,
        https=False,
    )


async def ensure_collection() -> None:
    """Create the collection if it doesn't exist. Idempotent."""
    s = get_settings()
    client = _client()
    try:
        await client.get_collection(s.qdrant_collection_name)
        return
    except (UnexpectedResponse, ValueError):
        pass

    await client.recreate_collection(
        collection_name=s.qdrant_collection_name,
        vectors_config=qm.VectorParams(
            size=s.embedding_dimensions,
            distance=qm.Distance.COSINE,
        ),
        optimizers_config=qm.OptimizersConfigDiff(default_segment_number=2),
    )
    # Indexed payload fields for fast filtering
    for field_name, schema in [
        ("service", qm.PayloadSchemaType.KEYWORD),
        ("environment", qm.PayloadSchemaType.KEYWORD),
        ("error_type", qm.PayloadSchemaType.KEYWORD),
        ("severity", qm.PayloadSchemaType.KEYWORD),
        ("resolved_at_ts", qm.PayloadSchemaType.INTEGER),
    ]:
        await client.create_payload_index(
            collection_name=s.qdrant_collection_name,
            field_name=field_name,
            field_schema=schema,
        )
    log.info("vector_store.collection.created", name=s.qdrant_collection_name)


# ── Public ops ──────────────────────────────────────────────────────────────


async def upsert_fix(
    *,
    fix_id: str,
    embedding: list[float],
    payload: dict[str, Any],
) -> None:
    """Insert or update a fix record in the vector store."""
    s = get_settings()
    client = _client()
    await client.upsert(
        collection_name=s.qdrant_collection_name,
        points=[
            qm.PointStruct(
                id=fix_id,
                vector=embedding,
                payload=payload,
            )
        ],
    )


async def similarity_search(
    *,
    embedding: list[float],
    top_k: int = 10,
    service: str | None = None,
    environment: str | None = None,
    error_type: str | None = None,
) -> list[dict[str, Any]]:
    """Cosine similarity search with optional payload pre-filtering.

    Returns: [{fix_id, score, payload}, ...] ordered by score desc.
    """
    s = get_settings()
    client = _client()

    # Build the pre-filter
    must_clauses: list[qm.FieldCondition] = []
    if service:
        must_clauses.append(qm.FieldCondition(key="service", match=qm.MatchValue(value=service)))
    if environment:
        must_clauses.append(
            qm.FieldCondition(key="environment", match=qm.MatchValue(value=environment))
        )
    if error_type:
        must_clauses.append(
            qm.FieldCondition(key="error_type", match=qm.MatchValue(value=error_type))
        )
    qfilter = qm.Filter(must=must_clauses) if must_clauses else None

    results = await client.search(
        collection_name=s.qdrant_collection_name,
        query_vector=embedding,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    )
    return [
        {
            "fix_id": str(p.id),
            "score": float(p.score),
            "payload": p.payload or {},
        }
        for p in results
    ]


async def points_exist(fix_ids: list[str]) -> set[str]:
    """Return the subset of fix_ids that already have a point in Qdrant.

    Used by the reconcile worker to decide which Postgres rows are missing
    a vector and need to be re-embedded. We use `retrieve` (not `search`)
    because we already know the IDs and want a yes/no answer per ID, not
    a similarity ranking.
    """
    if not fix_ids:
        return set()
    s = get_settings()
    client = _client()
    points = await client.retrieve(
        collection_name=s.qdrant_collection_name,
        ids=fix_ids,
        with_payload=False,
        with_vectors=False,
    )
    return {str(p.id) for p in points}


async def delete_fix(fix_id: str) -> None:
    s = get_settings()
    await _client().delete(
        collection_name=s.qdrant_collection_name,
        points_selector=qm.PointIdsList(points=[fix_id]),
    )


async def ping_qdrant() -> None:
    """Cheap reachability check for the readiness probe. Raises on failure."""
    await _client().get_collections()
