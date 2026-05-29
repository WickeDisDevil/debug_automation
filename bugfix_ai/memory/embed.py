"""In-process embeddings via sentence-transformers.

Why local sentence-transformers and not a hosted embedding API:
  * Constraint from the user: only GPT-oss-20B is allowed for chat.
  * sentence-transformers ships HF model weights — these are model weights,
    not a hosted "open-source AI service", and they run inside this Python
    process. Same compute envelope as the rest of the app.
  * BAAI/bge-small-en-v1.5 is 33M params, 384-dim, runs at ~2k texts/sec on CPU,
    and is consistently top-ranked on MTEB retrieval benchmarks at this size.

Production fixes:
  * Lazy load (no model load at import time).
  * In-memory LRU cache for repeated embeds (the same query is often run
    multiple times across a single ticket lifecycle).
  * Composite embedding helper for richer query signal.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)

_MODEL: SentenceTransformer | None = None
_MODEL_LOCK = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                settings = get_settings()
                log.info(
                    "embed.model.loading",
                    model=settings.embedding_model_name,
                    device=settings.embedding_device,
                )
                _MODEL = SentenceTransformer(
                    settings.embedding_model_name,
                    device=settings.embedding_device,
                )
                log.info("embed.model.ready")
    return _MODEL


# ── Public API ──────────────────────────────────────────────────────────────


def embed_text(text: str) -> list[float]:
    """Embed a single string and return a 384-dim list[float]."""
    cleaned = _preprocess(text)
    return _embed_cached(cleaned)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many strings in one forward pass (much faster than looping)."""
    cleaned = [_preprocess(t) for t in texts]
    model = _get_model()
    arr = model.encode(
        cleaned,
        normalize_embeddings=True,  # cosine ≈ dot product after normalize
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return arr.tolist()


def embed_composite(
    *,
    title: str = "",
    description: str = "",
    error_log: str = "",
    error_type: str = "",
) -> list[float]:
    """Build a richer query embedding by weighting multiple signals.

    Why composite > raw error log: a CodeQL alert title like
    "Use of uninitialized variable" carries strong retrieval signal that
    raw stack traces don't. Concatenating into a single tagged string lets
    the embedding model attend to all of them.
    """
    parts: list[str] = []
    if error_type:
        parts.append(f"[ERROR_TYPE] {error_type}")
    if title:
        parts.append(f"[TITLE] {title}")
    if description:
        parts.append(f"[DESCRIPTION] {description[:1000]}")
    if error_log:
        parts.append(f"[LOG] {error_log[:4000]}")
    composite = "\n".join(parts) if parts else error_log
    return embed_text(composite)


# ── Internals ────────────────────────────────────────────────────────────────


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
_WS_RE = re.compile(r"\s+")


def _preprocess(text: str) -> str:
    if not text:
        return ""
    text = _ANSI_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    # bge-small handles up to 512 tokens (~ 2000 chars typical English).
    return text[:8000]


@lru_cache(maxsize=4096)
def _embed_cached(text: str) -> tuple[float, ...]:
    """Single-text embedding, cached. Tuple for hashability of the cache value."""
    if not text:
        # Return zero-vector for empty string (avoids None handling everywhere)
        dims = get_settings().embedding_dimensions
        return tuple([0.0] * dims)
    model = _get_model()
    arr = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return tuple(float(x) for x in arr[0])
