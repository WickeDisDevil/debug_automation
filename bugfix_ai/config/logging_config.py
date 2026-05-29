"""Structured logging setup.

Why structlog and not stdlib logging directly:
  Every node in the LangGraph state machine emits events that need to
  be machine-parseable (JSON in production) AND human-readable in dev.
  structlog gives us both modes from one configuration, plus context
  variables — so a `correlation_id` bound at request entry shows up on
  every log line for the rest of that request without us threading it
  through every function call.

What this module exports:
  * configure_logging() — call ONCE at process start (FastAPI lifespan,
    CLI main(), worker entry). Idempotent in practice but not designed
    to be called per-request.
  * get_logger(name) — module-level loggers; the convention everywhere
    else in the codebase is `log = get_logger(__name__)`.
  * bind_context() / clear_context() — request-scoped key/value pairs
    that get merged into every subsequent log record on the same async
    context. Used by the API correlation-id middleware.

Output format:
  * development → ConsoleRenderer (colored, dev-friendly)
  * staging/production → JSONRenderer (one JSON object per line, ready
    for log shippers / Splunk / Loki).

Log level:
  Driven by `settings.environment`. DEBUG in dev, INFO everywhere else.
  We use a *filtering* bound logger so suppressed messages cost almost
  nothing — important because the retrieval/RRF inner loops emit a lot
  of optional debug events.
"""

from __future__ import annotations

import logging
import sys

import structlog

from bugfix_ai.config.settings import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = logging.DEBUG if settings.environment == "development" else logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if settings.environment == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_context(**kwargs: object) -> None:
    """Attach key/value pairs to the current async context's log record."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context vars — call at request boundaries."""
    structlog.contextvars.clear_contextvars()
