"""FastAPI lifespan: phase-aware startup.

Two runtime modes, picked by `settings.phase`:

  Phase 1 — categorization + Excel showcase (DEFAULT)
    * NO LangGraph, NO checkpointer (Postgres / SQLite), NO MLflow.
    * Starts the in-process daily scheduler that refreshes the
      Excel report (off if `categorization_scheduled_enabled=False`).
    * Closes pooled httpx clients (GraphQL + REST) on shutdown.
    * Keeps `app.state.graph = None` so any handler that touches it
      raises a clear 503 instead of crashing on attribute access.

  Phase 2 — fix-mode lanes (capture / assist / autonomous)
    * Opens the async checkpointer.
    * Builds the compiled LangGraph and stashes it on `app.state.graph`.
    * Closes everything cleanly on shutdown.

Why two modes (not "always build the graph"):
  The Phase-1 demo runs on a laptop with no Postgres/Qdrant/MLflow
  containers. Forcing the graph build at startup would brick the
  showcase. Gating it on `phase` keeps the Phase-2 code path intact
  for when the project graduates.

Request handlers should ALWAYS read the graph from `request.app.state
.graph`; they should never import a module-level graph object.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, TYPE_CHECKING

from bugfix_ai.config.logging_config import configure_logging, get_logger
from bugfix_ai.config.settings import get_settings

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)


@asynccontextmanager
async def graph_lifespan(app: "FastAPI") -> AsyncIterator[None]:
    """FastAPI lifespan handler. Branches on `settings.phase`."""
    configure_logging()
    s = get_settings()
    log.info("app.startup.begin", phase=s.phase)

    if s.phase == "1":
        # ── Phase 1 ────────────────────────────────────────────────────────
        # No LangGraph / checkpointer. Start the daily scheduler.
        from bugfix_ai.api.scheduler import start_scheduler, stop_scheduler
        from bugfix_ai.categorization.service import aclose_clients

        app.state.graph = None
        app.state.checkpointer = None
        scheduler_task = start_scheduler()
        app.state.scheduler_task = scheduler_task
        log.info("app.startup.ready", phase="1", scheduler=bool(scheduler_task))
        try:
            yield
        finally:
            log.info("app.shutdown.begin", phase="1")
            await stop_scheduler(scheduler_task)
            await aclose_clients()
            log.info("app.shutdown.complete", phase="1")
        return

    # ── Phase 2 ────────────────────────────────────────────────────────────
    # Lazy imports keep Phase-1 startups from importing Postgres / LangGraph.
    from bugfix_ai.core.checkpointer import open_checkpointer
    from bugfix_ai.core.graph import build_graph

    async with open_checkpointer() as cp:
        graph = build_graph(cp)
        app.state.graph = graph
        app.state.checkpointer = cp
        log.info("app.startup.ready", phase="2")
        try:
            yield
        finally:
            log.info("app.shutdown.begin", phase="2")

    log.info("app.shutdown.complete", phase="2")
