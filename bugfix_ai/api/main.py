"""FastAPI entry point — phase-aware router mounting.

  uvicorn bugfix_ai.api.main:app --host 0.0.0.0 --port 8000

Routers mounted:
  * Always:       health, issues (Phase-1 categorization showcase)
  * Phase 2 only: session, ingest (Phase-2 LangGraph + code-scanning)

The compiled LangGraph + checkpointer (Phase 2 only) is constructed
once in `graph_lifespan` and stashed on `app.state` so request
handlers can grab it cheaply.
"""

from __future__ import annotations

from fastapi import FastAPI

from bugfix_ai.api.middleware.auth import APIKeyMiddleware, CorrelationIdMiddleware
from bugfix_ai.api.routers import health, issues
from bugfix_ai.config.settings import get_settings
from bugfix_ai.core.app_factory import graph_lifespan


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="BugFix AI",
        version="0.1.0",
        description=(
            "Phase 1: ingests OPEN GitHub Issues, classifies each by the "
            "`/driver/...` URL in the bug body (LLM fallback for issues "
            "without one), and exports a downloadable Excel workbook with "
            "one sheet per category. Phase 2 layers a LangGraph-based "
            "fix-mode assistant on top."
        ),
        lifespan=graph_lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
    )

    # Order matters: correlation id first (so auth failures get logged
    # with cid), then auth.
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # Always-on routers.
    app.include_router(health.router)
    app.include_router(issues.router)

    # Phase-2 routers: only mount when the LangGraph stack is alive.
    if settings.phase == "2":
        from bugfix_ai.api.routers import ingest, session

        app.include_router(session.router)
        app.include_router(ingest.router)

    return app


app = create_app()
