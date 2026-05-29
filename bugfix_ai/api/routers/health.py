"""Liveness and readiness probes — phase-aware.

`/healthz` is a cheap liveness check (process is up, can serve a request).
`/health` is the readiness check — pings only the dependencies that are
EXPECTED to be live for the current `settings.phase`.

Phase 1 (categorization showcase):
  * NO LangGraph, NO Postgres, NO Qdrant.
  * The probe checks: settings load + (optionally) the latest Excel
    report on disk. Returns 200 even if no run has produced a workbook
    yet — the report is a delivery artifact, not a liveness dependency.

Phase 2 (fix-mode lanes):
  * The probe checks: settings + compiled graph + Postgres + Qdrant.
  * Any failure flips the response to 503 so a load-balancer can drop
    the instance from rotation.

We deliberately do NOT cache the readiness response — an uncached probe
is the whole point.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

router = APIRouter(tags=["health"])
log = get_logger(__name__)


@router.get("/healthz")
async def liveness() -> dict[str, str]:
    """Liveness — process can answer."""
    return {"status": "ok"}


@router.get("/health")
async def readiness(request: Request) -> JSONResponse:
    """Readiness — only checks dependencies the active phase actually uses."""
    checks: dict[str, dict] = {}
    overall_ok = True

    # Settings load (cheap, both phases).
    try:
        settings = get_settings()
        checks["settings"] = {
            "ok": True,
            "environment": settings.environment,
            "phase": settings.phase,
        }
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"status": "degraded", "checks": {"settings": {"ok": False, "error": str(e)[:200]}}},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    if settings.phase == "1":
        # Phase 1: no graph / Postgres / Qdrant. Surface the report file
        # (if any) for visibility, but its absence does NOT trip 503.
        report_path = Path(settings.excel_output_dir) / settings.excel_report_filename
        checks["excel_report"] = {
            "ok": True,
            "exists": report_path.exists(),
            "path": str(report_path),
        }
    else:
        # Phase 2: graph + dependency pings.
        graph = getattr(request.app.state, "graph", None)
        checks["graph"] = {"ok": graph is not None}
        overall_ok &= checks["graph"]["ok"]

        try:
            from sqlalchemy import text

            from bugfix_ai.memory.rules_db import session

            async with session() as s:
                await s.execute(text("SELECT 1"))
            checks["postgres"] = {"ok": True}
        except Exception as e:  # noqa: BLE001
            checks["postgres"] = {"ok": False, "error": str(e)[:200]}
            overall_ok = False

        try:
            from bugfix_ai.memory.vector_store import ping_qdrant

            await ping_qdrant()
            checks["qdrant"] = {"ok": True}
        except Exception as e:  # noqa: BLE001
            checks["qdrant"] = {"ok": False, "error": str(e)[:200]}
            overall_ok = False

    code = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        {"status": "ok" if overall_ok else "degraded", "checks": checks},
        status_code=code,
    )
