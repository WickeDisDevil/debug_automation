"""HTTP surface for the Phase-1 issue-categorization showcase.

Endpoints
---------
POST /issues/categorize/run
    Trigger an on-demand ingestion of OPEN issues from the configured
    repo, run the URL → LLM categorization pipeline, and refresh the
    Excel workbook on disk. Returns a JSON summary (counts per method,
    counts per category, duration, output path).

POST /issues/categorize/manual
    Categorize a single caller-supplied issue without hitting GitHub.
    Body accepts EITHER a normalized `IssueRecord` OR a raw GitHub
    issue dict (the parser converts it). Useful for ad-hoc one-offs
    and for re-classifying a single ticket from a paste.

GET  /issues/report.xlsx
    Stream the most recently generated workbook for download. 404 if
    no run has produced one yet.

GET  /issues/report/summary
    Lightweight JSON status: does a workbook exist, when was it
    written, how big, where on disk. Intended for the demo UI / curl
    sanity checks.

Why these endpoints are non-mutating w.r.t. GitHub:
  Phase-1 categorization is read-only. We never PATCH, comment, or
  label on GitHub from this module. That keeps the showcase surface
  safe to expose without the heavier HITL machinery the Phase-2
  fix-mode lanes carry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bugfix_ai.categorization.pipeline import categorize
from bugfix_ai.categorization.schema import CategorizedIssue, IssueRecord
from bugfix_ai.categorization.service import run_categorization
from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.integrations.github.issues_parser import parse_issue

log = get_logger(__name__)

router = APIRouter(prefix="/issues", tags=["issues"])


# ── Request / response models ───────────────────────────────────────────────


class RunCategorizationRequest(BaseModel):
    """Knobs for an on-demand categorization run."""

    max_issues: int | None = Field(
        default=None,
        ge=1,
        le=10_000,
        description="Hard cap on issues processed (after open-state filter).",
    )
    force_llm: bool = Field(
        default=False,
        description="Skip the URL short-circuit; always invoke the LLM.",
    )
    max_concurrency: int = Field(default=4, ge=1, le=16)


class RunCategorizationResponse(BaseModel):
    ingested: int
    categorized: int
    by_method: dict[str, int]
    by_category: dict[str, int]
    out_path: str
    started_at: str
    finished_at: str
    duration_seconds: float


class ManualCategorizeRequest(BaseModel):
    """Either a normalized IssueRecord OR a raw GitHub issue dict.

    Exactly one of `record` / `raw` must be supplied. The endpoint
    validates and rejects ambiguous input with 400.
    """

    record: IssueRecord | None = None
    raw: dict[str, Any] | None = None
    force_llm: bool = False


class ReportSummaryResponse(BaseModel):
    exists: bool
    path: str
    size_bytes: int | None = None
    modified_at: str | None = None


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/categorize/run", response_model=RunCategorizationResponse)
async def categorize_run(req: RunCategorizationRequest) -> RunCategorizationResponse:
    """Ingest open issues, categorize, refresh the Excel workbook."""
    try:
        summary = await run_categorization(
            max_issues=req.max_issues,
            force_llm=req.force_llm,
            max_concurrency=req.max_concurrency,
        )
    except Exception as e:  # noqa: BLE001
        log.error("issues_api.run_failed", error=str(e)[:500])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Categorization run failed: {e}",
        ) from e

    return RunCategorizationResponse(**summary)


@router.post("/categorize/manual", response_model=CategorizedIssue)
async def categorize_manual(req: ManualCategorizeRequest) -> CategorizedIssue:
    """Categorize a single ad-hoc issue without hitting GitHub."""
    if (req.record is None) == (req.raw is None):
        # Both supplied OR neither supplied → ambiguous.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide exactly one of `record` or `raw`.",
        )

    issue = req.record if req.record is not None else parse_issue(req.raw or {})
    try:
        return await categorize(issue, force_llm=req.force_llm)
    except Exception as e:  # noqa: BLE001
        log.error("issues_api.manual_failed", issue=issue.issue_id, error=str(e)[:500])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Categorization failed: {e}",
        ) from e


def _report_path() -> Path:
    s = get_settings()
    return Path(s.excel_output_dir) / s.excel_report_filename


@router.get("/report.xlsx")
async def download_report() -> FileResponse:
    """Stream the most recently generated workbook."""
    path = _report_path()
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No report yet. Trigger one with POST /issues/categorize/run "
                "or wait for the daily scheduled run."
            ),
        )
    return FileResponse(
        path=str(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


@router.get("/report/summary", response_model=ReportSummaryResponse)
async def report_summary() -> ReportSummaryResponse:
    """Lightweight status: does the workbook exist + when was it written."""
    path = _report_path()
    if not path.exists():
        return ReportSummaryResponse(exists=False, path=str(path))
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return ReportSummaryResponse(
        exists=True,
        path=str(path),
        size_bytes=stat.st_size,
        modified_at=modified,
    )
