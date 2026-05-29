"""High-level orchestration for the Phase-1 categorization run.

This is the single entry point used by:
  * the FastAPI on-demand endpoint (`POST /issues/categorize/run`)
  * the in-process daily scheduler (`api/scheduler.py`)
  * the standalone CLI (`scripts/categorize_issues.py`)

What `run_categorization()` does end-to-end:
  1. Pull OPEN issues from GitHub (GraphQL by default; REST if
     `categorization_use_graphql=False`).
  2. Parse each raw issue into an `IssueRecord`.
  3. Categorize the batch:
        URL extractor ─→ if no /driver URL ─→ LLM
  4. Write the workbook (one sheet per category) to
     `<excel_output_dir>/<excel_report_filename>`.
  5. Return a summary dict the caller can serialize / log.

Design notes:
  * The function is async (matches the rest of the stack).
  * It owns its own httpx client lifecycles via the module-level
    `aclose()` helpers — callers don't have to.
  * Failures in individual issues never abort the batch; bad rows
    become `uncategorized` so the workbook still renders.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bugfix_ai.categorization.excel_exporter import write_workbook
from bugfix_ai.categorization.pipeline import categorize_many
from bugfix_ai.categorization.schema import CategorizedIssue, IssueRecord
from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.integrations.github.issues_parser import parse_issue

log = get_logger(__name__)


# ── Source selection (GraphQL vs REST) ──────────────────────────────────────


async def _collect_open_issues(max_issues: int | None) -> list[IssueRecord]:
    """Fetch open issues using the configured client and parse them."""
    s = get_settings()
    records: list[IssueRecord] = []

    if s.categorization_use_graphql:
        from bugfix_ai.integrations.github.graphql_client import list_open_issues

        async for raw in list_open_issues():
            try:
                records.append(parse_issue(raw))
            except Exception as e:  # noqa: BLE001
                log.warning("service.parse_failed", error=str(e)[:300])
                continue
            if max_issues is not None and len(records) >= max_issues:
                break
    else:
        from bugfix_ai.integrations.github.issues_client import list_issues

        # Phase 1 ingests OPEN issues only — pin the state regardless of
        # caller-supplied defaults.
        async for raw in list_issues(state="open"):
            try:
                records.append(parse_issue(raw))
            except Exception as e:  # noqa: BLE001
                log.warning("service.parse_failed", error=str(e)[:300])
                continue
            if max_issues is not None and len(records) >= max_issues:
                break

    return records


# ── Public orchestrator ─────────────────────────────────────────────────────


async def run_categorization(
    *,
    max_issues: int | None = None,
    force_llm: bool = False,
    max_concurrency: int = 4,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full ingest → categorize → Excel pipeline.

    Parameters
    ----------
    max_issues:
        Hard cap; useful for demos. None = no cap.
    force_llm:
        Skip the URL short-circuit and always invoke the LLM. Used to
        re-classify the long tail when refreshing categorizations.
    max_concurrency:
        Bound on in-flight LLM calls.
    out_path:
        Override for the Excel destination. Defaults to
        `<excel_output_dir>/<excel_report_filename>`.

    Returns
    -------
    dict
        {
          "ingested": int,
          "categorized": int,
          "by_method": {method -> count},
          "by_category": {category -> count},
          "out_path": str,
          "started_at": ISO-8601,
          "finished_at": ISO-8601,
        }
    """
    s = get_settings()
    started = datetime.now(timezone.utc)
    log.info(
        "service.run.start",
        use_graphql=s.categorization_use_graphql,
        max_issues=max_issues,
        force_llm=force_llm,
    )

    issues = await _collect_open_issues(max_issues=max_issues)
    log.info("service.run.ingested", count=len(issues))

    rows: list[CategorizedIssue] = await categorize_many(
        issues, force_llm=force_llm, max_concurrency=max_concurrency
    )

    target = Path(out_path) if out_path else Path(s.excel_output_dir) / s.excel_report_filename
    written = write_workbook(rows, target)

    by_method: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for r in rows:
        by_method[r.method] = by_method.get(r.method, 0) + 1
        by_category[r.category] = by_category.get(r.category, 0) + 1

    finished = datetime.now(timezone.utc)
    summary = {
        "ingested": len(issues),
        "categorized": len(rows),
        "by_method": by_method,
        "by_category": by_category,
        "out_path": str(written),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    log.info("service.run.complete", **{k: v for k, v in summary.items() if k != "by_category"})
    return summary


# ── Resource cleanup (called from FastAPI lifespan) ────────────────────────


async def aclose_clients() -> None:
    """Close any pooled HTTP clients opened during a categorization run.

    Both the GraphQL and REST clients are imported lazily in
    `_collect_open_issues`; we mirror that here so we don't drag in
    httpx pools we never opened.
    """
    s = get_settings()
    if s.categorization_use_graphql:
        try:
            from bugfix_ai.integrations.github.graphql_client import aclose

            await aclose()
        except Exception as e:  # noqa: BLE001
            log.warning("service.aclose.graphql_failed", error=str(e)[:200])
    try:
        from bugfix_ai.integrations.github.issues_client import aclose as rest_aclose

        await rest_aclose()
    except Exception as e:  # noqa: BLE001
        log.warning("service.aclose.rest_failed", error=str(e)[:200])
