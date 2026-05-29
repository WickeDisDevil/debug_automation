"""CLI: ingest open GitHub Issues, categorize them, write the Excel report.

This is the Phase-1 showcase deliverable. It produces the exact
spreadsheet the user asked for:
  * one sheet per category
  * each sheet's columns: Issue ID | Title | Category | Priority | Status
  * a Summary sheet listing every category with its count

Designed to be the entry point a GitHub Actions workflow calls daily in
production. The FastAPI app's in-process scheduler invokes the same
underlying `run_categorization()` for laptop / staging demos.

Usage examples
--------------
    # Default: GraphQL ingest of all OPEN issues, write
    # ./out/issues_categorized.xlsx
    python -m scripts.categorize_issues

    # Cap to 200 issues (handy for a quick demo)
    python -m scripts.categorize_issues --max-issues 200

    # Force the LLM on every issue (skip /driver URL short-circuit)
    python -m scripts.categorize_issues --force-llm

    # Different output location
    python -m scripts.categorize_issues --out ./reports/today.xlsx

    # Read issues from a local JSON snapshot instead of GitHub.
    # Expects a JSON array of raw GitHub issue dicts.
    python -m scripts.categorize_issues --from-json ./samples/issues.json

Exit codes
----------
    0  success — workbook written
    1  no issues matched (empty repo, bad token, or filters)
    2  unexpected runtime error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from bugfix_ai.categorization.excel_exporter import write_workbook
from bugfix_ai.categorization.pipeline import categorize_many
from bugfix_ai.categorization.schema import IssueRecord
from bugfix_ai.categorization.service import aclose_clients, run_categorization
from bugfix_ai.config.logging_config import configure_logging, get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.integrations.github.issues_parser import parse_issues

log = get_logger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="categorize_issues",
        description="Ingest OPEN GitHub Issues, categorize, export Excel.",
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Output .xlsx path. Defaults to "
            "<excel_output_dir>/<excel_report_filename> from settings."
        ),
    )
    p.add_argument(
        "--max-issues",
        type=int,
        default=None,
        help="Hard cap on number of issues to process.",
    )
    p.add_argument(
        "--force-llm",
        action="store_true",
        help="Skip the /driver URL short-circuit; always invoke the LLM.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max in-flight LLM calls (default: 4).",
    )
    p.add_argument(
        "--from-json",
        default=None,
        help=(
            "Skip GitHub. Read raw issues from a local JSON file (array of "
            "raw GitHub issue dicts). Useful for demos / CI."
        ),
    )
    return p


# ── Local-JSON path (no GitHub) ─────────────────────────────────────────────


def _load_from_json(path: str, max_issues: int | None) -> list[IssueRecord]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--from-json path not found: {path}")
    raw_items = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise ValueError("--from-json file must contain a JSON array")
    if max_issues is not None:
        raw_items = raw_items[:max_issues]
    return parse_issues(raw_items)


async def _run_from_json(args: argparse.Namespace) -> int:
    issues = _load_from_json(args.from_json, args.max_issues)
    if not issues:
        log.warning("categorize_cli.no_issues", source="json")
        return 1
    log.info("categorize_cli.collected", source="json", count=len(issues))
    rows = await categorize_many(
        issues, force_llm=args.force_llm, max_concurrency=args.max_concurrency
    )
    s = get_settings()
    target = Path(args.out) if args.out else Path(s.excel_output_dir) / s.excel_report_filename
    written = write_workbook(rows, target)
    print(f"Wrote {len(rows)} categorized issues to {written}")
    return 0


# ── GitHub path (preferred) ─────────────────────────────────────────────────


async def _run_from_github(args: argparse.Namespace) -> int:
    summary = await run_categorization(
        max_issues=args.max_issues,
        force_llm=args.force_llm,
        max_concurrency=args.max_concurrency,
        out_path=args.out,
    )
    if summary["categorized"] == 0:
        log.warning("categorize_cli.no_issues", source="github")
        return 1
    print(
        f"Wrote {summary['categorized']} categorized issues to "
        f"{summary['out_path']} "
        f"(by_method={summary['by_method']}, "
        f"categories={len(summary['by_category'])}, "
        f"duration={summary['duration_seconds']:.1f}s)"
    )
    return 0


# ── Main ────────────────────────────────────────────────────────────────────


async def _run(args: argparse.Namespace) -> int:
    try:
        if args.from_json:
            return await _run_from_json(args)
        return await _run_from_github(args)
    finally:
        # Release pooled httpx clients inside this event loop.
        try:
            await aclose_clients()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging()
    try:
        rc = asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        log.error("categorize_cli.fatal", error=str(e)[:500])
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(rc)


if __name__ == "__main__":
    main()
