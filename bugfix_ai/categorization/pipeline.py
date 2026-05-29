"""End-to-end categorization pipeline (URL extractor + rule augmenter + LLM).

Phase-1 ordering (per spec):

    1. URL extraction
       Look for a /driver/<path> URL in the issue body/title/html_url.
       If found, the FULL path after `/driver/` becomes the category.
       Method = "url", confidence = 1.0 (deterministic).

    2. LLM fallback
       If no /driver URL exists, hand the issue to GPT-oss-20B (via
       `chat_structured`) which returns a free-form `<area>:<sub-area>`
       category plus a closed-Literal priority. Method = "nlp".

    3. Rule augmenter (every row)
       The legacy taxonomy classifier still runs to produce
       priority + component when neither extractor supplied them. It
       does NOT override the URL-derived category.

    4. Last-ditch fallback
       If the URL extractor abstained AND the LLM failed, the row is
       emitted with category = "uncategorized", method = "fallback".
       This keeps the batch alive — the row is visibly low-trust in the
       output so a reviewer can notice and re-run with --force-llm.

Concurrency:
  `categorize_many` runs LLM calls with bounded concurrency
  (asyncio.Semaphore) so we don't overwhelm a local Ollama instance.
  Default 4 in-flight; tune via `max_concurrency`. The URL extractor is
  pure CPU and runs inline.

Public surface:
    categorize(issue) -> CategorizedIssue
    categorize_many(issues) -> list[CategorizedIssue]
"""

from __future__ import annotations

import asyncio
from typing import Iterable

from bugfix_ai.categorization.nlp_classifier import classify_issue_with_llm
from bugfix_ai.categorization.rules import (
    classify_issue as rule_classify,
    extract_driver_category,
)
from bugfix_ai.categorization.schema import (
    CATEGORY_UNCATEGORIZED,
    CategorizedIssue,
    IssueRecord,
    LLMCategorization,
    Method,
    RuleVerdict,
)
from bugfix_ai.config.logging_config import get_logger

log = get_logger(__name__)


# ── Row builders ────────────────────────────────────────────────────────────


def _row_from_url(
    issue: IssueRecord, driver_path: str, rule: RuleVerdict
) -> CategorizedIssue:
    """Build a row whose category came from a /driver/... URL.

    The rule augmenter's priority/component still merge in (label-derived
    priority is more trustworthy than no priority at all), but the URL
    decides the category.
    """
    method: Method = "hybrid" if (rule.priority != "unspecified" or rule.component) else "url"
    return CategorizedIssue(
        issue_id=issue.issue_id,
        title=issue.title,
        category=driver_path,
        priority=rule.priority,
        status=issue.state,
        sub_category="",
        component=rule.component or "",
        method=method,
        confidence=1.0,
        matched_signals=[
            f"url:/driver/{driver_path}",
            *rule.matched_label_signals,
            *rule.matched_keyword_signals,
        ],
        html_url=issue.html_url,
        repository=issue.repository,
        author=issue.author,
        labels=list(issue.labels),
        created_at=issue.created_at,
        updated_at=issue.updated_at,
    )


def _row_from_llm(
    issue: IssueRecord, llm: LLMCategorization, rule: RuleVerdict
) -> CategorizedIssue:
    """Build a row whose category came from the LLM (no /driver URL was present).

    Priority preference: the rule layer's label-derived priority wins
    over the LLM's guess (operators trust their own labels). Component
    falls back to whichever layer found one.
    """
    final_priority = rule.priority if rule.priority != "unspecified" else llm.priority
    component = rule.component or llm.component or ""
    confidence = round(max(rule.confidence, llm.confidence), 3)
    return CategorizedIssue(
        issue_id=issue.issue_id,
        title=issue.title,
        category=llm.category,
        priority=final_priority,
        status=issue.state,
        sub_category=llm.sub_category,
        component=component,
        method="nlp",
        confidence=confidence,
        matched_signals=[
            *rule.matched_label_signals,
            *rule.matched_keyword_signals,
            *([f"llm:{llm.reasoning[:80]}"] if llm.reasoning else []),
        ],
        html_url=issue.html_url,
        repository=issue.repository,
        author=issue.author,
        labels=list(issue.labels),
        created_at=issue.created_at,
        updated_at=issue.updated_at,
    )


def _row_uncategorized(issue: IssueRecord, rule: RuleVerdict) -> CategorizedIssue:
    """Last-ditch row: URL extractor abstained AND LLM failed."""
    return CategorizedIssue(
        issue_id=issue.issue_id,
        title=issue.title,
        category=CATEGORY_UNCATEGORIZED,
        priority=rule.priority,
        status=issue.state,
        sub_category="",
        component=rule.component or "",
        method="fallback",
        confidence=0.0,
        matched_signals=[
            *rule.matched_label_signals,
            *rule.matched_keyword_signals,
        ],
        html_url=issue.html_url,
        repository=issue.repository,
        author=issue.author,
        labels=list(issue.labels),
        created_at=issue.created_at,
        updated_at=issue.updated_at,
    )


# ── Public API ──────────────────────────────────────────────────────────────


async def categorize(issue: IssueRecord, *, force_llm: bool = False) -> CategorizedIssue:
    """Categorize one issue.

    Parameters
    ----------
    force_llm:
        If True, skip the URL short-circuit and always invoke the LLM.
        Useful for evaluation or when refreshing categorizations after
        a model upgrade.
    """
    # Always run the rule augmenter so we have priority/component on hand,
    # regardless of which path produces the category.
    rule_v = rule_classify(issue)

    if not force_llm:
        driver_path = extract_driver_category(issue)
        if driver_path:
            log.info(
                "categorize.url_match",
                issue=issue.issue_id,
                category=driver_path,
            )
            return _row_from_url(issue, driver_path, rule_v)

    # LLM path (forced, or no /driver URL).
    llm_v = await classify_issue_with_llm(issue)
    if llm_v is not None:
        log.info(
            "categorize.llm_match",
            issue=issue.issue_id,
            category=llm_v.category,
            confidence=llm_v.confidence,
        )
        return _row_from_llm(issue, llm_v, rule_v)

    log.warning("categorize.uncategorized", issue=issue.issue_id)
    return _row_uncategorized(issue, rule_v)


async def categorize_many(
    issues: Iterable[IssueRecord],
    *,
    force_llm: bool = False,
    max_concurrency: int = 4,
) -> list[CategorizedIssue]:
    """Categorize a batch with bounded LLM concurrency.

    Order of the returned list matches the input order. Failures in
    individual issues are caught and converted to "uncategorized" rows
    so the batch never aborts mid-way.
    """
    issues_list = list(issues)
    if not issues_list:
        return []

    sem = asyncio.Semaphore(max_concurrency)

    async def _one(idx: int, issue: IssueRecord) -> tuple[int, CategorizedIssue]:
        async with sem:
            try:
                result = await categorize(issue, force_llm=force_llm)
            except Exception as e:  # noqa: BLE001 — defensive; never crash a batch
                log.error(
                    "categorize.unexpected_error",
                    issue=issue.issue_id,
                    error=str(e)[:300],
                )
                rule_v = rule_classify(issue)
                result = _row_uncategorized(issue, rule_v)
            return idx, result

    pairs = await asyncio.gather(*(_one(i, it) for i, it in enumerate(issues_list)))
    pairs.sort(key=lambda p: p[0])
    return [p[1] for p in pairs]
