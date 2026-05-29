"""Pydantic schemas for issue categorization.

Two layers of types:

  * `IssueRecord` — the normalized input row, derived from a raw GitHub
    Issue payload by `integrations.github.issues_parser`. Anything the
    classifiers need MUST live on this type so neither classifier needs
    to know the GitHub REST shape.

  * `CategorizedIssue` — the output row. This is the showcase deliverable
    for Phase 1: every column the user asked for (issue_id, title,
    category, priority, status) is here, plus the provenance fields
    (`method`, `confidence`, `sub_category`, `component`) so a reviewer
    can tell URL-driven decisions from rule- or LLM-driven ones.

  * `LLMCategorization` — the constrained output schema we hand to the
    GPT-oss-20B JSON-mode call. Field constraints (Literal priority,
    bounded length) are what make `chat_structured` retry-on-failure
    actually catch hallucinations.

Why `Category` is a FREE-FORM STRING (not a closed Literal):
  Phase 1 categorizes by the path segment after `/driver/` in the bug
  URL embedded in the issue body. That path is dynamic — it could be
  `audio/codec/foo.c`, `display/dp/link.c`, etc. Closing the taxonomy
  would force the URL extractor to invent a mapping the user didn't
  ask for. We keep the field free-form and let the Excel exporter
  group rows by exact category string. The LLM fallback (for issues
  that have no /driver URL) also returns a free-form category string.

A note on `priority`: GitHub Issues don't carry a built-in priority,
only labels. We map a curated set of label strings to a 4-level scale
(critical/high/medium/low) plus a sentinel "unspecified" for issues
that have no recognizable priority label.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Taxonomies ──────────────────────────────────────────────────────────────


# Category is a free-form string. The driver-URL extractor produces values
# like "audio/codec/foo.c"; the LLM fallback produces values like
# "audio:codec" or "build:dependencies". Sentinel values:
#   "uncategorized" — neither extractor produced anything (LLM failed too)
Category = str

# Sentinels referenced by code paths that need a definite literal.
CATEGORY_UNCATEGORIZED: Category = "uncategorized"

Priority = Literal["critical", "high", "medium", "low", "unspecified"]

Status = Literal["open", "closed"]

# Provenance for each row's classification.
#   url      — extracted from a /driver/... URL in the issue body
#   nlp      — produced by the LLM (no /driver URL was found)
#   hybrid   — URL provided category, rule taxonomy provided priority/component
#   rule     — legacy rule-only path (kept for backward compat)
#   fallback — rule + LLM both failed; row is "uncategorized"
Method = Literal["url", "nlp", "hybrid", "rule", "fallback"]


# ── Input ───────────────────────────────────────────────────────────────────


class IssueRecord(BaseModel):
    """Normalized input — GitHub-agnostic so the classifier doesn't care
    where the issue came from (REST API today, JSON dump tomorrow)."""

    issue_id: int = Field(ge=1, description="GitHub issue number, repo-scoped.")
    node_id: str = Field(default="", description="GitHub GraphQL node id; opaque.")
    title: str = Field(min_length=1, max_length=2000)
    body: str = Field(default="", max_length=200_000)
    labels: list[str] = Field(default_factory=list)
    state: Status = "open"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    closed_at: datetime | None = None
    html_url: str = ""
    repository: str = ""
    author: str = ""
    assignees: list[str] = Field(default_factory=list)


# ── Intermediate (rule layer's output) ──────────────────────────────────────


class RuleVerdict(BaseModel):
    """What the rule classifier produces. Confidence drives whether we
    bother the LLM."""

    category: Category
    priority: Priority
    component: str | None = None  # appended as sub-tag, e.g. "bug:auth-svc"
    matched_label_signals: list[str] = Field(default_factory=list)
    matched_keyword_signals: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ── LLM output (constrained for chat_structured + Pydantic retry loop) ──────


class LLMCategorization(BaseModel):
    """Schema returned by the GPT-oss-20B structured call.

    `category` is a short free-form label (e.g. "audio:codec",
    "display:dp", "kernel:memory"). The model is steered toward a
    `<area>:<sub-area>` style by the system prompt but free-form keeps
    the schema permissive — pydantic still bounds length to keep
    runaway outputs out of the spreadsheet.

    `priority` remains a closed Literal so JSON-mode + Pydantic catch
    hallucinated severities and the retry loop can re-prompt with the
    exact validation error.
    """

    category: Category = Field(min_length=1, max_length=120)
    priority: Priority
    sub_category: str = Field(default="", max_length=120)
    component: str = Field(default="", max_length=120)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=600)


# ── Output (showcase deliverable) ───────────────────────────────────────────


class CategorizedIssue(BaseModel):
    """The row that appears in the Excel sheet / API response.

    Mandatory showcase columns (in this order, per spec):
        issue_id | title | category | priority | status

    The remaining columns exist to make the categorization auditable
    and to feed downstream automation (component → routing, confidence
    → triage queue thresholds).
    """

    issue_id: int
    title: str
    category: Category
    priority: Priority
    status: Status

    # Provenance / auditing
    sub_category: str = ""
    component: str = ""
    method: Method = "rule"
    confidence: float = 0.0
    matched_signals: list[str] = Field(default_factory=list)

    # Useful context (also written to Excel as later columns)
    html_url: str = ""
    repository: str = ""
    author: str = ""
    labels: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def showcase_row(self) -> dict[str, object]:
        """The five-column row the user asked for, in display order."""
        return {
            "Issue ID": self.issue_id,
            "Title": self.title,
            "Category": self._display_category(),
            "Priority": self.priority,
            "Status": self.status,
        }

    def full_row(self) -> dict[str, object]:
        """All columns, used by the Excel exporter."""
        return {
            **self.showcase_row(),
            "Sub-Category": self.sub_category,
            "Component": self.component,
            "Method": self.method,
            "Confidence": round(self.confidence, 3),
            "Matched Signals": ", ".join(self.matched_signals),
            "Labels": ", ".join(self.labels),
            "Author": self.author,
            "Repository": self.repository,
            "URL": self.html_url,
            "Created At": self.created_at.isoformat() if self.created_at else "",
            "Updated At": self.updated_at.isoformat() if self.updated_at else "",
        }

    def _display_category(self) -> str:
        if self.sub_category:
            return f"{self.category}:{self.sub_category}"
        if self.component:
            return f"{self.category}:{self.component}"
        return self.category
