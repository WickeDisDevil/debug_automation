"""Rule-based categorization layer.

Two complementary passes live here:

  1. `extract_driver_category(issue)` — Phase-1 PRIMARY classifier.
     Scans the issue body / title / html_url for a URL containing
     `/driver/...` and returns the FULL PATH after `/driver/` as the
     category. This is the deterministic, zero-cost path the user asked
     for: every issue that includes a driver-source URL gets a
     reproducible, human-auditable category with no model in the loop.

  2. `classify_issue(issue)` — taxonomy-based AUGMENTER. Loads
     `issue_taxonomy.yaml` and returns priority + component (and a legacy
     category, retained for backward-compat callers). In the new pipeline
     ordering, this layer no longer drives the category — it only
     supplies priority and component hints when the URL or LLM path
     didn't.

Why a YAML config (still) for the augmenter:
  * Operators (not engineers) maintain the label vocabulary.
  * Label conventions drift over time; reloading is a one-line edit.
  * The taxonomy is the same artifact reviewed by leadership during
    triage process audits — keeping it data, not code, makes that loop
    fast.

Confidence semantics for `RuleVerdict` (used only when the URL extractor
abstains AND the LLM is going to run anyway):
  * +0.6 if any *label* keyword for the chosen category matched.
  * +0.3 if any *title/body* keyword matched (less reliable than a label).
  * +0.1 if a component matched (signals the issue is well-tagged).
  * +0.0 if everything is empty — we still return a category="other" with
         confidence=0.0.

The functions are intentionally pure and do no I/O once the YAML is
loaded — they can be unit-tested without a network or a DB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import yaml

from bugfix_ai.categorization.schema import (
    Category,
    IssueRecord,
    Priority,
    RuleVerdict,
)
from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)


# ── Loaded taxonomy ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CategoryRule:
    name: Category
    label_keywords: tuple[str, ...]
    title_keywords: tuple[str, ...]


@dataclass(frozen=True)
class _PriorityRule:
    level: Priority
    labels: tuple[str, ...]


@dataclass(frozen=True)
class _ComponentRule:
    name: str
    labels: tuple[str, ...]
    paths: tuple[str, ...]


@dataclass(frozen=True)
class _Taxonomy:
    categories: tuple[_CategoryRule, ...]
    priorities: tuple[_PriorityRule, ...]   # ordered, highest severity first
    components: tuple[_ComponentRule, ...]


@lru_cache(maxsize=1)
def _taxonomy() -> _Taxonomy:
    path = Path(get_settings().issue_taxonomy_path)
    if not path.exists():
        log.warning("issue_taxonomy.missing", path=str(path))
        return _Taxonomy(categories=(), priorities=(), components=())

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    categories = tuple(
        _CategoryRule(
            name=c["name"],
            label_keywords=tuple(s.lower() for s in (c.get("label_keywords") or [])),
            title_keywords=tuple(s.lower() for s in (c.get("title_keywords") or [])),
        )
        for c in (raw.get("categories") or [])
    )
    priorities = tuple(
        _PriorityRule(
            level=p["level"],
            labels=tuple(s.lower() for s in (p.get("labels") or [])),
        )
        for p in (raw.get("priority_labels") or [])
    )
    components = tuple(
        _ComponentRule(
            name=c["name"],
            labels=tuple(s.lower() for s in (c.get("labels") or [])),
            paths=tuple(s.lower() for s in (c.get("paths") or [])),
        )
        for c in (raw.get("components") or [])
    )
    log.info(
        "issue_taxonomy.loaded",
        categories=len(categories),
        priorities=len(priorities),
        components=len(components),
    )
    return _Taxonomy(categories=categories, priorities=priorities, components=components)


# ── Matching helpers ────────────────────────────────────────────────────────


def _normalized_labels(issue: IssueRecord) -> set[str]:
    return {lbl.strip().lower() for lbl in issue.labels if lbl}


_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Cached compiled pattern matching `phrase` with loose word boundaries.

    We use \b for ASCII boundaries when the phrase is alpha-only; otherwise
    a substring search avoids false negatives for keywords like 'cve-' that
    don't sit inside word boundaries.
    """
    pat = _WORD_BOUNDARY_CACHE.get(phrase)
    if pat is None:
        if re.fullmatch(r"[a-z0-9 ]+", phrase):
            pat = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        else:
            pat = re.compile(re.escape(phrase), re.IGNORECASE)
        _WORD_BOUNDARY_CACHE[phrase] = pat
    return pat


def _any_keyword_in_text(text: str, keywords: Iterable[str]) -> list[str]:
    """Return the keywords that occur in `text`."""
    if not text:
        return []
    matched: list[str] = []
    for kw in keywords:
        if not kw:
            continue
        if _phrase_pattern(kw).search(text):
            matched.append(kw)
    return matched


# ── Public entry point ─────────────────────────────────────────────────────


def classify_issue(issue: IssueRecord) -> RuleVerdict:
    """Apply the rule taxonomy to one issue. Pure function."""
    taxo = _taxonomy()
    issue_labels = _normalized_labels(issue)
    haystack = f"{issue.title}\n{issue.body[:4000]}"

    # ── Category ────────────────────────────────────────────────────────
    chosen: Category = "other"
    label_signals: list[str] = []
    keyword_signals: list[str] = []
    for cat in taxo.categories:
        ll = [lbl for lbl in cat.label_keywords if lbl in issue_labels]
        kk = _any_keyword_in_text(haystack, cat.title_keywords)
        if ll or kk:
            chosen = cat.name
            label_signals = ll
            keyword_signals = kk
            break

    # ── Priority (severity-ordered: first match in the priority list wins) ─
    priority: Priority = "unspecified"
    for prio in taxo.priorities:
        if any(lbl in issue_labels for lbl in prio.labels):
            priority = prio.level
            break

    # ── Component ──────────────────────────────────────────────────────
    component: str | None = None
    body_l = issue.body.lower()
    for comp in taxo.components:
        if any(lbl in issue_labels for lbl in comp.labels):
            component = comp.name
            break
        if any(p and p in body_l for p in comp.paths):
            component = comp.name
            break

    # ── Confidence ─────────────────────────────────────────────────────
    confidence = 0.0
    if label_signals:
        confidence += 0.6
    if keyword_signals:
        confidence += 0.3
    if component is not None:
        confidence += 0.1
    if priority != "unspecified":
        confidence = min(1.0, confidence + 0.05)
    confidence = round(min(1.0, confidence), 3)

    return RuleVerdict(
        category=chosen,
        priority=priority,
        component=component,
        matched_label_signals=label_signals,
        matched_keyword_signals=keyword_signals,
        confidence=confidence,
    )


# ── URL-based driver classification (Phase-1 primary) ──────────────────────


# Match http(s) URLs while excluding common trailing chars that aren't
# part of a URL when it appears in prose (closing parens, brackets, quotes,
# trailing punctuation). Markdown autolinks like <https://...> are also
# handled because the leading '<' and trailing '>' aren't matched as URL
# chars by this regex.
_URL_RE = re.compile(r"https?://[^\s<>)\]\"'`]+", re.IGNORECASE)
# Trim trailing punctuation that often clings to URLs in prose.
_URL_TRAIL_TRIM = ".,;:!?)]}>\""


def _iter_urls(text: str) -> Iterable[str]:
    """Yield each http(s) URL found in `text`, trailing punctuation stripped."""
    if not text:
        return
    for m in _URL_RE.finditer(text):
        yield m.group(0).rstrip(_URL_TRAIL_TRIM)


def extract_driver_category(issue) -> str | None:  # noqa: ANN001 — IssueRecord
    """Extract category = full path after `/driver/` from any URL in the issue.

    Looks at:
      * `issue.html_url` (the issue's own GitHub URL — usually NOT a /driver
        URL but cheap to check first)
      * `issue.title`
      * `issue.body`

    The first URL whose path contains `/driver/<something>` wins. The
    returned category is everything between `/driver/` and the URL's
    query/fragment, with leading/trailing slashes trimmed.

    Examples:
      .../driver/audio/codec/foo.c#L42       → "audio/codec/foo.c"
      .../driver/display/dp/link.c?ts=1      → "display/dp/link.c"
      .../driver/                            → None  (no path after)
      .../no-driver-segment/audio.c          → None

    Returns
    -------
    str | None
        The full path after `/driver/` (case-preserved), or None if no
        URL with a `/driver/<...>` segment is present.
    """
    haystack_parts: list[str] = []
    if getattr(issue, "html_url", ""):
        haystack_parts.append(issue.html_url)
    if getattr(issue, "title", ""):
        haystack_parts.append(issue.title)
    if getattr(issue, "body", ""):
        haystack_parts.append(issue.body)
    haystack = "\n".join(haystack_parts)

    for url in _iter_urls(haystack):
        try:
            parsed = urlparse(url)
        except ValueError:
            continue
        path = parsed.path or ""
        idx = path.find("/driver/")
        if idx < 0:
            continue
        after = path[idx + len("/driver/") :].strip("/")
        if after:
            return after
    return None


def reload_taxonomy() -> None:
    """Drop the cached taxonomy so the next call re-reads the YAML.

    Useful in long-running processes after editing the YAML in-place.
    """
    _taxonomy.cache_clear()
    _WORD_BOUNDARY_CACHE.clear()
    log.info("issue_taxonomy.reloaded")
