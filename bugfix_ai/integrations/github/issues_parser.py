"""Translate raw GitHub Issue payloads into our `IssueRecord` schema.

Why this exists as its own module:
  The classifiers (`bugfix_ai.categorization.rules` and `nlp_classifier`)
  must not know the GitHub REST shape. If we ever switch to the GraphQL
  endpoint, accept a JSON dump from a customer, or import from a CSV, we
  only change this parser — the categorization layer stays untouched.

What it does:
  * Pulls only the fields the categorization layer needs.
  * Coerces datetime strings to aware `datetime` objects.
  * Flattens label objects (`[{"name": "bug"}, ...]`) to a list of strings.
  * Resolves the author from `user.login`, assignees from `assignees[].login`.
  * Stamps the `repository` field from settings if the payload omits it.

What it deliberately does NOT do:
  * Filter pull requests — that responsibility lives in the client
    (`issues_client.list_issues` skips items with a `pull_request` key).
    The parser will still happily build an `IssueRecord` from a PR if
    one is passed in, but in normal flow it never sees one.
  * Touch labels' colors / descriptions / IDs — the rule layer matches
    on the lowercase label NAME only, so we drop the rest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bugfix_ai.categorization.schema import IssueRecord, Status
from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string into an aware UTC datetime.

    Returns None for falsy input. GitHub uses 'Z' for UTC; we normalize
    that to the explicit `+00:00` form so `fromisoformat` accepts it on
    older Pythons too.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        # Already a datetime — make sure it's tz-aware.
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        log.warning("issues_parser.bad_timestamp", value=str(value)[:80])
        return None


def _extract_labels(raw_labels: Any) -> list[str]:
    """Flatten label objects to lowercase trimmed name strings.

    GitHub returns a list of dicts (`{"name": "bug", ...}`), but some
    JSON dumps use plain strings. We accept both.
    """
    out: list[str] = []
    if not raw_labels:
        return out
    for entry in raw_labels:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = entry.get("name", "")
        else:
            continue
        name = (name or "").strip()
        if name:
            out.append(name)
    return out


def _extract_login(user_obj: Any) -> str:
    """Pull `login` from a user dict; tolerate missing/null user."""
    if isinstance(user_obj, dict):
        return str(user_obj.get("login") or "")
    return ""


def _extract_assignees(raw: Any) -> list[str]:
    """Return a list of assignee logins (deduped, order-preserving)."""
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for u in raw:
        login = _extract_login(u)
        if login and login not in seen:
            seen.add(login)
            out.append(login)
    return out


def _normalize_state(value: Any) -> Status:
    """Coerce GitHub's state field to our Status literal.

    GitHub returns "open" or "closed". Anything else falls back to "open"
    with a warning so the row still validates.
    """
    s = str(value or "").strip().lower()
    if s == "closed":
        return "closed"
    if s == "open":
        return "open"
    log.warning("issues_parser.unknown_state", value=str(value)[:40])
    return "open"


def parse_issue(raw: dict[str, Any]) -> IssueRecord:
    """Convert a single raw GitHub Issue payload into an `IssueRecord`.

    Parameters
    ----------
    raw:
        Dict from the GitHub REST API (e.g. an item yielded by
        `issues_client.list_issues`).

    Returns
    -------
    IssueRecord
        Validated by Pydantic. Required field: `issue_id` (>= 1) and
        `title` (1..2000 chars). If either is missing, Pydantic will
        raise — that's intentional, we don't want to silently emit
        garbage rows.
    """
    settings = get_settings()

    issue_id = int(raw.get("number") or 0)
    title = str(raw.get("title") or "").strip()
    body = raw.get("body") or ""

    repo_name = ""
    repo_obj = raw.get("repository")
    if isinstance(repo_obj, dict):
        repo_name = str(repo_obj.get("full_name") or "")
    if not repo_name:
        repo_name = f"{settings.github_owner}/{settings.github_repo}"

    return IssueRecord(
        issue_id=issue_id,
        node_id=str(raw.get("node_id") or ""),
        title=title or "(untitled)",
        body=body,
        labels=_extract_labels(raw.get("labels")),
        state=_normalize_state(raw.get("state")),
        created_at=_parse_dt(raw.get("created_at")),
        updated_at=_parse_dt(raw.get("updated_at")),
        closed_at=_parse_dt(raw.get("closed_at")),
        html_url=str(raw.get("html_url") or ""),
        repository=repo_name,
        author=_extract_login(raw.get("user")),
        assignees=_extract_assignees(raw.get("assignees")),
    )


def parse_issues(raw_items: list[dict[str, Any]]) -> list[IssueRecord]:
    """Parse a batch, skipping any item that fails validation.

    Parsing failures are logged but do not abort the batch — the showcase
    spreadsheet should still render the issues that DID parse cleanly.
    """
    out: list[IssueRecord] = []
    for raw in raw_items:
        try:
            out.append(parse_issue(raw))
        except Exception as e:  # noqa: BLE001 — defensive boundary
            log.warning(
                "issues_parser.parse_failed",
                issue_id=raw.get("number"),
                error=str(e)[:300],
            )
    return out
