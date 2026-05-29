"""Async GitHub Issues GraphQL client (Phase 1).

Why GraphQL (instead of the REST `issues_client.py`):
  * One round-trip pulls 100 issues with all the fields the categorizer
    needs (labels, body, html_url, author, timestamps, state). REST
    needs follow-up calls for each issue's labels in some cases.
  * Cursor-based pagination is more reliable across concurrent inserts
    than REST `page=N` numbering.
  * Lets us scope to OPEN issues at the server side via the search
    expression `repo:OWNER/REPO is:issue is:open` — matching the
    user's "scope of ingestion: only open issues" requirement.

Auth:
  Fine-Grained PAT with the minimum permissions:
      Repository permissions →
          Issues:    Read-only
          Metadata:  Read-only
  The classic `github_token` (if set) is used as a fallback so dev
  setups that already had a single token keep working — see
  `Settings.issues_token()`.

Public surface:
    list_open_issues(owner=..., repo=...) -> AsyncIterator[dict]
        Yield raw GitHub issue payloads (REST-shaped, so the existing
        `issues_parser.parse_issue` keeps working without a rewrite).

    aclose() -> None
        Release the pooled httpx client. Call from the FastAPI lifespan.

What this module deliberately does NOT do:
  * Mutate state — no `mutation { ... }` blocks. Read-only by design.
  * Filter by label / since — Phase 1's spec says "only open issues",
    period. Filters can be added later as kwargs without breaking the
    caller, but adding them now would be premature.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncIterator

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)


_GRAPHQL_URL = "https://api.github.com/graphql"


class GitHubGraphQLError(RuntimeError):
    """Raised when the GraphQL endpoint returns errors or non-2xx."""


# ── HTTP plumbing ───────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _http() -> httpx.AsyncClient:
    """Process-wide pooled async client for the GraphQL endpoint."""
    s = get_settings()
    token = s.issues_token()
    if not token:
        raise GitHubGraphQLError(
            "No GitHub PAT configured: set GITHUB_PAT_FINE_GRAINED (preferred) "
            "or GITHUB_TOKEN."
        )
    return httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "bugfix-ai/0.1 (issues-graphql)",
            "Content-Type": "application/json",
        },
        timeout=30.0,
        limits=httpx.Limits(max_connections=10),
    )


async def _post_graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """POST a GraphQL query with bounded retries.

    Retries on transport errors and 5xx; treats GitHub secondary
    rate-limiting (an explicit `errors[].type == "RATE_LIMITED"` or 403
    with "rate limit" in body) as retryable.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    ):
        with attempt:
            client = _http()
            resp = await client.post(
                _GRAPHQL_URL, json={"query": query, "variables": variables}
            )
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"GitHub GraphQL {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                raise httpx.HTTPStatusError(
                    "Secondary rate limit", request=resp.request, response=resp
                )
            if resp.status_code != 200:
                raise GitHubGraphQLError(
                    f"GraphQL non-200: {resp.status_code} {resp.text[:200]}"
                )
            payload = resp.json()
            errors = payload.get("errors")
            if errors:
                # If the error is a transient RATE_LIMITED, raise to trigger retry.
                if any((e.get("type") == "RATE_LIMITED") for e in errors):
                    raise httpx.HTTPStatusError(
                        "Rate limited", request=resp.request, response=resp
                    )
                raise GitHubGraphQLError(f"GraphQL errors: {errors}")
            return payload["data"]
    raise GitHubGraphQLError("unreachable")  # for type checker


# ── Query ───────────────────────────────────────────────────────────────────


# We use the `search` connection because it lets us scope to issues only
# (excluding PRs) and to open state in one expression. The fields
# requested are exactly what `issues_parser.parse_issue` consumes.
_OPEN_ISSUES_QUERY = """
query ListOpenIssues($q: String!, $first: Int!, $after: String) {
  search(query: $q, type: ISSUE, first: $first, after: $after) {
    issueCount
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      __typename
      ... on Issue {
        number
        id
        title
        body
        url
        state
        createdAt
        updatedAt
        closedAt
        author { login }
        repository { nameWithOwner }
        labels(first: 100) {
          nodes { name }
        }
        assignees(first: 20) {
          nodes { login }
        }
      }
    }
  }
}
"""


def _node_to_rest_shape(node: dict[str, Any]) -> dict[str, Any]:
    """Map a GraphQL Issue node to the REST shape `issues_parser` expects.

    The parser keys it reads:
      number, node_id, title, body, html_url, state, created_at,
      updated_at, closed_at, user.login, assignees[].login,
      labels[].name, repository.full_name
    """
    author = node.get("author") or {}
    repo = node.get("repository") or {}
    labels = ((node.get("labels") or {}).get("nodes")) or []
    assignees = ((node.get("assignees") or {}).get("nodes")) or []
    return {
        "number": node.get("number"),
        "node_id": node.get("id") or "",
        "title": node.get("title") or "",
        "body": node.get("body") or "",
        "html_url": node.get("url") or "",
        # GraphQL returns "OPEN" / "CLOSED"; parser lowercases.
        "state": (node.get("state") or "OPEN").lower(),
        "created_at": node.get("createdAt"),
        "updated_at": node.get("updatedAt"),
        "closed_at": node.get("closedAt"),
        "user": {"login": author.get("login") or ""},
        "assignees": [{"login": a.get("login") or ""} for a in assignees],
        "labels": [{"name": l.get("name") or ""} for l in labels],
        "repository": {"full_name": repo.get("nameWithOwner") or ""},
    }


# ── Public API ──────────────────────────────────────────────────────────────


async def list_open_issues(
    *,
    owner: str | None = None,
    repo: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield every OPEN issue in the configured repo, REST-shaped.

    Uses the `search` connection with `repo:OWNER/REPO is:issue is:open`.
    Pagination is cursor-based; capped by `settings.graphql_max_pages` as
    a safety net so a misconfigured query can't paginate forever.
    """
    s = get_settings()
    owner = owner or s.github_owner
    repo = repo or s.github_repo
    page_size = min(max(s.graphql_page_size, 1), 100)
    max_pages = max(s.graphql_max_pages, 1)

    q = f"repo:{owner}/{repo} is:issue is:open"
    after: str | None = None
    pages = 0
    yielded = 0

    while True:
        if pages >= max_pages:
            log.warning(
                "github.graphql.page_cap_reached",
                pages=pages,
                cap=max_pages,
                yielded=yielded,
            )
            return

        data = await _post_graphql(
            _OPEN_ISSUES_QUERY,
            {"q": q, "first": page_size, "after": after},
        )
        search = data.get("search") or {}
        nodes = search.get("nodes") or []
        for node in nodes:
            # Defensive: `search type:ISSUE` should only return Issues, but
            # the union return type means we double-check.
            if node.get("__typename") != "Issue":
                continue
            yield _node_to_rest_shape(node)
            yielded += 1

        page_info = search.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        after = page_info.get("endCursor")
        pages += 1

    log.info("github.graphql.list_open_issues.done", pages=pages + 1, yielded=yielded)


async def aclose() -> None:
    """Release the pooled connection — call this from FastAPI lifespan."""
    if _http.cache_info().currsize:  # type: ignore[attr-defined]
        await _http().aclose()
