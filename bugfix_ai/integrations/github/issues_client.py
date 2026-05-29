"""Async GitHub Issues client (REST API).

Why this module exists:
  Part 1 of the project — the "showcase" deliverable — needs to pull every
  GitHub Issue from the configured repo so the categorization pipeline can
  produce its Issue ID / Title / Category / Priority / Status sheet. The
  Code Scanning client (`client.py`) covers a different endpoint
  (/code-scanning/alerts) and is used for the autonomous-fix half of the
  product, so we keep the two surfaces separate.

API surface used:
  GET /repos/{owner}/{repo}/issues
      → list issues with pagination, state, label, since filters
  GET /repos/{owner}/{repo}/issues/{issue_number}
      → fetch a single issue by number

Important quirk of the GitHub Issues endpoint:
  `GET /issues` returns BOTH issues and pull requests (PRs are a special
  kind of issue under the hood). We exclude PRs by skipping any item that
  has a `pull_request` key. This is the documented filter — there is no
  server-side `is:issue` parameter on REST.

Auth, retries, ETag, pagination follow the same contract as `client.py`
so operators only learn one mental model:
  * httpx.AsyncClient cached via `lru_cache` (one pool per process).
  * tenacity exponential backoff on transport errors and 5xx.
  * GitHub secondary rate-limit (returned as 403 with "rate limit" in
    the body) is treated as retryable.
  * `Link: <...>; rel="next"` header drives pagination.
  * `If-None-Match` header lets the caller cheaply poll — a 304 ends
    iteration without yielding.

Defensive boundaries:
  * `issues_max_pages` from settings caps the total pages fetched, so a
    runaway loop can never wedge the process.
  * `per_page` is bounded to GitHub's documented 100 maximum.
  * The client never mutates state — no PATCH / POST / DELETE here.
"""

from __future__ import annotations

import re
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


_API = "https://api.github.com"
_ACCEPT = "application/vnd.github+json"
_GH_VERSION = "2022-11-28"


class GitHubIssuesError(RuntimeError):
    """Raised on any non-2xx terminal response from the Issues endpoint."""


# ── HTTP plumbing ───────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _http() -> httpx.AsyncClient:
    """Return a process-wide pooled async client.

    Cached so we don't open a new TCP connection per call. The pool is
    cleaned up via `aclose()` on shutdown.
    """
    s = get_settings()
    if not s.github_token:
        raise GitHubIssuesError("GITHUB_TOKEN is not set")
    return httpx.AsyncClient(
        base_url=_API,
        headers={
            "Authorization": f"Bearer {s.github_token}",
            "Accept": _ACCEPT,
            "X-GitHub-Api-Version": _GH_VERSION,
            "User-Agent": "bugfix-ai/0.1 (issues)",
        },
        timeout=30.0,
        limits=httpx.Limits(max_connections=10),
    )


async def _request(method: str, path_or_url: str, **kwargs) -> httpx.Response:
    """Issue a request with bounded retries.

    Retries on network errors and 5xx; treats GitHub secondary-rate-limit
    (403 + "rate limit" in body) as retryable. Caller decides what to do
    with non-retryable 4xx.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    ):
        with attempt:
            client = _http()
            resp = await client.request(method, path_or_url, **kwargs)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"GitHub {resp.status_code}", request=resp.request, response=resp
                )
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                raise httpx.HTTPStatusError(
                    "Secondary rate limit", request=resp.request, response=resp
                )
            return resp
    raise GitHubIssuesError("unreachable")  # for type checker


# ── Public API ──────────────────────────────────────────────────────────────


async def list_issues(
    *,
    state: str | None = None,
    labels: list[str] | None = None,
    since: str | None = None,
    per_page: int | None = None,
    etag: str | None = None,
    include_pull_requests: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Iterate every GitHub Issue matching the filters.

    Parameters
    ----------
    state:
        "open" | "closed" | "all". Defaults to `settings.issues_default_state`.
    labels:
        Optional list of label names; GitHub joins them with AND.
    since:
        ISO-8601 timestamp; only issues updated at or after this time are
        returned. Useful for incremental syncs.
    per_page:
        1..100, default `settings.issues_per_page`.
    etag:
        If provided and the index is unchanged, no items are yielded
        (GitHub returns 304).
    include_pull_requests:
        Default False — PRs are excluded because the showcase output is
        about *issues*. Pass True only if the caller really wants both.

    Yields
    ------
    dict
        Raw GitHub issue payload, one per item.

    Notes
    -----
    Pagination cap: a maximum of `settings.issues_max_pages` pages is
    fetched, after which iteration stops with a warning. This is a safety
    net for misconfigured filters that would otherwise paginate forever.
    """
    s = get_settings()
    state = state or s.issues_default_state
    per_page = min(per_page or s.issues_per_page, 100)

    params: dict[str, Any] = {
        "state": state,
        "per_page": per_page,
        # Sort by updated DESC so callers using `since` see fresh items first.
        "sort": "updated",
        "direction": "desc",
    }
    if labels:
        params["labels"] = ",".join(labels)
    if since:
        params["since"] = since

    url: str | None = f"/repos/{s.github_owner}/{s.github_repo}/issues"
    headers = {"If-None-Match": etag} if etag else {}

    pages = 0
    yielded = 0
    while url:
        if pages >= s.issues_max_pages:
            log.warning(
                "github.issues.page_cap_reached",
                pages=pages,
                cap=s.issues_max_pages,
                yielded=yielded,
            )
            return

        resp = await _request(
            "GET",
            url,
            params=params if "?" not in url else None,
            headers=headers,
        )

        if resp.status_code == 304:
            log.info("github.issues.unchanged", etag=etag)
            return
        if resp.status_code != 200:
            raise GitHubIssuesError(
                f"List issues failed: {resp.status_code} {resp.text[:200]}"
            )

        page_items = resp.json()
        if not isinstance(page_items, list):
            raise GitHubIssuesError(
                f"Unexpected payload (not a list): {str(page_items)[:200]}"
            )

        for item in page_items:
            # GitHub's /issues endpoint returns PRs too — filter unless asked.
            if not include_pull_requests and "pull_request" in item:
                continue
            yield item
            yielded += 1

        pages += 1
        url = _next_link(resp.headers.get("Link"))
        # Subsequent paginated URLs already include query params in the link.
        params = None
        headers = {}

    log.info("github.issues.list_complete", pages=pages, yielded=yielded)


async def get_issue(issue_number: int) -> dict[str, Any]:
    """Fetch a single issue by its repo-scoped number.

    Raises
    ------
    GitHubIssuesError
        If the response is not 200. Includes a status-code prefix so
        callers can distinguish 404 (gone) from auth / scope issues.
    """
    s = get_settings()
    resp = await _request(
        "GET",
        f"/repos/{s.github_owner}/{s.github_repo}/issues/{issue_number}",
    )
    if resp.status_code != 200:
        raise GitHubIssuesError(
            f"Get issue {issue_number} failed: {resp.status_code} {resp.text[:200]}"
        )
    return resp.json()


async def get_issues_index_etag() -> str | None:
    """HEAD the issues list to retrieve the current ETag for cheap polling.

    Returned value can be passed back as `etag=` to `list_issues()` on the
    next poll cycle; a 304 means nothing has changed since the prior fetch.
    """
    s = get_settings()
    resp = await _request(
        "HEAD",
        f"/repos/{s.github_owner}/{s.github_repo}/issues",
    )
    return resp.headers.get("ETag")


# ── Helpers ─────────────────────────────────────────────────────────────────


_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def _next_link(link_header: str | None) -> str | None:
    """Extract the `rel="next"` URL from a GitHub Link header, or None."""
    if not link_header:
        return None
    m = _LINK_NEXT_RE.search(link_header)
    return m.group(1) if m else None


async def aclose() -> None:
    """Release the pooled connection — call this from FastAPI lifespan."""
    if _http.cache_info().currsize:  # type: ignore[attr-defined]
        await _http().aclose()
