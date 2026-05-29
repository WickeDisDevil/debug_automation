"""Async GitHub Code Scanning client (REST API).

API reference:
  GET /repos/{owner}/{repo}/code-scanning/alerts
  GET /repos/{owner}/{repo}/code-scanning/alerts/{alert_number}
  GET /repos/{owner}/{repo}/code-scanning/alerts/{alert_number}/instances

Auth: PAT with `security_events` scope (read), or `repo` for private repos.

Production fixes:
  * Async httpx with connection pooling.
  * ETag handling for cheap polling (304 means nothing changed).
  * Pagination via Link header.
  * Retry with backoff on 5xx and on the secondary-rate-limit responses GitHub
    returns when you hammer them.
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


class GitHubError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _http() -> httpx.AsyncClient:
    s = get_settings()
    if not s.github_token:
        raise GitHubError("GITHUB_TOKEN is not set")
    return httpx.AsyncClient(
        base_url=_API,
        headers={
            "Authorization": f"Bearer {s.github_token}",
            "Accept": _ACCEPT,
            "X-GitHub-Api-Version": _GH_VERSION,
            "User-Agent": "bugfix-ai/0.1",
        },
        timeout=30.0,
        limits=httpx.Limits(max_connections=10),
    )


async def _request(method: str, path: str, **kwargs) -> httpx.Response:
    """Issue a request with backoff on transient errors."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    ):
        with attempt:
            client = _http()
            resp = await client.request(method, path, **kwargs)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"GitHub {resp.status_code}", request=resp.request, response=resp
                )
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                raise httpx.HTTPStatusError(
                    "Secondary rate limit", request=resp.request, response=resp
                )
            return resp
    raise GitHubError("unreachable")  # for type checker


# ── Public API ──────────────────────────────────────────────────────────────


async def list_code_scanning_alerts(
    *,
    state: str = "open",
    severity: list[str] | None = None,
    tool_name: str | None = "CodeQL",
    ref: str | None = None,
    per_page: int = 100,
    etag: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Iterate every Code Scanning alert matching the filters.

    Yields one dict per alert. Handles pagination via the Link header.
    If `etag` is provided and the index hasn't changed, no alerts are yielded.
    """
    s = get_settings()
    severity_csv = ",".join(severity) if severity else None

    params: dict[str, Any] = {
        "state": state,
        "per_page": per_page,
        "tool_name": tool_name,
    }
    if severity_csv:
        params["severity"] = severity_csv
    if ref:
        params["ref"] = ref

    url = f"/repos/{s.github_owner}/{s.github_repo}/code-scanning/alerts"
    headers = {"If-None-Match": etag} if etag else {}

    while url:
        resp = await _request("GET", url, params=params if "?" not in url else None, headers=headers)
        if resp.status_code == 304:
            log.info("github.alerts.unchanged", etag=etag)
            return
        if resp.status_code != 200:
            raise GitHubError(f"List alerts failed: {resp.status_code} {resp.text[:200]}")

        for alert in resp.json():
            yield alert

        # Pagination via Link header
        url = _next_link(resp.headers.get("Link"))
        params = None  # subsequent URLs already have query params baked in
        headers = {}


async def get_alert(alert_number: int) -> dict[str, Any]:
    s = get_settings()
    resp = await _request(
        "GET",
        f"/repos/{s.github_owner}/{s.github_repo}/code-scanning/alerts/{alert_number}",
    )
    if resp.status_code != 200:
        raise GitHubError(f"Get alert {alert_number} failed: {resp.status_code}")
    return resp.json()


async def get_alert_instances(alert_number: int) -> list[dict[str, Any]]:
    """Returns all instances of an alert across refs (different file/line locations)."""
    s = get_settings()
    resp = await _request(
        "GET",
        f"/repos/{s.github_owner}/{s.github_repo}/code-scanning/alerts/{alert_number}/instances",
    )
    if resp.status_code != 200:
        raise GitHubError(f"Get alert {alert_number} instances failed: {resp.status_code}")
    return resp.json()


async def get_index_etag() -> str | None:
    """HEAD the alerts list to retrieve the current ETag for cheap polling."""
    s = get_settings()
    resp = await _request(
        "HEAD",
        f"/repos/{s.github_owner}/{s.github_repo}/code-scanning/alerts",
    )
    return resp.headers.get("ETag")


# ── Helpers ─────────────────────────────────────────────────────────────────


_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    m = _LINK_NEXT_RE.search(link_header)
    return m.group(1) if m else None


async def aclose() -> None:
    """Call from shutdown to release the connection pool."""
    # `_http` is an lru_cache wrapper; check currsize directly to see if a
    # client was ever instantiated. The previous `"_http" in dir()` check
    # always returned False (dir() with no args inspects local scope), so
    # the pool was never actually closed on shutdown.
    if _http.cache_info().currsize:
        await _http().aclose()
