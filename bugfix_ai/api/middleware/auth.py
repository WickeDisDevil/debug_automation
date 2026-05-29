"""API key auth + a simple correlation-id middleware.

Production fixes baked in:
  * Constant-time comparison via `hmac.compare_digest` to avoid timing oracles.
  * Header is `X-API-Key`, never `Authorization` (we don't want to be confused
    with bearer-token semantics).
  * `/health` and `/healthz` are open so probes work without keys.
  * Every request gets a correlation id (echoed back in the response and
    bound to the structlog context) so logs can be joined per-request.
"""

from __future__ import annotations

import hmac
import uuid
from typing import Awaitable, Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from bugfix_ai.config.logging_config import bind_context, clear_context, get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)


_OPEN_PATHS = {"/health", "/healthz", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in _OPEN_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        provided = request.headers.get("x-api-key", "")
        expected = get_settings().api_key
        if not provided or not hmac.compare_digest(provided, expected):
            log.warning("auth.rejected", path=request.url.path)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing X-API-Key",
            )
        return await call_next(request)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        cid = request.headers.get("x-correlation-id") or uuid.uuid4().hex
        bind_context(correlation_id=cid, path=request.url.path, method=request.method)
        try:
            response = await call_next(request)
        finally:
            clear_context()
        response.headers["x-correlation-id"] = cid
        return response
