"""Ingestion router — webhook + manual trigger for GitHub Code Scanning alerts.

We deliberately don't trust the webhook payload as the SOURCE OF TRUTH; we
re-fetch from the GitHub API using the alert_number. This protects us from
spoofed payloads and from race conditions where the webhook fires before the
alert is fully written server-side.

Idempotency: the poller / handler both use `has_seen_alert` / `mark_alert_seen`
so a replayed webhook is a no-op.
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.integrations.github.client import get_alert
from bugfix_ai.integrations.github.poller import ingest_one_alert, poll_once

router = APIRouter(prefix="/ingest", tags=["ingest"])
log = get_logger(__name__)


async def _ingest_by_number(graph, alert_number: int) -> None:
    """Background task: fetch the full alert payload, then ingest it."""
    try:
        alert = await get_alert(alert_number)
    except Exception as e:  # noqa: BLE001
        log.error("ingest.fetch_failed", alert=alert_number, error=str(e))
        return
    await ingest_one_alert(graph, alert)


# ── Manual trigger ──────────────────────────────────────────────────────────


class IngestOneRequest(BaseModel):
    alert_number: int = Field(ge=1)


@router.post("/github/alert", status_code=status.HTTP_202_ACCEPTED)
async def ingest_alert(
    req: IngestOneRequest, request: Request, background: BackgroundTasks
) -> dict:
    """Trigger a single-alert ingestion. The actual graph run is backgrounded."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="graph not initialized"
        )
    background.add_task(_ingest_by_number, graph, req.alert_number)
    return {"queued_alert_number": req.alert_number}


@router.post("/github/poll", status_code=status.HTTP_202_ACCEPTED)
async def trigger_poll(request: Request, background: BackgroundTasks) -> dict:
    """Run one poll cycle (admin / manual catch-up)."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="graph not initialized"
        )
    background.add_task(poll_once, graph)
    return {"status": "polling"}


# ── Webhook ─────────────────────────────────────────────────────────────────


def _verify_signature(secret: str, body: bytes, signature_header: str) -> bool:
    """GitHub sends `X-Hub-Signature-256: sha256=<hex>`. Verify in constant time."""
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    provided = signature_header[len("sha256="):]
    return hmac.compare_digest(expected, provided)


@router.post("/github/webhook")
async def github_webhook(
    request: Request,
    background: BackgroundTasks,
    x_github_event: str = Header(default=""),
    x_hub_signature_256: str = Header(default=""),
) -> dict:
    """Handle the `code_scanning_alert` event."""
    body = await request.body()
    settings = get_settings()

    # If a webhook secret is configured, enforce signature verification.
    secret = getattr(settings, "github_webhook_secret", "") or ""
    if secret:
        if not _verify_signature(secret, body, x_hub_signature_256):
            log.warning("webhook.invalid_signature", event=x_github_event)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid webhook signature"
            )

    if x_github_event != "code_scanning_alert":
        # Ack other events so GitHub doesn't retry, but do nothing.
        log.info("webhook.ignored_event", event=x_github_event)
        return {"status": "ignored", "event": x_github_event}

    import json as _json

    try:
        payload = _json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid JSON payload: {e}",
        )

    alert = payload.get("alert") or {}
    alert_number = alert.get("number")
    if not isinstance(alert_number, int):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="payload missing alert.number",
        )

    # Background the actual ingestion — the webhook handler must return fast.
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="graph not initialized"
        )
    background.add_task(_ingest_by_number, graph, alert_number)
    log.info("webhook.queued", alert_number=alert_number, action=payload.get("action"))
    return {"status": "queued", "alert_number": alert_number}
