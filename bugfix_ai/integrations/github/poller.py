"""Background poller that pulls open Code Scanning alerts into the graph.

How it fits the system:
  This module is the long-running background worker that keeps the
  fix store fresh against the source of truth (GitHub Code Scanning).
  It can be driven two ways:
    * Periodic poll — `run_poller()` started as an asyncio task on
      app startup; sleeps `github_poll_interval_seconds` between
      passes.
    * Webhook-triggered — the `/ingest/github/webhook` endpoint
      verifies the HMAC signature and calls `ingest_one_alert(...)`
      directly. The poll loop is the catch-all for missed events.

Idempotency story:
  * `has_seen_alert()` / `mark_alert_seen()` (rules_db) maintain an
    `ingestion_seen` table keyed on (provider, alert_number). A
    duplicate webhook or a re-poll right after a successful run is a
    no-op — we never start two graph runs for the same alert.
  * ETag on `get_index_etag()` lets us short-circuit the whole LIST
    call when nothing changed since the last poll. This matters for
    rate limits — GitHub's secondary limits punish hammering.

Threading model:
  We pin `thread_id = ticket_id` (where ticket_id is `gh-{number}`).
  This means each alert has exactly ONE persistent state thread that
  can be resumed at HITL interrupts. If the same alert is re-scanned
  and re-emitted with the same number, we'd skip it via the dedup
  table — but if a human already started a session on that thread,
  the resumption logic in the API still works because the thread_id
  is stable.

Failure handling:
  Both `ingest_one_alert` and `poll_once` swallow exceptions and log
  them. The poll loop NEVER raises — a failed pass just sleeps and
  retries on the next cycle. This is intentional: we'd rather miss
  one alert than tear down the worker.
"""

from __future__ import annotations

import asyncio

from langgraph.graph.state import CompiledStateGraph

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.integrations.github.client import (
    get_index_etag,
    list_code_scanning_alerts,
)
from bugfix_ai.integrations.github.parser import (
    build_pseudo_log_from_alert,
    parse_alert,
)
from bugfix_ai.memory.rules_db import has_seen_alert, mark_alert_seen

log = get_logger(__name__)

_PROVIDER = "github_code_scanning"


async def ingest_one_alert(graph: CompiledStateGraph, alert: dict) -> str | None:
    """Ingest a single alert, idempotently. Returns thread_id if started."""
    alert_number = int(alert["number"])
    if await has_seen_alert(_PROVIDER, alert_number):
        log.debug("alert.skip.already_seen", alert=alert_number)
        return None

    initial_state = parse_alert(alert)
    initial_state["error_logs_raw"] = build_pseudo_log_from_alert(alert)
    # CodeQL alerts have no concept of "fix narrative yet" so default to assist
    initial_state["mode"] = "assist"

    thread_id = initial_state["ticket_id"]
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # ainvoke runs until the first interrupt (consent_gate in assist mode)
        await graph.ainvoke(initial_state, config=config)
        await mark_alert_seen(_PROVIDER, alert_number, alert.get("state", "open"))
        log.info("alert.ingested", thread_id=thread_id, alert=alert_number)
        return thread_id
    except Exception as e:  # noqa: BLE001
        log.error("alert.ingest.failed", alert=alert_number, error=str(e))
        return None


async def poll_once(graph: CompiledStateGraph, etag: str | None = None) -> str | None:
    """Single polling pass. Returns the new ETag for use in the next call."""
    settings = get_settings()
    new_etag = await get_index_etag()
    if etag and new_etag and new_etag == etag:
        log.debug("poll.unchanged")
        return etag

    count = 0
    async for alert in list_code_scanning_alerts(
        state="open",
        severity=settings.severity_list(),
        tool_name=settings.github_tool_filter,
        ref=settings.github_ref,
        etag=etag,
    ):
        await ingest_one_alert(graph, alert)
        count += 1
    log.info("poll.complete", processed=count)
    return new_etag


async def run_poller(graph: CompiledStateGraph) -> None:
    """Long-running poll loop. Call from a background task on app startup."""
    settings = get_settings()
    etag: str | None = None
    while True:
        try:
            etag = await poll_once(graph, etag)
        except Exception as e:  # noqa: BLE001
            log.error("poll.error", error=str(e))
        await asyncio.sleep(settings.github_poll_interval_seconds)
