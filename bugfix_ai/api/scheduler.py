"""In-process daily scheduler for the categorization run.

Why a tiny custom loop instead of APScheduler / Celery:
  * The Phase-1 showcase deliberately avoids extra runtime
    dependencies — operators can run the FastAPI process and the daily
    refresh "just works."
  * Production deployments are expected to use **GitHub Actions** to
    drive the daily ingest (which calls the same `run_categorization`
    via `python -m scripts.categorize_issues`). The in-process loop is
    a convenience for laptop demos and staging — kept off in prod by
    setting `categorization_scheduled_enabled=False`.

What the loop does:
  1. Compute the next fire time at H:M UTC (config knobs).
  2. Sleep until then (`asyncio.sleep`).
  3. Invoke `run_categorization()`. Any exception is logged and
     swallowed — a single failed run must not stop tomorrow's.
  4. Repeat.

The loop is started from `core.app_factory.graph_lifespan` only when
`settings.phase == "1"`. The Phase-2 LangGraph stack runs separately.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta, timezone

from bugfix_ai.categorization.service import run_categorization
from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)


def _next_fire_time(hour: int, minute: int, *, now: datetime | None = None) -> datetime:
    """Return the next UTC datetime matching H:M (today if still ahead, else tomorrow)."""
    now = now or datetime.now(timezone.utc)
    target_today = datetime.combine(now.date(), time(hour, minute), tzinfo=timezone.utc)
    if target_today > now:
        return target_today
    return target_today + timedelta(days=1)


async def _scheduler_loop() -> None:
    """Sleep-until-next-fire, run categorization, repeat. Never raises."""
    s = get_settings()
    hour = s.categorization_scheduled_hour_utc
    minute = s.categorization_scheduled_minute_utc

    while True:
        next_fire = _next_fire_time(hour, minute)
        delay = (next_fire - datetime.now(timezone.utc)).total_seconds()
        log.info(
            "scheduler.sleep",
            next_fire_utc=next_fire.isoformat(),
            delay_seconds=int(delay),
        )
        try:
            await asyncio.sleep(max(delay, 1))
        except asyncio.CancelledError:
            log.info("scheduler.cancelled")
            raise

        try:
            log.info("scheduler.run.begin")
            summary = await run_categorization()
            log.info(
                "scheduler.run.complete",
                ingested=summary.get("ingested"),
                categorized=summary.get("categorized"),
                out_path=summary.get("out_path"),
            )
        except Exception as e:  # noqa: BLE001 — never let one failure stop tomorrow's run
            log.error("scheduler.run.failed", error=str(e)[:500])


def start_scheduler() -> asyncio.Task | None:
    """Start the daily scheduler as a background task.

    Returns the task handle so the lifespan can `cancel()` + `await` it
    on shutdown. Returns None when the scheduler is disabled.
    """
    s = get_settings()
    if not s.categorization_scheduled_enabled:
        log.info("scheduler.disabled")
        return None
    log.info(
        "scheduler.start",
        hour=s.categorization_scheduled_hour_utc,
        minute=s.categorization_scheduled_minute_utc,
    )
    return asyncio.create_task(_scheduler_loop(), name="categorization-scheduler")


async def stop_scheduler(task: asyncio.Task | None) -> None:
    """Cancel and await the scheduler task. Safe to call with None."""
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass
