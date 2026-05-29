"""In-process background trace recorder for Lane A (capture mode).

Why an in-process registry (not Postgres / Redis) for the live buffer:
  Capture-mode events are high-frequency, ephemeral, and only matter
  until the bug is resolved. Round-tripping every keystroke-class event
  to Postgres would dominate latency and leave half-finished traces in
  the canonical store. The recorder keeps a per-thread deque in memory;
  the LangGraph checkpointer mirrors the events into `capture_events`
  on every API touch so a process restart can rehydrate the trace from
  the durable state. Only when `finalize_capture` runs does the trace
  flow into Postgres + Qdrant via `memory.fix_store.store_fix`.

Concurrency model:
  One asyncio.Lock per thread_id, lazily created. Append, snapshot, and
  clear all hold the same lock so the trace cannot be observed mid-write.

Bounded buffer:
  Each thread's buffer is capped at `_MAX_EVENTS_PER_THREAD`. Once full
  the oldest event is dropped and a single warning is logged per overflow
  episode — Lane A is meant to capture the *shape* of a fix, not every
  keystroke, and a misbehaving hook should not balloon process memory.

What feeds events into here:
  * `POST /sessions/{thread_id}/capture/event` — driven by an editor /
    shell / git hook the engineer installs once.
  * `integrations.terminal.safe_executor` mirrors any commands run
    through the assistant during the same session.

Public surface:
  * `record_event(thread_id, event)` — append a CaptureEvent.
  * `snapshot(thread_id)`            — copy current buffer (read-only view).
  * `mark_resolved(thread_id, note)` — append a synthetic "resolution" event
                                       and flag the trace ready for finalize.
  * `clear(thread_id)`               — drop the buffer (called after persist).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import CaptureEvent

log = get_logger(__name__)


_MAX_EVENTS_PER_THREAD = 2_000


class _Recorder:
    """Per-process registry of in-flight capture traces."""

    def __init__(self) -> None:
        self._buffers: dict[str, Deque[CaptureEvent]] = defaultdict(
            lambda: deque(maxlen=_MAX_EVENTS_PER_THREAD)
        )
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._resolved: dict[str, bool] = defaultdict(bool)
        self._overflowed: set[str] = set()

    async def record_event(self, thread_id: str, event: CaptureEvent) -> CaptureEvent:
        async with self._locks[thread_id]:
            ev: CaptureEvent = {
                **event,
                "timestamp": event.get("timestamp")
                or datetime.now(timezone.utc).isoformat(),
                "kind": event.get("kind") or "note",
            }
            buf = self._buffers[thread_id]
            if len(buf) == buf.maxlen and thread_id not in self._overflowed:
                log.warning(
                    "capture.recorder.overflow",
                    thread_id=thread_id,
                    cap=buf.maxlen,
                )
                self._overflowed.add(thread_id)
            buf.append(ev)
            return ev

    async def snapshot(self, thread_id: str) -> list[CaptureEvent]:
        async with self._locks[thread_id]:
            return list(self._buffers[thread_id])

    async def mark_resolved(
        self, thread_id: str, note: str = ""
    ) -> CaptureEvent:
        marker: CaptureEvent = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": "resolution",
            "tool": None,
            "command": None,
            "files": [],
            "stdout": None,
            "stderr": None,
            "exit_code": None,
            "note": note,
        }
        ev = await self.record_event(thread_id, marker)
        self._resolved[thread_id] = True
        return ev

    def is_resolved(self, thread_id: str) -> bool:
        return self._resolved.get(thread_id, False)

    async def clear(self, thread_id: str) -> None:
        async with self._locks[thread_id]:
            self._buffers.pop(thread_id, None)
            self._resolved.pop(thread_id, None)
            self._overflowed.discard(thread_id)


# Module-level singleton. Cheap to import; no I/O at construction time.
recorder = _Recorder()


# ── Convenience entrypoints (imported by API + safe_executor) ───────────────


async def record_event(thread_id: str, event: CaptureEvent) -> CaptureEvent:
    return await recorder.record_event(thread_id, event)


async def snapshot(thread_id: str) -> list[CaptureEvent]:
    return await recorder.snapshot(thread_id)


async def mark_resolved(thread_id: str, note: str = "") -> CaptureEvent:
    return await recorder.mark_resolved(thread_id, note)


async def clear(thread_id: str) -> None:
    await recorder.clear(thread_id)


def is_resolved(thread_id: str) -> bool:
    return recorder.is_resolved(thread_id)
