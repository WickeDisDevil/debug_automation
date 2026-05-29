"""Tests for the Lane-A in-process capture recorder.

The recorder is a module-level singleton, so every test uses a unique
`thread_id` (uuid4) to avoid cross-test pollution without needing to
reset internal state.

Pinned contract:
  * `record_event` appends and stamps a `timestamp` + default `kind`.
  * `snapshot` returns a *copy* (mutating it does not affect the buffer).
  * `mark_resolved` appends a synthetic event and flips `is_resolved`.
  * `clear` empties the buffer and resets the resolved flag.
"""

from __future__ import annotations

import uuid

import pytest

from bugfix_ai.nodes.capture import recorder as rec


def _tid() -> str:
    return f"t-{uuid.uuid4()}"


@pytest.mark.asyncio
async def test_record_event_appends_and_stamps_defaults():
    tid = _tid()
    ev = await rec.record_event(tid, {"command": "pytest -q"})
    assert ev["command"] == "pytest -q"
    assert ev["kind"] == "note"
    assert ev["timestamp"]  # ISO-format string is truthy
    snap = await rec.snapshot(tid)
    assert len(snap) == 1
    await rec.clear(tid)


@pytest.mark.asyncio
async def test_snapshot_is_a_copy():
    tid = _tid()
    await rec.record_event(tid, {"command": "ls"})
    snap = await rec.snapshot(tid)
    snap.append({"command": "tampered"})  # must not affect the real buffer
    assert len(await rec.snapshot(tid)) == 1
    await rec.clear(tid)


@pytest.mark.asyncio
async def test_mark_resolved_sets_flag_and_appends_marker():
    tid = _tid()
    await rec.record_event(tid, {"command": "git diff"})
    assert rec.is_resolved(tid) is False
    await rec.mark_resolved(tid, note="patched header")
    assert rec.is_resolved(tid) is True
    snap = await rec.snapshot(tid)
    assert snap[-1]["kind"] == "resolution"
    assert snap[-1]["note"] == "patched header"
    await rec.clear(tid)


@pytest.mark.asyncio
async def test_clear_drops_buffer_and_resolved_flag():
    tid = _tid()
    await rec.record_event(tid, {"command": "x"})
    await rec.mark_resolved(tid)
    await rec.clear(tid)
    assert await rec.snapshot(tid) == []
    assert rec.is_resolved(tid) is False


@pytest.mark.asyncio
async def test_multiple_threads_are_isolated():
    tid_a, tid_b = _tid(), _tid()
    await rec.record_event(tid_a, {"command": "A"})
    await rec.record_event(tid_b, {"command": "B"})
    snap_a = await rec.snapshot(tid_a)
    snap_b = await rec.snapshot(tid_b)
    assert [e["command"] for e in snap_a] == ["A"]
    assert [e["command"] for e in snap_b] == ["B"]
    await rec.clear(tid_a)
    await rec.clear(tid_b)
