"""Test for the Lane-A entry node `emit_new_issue_node`.

Pinned contract:
  * Returns `capture_active=True` (arms the silent-observer recorder).
  * Stamps `capture_started_ts` with an ISO timestamp (truthy is enough).
  * Initialises `capture_events=[]` so the API can append without a None
    guard.
  * Emits the well-known signal "New issue" (constant `NEW_ISSUE_SIGNAL`).
  * Writes exactly one obs_log entry.
"""

from __future__ import annotations

import pytest

from bugfix_ai.nodes.capture.emit_new_issue import (
    NEW_ISSUE_SIGNAL,
    emit_new_issue_node,
)


@pytest.mark.asyncio
async def test_emit_new_issue_arms_recorder_and_returns_signal():
    state = {"ticket_id": "BUG-123"}
    out = await emit_new_issue_node(state)

    assert out["capture_active"] is True
    assert out["capture_started_ts"]
    assert out["capture_events"] == []
    assert out["capture_resolved"] is False
    assert out["capture_resolution_note"] == ""
    assert len(out["obs_log"]) == 1


def test_new_issue_signal_constant():
    # The exact phrase is part of the public client contract.
    assert NEW_ISSUE_SIGNAL == "New issue"
