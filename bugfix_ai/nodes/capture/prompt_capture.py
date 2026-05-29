"""DEPRECATED — replaced by the silent-observer pattern in `emit_new_issue`.

Lane A used to pause the graph here and ask the engineer to type a
narrative of the fix they had just performed. That contract has been
retired: the new Lane A flags the ticket as "New issue", returns
control to the engineer immediately, and records what they do in the
background. The trace is later turned into a structured fix plan by
the finalize coordinator without any narration step.

This module is kept as a thin shim so any external imports of
`prompt_capture_node` (tests, scripts, older deployments) keep working
during the transition. New code should import
`bugfix_ai.nodes.capture.emit_new_issue.emit_new_issue_node` directly.

The shim emits a deprecation log line on first use and forwards to the
new node so the graph behaviour stays identical regardless of which
import path callers use.
"""

from __future__ import annotations

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.core.state import BugFixState
from bugfix_ai.nodes.capture.emit_new_issue import emit_new_issue_node

log = get_logger(__name__)

_warned = False


async def prompt_capture_node(state: BugFixState) -> dict:
    """Deprecated alias — forwards to `emit_new_issue_node`."""
    global _warned
    if not _warned:
        log.warning(
            "capture.prompt_capture.deprecated",
            note="prompt_capture_node is replaced by emit_new_issue_node "
            "(silent observer); update your imports.",
        )
        _warned = True
    return await emit_new_issue_node(state)
