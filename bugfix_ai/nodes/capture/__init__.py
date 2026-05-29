"""Capture lane (Lane A) — silent observer pattern.

Public surface:
  * `emit_new_issue_node` — graph node; flags ticket as "New issue" and
    arms the recorder. Synchronous, no HITL.
  * `recorder` — module-level in-process buffer for live capture events.
  * `finalize_capture` — out-of-band coordinator that synthesizes the
    captured trace into a structured fix plan and persists it.
  * `extract_steps_node`, `store_fix_node` — invoked by the finalize
    coordinator (NOT wired into the graph anymore).
  * `prompt_capture_node` — DEPRECATED alias kept for backward compat.
"""

from bugfix_ai.nodes.capture import recorder
from bugfix_ai.nodes.capture.emit_new_issue import (
    NEW_ISSUE_SIGNAL,
    emit_new_issue_node,
)
from bugfix_ai.nodes.capture.extract_steps import extract_steps_node
from bugfix_ai.nodes.capture.finalize import finalize_capture
from bugfix_ai.nodes.capture.prompt_capture import prompt_capture_node
from bugfix_ai.nodes.capture.store_fix import store_fix_node

__all__ = [
    "NEW_ISSUE_SIGNAL",
    "emit_new_issue_node",
    "extract_steps_node",
    "finalize_capture",
    "prompt_capture_node",
    "recorder",
    "store_fix_node",
]
