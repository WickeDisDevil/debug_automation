"""Assemble and compile the LangGraph StateGraph.

Topology at a glance (post silent-observer refactor):
        intake → classify
                   │
        ┌──────────┼──────────────┐
        ▼          ▼              ▼
     capture     assist        autonomous
        │          │              │
   emit_new_issue  semantic_retrieve   load_fix_plan
        │          ↓              ↓
        └─▶ END    rule_filter    pre_execute_review (HITL)
                   ↓              ↓
                 rrf_rank         execute_step
                   ↓              ↓
                 present_similar  hitl_checkpoint (HITL)
                   ↓              │
                consent_gate(HITL)│  ┌── continue → advance_step → pre_execute
                                  │  ├── redo     → pre_execute_review
                                  │  └── manual   → manual_fallback → END
                                  │     complete  → END

Capture lane is intentionally a single-node terminator now. The
`emit_new_issue` node flags the ticket as "New issue" and returns
control to the engineer immediately — there is NO HITL pause here.
A background recorder (`nodes.capture.recorder`) collects events as
the engineer fixes the bug, and a separate coordinator
(`nodes.capture.finalize.finalize_capture`) is invoked out-of-band
when the engineer marks the bug resolved (or an editor hook does it
for them). The coordinator runs `extract_steps` then `store_fix`
without ever re-entering the graph, so the engineer never has to
narrate the fix.

Why `build_graph(checkpointer)` is a function (and not a module-level
singleton compiled at import time):
  * The compiled graph captures a checkpointer connection. We MUST
    create that connection on the running event loop, not at import.
  * Tests build their own graph against an in-memory SQLite — that's
    only possible if construction is parameterized.

Why THREE interrupts now (down from four):
  * consent_gate — wait for the developer to confirm "yes, replay
    that similar fix autonomously" vs. "no, I'll do it manually".
  * pre_execute_review — wait for the developer to approve / edit
    the LLM-adapted command BEFORE we run it. This is the deviation
    from the original architecture: separating "model adapts" from
    "human approves" from "machine runs" makes the safety boundary
    crystal-clear.
  * hitl_checkpoint — wait for the developer to verify the result
    of the executed step (success / redo / abort).
  The previous `prompt_capture` interrupt has been retired — capture
  mode is now silent.

Conditional edges:
  Routing decisions (`_route_after_*`) are pure functions of state —
  no I/O, no LLM calls. They run inside LangGraph's edge resolver
  and must be cheap and deterministic.

Internal helpers:
  `_advance_step_node` and `_complete_autonomous_node` are tiny
  state-mutation nodes inlined here because they're glue, not
  business logic. Promoting them to their own files would just add
  navigation cost.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from bugfix_ai.core.state import BugFixState
from bugfix_ai.nodes.assist.consent_gate import consent_gate_node
from bugfix_ai.nodes.assist.present_similar import present_similar_node
from bugfix_ai.nodes.assist.rrf_rank import rrf_rank_node
from bugfix_ai.nodes.assist.rule_filter import rule_filter_node
from bugfix_ai.nodes.assist.semantic_retrieve import semantic_retrieve_node
from bugfix_ai.nodes.autonomous.execute_step import execute_step_node
from bugfix_ai.nodes.autonomous.hitl_checkpoint import hitl_checkpoint_node
from bugfix_ai.nodes.autonomous.load_fix_plan import load_fix_plan_node
from bugfix_ai.nodes.autonomous.manual_fallback import manual_fallback_node
from bugfix_ai.nodes.autonomous.pre_execute_review import pre_execute_review_node
from bugfix_ai.nodes.capture.emit_new_issue import emit_new_issue_node
from bugfix_ai.nodes.classify import classify_node
from bugfix_ai.nodes.intake import intake_node


# ── Conditional edge routers ─────────────────────────────────────────────────


def _route_after_classify(state: BugFixState) -> str:
    return state["mode"]


def _route_after_consent(state: BugFixState) -> str:
    decision = state.get("hitl_decision")
    if decision == "autonomous":
        return "load_fix_plan"
    return "manual_fallback"


def _route_after_pre_execute(state: BugFixState) -> str:
    """After human reviews the adapted command, decide whether to actually run it."""
    decision = state.get("hitl_decision")
    if decision == "continue":
        return "execute_step"
    return "manual_fallback"


def _route_after_hitl(state: BugFixState) -> str:
    decision = state.get("hitl_decision")
    if decision == "continue":
        if state["current_step_idx"] + 1 < len(state["fix_plan"]):
            return "advance_step"
        return "complete_autonomous"
    if decision == "redo":
        return "pre_execute_review"
    return "manual_fallback"


def _advance_step_node(state: BugFixState) -> dict:
    """Pure state-mutation node: increment current_step_idx, reset redo counter."""
    return {
        "current_step_idx": state["current_step_idx"] + 1,
        "redo_count": 0,
        "hitl_decision": None,
    }


def _complete_autonomous_node(state: BugFixState) -> dict:
    return {"autonomous_success": True}


# ── Builder ─────────────────────────────────────────────────────────────────


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Construct, validate, and compile the StateGraph.

    Compilation:
      - Bakes in the checkpointer (durable HITL state).
      - Bakes in the interrupt points so they cannot be forgotten at call sites.
    """
    builder = StateGraph(BugFixState)

    # Common
    builder.add_node("intake", intake_node)
    builder.add_node("classify", classify_node)

    # Capture (silent observer — no HITL interrupt, no in-graph extract/persist).
    # extract_steps + store_fix run out-of-band via
    # `nodes.capture.finalize.finalize_capture` when the engineer marks the
    # bug resolved.
    builder.add_node("emit_new_issue", emit_new_issue_node)

    # Assist
    builder.add_node("semantic_retrieve", semantic_retrieve_node)
    builder.add_node("rule_filter", rule_filter_node)
    builder.add_node("rrf_rank", rrf_rank_node)
    builder.add_node("present_similar", present_similar_node)
    builder.add_node("consent_gate", consent_gate_node)

    # Autonomous
    builder.add_node("load_fix_plan", load_fix_plan_node)
    builder.add_node("pre_execute_review", pre_execute_review_node)
    builder.add_node("execute_step", execute_step_node)
    builder.add_node("hitl_checkpoint", hitl_checkpoint_node)
    builder.add_node("advance_step", _advance_step_node)
    builder.add_node("complete_autonomous", _complete_autonomous_node)
    builder.add_node("manual_fallback", manual_fallback_node)

    # Entry
    builder.set_entry_point("intake")
    builder.add_edge("intake", "classify")

    # Capture sub-graph — single-node terminator. The recorder + finalize
    # coordinator handle the rest outside the graph.
    builder.add_edge("emit_new_issue", END)

    # Assist sub-graph
    builder.add_edge("semantic_retrieve", "rule_filter")
    builder.add_edge("rule_filter", "rrf_rank")
    builder.add_edge("rrf_rank", "present_similar")
    builder.add_edge("present_similar", "consent_gate")

    # Autonomous sub-graph
    builder.add_edge("load_fix_plan", "pre_execute_review")
    builder.add_edge("execute_step", "hitl_checkpoint")
    builder.add_edge("advance_step", "pre_execute_review")
    builder.add_edge("complete_autonomous", END)
    builder.add_edge("manual_fallback", END)

    # Conditional edges
    builder.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "capture": "emit_new_issue",
            "assist": "semantic_retrieve",
            "autonomous": "load_fix_plan",
        },
    )
    builder.add_conditional_edges(
        "consent_gate",
        _route_after_consent,
        {"load_fix_plan": "load_fix_plan", "manual_fallback": "manual_fallback"},
    )
    builder.add_conditional_edges(
        "pre_execute_review",
        _route_after_pre_execute,
        {"execute_step": "execute_step", "manual_fallback": "manual_fallback"},
    )
    builder.add_conditional_edges(
        "hitl_checkpoint",
        _route_after_hitl,
        {
            "advance_step": "advance_step",
            "pre_execute_review": "pre_execute_review",  # redo path
            "complete_autonomous": "complete_autonomous",
            "manual_fallback": "manual_fallback",
        },
    )

    return builder.compile(
        checkpointer=checkpointer,
        # Three interrupts (capture mode is silent — no HITL pause):
        #   consent_gate       — wait for dev to choose autonomous vs manual
        #   pre_execute_review — wait for dev to approve adapted command
        #   hitl_checkpoint    — wait for dev to verify step result
        interrupt_before=[
            "consent_gate",
            "pre_execute_review",
            "hitl_checkpoint",
        ],
    )
