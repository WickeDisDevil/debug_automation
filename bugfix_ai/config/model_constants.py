"""Pydantic schemas for the structured outputs we expect from GPT-oss-20B.

Why this module is the single source of truth:
  Every LLM call in this codebase passes through `core.llm_client
  .chat_structured(..., schema=SomeModel)`. That function uses JSON-mode
  + Pydantic validation + a parse-retry loop (re-prompts the model with
  the validation error). The schemas below are what those calls validate
  against, so any change here is a contract change for the model.

Why we don't use OpenAI strict json_schema response_format:
  GPT-oss-20B is served through Ollama / vLLM / llama.cpp depending on
  the deployment. Strict json_schema mode is supported inconsistently
  across those servers — some accept it, some accept the field but
  silently ignore it, some reject the call. JSON-mode + Pydantic gives
  the same correctness guarantee with portable behavior.

Schema groups:
  * FixStepModel / FixStepsResponse — extract_steps node. The `tool`
    Literal pins the model to the four tools we actually support; any
    other tool string would fail Pydantic and trigger a re-prompt.
  * ClassifyResponse — classify node. `suggested_mode` drives the
    capture / assist / autonomous routing decision.
  * AdaptedCommandResponse — pre_execute_review node. The `unchanged`
    bool forces the model to *explicitly* claim "no edit needed",
    which eliminates the failure mode of subtly rewording a command
    when no change was warranted.
  * RulerRanking / RulerScore — RL judge output. Used by `rl/ruler.py`
    to convert a list of trajectories into a ranking + per-trajectory
    scores; that ranking then becomes adjacent-pair preferences in
    `rl/preference_store.py`.

Field bounds (`min_length`, `max_length`, `ge`, `le`) are intentional —
they catch hallucinated empty strings and runaway essays before either
can reach a downstream consumer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── extract_steps output ─────────────────────────────────────────────────────


class FixStepModel(BaseModel):
    """A single atomic, executable step in a bug fix procedure."""

    step_idx: int = Field(ge=0)
    description: str = Field(min_length=1, max_length=500)
    tool: Literal["terminal", "github", "code_edit", "manual"]
    command: str | None = None  # null only when tool == "manual"
    expected_outcome: str = Field(min_length=1, max_length=500)
    is_reversible: bool


class FixStepsResponse(BaseModel):
    steps: list[FixStepModel]
    root_cause: str = Field(min_length=1, max_length=2000)
    fix_summary: str = Field(min_length=1, max_length=1000)


# ── classify output ──────────────────────────────────────────────────────────


class ClassifyResponse(BaseModel):
    error_type: str = Field(min_length=1, max_length=100)
    service: str = Field(min_length=1, max_length=100)
    severity: Literal["low", "medium", "high", "critical"]
    stack_pattern: str = Field(default="", max_length=500)
    suggested_mode: Literal["capture", "assist", "autonomous"]
    confidence: float = Field(ge=0.0, le=1.0)


# ── command adaptation output ────────────────────────────────────────────────


class AdaptedCommandResponse(BaseModel):
    """Output of the command-adapter LLM call.

    `unchanged` flag forces the model to be explicit when no edit is needed,
    which prevents drift from over-eager rewriting.
    """

    adapted_command: str = Field(min_length=1, max_length=2000)
    unchanged: bool
    reasoning: str = Field(default="", max_length=500)


# ── RULER (LLM-as-judge) output ──────────────────────────────────────────────


class RulerScore(BaseModel):
    fix_id: str
    score: float = Field(ge=0.0, le=1.0)


class RulerRanking(BaseModel):
    ranking: list[str]  # fix_ids ordered best-to-worst
    scores: list[RulerScore]
    reasoning: str = Field(max_length=2000)
