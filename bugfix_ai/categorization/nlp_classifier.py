"""NLP categorization layer powered by GPT-oss-20B.

Why this exists in Phase 1:
  The URL extractor handles every issue that has a `/driver/...` link
  pointing at the offending source file. Many issues won't — the
  reporter pasted a stack trace, described the symptom in prose, or
  linked an internal doc instead. For those, we hand the issue to
  GPT-oss-20B and ask it to invent a short `<area>:<sub-area>` label.

Why GPT-oss-20B specifically:
  Hard client constraint — the only foundation model permitted is
  OpenAI's open-weight 20B-parameter MoE (`gpt-oss:20b`) served by a
  local OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp). Calls
  go through `core.llm_client.chat_structured`, which uses JSON-mode +
  Pydantic validation with a parse-retry loop — robust enough to
  compensate for local servers' inconsistent strict-schema support.

Defensive design:
  * Body text is truncated to `categorization_max_body_chars` to bound the
    prompt size; long stack traces don't help category selection.
  * On LLM failure (transport, validation), we DO NOT fabricate a result;
    we return None and let the pipeline fall back to "uncategorized".
  * `category` is FREE-FORM (length-bounded by the schema) so the model
    can produce driver-area labels we didn't anticipate. `priority`
    stays a closed Literal — that's the field the schema validator
    actually catches hallucinations on.
"""

from __future__ import annotations

from bugfix_ai.categorization.schema import IssueRecord, LLMCategorization
from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.core.llm_client import LLMError, chat_structured
from bugfix_ai.integrations.logs.redactor import redact_text

log = get_logger(__name__)


_SYSTEM_PROMPT = (
    "You are a driver-software triage assistant. Given a GitHub issue's "
    "title, body, and labels, produce a SHORT category label and a "
    "PRIORITY for the issue. Use the issue text and labels as your "
    "evidence — do not invent details.\n\n"
    "CATEGORY — return a short, lowercase, slash- or colon-separated "
    "label that names the driver area and (when clear) the sub-area "
    "involved. Examples:\n"
    "  audio/codec\n"
    "  audio:dma\n"
    "  display/dp\n"
    "  display:hdmi\n"
    "  kernel/memory\n"
    "  build:dependencies\n"
    "  power/thermal\n"
    "  test/regression\n"
    "Keep it under 80 characters. If the area is genuinely unclear, "
    "return `unknown`.\n\n"
    "PRIORITY — pick EXACTLY ONE of:\n"
    "  critical    — production outage, data loss, security breach in progress\n"
    "  high        — major broken functionality, no workaround, urgent\n"
    "  medium      — significant impact but workaround exists\n"
    "  low         — minor / cosmetic / nice-to-have\n"
    "  unspecified — nothing in the issue suggests a priority\n\n"
    "Return your decision in the structured JSON schema below. Keep "
    "`reasoning` to ONE short paragraph citing the specific signals you used."
)


def _build_user_prompt(issue: IssueRecord) -> str:
    settings = get_settings()
    body = redact_text(issue.body or "")[: settings.categorization_max_body_chars]
    labels = ", ".join(issue.labels) if issue.labels else "(none)"
    return (
        f"Issue #{issue.issue_id} — state={issue.state}\n"
        f"Title: {issue.title}\n"
        f"Labels: {labels}\n"
        f"Author: {issue.author or '(unknown)'}\n"
        f"Repository: {issue.repository or '(unknown)'}\n\n"
        f"Body (redacted, possibly truncated):\n{body or '(empty)'}"
    )


async def classify_issue_with_llm(issue: IssueRecord) -> LLMCategorization | None:
    """Categorize a single issue via the LLM. Returns None on failure."""
    user = _build_user_prompt(issue)
    try:
        result = await chat_structured(
            system=_SYSTEM_PROMPT,
            user=user,
            schema=LLMCategorization,
            temperature=0.0,
            max_tokens=512,
        )
    except LLMError as e:
        log.warning("nlp_classifier.failed", issue=issue.issue_id, error=str(e)[:300])
        return None

    log.info(
        "nlp_classifier.classified",
        issue=issue.issue_id,
        category=result.category,
        priority=result.priority,
        confidence=result.confidence,
    )
    return result
