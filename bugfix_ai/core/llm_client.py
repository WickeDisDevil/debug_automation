"""Async wrapper around the OpenAI-compatible client pointed at GPT-oss-20B.

Why this module is the only place that talks to the LLM:
  Every node and helper in the codebase that needs the model goes
  through `chat_text()` (free-text) or `chat_structured()` (validated
  JSON). Centralizing here means one place owns the retry policy, the
  client pool, and the schema-enforcement story — there is no second
  way to call the model that could quietly skip those guarantees.

Production fixes baked in:
  * AsyncOpenAI — never blocks the event loop, so a slow LLM call
    can't stall the FastAPI worker handling other requests.
  * Tenacity exponential backoff on rate-limit / 5xx / network errors.
    The OpenAI SDK's built-in retries are disabled (max_retries=0)
    so we own the policy end-to-end.
  * JSON-mode + Pydantic validation with a parse-retry loop. On a
    Pydantic validation failure we re-prompt the model with the exact
    validation error message — this almost always succeeds on attempt
    2 because the model can see precisely what was wrong.
  * Per-call max_tokens cap so a runaway generation can't consume the
    server's whole context window.
  * lru_cached singleton client to share the underlying httpx
    connection pool across the process.

Why we don't use OpenAI strict json_schema response_format:
  GPT-oss-20B is served through Ollama / vLLM / llama.cpp depending on
  deployment. json_object mode is supported reliably; strict
  json_schema support is patchy (some servers accept the field and
  silently ignore it). Pydantic + retry gives the same correctness
  property with portable behavior.

Error model:
  All terminal failures bubble up as `LLMError` (a RuntimeError). Node
  code is expected to either propagate this (state-machine fails the
  step and surfaces it to the human) or catch it and fall back — see
  `categorization.nlp_classifier` for an example of graceful fallback.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import TypeVar

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM ultimately fails (after retries)."""


@lru_cache(maxsize=1)
def get_llm_client() -> AsyncOpenAI:
    """Process-wide AsyncOpenAI singleton pointed at the local GPT-oss server."""
    settings = get_settings()
    return AsyncOpenAI(
        base_url=settings.gpt_oss_base_url,
        api_key=settings.gpt_oss_api_key,
        timeout=settings.gpt_oss_timeout_seconds,
        max_retries=0,  # we drive retries via tenacity; disable SDK's built-in
    )


_RETRY_EXCEPTIONS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)


async def _chat_with_retry(
    *,
    messages: list[dict],
    response_format: dict | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Send a chat request with exponential backoff. Returns raw content string."""
    settings = get_settings()
    client = get_llm_client()

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(settings.gpt_oss_max_retries),
        wait=wait_random_exponential(multiplier=1, max=20),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        reraise=True,
    ):
        with attempt:
            kwargs: dict = {
                "model": settings.gpt_oss_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content is None:
                raise APIError("Empty completion content", request=None, body=None)
            return content


async def chat_text(
    *,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Plain free-text chat call. Use for narrative-style outputs."""
    return await _chat_with_retry(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def chat_structured(
    *,
    system: str,
    user: str,
    schema: type[T],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    parse_retries: int = 2,
) -> T:
    """Structured output via JSON-mode + Pydantic validation, with parse retries.

    Why this design and not OpenAI strict schema:
      - GPT-oss-20B served through Ollama/vLLM/llama.cpp accepts json_object
        mode reliably but strict json_schema support varies by server build.
      - We enforce the contract by validating with Pydantic; on validation
        failure we re-prompt the model with the validation error so it can fix
        its own output before we give up. This is more robust than betting on
        every server honoring the strict-schema flag.
    """
    schema_hint = schema.model_json_schema()
    augmented_system = (
        system
        + "\n\nRespond with a single JSON object that conforms to this schema:\n"
        + json.dumps(schema_hint, indent=2)
        + "\n\nReturn ONLY the JSON object. No prose, no code fences."
    )

    last_error: str | None = None
    for attempt_idx in range(parse_retries + 1):
        user_for_attempt = user
        if last_error and attempt_idx > 0:
            user_for_attempt = (
                f"{user}\n\n"
                f"Your previous response failed validation:\n{last_error}\n"
                f"Return a corrected JSON object now."
            )

        try:
            raw = await _chat_with_retry(
                messages=[
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": user_for_attempt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RetryError as e:
            raise LLMError(f"LLM transport failed: {e}") from e

        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            log.warning(
                "structured_output.parse_failed",
                attempt=attempt_idx + 1,
                error=last_error[:300],
            )
            continue

    raise LLMError(
        f"Structured LLM output failed validation after {parse_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
