"""Centralized settings, validated once at startup via Pydantic.

All other modules import `get_settings()` instead of reading env vars directly.
This keeps the .env contract in one place and gives us type-checked access.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── GPT-oss-20B (local OpenAI-compatible server) ─────────────────────────
    gpt_oss_base_url: str = "http://localhost:11434/v1"
    gpt_oss_api_key: str = "ollama"
    gpt_oss_model: str = "gpt-oss:20b"
    gpt_oss_timeout_seconds: float = 120.0
    gpt_oss_max_retries: int = 3

    # ── Embedding model ──────────────────────────────────────────────────────
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "bugfix_embeddings"
    qdrant_api_key: str | None = None

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_url: str = "postgresql+asyncpg://bugfix:bugfix@localhost:5432/bugfix_ai"

    # ── GitHub Code Scanning (Phase 2 — autonomous-fix half) ─────────────────
    github_token: str = ""
    github_owner: str = "AMD-SWV-Driver"
    github_repo: str = "drivers"
    github_ref: str = "amd/release/25.30.01.67-SWV"
    github_severity_filter: str = "critical,high,medium,low"
    github_tool_filter: str = "CodeQL"
    github_poll_interval_seconds: int = 900
    github_webhook_secret: str = ""  # set if exposing /ingest/github/webhook publicly

    # ── GitHub Issues (Phase 1 — categorization & Excel showcase) ───────────
    # Fine-Grained PAT with the minimum permissions:
    #   Repository permissions → Issues: Read-only, Metadata: Read-only
    # Used by the GraphQL client. Falls back to `github_token` if unset so a
    # single classic PAT keeps working for dev.
    github_pat_fine_grained: str = ""
    # When True, ingestion goes through the GraphQL client (preferred for
    # bulk: one round-trip pulls 100 issues with all fields needed).
    # When False, the legacy REST client is used.
    categorization_use_graphql: bool = True
    # GraphQL pagination knobs (open-issue search).
    graphql_page_size: int = 100         # GitHub max per page
    graphql_max_pages: int = 100         # safety cap → up to 10k issues per run

    # ── Scheduled categorization (showcase: in-process; prod: GitHub Actions)
    # When the FastAPI app is the runtime, a background coroutine wakes
    # daily at this UTC hour to ingest open issues, categorize them, and
    # refresh the Excel report on disk.
    categorization_scheduled_enabled: bool = True
    categorization_scheduled_hour_utc: int = Field(default=2, ge=0, le=23)
    categorization_scheduled_minute_utc: int = Field(default=0, ge=0, le=59)

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "bugfix-ai-production"

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "bugfix-ai"

    # ── App ──────────────────────────────────────────────────────────────────
    api_key: str = Field(default="dev-replace-me", min_length=8)
    environment: Literal["development", "staging", "production"] = "development"
    checkpointer_type: Literal["sqlite", "postgres"] = "postgres"
    sqlite_db_path: str = "./checkpoints.db"

    # ── Project phase ────────────────────────────────────────────────────────
    # Phase 1 = ingest + categorize + Excel report (showcase deliverable).
    #          The LangGraph stack (checkpointer, MLflow, Qdrant, etc.) is
    #          NOT initialized — keeps the demo runtime light.
    # Phase 2 = developer-facing fix-mode lanes (capture / assist / autonomous)
    #          built on top of the categorized output. Switches the lifespan
    #          to the full graph build.
    phase: Literal["1", "2"] = "1"

    # ── Log sources ──────────────────────────────────────────────────────────
    log_source: Literal["file", "s3", "elk", "splunk", "none"] = "file"
    log_base_path: str = "/var/log/services"

    # ── Safety ───────────────────────────────────────────────────────────────
    terminal_dry_run_default: bool = True
    terminal_allowlist_path: str = "./config/terminal_allowlist.yaml"
    terminal_timeout_seconds: float = 60.0
    max_redo_per_step: int = 3

    # ── Issue categorization (Part 1 — showcase) ─────────────────────────────
    issue_taxonomy_path: str = "./bugfix_ai/config/issue_taxonomy.yaml"
    categorization_llm_threshold: float = 0.75   # rule confidence below this → call LLM
    categorization_max_body_chars: int = 4000    # truncation bound for LLM prompt
    issues_default_state: Literal["open", "closed", "all"] = "open"
    issues_per_page: int = 100
    issues_max_pages: int = 50
    # Where the latest generated Excel report is written / served from.
    excel_output_dir: str = "./out"
    excel_report_filename: str = "issues_categorized.xlsx"

    # ── RL / training data ──────────────────────────────────────────────────
    trajectory_jsonl_path: str = "./data/trajectories.jsonl"
    sft_dataset_path: str = "./data/sft_dataset.jsonl"
    dpo_dataset_path: str = "./data/dpo_dataset.jsonl"

    @field_validator("github_severity_filter")
    @classmethod
    def _split_csv(cls, v: str) -> str:  # noqa: D401
        # Accept comma-separated, normalize whitespace
        return ",".join(part.strip().lower() for part in v.split(",") if part.strip())

    def severity_list(self) -> list[str]:
        return [s for s in self.github_severity_filter.split(",") if s]

    def is_production(self) -> bool:
        return self.environment == "production"

    def issues_token(self) -> str:
        """The PAT used by the Phase-1 issues GraphQL/REST clients.

        Prefers the fine-grained PAT (least-privilege: Issues:read,
        Metadata:read). Falls back to the classic `github_token` so a
        single-token dev setup keeps working.
        """
        return self.github_pat_fine_grained or self.github_token


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. The .env file is read exactly once per process."""
    return Settings()  # type: ignore[call-arg]
