"""initial schema: fixes, error_type_baselines, ingestion_seen, preference_pairs

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-05-11
"""
from __future__ import annotations

from alembic import op


revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── fixes: canonical row per resolved bug ──────────────────────────────
    op.execute(
        """
        CREATE TABLE fixes (
            id                    TEXT PRIMARY KEY,
            run_id                TEXT,
            ticket_id             TEXT,
            ticket_title          TEXT,
            ticket_description    TEXT,
            alert_number          INTEGER,
            repository            TEXT,
            ref                   TEXT,
            service               TEXT,
            environment           TEXT,
            error_type            TEXT,
            severity              TEXT,
            stack_pattern         TEXT,
            classify_confidence   DOUBLE PRECISION,
            mode                  TEXT,
            root_cause            TEXT,
            fix_summary           TEXT,
            steps_json            JSONB NOT NULL DEFAULT '[]'::jsonb,
            steps                 JSONB NOT NULL DEFAULT '[]'::jsonb,
            dev_narrative         TEXT,
            error_logs_redacted   TEXT,
            resolved_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            time_to_resolve_min   INTEGER NOT NULL DEFAULT 0,
            redo_count            INTEGER NOT NULL DEFAULT 0,
            autonomous_success    BOOLEAN NOT NULL DEFAULT FALSE,
            ruler_score           DOUBLE PRECISION,
            created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute("CREATE INDEX ix_fixes_error_type ON fixes (error_type);")
    op.execute("CREATE INDEX ix_fixes_service ON fixes (service);")
    op.execute("CREATE INDEX ix_fixes_resolved_at ON fixes (resolved_at DESC);")
    op.execute("CREATE INDEX ix_fixes_alert_number ON fixes (alert_number);")

    # ── error_type_baselines: for time-saved metric ────────────────────────
    op.execute(
        """
        CREATE TABLE error_type_baselines (
            error_type              TEXT NOT NULL,
            service                 TEXT,
            avg_resolution_minutes  DOUBLE PRECISION NOT NULL,
            sample_count            INTEGER NOT NULL DEFAULT 0,
            updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (error_type, COALESCE(service, ''))
        );
        """
    )

    # ── ingestion_seen: idempotency for webhooks / poller ──────────────────
    op.execute(
        """
        CREATE TABLE ingestion_seen (
            provider       TEXT NOT NULL,
            alert_number   BIGINT NOT NULL,
            last_state     TEXT,
            etag           TEXT,
            last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (provider, alert_number)
        );
        """
    )

    # ── preference_pairs: DPO/GRPO training data ───────────────────────────
    op.execute(
        """
        CREATE TABLE preference_pairs (
            pair_id           TEXT PRIMARY KEY,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            source            TEXT NOT NULL,
            error_type        TEXT,
            run_id            TEXT,
            chosen_payload    TEXT NOT NULL,
            rejected_payload  TEXT NOT NULL,
            margin            DOUBLE PRECISION
        );
        """
    )
    op.execute("CREATE INDEX ix_preference_pairs_source ON preference_pairs (source);")
    op.execute(
        "CREATE INDEX ix_preference_pairs_error_type ON preference_pairs (error_type);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS preference_pairs;")
    op.execute("DROP TABLE IF EXISTS ingestion_seen;")
    op.execute("DROP TABLE IF EXISTS error_type_baselines;")
    op.execute("DROP TABLE IF EXISTS fixes;")
