#!/usr/bin/env bash
# Phase-2 codespace bootstrap.
#
# Idempotent — safe to re-run. Steps:
#   1. Install CPU-only Torch (saves ~2 GB vs the default CUDA build —
#      Codespaces has no GPU).
#   2. Install bugfix_ai/requirements.txt + project in editable mode.
#   3. Seed .env from .env.example with Phase-2 service hostnames.
#   4. Wait for Postgres to accept connections (compose healthcheck
#      already gates this, but belt-and-braces for slow first-boot).
#   5. Run alembic migrations if an alembic config is present.
#   6. Print a banner with the entry points.

set -euo pipefail

echo "==> Phase-2 codespace bootstrap"

python -m pip install --upgrade pip setuptools wheel

echo "==> Installing CPU-only Torch"
pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "==> Installing project requirements"
pip install -r bugfix_ai/requirements.txt
pip install -e .
pip install ruff pytest pytest-asyncio

ENV_FILE="bugfix_ai/.env"
if [ ! -f "$ENV_FILE" ] && [ -f "bugfix_ai/.env.example" ]; then
  echo "==> Seeding $ENV_FILE from .env.example"
  cp bugfix_ai/.env.example "$ENV_FILE"
  {
    echo ""
    echo "# Phase-2 codespace overrides (added by postCreate.sh)"
    echo "PHASE=2"
    echo "CHECKPOINTER_TYPE=postgres"
    echo "POSTGRES_HOST=postgres"
    echo "POSTGRES_PORT=5432"
    echo "POSTGRES_USER=bugfix"
    echo "POSTGRES_PASSWORD=bugfix"
    echo "POSTGRES_DB=bugfix"
    echo "QDRANT_URL=http://qdrant:6333"
    echo "MLFLOW_TRACKING_URI=http://mlflow:5000"
  } >> "$ENV_FILE"
fi

echo "==> Waiting for Postgres on postgres:5432"
for i in $(seq 1 30); do
  if (echo > /dev/tcp/postgres/5432) 2>/dev/null; then
    echo "    Postgres is up."
    break
  fi
  sleep 2
done

if [ -f "bugfix_ai/alembic.ini" ]; then
  echo "==> Running alembic migrations"
  (cd bugfix_ai && alembic upgrade head) || echo "    (alembic failed — skipping)"
fi

cat <<'BANNER'

==============================================================
 BugFix AI - Phase 2 codespace ready
==============================================================
 Services (sibling containers on the compose network):
   * postgres : postgres:5432   (user/pass/db: bugfix)
   * qdrant   : http://qdrant:6333
   * mlflow   : http://mlflow:5000  (auto-opens in browser)

 Try one of:
   uvicorn bugfix_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
   pytest -q

 Open the **Ports** tab in VS Code to hit the FastAPI Swagger UI
 (port 8000) and the MLflow tracking UI (port 5000).
==============================================================

BANNER
