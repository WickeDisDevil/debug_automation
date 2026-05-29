# Phase 2 Codespace

This folder is a **separate** dev container config from the default
Phase 1 one (`.devcontainer/devcontainer.json`). When you click
**Code → Codespaces → New with options** on GitHub, the dropdown lets
you pick *which* dev container to use — pick this one to get the full
Phase 2 stack.

## What it provisions

Unlike the lightweight Phase 1 container, this one uses **Docker
Compose** to bring up four sibling containers on the same network:

| Service    | Image                              | Purpose                                 |
|------------|------------------------------------|-----------------------------------------|
| `app`      | `python:1-3.11-bookworm` (devcontainer) | VS Code attaches here                   |
| `postgres` | `postgres:16`                      | LangGraph checkpointer + structured fix store |
| `qdrant`   | `qdrant/qdrant:v1.11.0`            | Vector store for fix-memory retrieval   |
| `mlflow`   | `ghcr.io/mlflow/mlflow:v2.15.0`    | Experiment tracking + run history       |

App reaches the others by service name (`postgres`, `qdrant`,
`mlflow`) — no `localhost` games needed.

**Forwarded ports:** `8000` (FastAPI), `5432` (Postgres), `6333`
(Qdrant HTTP), `5000` (MLflow UI — auto-opens in your browser).

## What `postCreate.sh` does

Runs once when the codespace is first created:

1. Upgrades pip / setuptools / wheel.
2. Installs the CPU-only Torch wheel.
3. Installs `bugfix_ai/requirements.txt`, the project in editable mode,
   plus `ruff` / `pytest` / `pytest-asyncio` for the test suite.
4. Seeds `bugfix_ai/.env` from `.env.example` with Phase-2 service
   hostnames (`POSTGRES_HOST=postgres`, `QDRANT_URL=http://qdrant:6333`,
   `MLFLOW_TRACKING_URI=http://mlflow:5000`).
5. Waits for Postgres to accept connections.
6. Runs `alembic upgrade head` if `bugfix_ai/alembic.ini` exists.
7. Prints a "what to try next" banner.

## How to use the codespace

1. On GitHub: **Code → Codespaces → New with options**.
2. Under **Dev container configuration**, pick
   *"BugFix AI - Codespaces (Phase 2 full stack)"*.
3. Choose a machine type with **at least 4 cores / 8 GB RAM** — the
   four-service stack will not fit on the default 2-core box.
4. Wait for `postCreate.sh` to finish (~5-7 minutes first time).
5. In the terminal, run one of:
   ```bash
   uvicorn bugfix_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
   pytest -q
   ```
6. Open the **Ports** tab to launch Swagger (port 8000) or the MLflow
   UI (port 5000).

## Cost / quota note

Phase 2 burns codespace hours faster than Phase 1 because of the
larger machine type and the four always-on containers. Stop the
codespace from the GitHub UI when you're done.
