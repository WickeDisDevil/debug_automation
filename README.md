# BugFix AI

A LangGraph-based debugging assistant that ingests **GitHub Code Scanning (CodeQL) alerts**, retrieves similar past fixes via hybrid semantic + rules retrieval, and either presents them to a developer or replays them autonomously under human-in-the-loop checkpoints.

The system is designed around a single hard rule: **all LLM calls go through GPT-oss-20B** (served by an OpenAI-compatible local endpoint such as Ollama, vLLM, or llama.cpp). No other foundation model is permitted. The only other ML weights in the stack are the in-process sentence-transformers embedding model.

---

## Architecture at a glance

```
┌────────────────────────┐
│ GitHub Code Scanning   │  alerts via REST API + webhook
└────────────┬───────────┘
             │
             ▼
   ┌─────────────────┐    ┌──────────────────────┐
   │  Poller / API   │───▶│  LangGraph StateGraph │
   └─────────────────┘    │   (durable HITL)     │
                          └────┬─────────┬────────┘
                               │         │
       ┌───────────────────────┘         └──────────────────────┐
       ▼                                                        ▼
┌─────────────────────────────┐                ┌────────────────────────────┐
│ Hybrid retrieval            │                │ Autonomous executor        │
│   semantic (Qdrant)         │                │   pre-execute review (HIL) │
│   rules    (Postgres)       │                │   safe shell (allowlist)   │
│   RRF + recency decay       │                │   per-step HITL checkpoint │
└─────────────┬───────────────┘                └────────────┬───────────────┘
              │                                             │
              └──────────────► fixes table + Qdrant ◀───────┘
                                       │
                                       ▼
                             ┌──────────────────┐
                             │  RL training     │
                             │   trajectories   │
                             │   RULER judge    │
                             │   DPO/GRPO       │
                             └──────────────────┘
```

The graph has **four interrupt points** so a human is in the loop at every irreversible decision:

1. `prompt_capture` — wait for the dev to write the fix narrative (capture mode).
2. `consent_gate` — wait for the dev to choose "autonomous" vs "manual".
3. `pre_execute_review` — wait for the dev to approve (or edit) the adapted command BEFORE it runs.
4. `hitl_checkpoint` — wait for the dev to confirm the result and choose continue / redo / manual.

---

## Local setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Bring up Postgres, Qdrant, MLflow
docker compose up -d

# 3. Configure
cp bugfix_ai/.env.example .env
$EDITOR .env   # set GITHUB_TOKEN, GPT_OSS_BASE_URL, etc.

# 4. Migrate the database
alembic upgrade head

# 5. Seed baselines (optional, drives the time-saved metric)
python -m scripts.seed_rules_db

# 6. Start the API
uvicorn bugfix_ai.api.main:app --host 0.0.0.0 --port 8000
```

Hit `GET /health` (no auth) to confirm Postgres + Qdrant + the compiled graph are all healthy.

---

## Required environment

The `.env` contract is enforced by `bugfix_ai/config/settings.py`. Key variables:

| Variable | Purpose |
|---|---|
| `GPT_OSS_BASE_URL` | OpenAI-compatible endpoint for GPT-oss-20B (default `http://localhost:11434/v1` for Ollama). |
| `GPT_OSS_API_KEY` | Token for the local server (Ollama accepts the literal string `ollama`). |
| `GPT_OSS_MODEL` | Model id, default `gpt-oss:20b`. |
| `EMBEDDING_MODEL_NAME` | Sentence-transformers model, default `BAAI/bge-small-en-v1.5` (384-dim). |
| `EMBEDDING_DIMENSIONS` | Must match the model. |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant connection. |
| `POSTGRES_URL` | Async URL: `postgresql+asyncpg://...`. |
| `GITHUB_TOKEN` | Personal access token with `security_events:read` on the target repo. |
| `GITHUB_OWNER` / `GITHUB_REPO` / `GITHUB_REF` | Where to pull alerts from. |
| `GITHUB_WEBHOOK_SECRET` | Optional; required if you expose `/ingest/github/webhook`. |
| `MLFLOW_TRACKING_URI` | MLflow tracking server. |
| `API_KEY` | Required for every non-`/health` HTTP call (`X-API-Key` header). |
| `TERMINAL_DRY_RUN_DEFAULT` | Default `true`. Flip to `false` only after operational confidence. |
| `TERMINAL_ALLOWLIST_PATH` | YAML allowlist of binaries + deny tokens. |

---

## API surface

All endpoints (except `/health` / `/healthz` / `/docs`) require `X-API-Key`.

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/healthz` | Liveness. |
| `GET`  | `/health`  | Readiness — pings Postgres and Qdrant. |
| `POST` | `/sessions` | Start a manual run with an arbitrary ticket payload. |
| `GET`  | `/sessions/{thread_id}` | Fetch the latest snapshot. |
| `POST` | `/sessions/{thread_id}/decision` | Resume from an interrupt with a decision. |
| `POST` | `/ingest/github/alert` | Ingest one alert by number. |
| `POST` | `/ingest/github/poll` | Trigger a one-off poll of open alerts. |
| `POST` | `/ingest/github/webhook` | GitHub `code_scanning_alert` webhook. |

The decision payload covers all four interrupts:

```jsonc
{
  "decision": "continue | redo | manual | autonomous",
  "selected_fix_id": "...",       // at consent_gate
  "dev_narrative":  "...",        // at prompt_capture
  "edited_command": "..."         // at pre_execute_review (overrides the LLM's adapted command)
}
```

---

## Safety model

Two-layer defense around autonomous execution:

1. **Static** — `bugfix_ai/integrations/terminal/safe_executor.py` rejects any command before spawning a process if it: contains shell metacharacters (`|`, `;`, `&&`, backticks, `$()`), matches a deny token from the allowlist YAML (e.g. `rm -rf`, `DROP TABLE`), uses a binary not on the allowlist, or uses a disallowed subcommand of an allowlisted binary.
2. **Human** — `pre_execute_review_node` runs the LLM-based command adaptation, runs the static check for informational purposes, and the graph then **interrupts** so the human can approve, edit, or reject.

Defaults are conservative: `terminal_dry_run_default=True`, the allowlist starts with read-only binaries (`ls`, `cat`, `kubectl get`, `systemctl status`), and the redo limit prevents loops on a single step.

PII / secret redaction (`bugfix_ai/integrations/logs/redactor.py`) runs before any log content reaches the LLM. JWTs, AWS keys, GitHub PATs, OpenAI keys, bearer tokens, DB DSNs, emails, IPv4 addresses, hex tokens, and SSNs are all stripped.

---

## Reinforcement learning loop

Every completed run produces a `Trajectory` (`bugfix_ai/rl/trajectory_collector.py`) appended to `data/trajectories.jsonl`. The `RulerRanking` judge (`bugfix_ai/rl/ruler.py`) and recorded preference pairs (`bugfix_ai/rl/preference_store.py`) feed a TRL-compatible export:

```bash
python -m scripts.export_training_data
# writes data/sft_dataset.jsonl  (HuggingFace messages format)
# writes data/dpo_dataset.jsonl  ({prompt, chosen, rejected})
```

Recommended rollout:

1. Collect 200–500 trajectories under capture/assist mode.
2. Run **SFT** on `sft_dataset.jsonl` first — it's the safest warm-start and almost always wins outright on this kind of structured-output task.
3. Only after SFT plateaus, add **DPO** (or GRPO) with `dpo_dataset.jsonl`. Without an SFT warm-start GPT-oss-20B's reasoning chains are too noisy for stable RL.

---

## Observability

Each run produces:

- A structlog JSON log stream with a per-request `correlation_id`.
- An `obs_log` field on the graph state — the timeline of every node decision.
- An MLflow run (`bugfix_ai/observability/mlflow_tracker.py`) with the run id stored on state, so all node-level metrics tag back to one parent run.
- Final metrics: `time_to_resolve_min`, `time_saved_min` (vs `error_type_baselines`), `lines_saved` (when `GIT_REPO_PATH` is set), `redo_count`, `autonomous_success`.

---

## Project layout

```
bugfix_ai/
  api/                 FastAPI app, routers, middleware
  config/              pydantic settings, allowlist YAML, structlog
  core/                state schema, llm client, checkpointer, graph builder
  integrations/
    github/            Code Scanning REST client + poller (idempotent)
    logs/              fetch + preprocess + redact
    terminal/          safe shell executor
  memory/
    embed.py           sentence-transformers (in-process)
    vector_store.py    Qdrant async client
    rules_db.py        Postgres async (raw SQL via `text()`)
    fix_store.py       writes both Postgres + Qdrant
    retrieval/         RRF + hybrid retriever
  nodes/               LangGraph node functions
    capture/           prompt_capture, extract_steps, store_fix
    assist/            semantic_retrieve, rule_filter, rrf_rank, present_similar, consent_gate
    autonomous/        load_fix_plan, pre_execute_review, execute_step, hitl_checkpoint, manual_fallback
  observability/       mlflow_tracker, decision_logger, metrics
  rl/                  ruler, trajectory_collector, preference_store, sft_exporter
alembic/               db migrations (raw SQL)
scripts/               ingest_github, seed_rules_db, backfill_embeddings, export_training_data
docker-compose.yml     Postgres + Qdrant + MLflow
```

---

## Development

```bash
# format + lint
ruff check . && ruff format .

# type check
mypy bugfix_ai

# run a one-off ingestion against a single CodeQL alert
python -m scripts.ingest_github --alert 1234
```
