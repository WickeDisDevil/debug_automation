# BugFix AI — How the System Works
### A complete, plain-English walkthrough for a non-technical audience

---

## The one-paragraph summary

BugFix AI ingests every open issue from your GitHub repository, automatically classifies each one by the driver area it touches, and produces a downloadable Excel report grouped by category — one tab per driver area, with the columns your reviewers actually want (Issue ID, Title, Category, Priority, Status). Phase 1 delivers that classifier-and-spreadsheet showcase end-to-end. Phase 2 (planned) takes the categorized output and feeds it into a developer-facing assistant that recognizes recurring bug patterns, suggests prior fixes, and — for known mechanical fixes — applies the fix itself with human approval gates in front of every action.

---

## The big picture

Imagine the system as a triage analyst who:

1. Logs into your GitHub repository every morning.
2. Pulls every open issue ticket.
3. Reads the bug URL inside each ticket and groups the issue under the driver-area path it points to (e.g. `audio/codec`, `display/dp`, `kernel/memory`).
4. For tickets that don't carry a driver URL, asks a self-hosted AI model to invent a short label.
5. Writes the result to an Excel workbook and hands it over.
6. (Phase 2) When an engineer picks a bucket to start fixing, the assistant springs into action and routes them through one of three lanes — capture, assist, or autonomous.

Phase 1 covers steps 1–5. Phase 2 covers step 6. The Phase-1 deliverable is what your client sees first.

---

## Step 0 — The inputs

The system listens to two sources:

- **GitHub Issues (bulk ingest via GraphQL).** Authenticated with a **Fine-Grained Personal Access Token** carrying the absolute minimum permissions: `Issues: Read-only` and `Metadata: Read-only`. The GraphQL `search` connection is scoped to `repo:OWNER/REPO is:issue is:open`, so we only ever pull issues that are still open. One round-trip returns 100 issues with all the fields the categorizer needs (title, body, labels, html_url, author, timestamps, state). Cursor-based pagination handles thousands of issues reliably.
- **Manual ticket submission (single-issue API).** A `POST /issues/categorize/manual` endpoint accepts either a normalized record or a raw GitHub issue dict. Useful when an operator wants to drop in one ticket on demand without hitting GitHub.

**Why GraphQL instead of REST:** for a few hundred issues either is fine; for thousands, GraphQL pulls everything in a single typed query per page, while REST needs follow-up calls for label expansion in some shapes. Cursor-based pagination is also more reliable than REST's `page=N` numbering when the index is changing under us.

**Why a fine-grained PAT instead of a classic token:** least privilege. The fine-grained token is scoped to one repository and two read scopes. If it leaks, the blast radius is "someone read your open issues" — they cannot mutate state, cannot see private code, cannot see secrets.

---

## Step 1 — Intake and cleaning

GitHub Issues arrive as structured JSON objects. They already carry the fields we need — title, body, labels, state, URLs, author, timestamps. **No preprocessing is required.** The parser (`integrations/github/issues_parser.py`) flattens the payload into our internal `IssueRecord` schema and the categorizer reads from there.

(We left a hook for cleaning to be added later if a customer's repo needs it — body normalization, deduping, etc. — but it isn't on the Phase-1 critical path.)

---

## Step 2 — Classification (deciding what kind of bug it is)

The classifier runs in two stages, in order, with the cheap deterministic stage first.

### Stage A — URL-based classification (the primary path)

The system scans the issue's `html_url`, title, and body for any URL containing a `/driver/` segment. The **full path after `/driver/`** becomes the category.

Examples:

```
.../driver/audio/codec/foo.c#L42       → category = "audio/codec/foo.c"
.../driver/display/dp/link.c?ts=1      → category = "display/dp/link.c"
.../driver/kernel/memory/alloc.c       → category = "kernel/memory/alloc.c"
```

This is **deterministic, zero-cost, and human-auditable.** Every issue that includes a driver-source link gets a reproducible category with no model in the loop. The classifier confidence for this path is `1.0`.

### Stage B — Self-hosted AI fallback

For issues that don't have a `/driver/...` URL (the reporter pasted a stack trace, described the symptom in prose, linked an internal doc, etc.), the system hands the issue text to a **self-hosted GPT-oss-20B** model running on a local OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp). The model returns a short `<area>:<sub-area>` label (e.g. `audio:codec`, `display:hdmi`, `power:thermal`) plus a closed-set priority (`critical`/`high`/`medium`/`low`/`unspecified`).

The structured call uses JSON-mode with Pydantic validation and a parse-retry loop, so hallucinated priorities or runaway labels are caught and re-prompted automatically.

### Why URL first, AI second

- **The URL is ground truth.** If the reporter linked the offending source file, no model can do better than reading the path.
- **The AI is an expensive insurance policy** for the long tail. We pay for it only when the cheap path can't answer.
- **Net result:** for a typical batch where most issues link a driver source file, the AI is invoked on a small minority — fast, cheap, and the deterministic categories dominate the report.

### Why a self-hosted AI, not a hosted one

Your codebase, your bug reports, your customer environments — all confidential. A self-hosted model means nothing leaves your network. There is no per-token bill from a third party. The only foundation model the project uses is the OpenAI open-weight 20B-parameter MoE (`gpt-oss:20b`).

---

## Step 3 — Excel report (the Phase-1)

This is the main showcase artifact. The categorized rows are written to a multi-sheet `.xlsx` workbook.

### Layout

- **One sheet per category.** Each tab carries the five spec columns:
  ```
  Issue ID | Title | Category | Priority | Status
  ```
- **A leading "Summary" sheet** lists every category with its issue count, plus a TOTAL row, so a reviewer can navigate the workbook without scrolling tabs.
- **Sheet names are sanitized** for Excel's constraints: 31-character limit, no `:\/?*[]`, leading/trailing apostrophes stripped, and duplicates after sanitization get a numeric suffix.

### How a reviewer uses it

1. Open the workbook. The Summary tab shows the category breakdown at a glance.
2. Click into a category tab (e.g. `audio_codec_foo.c`) to see the issues in that bucket.
3. Sort or filter by Priority to focus on `critical`/`high` first.
4. Use the Issue ID column to jump back into GitHub.

### How a reviewer obtains it

Three ways:

- **HTTP download.** `GET /issues/report.xlsx` streams the most recently generated workbook.
- **CLI.** `python -m scripts.categorize_issues` runs the full pipeline and writes the workbook to `<excel_output_dir>/<excel_report_filename>`.
- **Daily refresh.** When the FastAPI process is the runtime, an in-process scheduler regenerates the report once per day (UTC hour:minute is configurable). In production we recommend driving the daily refresh from **GitHub Actions** instead — same code, more visible runs, no long-lived process required.

---

## Step 4 — The mode decision (Phase 2)

When a developer picks an issue out of the Excel report and starts fixing, **Phase 2** comes into play. The assistant routes each fix into one of three lanes based on whether it has seen a similar bug before:

- **Lane A — Capture mode (we've never seen this before).** The system simply flags the ticket as **"New issue"** and lets the engineer get on with their work — it does NOT pause to ask them to walk through the fix. While the engineer debugs, the assistant silently observes in the background and records what is going on: the steps taken, the tools invoked, the commands run, the files touched, and the eventual resolution. That captured trace is automatically persisted into the system's memory (structured + semantic stores) so the same problem, when it appears again, can be recognized and routed straight into Lane B or Lane C without anyone having to re-narrate the fix. The whole point of capture mode is that it costs the engineer nothing extra — the act of fixing the bug is itself the act of teaching the system.
- **Lane B — Assist mode (we've seen something similar).** The system finds prior similar fixes in its memory and shows the top candidates ranked by relevance. The engineer picks one or rejects all of them.
- **Lane C — Autonomous mode (we've seen this exact pattern many times).** For mechanical, low-risk fixes (restart a stuck service, rotate a stale credential), the system loads the saved fix plan, asks for one approval, and runs the steps itself.

Phase 2 layers on top of the categorized Excel report — the categories Phase 1 produces become the index the fix-mode assistant routes against. None of Phase 2's runtime (LangGraph, Postgres, Qdrant, MLflow) is required for Phase 1 to work; the lifespan branches on a single `phase` setting.

---

## How the pieces fit together (request flow)

```
                ┌────────────────────────────────────────┐
                │  GitHub Issues (REST + GraphQL APIs)   │
                └──────────────┬─────────────────────────┘
                               │ Fine-Grained PAT
                               │ (Issues:read, Metadata:read)
                               ▼
        ┌──────────────────────────────────────────────┐
        │  GraphQL client — list_open_issues()         │
        │  scope: repo:OWNER/REPO is:issue is:open     │
        └──────────────┬───────────────────────────────┘
                       │ raw issue dicts (REST-shaped)
                       ▼
        ┌──────────────────────────────────────────────┐
        │  issues_parser.parse_issue → IssueRecord     │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │  categorize(issue):                          │
        │    1. URL extractor  → /driver/<path>        │
        │    2. LLM fallback   → <area>:<sub-area>     │
        │    3. Rule augmenter → priority + component  │
        │    4. Else           → "uncategorized"       │
        └──────────────┬───────────────────────────────┘
                       │ CategorizedIssue rows
                       ▼
        ┌──────────────────────────────────────────────┐
        │  excel_exporter.write_workbook               │
        │  → Summary + one sheet per category          │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │  /issues/report.xlsx  (FileResponse)         │
        └──────────────────────────────────────────────┘
```

Three triggers funnel into the same `run_categorization()` orchestrator:

- `POST /issues/categorize/run` — on-demand HTTP call
- `python -m scripts.categorize_issues` — CLI / GitHub Actions
- in-process daily scheduler — laptop / staging demos

---

## Configuration knobs that matter

| Setting | What it does | Default |
|---|---|---|
| `phase` | `"1"` runs the categorization-and-Excel showcase; `"2"` boots the LangGraph fix-mode lanes. | `"1"` |
| `github_pat_fine_grained` | Fine-grained PAT used by the GraphQL client. Falls back to `github_token` if unset. | `""` |
| `github_owner` / `github_repo` | The repository to ingest. | `AMD-SWV-Driver` / `drivers` |
| `categorization_use_graphql` | `True` uses GraphQL; `False` uses the legacy REST client. | `True` |
| `graphql_page_size` | Issues per GraphQL page (max 100). | `100` |
| `graphql_max_pages` | Safety cap on pagination — at default this is up to 10,000 issues per run. | `100` |
| `categorization_scheduled_enabled` | Run the in-process daily refresh? Off in production (use GitHub Actions). | `True` |
| `categorization_scheduled_hour_utc` / `_minute_utc` | When the daily refresh fires, in UTC. | `02:00` |
| `excel_output_dir` / `excel_report_filename` | Where the workbook is written and served from. | `./out/issues_categorized.xlsx` |
| `gpt_oss_base_url` / `gpt_oss_model` | The local OpenAI-compatible endpoint and model name. | `http://localhost:11434/v1` / `gpt-oss:20b` |

---

## What Phase 1 deliberately does NOT do

- **No state mutation on GitHub.** No PATCH, no comments, no label edits. Phase 1 is read-only.
- **No Postgres, no Qdrant, no MLflow.** Those are Phase-2 dependencies. Phase 1 boots clean on a laptop with just Python and a local LLM endpoint.
- **No autonomous shell execution.** That's the Phase-2 sandbox; Phase 1 doesn't need it.
- **No JIRA.** Issues come from GitHub only.
- **No log scraping or redaction.** Issue bodies pass through a redactor only when handed to the LLM (so secrets in pasted stack traces don't reach the model).

---

## Reliability features that ARE in Phase 1

- **Bounded retries on every external call.** The GraphQL client retries on transport errors, 5xx, and GitHub's secondary rate-limit response with exponential backoff.
- **Pagination caps.** A misconfigured query cannot paginate forever — `graphql_max_pages` is the safety net.
- **Per-issue isolation.** A failure on one issue (parse error, LLM timeout) becomes an `uncategorized` row instead of aborting the batch.
- **Pooled HTTP clients.** One async httpx pool per process, closed cleanly on shutdown.
- **Structured logs with correlation IDs.** Every request carries a UUID so a single ticket's path through the system can be reconstructed from logs.
- **Phase-aware health probe.** `/health` only checks the dependencies the active phase actually uses — no false 503s when Postgres isn't running because Phase 1 doesn't need it.
- **Constant-time API key auth.** `X-API-Key` header on every non-health endpoint, compared with `hmac.compare_digest`.

---

## Security boundaries

- **Authentication.** API key on every `/issues/*` endpoint, compared in constant time. `/health`, `/healthz`, `/docs` are open so probes work without keys.
- **Least-privilege GitHub access.** Fine-grained PAT scoped to one repo, two read permissions. No write, no admin.
- **PII / secret redaction** before any text reaches the LLM. The model sees `<EMAIL>`, `<JWT>`, etc. — never the real values.
- **No outbound calls outside GitHub + your local LLM.** No telemetry to third parties. No data leaves the perimeter.

---

## What this delivers, in business terms

- **A single Excel workbook your team can open today** that shows every open issue grouped by the driver area it touches.
- **Reproducibility.** Re-running the pipeline against the same issues produces the same categories — auditors can verify the categorization without rerunning the AI.
- **Operator control.** The category vocabulary is whatever the `/driver/...` paths in your repository naturally produce; you don't have to maintain a hand-curated taxonomy.
- **A foundation for Phase 2.** The same `IssueRecord` and category outputs feed the fix-mode assistant when you turn it on.

---

## A few concrete examples

### Example 1 — A driver bug with a source link
An issue body contains: *"Crash when sample rate switches — see https://github.com/your/repo/blob/main/driver/audio/codec/wm8804.c#L284"*. The URL extractor picks `audio/codec/wm8804.c` as the category. No AI call; the row appears under that tab in the workbook with confidence `1.0`.

### Example 2 — A symptom-only report
An issue body contains: *"My display flickers after waking from suspend on HDMI"* with no `/driver/...` URL. The URL extractor abstains; the GPT-oss-20B model returns `display:hdmi` with priority `medium`. The row appears under the `display_hdmi` tab.

### Example 3 — Manual single-issue categorization
A triage engineer pastes a raw GitHub issue dict into `POST /issues/categorize/manual`. The endpoint runs the same pipeline on that one record and returns the categorized row as JSON — useful for spot-checking or when the engineer wants a category without running the full ingest.

### Example 4 — Daily refresh
The scheduler fires at 02:00 UTC, calls `run_categorization()`, the workbook is rewritten in place. The next morning the team opens `/issues/report.xlsx` and sees the latest grouping.

---

## In summary

Phase 1 is built around three ideas:

1. **Determinism beats cleverness.** The URL extractor handles the issues that have a definitive answer; the AI handles only the long tail.
2. **Excel is the universal language of review.** The deliverable is a workbook your team can open today, not a JSON dump.
3. **Phase 1 must boot on a laptop.** No databases, no orchestration backends — just Python, a local LLM, and a GitHub PAT. Phase 2's heavier stack stays dormant until you flip a single setting.

Phase 2 (the fix-mode lanes — capture, assist, autonomous — with persistent memory, retrieval, and the safe-execution sandbox) layers on top of the same categorized output without rewriting any of Phase 1.
