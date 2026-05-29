"""Issue categorization package — Phase 1 of the BugFix AI project.

Inputs:  raw GitHub Issue payloads (REST or GraphQL shape — both pass
         through `integrations.github.issues_parser`).
Outputs: `CategorizedIssue` rows containing
         (issue_id, title, category, priority, status, ...).

Pipeline (see `pipeline.py`):
    1. URL extractor — scans the issue body / title / html_url for any
       URL containing `/driver/<path>` and uses the FULL path after
       `/driver/` as the category. Deterministic, zero-cost, and the
       primary classifier per the Phase-1 spec.
    2. LLM fallback — issues with NO /driver URL are sent to
       GPT-oss-20B via `core.llm_client.chat_structured`. The model
       returns a short `<area>:<sub-area>` label plus a closed-Literal
       priority.
    3. Rule augmenter — `rules.classify_issue` runs on every issue to
       supply priority + component when the URL/LLM paths didn't.
    4. `service.run_categorization()` — high-level orchestrator used by
       both the FastAPI on-demand endpoint and the daily scheduler.
       Calls the pipeline, then `excel_exporter.write_workbook()` to
       persist the showcase report (one sheet per category).

The package never mutates GitHub state and never opens shells. It only
reads issues and emits structured rows.
"""
