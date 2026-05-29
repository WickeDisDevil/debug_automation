"""Convert raw GitHub Code Scanning alert payloads into our internal shape.

Why this is its own module:
  The classifier and graph nodes should never see GitHub's REST shape
  directly. If GitHub renames a field tomorrow, we change one file
  here, not every consumer. Same isolation principle as
  `issues_parser.py` does for the Issues endpoint.

What CodeQL alerts look like (relevant subset):
  {
    "number": 1234,                           ← repo-scoped alert id
    "state": "open" | "dismissed" | "fixed",
    "rule": {
      "id": "cpp/uninitialized-local",
      "severity": "warning" | "error" | "note",       ← legacy
      "security_severity_level": "high" | ...,         ← preferred
      "name": "...", "description": "...", "tags": [...]
    },
    "tool": {"name": "CodeQL", "version": "..."},
    "most_recent_instance": {
      "ref": "refs/heads/...",
      "location": {"path": "...", "start_line": N, "end_line": N},
      "message": {"text": "..."},
      "classifications": [...]
    },
    "html_url": "...",
    "created_at": "..."
  }

Two outputs:
  * parse_alert(alert)             — flattened dict suitable for
                                      seeding `BugFixState` (field
                                      names match `core.state.BugFixState`).
  * build_pseudo_log_from_alert()  — synthesizes an "error log" from
                                      the alert text. CodeQL alerts
                                      come from static analysis and
                                      have no runtime log; the rest
                                      of the pipeline expects one, so
                                      we manufacture a coherent stand-in
                                      so node code stays uniform.

Severity normalization:
  GitHub uses two parallel scales (`security_severity_level` for newer
  scanners, legacy `severity` for older). `_SEVERITY_MAP` collapses
  both into our normalized {critical, high, medium, low} so
  downstream code never branches on the raw GitHub strings.

Service heuristic:
  Top-level directory of the file path. Crude but consistent — works
  well for monorepos where folder roots correspond to services. For
  layouts where this doesn't hold, callers can override `service`
  before invoking the graph.
"""

from __future__ import annotations

from typing import Any

from bugfix_ai.config.settings import get_settings


# CodeQL `security_severity_level` is the value GH shows in the UI; rule.severity
# is the legacy compiler-style level. We map both to our normalized values.
_SEVERITY_MAP = {
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "moderate": "medium",
    "low": "low",
    "warning": "medium",
    "error": "high",
    "note": "low",
}


def parse_alert(alert: dict[str, Any]) -> dict[str, Any]:
    """Flatten a CodeQL alert into a dict ready to seed BugFixState.

    The returned dict is the value passed to graph.invoke as initial state.
    Keys map directly onto BugFixState fields (see core/state.py).
    """
    s = get_settings()
    rule = alert.get("rule") or {}
    instance = alert.get("most_recent_instance") or {}
    location = instance.get("location") or {}
    tool = alert.get("tool") or {}

    severity_raw = (
        rule.get("security_severity_level")
        or rule.get("severity")
        or "medium"
    ).lower()
    severity = _SEVERITY_MAP.get(severity_raw, "medium")

    file_path = location.get("path", "")
    start_line = int(location.get("start_line") or 0)
    end_line = int(location.get("end_line") or start_line)

    title = rule.get("name") or rule.get("id") or "Code scanning alert"
    description_parts = [
        rule.get("description") or "",
        instance.get("message", {}).get("text") or "",
    ]
    description = "\n\n".join(p for p in description_parts if p)

    # Service heuristic: top-level directory of the file path.
    service = file_path.split("/", 1)[0] if "/" in file_path else "unknown"

    return {
        "ticket_id": f"gh-{alert['number']}",
        "ticket_title": f"[{rule.get('id', 'codeql')}] {title}",
        "ticket_description": description,
        "service": service,
        "environment": "code",  # this is static analysis, not a runtime env
        "ticket_created_ts": alert.get("created_at", ""),
        "alert_source": {
            "provider": "github_code_scanning",
            "alert_number": int(alert["number"]),
            "rule_id": rule.get("id", ""),
            "rule_severity": severity,
            "repository": f"{s.github_owner}/{s.github_repo}",
            "ref": instance.get("ref", s.github_ref),
            "html_url": alert.get("html_url", ""),
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
        },
        # Pre-classify hint so classify_node can be more confident — these come
        # straight from CodeQL and are more reliable than what we'd infer.
        "severity": severity,
        "error_type": _normalize_rule_id(rule.get("id", "")),
        "stack_pattern": "",
    }


def _normalize_rule_id(rule_id: str) -> str:
    """`cpp/uninitialized-local` → `uninitialized_local` etc."""
    if not rule_id:
        return "unknown"
    last = rule_id.rsplit("/", 1)[-1]
    return last.replace("-", "_").lower()


def build_pseudo_log_from_alert(alert: dict[str, Any]) -> str:
    """CodeQL alerts have no runtime log. We synthesize one from the alert text
    so the rest of the pipeline (which expects 'error_logs_raw') has something
    coherent to work with.
    """
    rule = alert.get("rule") or {}
    instance = alert.get("most_recent_instance") or {}
    location = instance.get("location") or {}
    return (
        f"[CodeQL] rule={rule.get('id','')} severity={rule.get('security_severity_level','')}\n"
        f"file={location.get('path','')} "
        f"line={location.get('start_line','')}-{location.get('end_line','')}\n"
        f"description: {rule.get('description','')}\n"
        f"message: {instance.get('message', {}).get('text','')}\n"
    )
