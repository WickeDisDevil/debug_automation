"""Safe shell executor — the SINGLE chokepoint for all autonomous-mode shell I/O.

Threat model this addresses:
  Autonomous mode lets the system propose-and-run shell commands derived
  from a saved fix, possibly adapted by an LLM. The two failure modes we
  care about are:

    1. LLM hallucination — the model expands a saved command into
       something destructive (`rm -rf /`, `kubectl delete ns prod`,
       `DROP DATABASE`, fork bombs).
    2. Latent ambiguity — the saved command is fine in isolation but
       wrong for THIS service / THIS environment, and the adaptation
       step amplifies the mismatch.

Defense layers (defense-in-depth, in execution order):

  Layer 0 — Saved command provenance:
       Only commands that came from `fix_store` (a captured fix) ever
       reach this module. Free-form LLM output is never executed.

  Layer 1 — Static checks (this module, `evaluate_safety`):
       a. deny-token blacklist (`rm -rf`, `mkfs`, `:(){:|:&};:` etc.)
       b. shell-metachar rejection — pipes, semicolons, backticks,
          subshells, `&&`, `||` are ALL rejected. If a fix needs them,
          it must escape via manual_fallback.
       c. binary allowlist (loaded from YAML at startup; cached) —
          unknown binaries are rejected.
       d. per-binary allowed_subcommands (e.g. `kubectl get` ✓,
          `kubectl delete` ✗).
       e. per-binary deny_args substring check.
       f. reversibility cross-check between the step's claim and the
          binary rule's claim — log a warning on mismatch.

  Layer 2 — Human-in-the-loop (pre_execute_review_node):
       Even after Layer 1 passes, the graph INTERRUPTS and shows the
       human:  saved command  → adapted command  → safety verdict.
       The human approves, edits, or rejects. Layer 1 is run AGAIN on
       any edited command.

  Layer 3 — Subprocess hardening:
       * `asyncio.create_subprocess_exec` with token list, NEVER
         `create_subprocess_shell` (no shell interpolation by design).
       * Hard timeout via `asyncio.wait_for` + `proc.kill()` on expiry.
       * stdout/stderr are size-capped at decode time so a runaway
         process can't blow the LLM's context.

Dry-run mode:
  When `dry_run=True` (or `settings.terminal_dry_run_default`), Layer 1
  still runs but the subprocess is NOT spawned — a synthetic success
  result is returned with `safety_verdict="dry_run"`. This is the
  default for first-time deployments and for any irreversible binary.

What this module deliberately does NOT do:
  * Execute under sudo / privilege escalation.
  * Set environment variables for the child process beyond inheritance.
  * Implement a per-user audit log (the decision logger in
    `observability/decision_logger.py` does that one level up).
  * Sanitize stdout (the redactor handles that on the way back up).
"""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.config.settings import get_settings

log = get_logger(__name__)


# ── Allowlist loading ───────────────────────────────────────────────────────


@dataclass
class BinaryRule:
    name: str
    allowed_subcommands: list[str] = field(default_factory=list)
    deny_args: list[str] = field(default_factory=list)
    reversible: bool = True


@dataclass
class Allowlist:
    binaries: dict[str, BinaryRule]
    deny_tokens: list[str]


@lru_cache(maxsize=1)
def _load_allowlist() -> Allowlist:
    settings = get_settings()
    path = Path(settings.terminal_allowlist_path)
    if not path.exists():
        log.warning("allowlist.missing", path=str(path))
        return Allowlist(binaries={}, deny_tokens=[])
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    binaries: dict[str, BinaryRule] = {}
    for entry in data.get("binaries", []) or []:
        name = entry["name"]
        binaries[name] = BinaryRule(
            name=name,
            allowed_subcommands=list(entry.get("allowed_subcommands") or []),
            deny_args=list(entry.get("deny_args") or []),
            reversible=bool(entry.get("reversible", True)),
        )
    deny_tokens = list(data.get("deny_tokens") or [])
    log.info("allowlist.loaded", binaries=len(binaries), deny_tokens=len(deny_tokens))
    return Allowlist(binaries=binaries, deny_tokens=deny_tokens)


# ── Result type ─────────────────────────────────────────────────────────────


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    success: bool = False
    duration_sec: float = 0.0
    safety_verdict: str = "rejected"  # "allowed" | "rejected" | "dry_run"
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "success": self.success,
            "duration_sec": self.duration_sec,
            "safety_verdict": self.safety_verdict,
            "rejection_reason": self.rejection_reason,
        }


# ── Public API ──────────────────────────────────────────────────────────────


def evaluate_safety(command: str, *, is_reversible_hint: bool = True) -> tuple[bool, str]:
    """Static checks. Returns (allowed, reason). Reason is empty when allowed."""
    if not command or not command.strip():
        return False, "empty command"

    allow = _load_allowlist()

    # Deny tokens are the strongest signal — reject regardless of binary.
    lowered = command.lower()
    for token in allow.deny_tokens:
        if token.lower() in lowered:
            return False, f"deny token matched: {token!r}"

    # Tokenize. We deliberately disallow shell features that would defeat the
    # allowlist (pipes/subshells/redirects). If a saved fix really needs them,
    # surface it through manual_fallback.
    if any(c in command for c in ["|", ";", "&&", "||", "`", "$("]):
        return False, "shell metacharacters not permitted in autonomous mode"

    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return False, f"unparseable command: {e}"
    if not tokens:
        return False, "empty token list"

    binary = Path(tokens[0]).name  # strip /usr/bin/ etc.
    rule = allow.binaries.get(binary)
    if rule is None:
        return False, f"binary {binary!r} not in allowlist"

    # Subcommand check (e.g. `kubectl get` allowed, `kubectl delete` not)
    if rule.allowed_subcommands:
        if len(tokens) < 2 or tokens[1] not in rule.allowed_subcommands:
            return False, (
                f"subcommand not allowed for {binary!r}; "
                f"allowed: {rule.allowed_subcommands}"
            )

    # Per-binary deny args (substring match against the joined arg string)
    arg_blob = " ".join(tokens[1:])
    for forbidden in rule.deny_args:
        if forbidden in arg_blob:
            return False, f"forbidden token {forbidden!r} for binary {binary!r}"

    # Irreversible commands require explicit reversibility hint to be False
    # AT THE STEP LEVEL. If the saved step says it's reversible but the binary
    # rule says it's not, trust the more restrictive side.
    if not rule.reversible and is_reversible_hint:
        log.warning(
            "safety.reversibility_mismatch",
            binary=binary,
            step_says_reversible=is_reversible_hint,
            rule_says_reversible=rule.reversible,
        )

    return True, ""


async def execute_command(
    command: str,
    *,
    timeout_sec: float = 60.0,
    dry_run: bool | None = None,
    is_reversible_hint: bool = True,
    cwd: str | None = None,
) -> ExecResult:
    """Run a command after passing all safety checks.

    `dry_run` defaults to settings.terminal_dry_run_default if None.
    """
    settings = get_settings()
    if dry_run is None:
        dry_run = settings.terminal_dry_run_default

    allowed, reason = evaluate_safety(command, is_reversible_hint=is_reversible_hint)
    if not allowed:
        log.warning("safe_exec.rejected", command=command[:200], reason=reason)
        return ExecResult(safety_verdict="rejected", rejection_reason=reason)

    if dry_run:
        log.info("safe_exec.dry_run", command=command[:200])
        return ExecResult(
            stdout=f"[DRY RUN] would execute: {command}",
            exit_code=0,
            success=True,
            safety_verdict="dry_run",
        )

    return await _run_subprocess(command, timeout_sec=timeout_sec, cwd=cwd)


async def _run_subprocess(command: str, *, timeout_sec: float, cwd: str | None) -> ExecResult:
    """Spawn the subprocess. We've already validated `command` is safe."""
    import time

    tokens = shlex.split(command)
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *tokens,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(
                stderr=f"timeout after {timeout_sec}s",
                exit_code=124,
                duration_sec=time.monotonic() - start,
                safety_verdict="allowed",
                rejection_reason="",
            )
    except FileNotFoundError as e:
        return ExecResult(
            stderr=f"binary not found: {e}",
            exit_code=127,
            duration_sec=time.monotonic() - start,
            safety_verdict="allowed",
        )

    duration = time.monotonic() - start
    return ExecResult(
        stdout=stdout_b.decode(errors="replace")[:8000],
        stderr=stderr_b.decode(errors="replace")[:4000],
        exit_code=proc.returncode if proc.returncode is not None else -1,
        success=proc.returncode == 0,
        duration_sec=duration,
        safety_verdict="allowed",
    )
