"""Export trajectories and preference pairs as HuggingFace-friendly datasets.

Two outputs:

  1. SFT (supervised fine-tuning) dataset — JSONL of (system, user, assistant)
     turns, derived from successful trajectories. This is the recommended
     warm-start for any RL phase.

  2. DPO/GRPO preference dataset — JSONL of (prompt, chosen, rejected) tuples,
     derived from `preference_store.PreferencePair` rows.

Both formats are compatible with `datasets.load_dataset("json", ...)` and
TRL's `SFTTrainer` / `DPOTrainer`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bugfix_ai.config.logging_config import get_logger
from bugfix_ai.rl.preference_store import list_pairs

log = get_logger(__name__)


# ── SFT export ──────────────────────────────────────────────────────────────


_SFT_SYSTEM = (
    "You are an expert SRE assistant. Given an error log and ticket context, "
    "produce a structured fix consisting of a root cause, a one-paragraph "
    "summary, and an ordered list of executable steps."
)


@dataclass
class SFTExample:
    system: str
    user: str
    assistant: str

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "messages": [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": self.user},
                    {"role": "assistant", "content": self.assistant},
                ]
            },
            ensure_ascii=False,
        )


def _build_user(traj: dict) -> str:
    return (
        f"Ticket: {traj.get('ticket_title', '')}\n"
        f"Error type: {traj.get('error_type', 'unknown')}\n"
        f"Severity: {traj.get('severity', 'unknown')}\n\n"
        f"Description:\n{traj.get('ticket_description', '')}\n\n"
        f"Redacted error log:\n{traj.get('error_log_redacted', '')[:4000]}"
    )


def _build_assistant(traj: dict) -> str:
    payload = {
        "root_cause": traj.get("root_cause", ""),
        "fix_summary": traj.get("fix_summary", ""),
        "steps": traj.get("fix_steps") or [],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _quality_filter(traj: dict) -> bool:
    """Keep only trajectories that look like teachable signal."""
    if not traj.get("fix_steps"):
        return False
    if not traj.get("root_cause") or not traj.get("fix_summary"):
        return False
    if traj.get("redo_count", 0) > 2:
        # Lots of dev rejections → probably bad reasoning, don't teach this.
        return False
    return True


def export_sft(
    *,
    trajectories_jsonl: str,
    output_path: str,
    only_successful: bool = True,
) -> int:
    """Read trajectories.jsonl, emit an SFT JSONL. Returns example count."""
    src = Path(trajectories_jsonl)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        log.warning("sft.export.no_source", path=str(src))
        return 0

    written = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                log.warning("sft.export.bad_line")
                continue

            if only_successful and not traj.get("autonomous_success", False):
                # Capture-mode runs don't have "autonomous_success" → keep them
                # if they have steps. Autonomous runs must succeed.
                if traj.get("mode") == "autonomous":
                    continue

            if not _quality_filter(traj):
                continue

            example = SFTExample(
                system=_SFT_SYSTEM,
                user=_build_user(traj),
                assistant=_build_assistant(traj),
            )
            fout.write(example.to_jsonl() + "\n")
            written += 1

    log.info("sft.export.done", count=written, output=str(dst))
    return written


# ── DPO/GRPO export ─────────────────────────────────────────────────────────


async def export_dpo(
    *,
    output_path: str,
    source: str | None = None,
    error_type: str | None = None,
    limit: int = 5000,
) -> int:
    """Export preference pairs as TRL-compatible (prompt, chosen, rejected) JSONL.

    Each row:
      {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    pairs = await list_pairs(source=source, error_type=error_type, limit=limit)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with dst.open("w", encoding="utf-8") as fout:
        for p in pairs:
            row = {
                # TRL DPO prompt is conceptually "the input the policy saw".
                # Without per-pair input persistence we use the error_type as
                # a coarse anchor; for higher signal, capture the input on the
                # PreferencePair row in a future migration.
                "prompt": _dpo_prompt(p.get("error_type")),
                "chosen": p["chosen_payload"],
                "rejected": p["rejected_payload"],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    log.info("dpo.export.done", count=written, output=str(dst))
    return written


def _dpo_prompt(error_type: str | None) -> str:
    et = error_type or "unknown"
    return (
        f"You are diagnosing a bug of type `{et}`. "
        f"Produce the best fix (root cause + ordered steps) you can."
    )


# ── Combined entry-point ────────────────────────────────────────────────────


async def export_all(
    *,
    trajectories_jsonl: str,
    sft_output: str,
    dpo_output: str,
) -> dict[str, int]:
    sft_count = export_sft(trajectories_jsonl=trajectories_jsonl, output_path=sft_output)
    dpo_count = await export_dpo(output_path=dpo_output)
    return {"sft_examples": sft_count, "dpo_pairs": dpo_count}


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    """Tiny helper used by training scripts."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
