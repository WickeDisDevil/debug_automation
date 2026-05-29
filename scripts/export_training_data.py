"""Export SFT and DPO datasets from collected trajectories + preference pairs.

    python -m scripts.export_training_data
"""

from __future__ import annotations

import asyncio

from bugfix_ai.config.logging_config import configure_logging, get_logger
from bugfix_ai.config.settings import get_settings
from bugfix_ai.rl.sft_exporter import export_all

log = get_logger(__name__)


async def main() -> int:
    configure_logging()
    settings = get_settings()
    counts = await export_all(
        trajectories_jsonl=settings.trajectory_jsonl_path,
        sft_output=settings.sft_dataset_path,
        dpo_output=settings.dpo_dataset_path,
    )
    log.info("export.done", **counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
