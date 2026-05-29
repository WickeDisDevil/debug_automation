"""Operator CLI: backfill Qdrant points for fixes that didn't get indexed.

When `store_fix` commits a Postgres row but the Qdrant upsert fails (and
exhausts its retry budget), the row is left in a "vector pending" state.
This script walks the `fixes` table, asks Qdrant which IDs already have a
point, and re-embeds + upserts the gaps.

Usage:
    python -m scripts.reconcile_vectors                   # full table
    python -m scripts.reconcile_vectors --batch 500       # bigger batches
    python -m scripts.reconcile_vectors --max-batches 4   # quick spot-check

Exit codes:
    0  success
    1  reconcile completed but at least one fix could not be indexed
    2  unexpected error
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from bugfix_ai.config.logging_config import configure_logging, get_logger
from bugfix_ai.memory.fix_store import reconcile_missing_vectors
from bugfix_ai.memory.vector_store import ensure_collection

log = get_logger(__name__)


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=200, help="Batch size (default: 200)")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after N batches (default: unbounded)",
    )
    args = parser.parse_args(argv)

    configure_logging()
    await ensure_collection()

    try:
        result = await reconcile_missing_vectors(
            batch_size=args.batch,
            max_batches=args.max_batches,
        )
    except Exception as e:  # noqa: BLE001
        log.error("reconcile.unexpected_error", error=str(e))
        return 2

    log.info("reconcile.summary", **result)
    print(
        f"scanned={result['scanned']} missing={result['missing']} "
        f"reindexed={result['reindexed']} failed={result['failed']}"
    )
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
