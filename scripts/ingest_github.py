"""One-shot GitHub Code Scanning ingestion.

Usage:
    python -m scripts.ingest_github             # poll once, ingest all open alerts
    python -m scripts.ingest_github --alert N   # ingest a single alert by number
    python -m scripts.ingest_github --watch     # long-running poll loop
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from bugfix_ai.config.logging_config import configure_logging, get_logger
from bugfix_ai.core.checkpointer import open_checkpointer
from bugfix_ai.core.graph import build_graph
from bugfix_ai.integrations.github.client import get_alert
from bugfix_ai.integrations.github.poller import (
    ingest_one_alert,
    poll_once,
    run_poller,
)
from bugfix_ai.memory.vector_store import ensure_collection

log = get_logger(__name__)


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="GitHub Code Scanning ingestion")
    parser.add_argument(
        "--alert",
        type=int,
        default=None,
        help="Ingest exactly one alert by number, then exit.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Long-running poll loop (interval from settings.github_poll_interval_seconds).",
    )
    args = parser.parse_args(argv)

    configure_logging()
    await ensure_collection()

    async with open_checkpointer() as cp:
        graph = build_graph(cp)

        if args.alert is not None:
            alert = await get_alert(args.alert)
            tid = await ingest_one_alert(graph, alert)
            log.info("script.ingest.done", thread_id=tid, alert=args.alert)
            return 0

        if args.watch:
            log.info("script.poller.watch")
            await run_poller(graph)
            return 0

        # Default: one poll cycle
        new_etag = await poll_once(graph)
        log.info("script.poll.done", new_etag=new_etag)
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
