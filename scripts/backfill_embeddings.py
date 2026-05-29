"""Recompute and upsert embeddings for every fix in Postgres.

Use this when you change the embedding model (e.g. bge-small → bge-large) and
need to re-index the vector store. It walks the `fixes` table in batches and
calls `embed_composite` + `upsert_fix` for each row.

    python -m scripts.backfill_embeddings [--limit N] [--batch 50]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import text

from bugfix_ai.config.logging_config import configure_logging, get_logger
from bugfix_ai.memory.embed import embed_composite
from bugfix_ai.memory.rules_db import session
from bugfix_ai.memory.vector_store import ensure_collection, upsert_fix

log = get_logger(__name__)


async def _fetch_batch(offset: int, batch_size: int) -> list[dict]:
    sql = """
        SELECT id::text AS fix_id,
               ticket_title, ticket_description,
               error_logs_redacted, error_type,
               service, environment, severity,
               root_cause, fix_summary,
               EXTRACT(EPOCH FROM resolved_at) AS resolved_at_ts
          FROM fixes
         ORDER BY resolved_at DESC
         LIMIT :n OFFSET :o
    """
    async with session() as s:
        rows = (await s.execute(text(sql), {"n": batch_size, "o": offset})).mappings().all()
    return [dict(r) for r in rows]


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Total rows to process")
    parser.add_argument("--batch", type=int, default=50, help="Batch size")
    args = parser.parse_args(argv)

    configure_logging()
    await ensure_collection()

    processed = 0
    offset = 0
    while True:
        batch = await _fetch_batch(offset, args.batch)
        if not batch:
            break
        for row in batch:
            embedding = await embed_composite(
                title=row.get("ticket_title") or "",
                description=row.get("ticket_description") or "",
                error_log=row.get("error_logs_redacted") or "",
                error_type=row.get("error_type") or "",
            )
            await upsert_fix(
                fix_id=row["fix_id"],
                embedding=embedding,
                payload={
                    "service": row.get("service"),
                    "environment": row.get("environment"),
                    "error_type": row.get("error_type"),
                    "severity": row.get("severity"),
                    "resolved_at_ts": row.get("resolved_at_ts"),
                    "root_cause": row.get("root_cause"),
                    "fix_summary": row.get("fix_summary"),
                },
            )
            processed += 1
            if args.limit is not None and processed >= args.limit:
                log.info("backfill.done", processed=processed)
                return 0
        offset += len(batch)
        log.info("backfill.batch", processed=processed)

    log.info("backfill.done", processed=processed)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
