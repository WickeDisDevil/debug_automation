"""Seed initial baselines into `error_type_baselines`.

The time-saved metric needs a comparison number. Without seed data, every fix
reports `time_saved_min=0.0` until you've manually completed enough runs of the
same error_type to compute a baseline.

Edit the BASELINES dict for your environment, then:

    python -m scripts.seed_rules_db
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy import text

from bugfix_ai.config.logging_config import configure_logging, get_logger
from bugfix_ai.memory.rules_db import session

log = get_logger(__name__)


# (error_type, service_or_None) -> avg minutes
BASELINES: dict[tuple[str, str | None], float] = {
    ("py/sql-injection", None): 90.0,
    ("py/path-injection", None): 60.0,
    ("py/clear-text-logging-sensitive-data", None): 45.0,
    ("py/code-injection", None): 120.0,
    ("py/unsafe-deserialization", None): 90.0,
    ("py/server-side-request-forgery", None): 75.0,
    ("py/missing-rate-limiting", None): 30.0,
    ("py/weak-cryptographic-algorithm", None): 60.0,
    # Add per-service overrides as you go:
    # ("py/sql-injection", "auth-svc"): 30.0,
}


async def main() -> int:
    configure_logging()
    sql = """
        INSERT INTO error_type_baselines
            (error_type, service, avg_resolution_minutes, sample_count, updated_at)
        VALUES (:et, :svc, :avg, :n, :ts)
        ON CONFLICT (error_type, COALESCE(service, ''))
        DO UPDATE SET avg_resolution_minutes = excluded.avg_resolution_minutes,
                      updated_at = excluded.updated_at
    """
    now = datetime.now(timezone.utc)

    async with session() as s:
        for (et, svc), minutes in BASELINES.items():
            await s.execute(
                text(sql),
                {"et": et, "svc": svc, "avg": minutes, "n": 1, "ts": now},
            )
        await s.commit()
    log.info("seed.complete", count=len(BASELINES))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
