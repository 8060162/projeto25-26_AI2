"""
Scheduler — Memory Reporting.

Um único job diário — sem realtime nem hourly (não fazem sentido
para métricas de comportamento de utilizadores).

Reutiliza RedisSchedulerLock do reporting existente — sem duplicação.
"""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from rag_api.reporting import RedisSchedulerLock

from .aggregator import MemorySnapshotAggregator, _daily_window

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class MemorySchedulerConfig:
    daily_hour:   int
    daily_minute: int
    purge_hour:   int
    purge_days:   int   # snapshots mais antigos que N dias são eliminados
    timezone:     str


async def _run_daily_snapshot(
    aggregator: MemorySnapshotAggregator,
    lock:       RedisSchedulerLock,
) -> None:
    if not await lock.acquire("memory_snapshot_daily"):
        return

    window = _daily_window()
    try:
        await aggregator.compute_and_store(window)
        logger.info("memory_snapshot.computed", period=window.period)
    except Exception:
        logger.warning("memory_snapshot.job_failed", period=window.period)


async def _purge_old_snapshots(
    aggregator: MemorySnapshotAggregator,
    lock:       RedisSchedulerLock,
    purge_days: int,
) -> None:
    if not await lock.acquire("memory_snapshot_purge"):
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=purge_days)
    await aggregator.purge_before(cutoff)


def create_memory_scheduler(
    aggregator: MemorySnapshotAggregator,
    lock:       RedisSchedulerLock,
    config:     MemorySchedulerConfig,
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=config.timezone)

    scheduler.add_job(
        _run_daily_snapshot,
        "cron",
        hour   = config.daily_hour,
        minute = config.daily_minute,
        args   = [aggregator, lock],
        id     = "memory_snapshot_daily",
    )
    scheduler.add_job(
        _purge_old_snapshots,
        "cron",
        hour = config.purge_hour,
        args = [aggregator, lock, config.purge_days],
        id   = "memory_snapshot_purge",
    )

    return scheduler