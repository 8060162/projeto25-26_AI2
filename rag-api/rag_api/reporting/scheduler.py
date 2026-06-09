from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .aggregator import SnapshotAggregator, _window_for
from .lock import RedisSchedulerLock
from .models import Granularity

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SchedulerConfig:
    realtime_minutes:  int
    hourly_minutes:    int
    daily_hour:        int
    daily_minute:      int
    monthly_day:       int
    monthly_hour:      int
    purge_hour:        int
    timezone:          str


def _ttl_cutoffs() -> dict[Granularity, datetime | None]:
    now = datetime.now(timezone.utc)
    return {
        Granularity.HOURLY:  now - timedelta(days=7),
        Granularity.DAILY:   now - timedelta(days=548),
        Granularity.MONTHLY: None,
    }


async def _run_job(
    aggregator: SnapshotAggregator,
    lock:       RedisSchedulerLock,
    granularity: Granularity,
) -> None:
    if not await lock.acquire(f"snapshot_{granularity}"):
        return   # outra instância já está a executar este job

    window = _window_for(granularity)
    try:
        await aggregator.compute_and_store(window)
        logger.info("snapshot.computed", granularity=granularity, period=window.period)
    except Exception:
        logger.warning("snapshot.job_failed", granularity=granularity)


async def _purge_old_snapshots(
    aggregator: SnapshotAggregator,
    lock:       RedisSchedulerLock,
) -> None:
    if not await lock.acquire("snapshot_purge"):
        return

    for granularity, cutoff in _ttl_cutoffs().items():
        if cutoff is None:
            continue
        await aggregator.purge_before(granularity, cutoff)


def create_scheduler(
    aggregator: SnapshotAggregator,
    lock:       RedisSchedulerLock,
    config:     SchedulerConfig,
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=config.timezone)

    scheduler.add_job(
        _run_job, "interval", minutes=config.realtime_minutes,
        args=[aggregator, lock, Granularity.REALTIME],
        id="snapshot_realtime",
    )
    scheduler.add_job(
        _run_job, "interval", minutes=config.hourly_minutes,
        args=[aggregator, lock, Granularity.HOURLY],
        id="snapshot_hourly",
    )
    scheduler.add_job(
        _run_job, "cron",
        hour=config.daily_hour, minute=config.daily_minute,
        args=[aggregator, lock, Granularity.DAILY],
        id="snapshot_daily",
    )
    scheduler.add_job(
        _run_job, "cron",
        day=config.monthly_day, hour=config.monthly_hour,
        args=[aggregator, lock, Granularity.MONTHLY],
        id="snapshot_monthly",
    )
    scheduler.add_job(
        _purge_old_snapshots, "cron",
        hour=config.purge_hour,
        args=[aggregator, lock],
        id="snapshot_purge",
    )

    return scheduler