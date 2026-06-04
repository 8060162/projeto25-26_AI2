import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from rag_api.reporting.scheduler import _run_job, _purge_old_snapshots, SchedulerConfig, create_scheduler
from rag_api.reporting.models import Granularity
from rag_api.reporting.aggregator import StubSnapshotAggregator
from rag_api.reporting.lock import RedisSchedulerLock


def make_lock(acquired: bool) -> RedisSchedulerLock:
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=acquired or None)
    return RedisSchedulerLock(redis, ttl_seconds=60)


@pytest.mark.asyncio
async def test_run_job_skips_when_lock_not_acquired():
    aggregator = AsyncMock()
    lock       = make_lock(acquired=False)

    await _run_job(aggregator, lock, Granularity.REALTIME)

    aggregator.compute_and_store.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_job_computes_when_lock_acquired():
    aggregator = AsyncMock()
    aggregator.compute_and_store.return_value = {}
    lock = make_lock(acquired=True)

    await _run_job(aggregator, lock, Granularity.REALTIME)

    aggregator.compute_and_store.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_job_swallows_aggregator_error():
    aggregator = AsyncMock()
    aggregator.compute_and_store.side_effect = Exception("db timeout")
    lock = make_lock(acquired=True)

    await _run_job(aggregator, lock, Granularity.REALTIME)


@pytest.mark.asyncio
async def test_purge_skips_when_lock_not_acquired():
    aggregator = AsyncMock()
    lock       = make_lock(acquired=False)

    await _purge_old_snapshots(aggregator, lock)

    aggregator.purge_before.assert_not_awaited()


def test_create_scheduler_registers_five_jobs():
    aggregator = StubSnapshotAggregator()
    lock       = make_lock(acquired=True)
    config     = SchedulerConfig(
        realtime_minutes=15, hourly_minutes=60,
        daily_hour=0, daily_minute=0,
        monthly_day=1, monthly_hour=0,
        purge_hour=2, timezone="Europe/Lisbon",
    )
    scheduler = create_scheduler(aggregator, lock, config)
    assert len(scheduler.get_jobs()) == 5