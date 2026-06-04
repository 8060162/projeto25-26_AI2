import pytest
from unittest.mock import AsyncMock, MagicMock
from rag_api.reporting.lock import RedisSchedulerLock


@pytest.mark.asyncio
async def test_acquire_returns_true_when_lock_free():
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    lock = RedisSchedulerLock(redis, ttl_seconds=60)

    assert await lock.acquire("snapshot_realtime") is True
    redis.set.assert_awaited_once()
    _, kwargs = redis.set.call_args
    assert kwargs["nx"] is True
    assert kwargs["px"] == 60_000


@pytest.mark.asyncio
async def test_acquire_returns_false_when_lock_taken():
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=None)
    lock = RedisSchedulerLock(redis, ttl_seconds=60)

    assert await lock.acquire("snapshot_realtime") is False


@pytest.mark.asyncio
async def test_different_jobs_use_different_keys():
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    lock = RedisSchedulerLock(redis, ttl_seconds=60)

    await lock.acquire("snapshot_realtime")
    await lock.acquire("snapshot_hourly")

    keys = [call.args[0] for call in redis.set.call_args_list]
    assert keys[0] != keys[1]
    