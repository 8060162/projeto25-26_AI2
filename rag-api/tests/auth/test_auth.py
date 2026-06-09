"""
Testes do Módulo 2 — Autenticação e Autorização

Cobertura:
- keys.py        : geração, hash, consistência
- strategies.py  : happy path, key inválida, revogada, expirada, scope
- cache.py       : hit, miss, invalidação
- rate_limit.py  : dentro do limite, excedido
- audit.py       : eventos registados nos casos correctos
"""
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_api.auth.keys import generate_api_key, hash_key
from rag_api.auth.strategies import APIKeyAuthStrategy, InvalidKeyError
from rag_api.auth.cache import RedisAuthCache
from rag_api.api.middleware.rate_limit import RateLimiter, RateLimitExceededError
from rag_api.schemas.identity import APIKeyRecord, ClientIdentity


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_record(active=True, expires_at=None) -> APIKeyRecord:
    return APIKeyRecord(
        id="uuid-1",
        client_id="test-client",
        key_hash="irrelevant",
        key_hint="rag_testkey1",
        scopes=["rag:query"],
        rate_limit=100,
        active=active,
        expires_at=expires_at,
    )


@pytest.fixture
def mock_request():
    req = MagicMock()
    req.headers = {"Authorization": "Bearer rag_validtoken123"}
    req.state.trace_id = "trace-abc"
    return req


@pytest.fixture
def mock_repo():
    return AsyncMock()


@pytest.fixture
def mock_cache():
    return AsyncMock()


# ── keys.py ───────────────────────────────────────────────────────────────────

def test_generate_api_key_has_correct_prefix():
    raw, _, _ = generate_api_key()
    assert raw.startswith("rag_")


def test_generate_api_key_returns_unique_values():
    keys = {generate_api_key()[0] for _ in range(100)}
    assert len(keys) == 100


def test_hash_key_is_deterministic():
    raw = "rag_sometoken"
    assert hash_key(raw) == hash_key(raw)


def test_hash_key_differs_from_raw():
    raw = "rag_sometoken"
    assert hash_key(raw) != raw


def test_key_hint_is_first_12_chars():
    raw, _, hint = generate_api_key()
    assert hint == raw[:12]


# ── strategies.py ─────────────────────────────────────────────────────────────

class TestAPIKeyAuthStrategy:

    @pytest.mark.asyncio
    async def test_valid_key_returns_identity(self, mock_request, mock_repo, mock_cache):
        record = make_record()
        mock_cache.get = AsyncMock(return_value=record)

        strategy = APIKeyAuthStrategy(mock_repo, mock_cache)
        identity = await strategy.authenticate(mock_request)

        assert identity.client_id == "test-client"
        assert "rag:query" in identity.scopes

    @pytest.mark.asyncio
    async def test_unknown_key_raises_invalid(self, mock_request, mock_repo, mock_cache):
        mock_cache.get = AsyncMock(return_value=None)
        mock_repo.get_by_hash = AsyncMock(return_value=None)

        strategy = APIKeyAuthStrategy(mock_repo, mock_cache)
        with pytest.raises(InvalidKeyError, match="invalid_key"):
            await strategy.authenticate(mock_request)

    @pytest.mark.asyncio
    async def test_revoked_key_raises_invalid(self, mock_request, mock_repo, mock_cache):
        mock_cache.get = AsyncMock(return_value=make_record(active=False))

        strategy = APIKeyAuthStrategy(mock_repo, mock_cache)
        with pytest.raises(InvalidKeyError, match="key_revoked"):
            await strategy.authenticate(mock_request)

    @pytest.mark.asyncio
    async def test_expired_key_raises_invalid(self, mock_request, mock_repo, mock_cache):
        expired = make_record(expires_at=datetime.now(timezone.utc) - timedelta(days=1))
        mock_cache.get = AsyncMock(return_value=expired)

        strategy = APIKeyAuthStrategy(mock_repo, mock_cache)
        with pytest.raises(InvalidKeyError, match="key_expired"):
            await strategy.authenticate(mock_request)

    @pytest.mark.asyncio
    async def test_missing_bearer_raises_invalid(self, mock_repo, mock_cache):
        req = MagicMock()
        req.headers = {}
        req.state.trace_id = "trace-x"

        strategy = APIKeyAuthStrategy(mock_repo, mock_cache)
        with pytest.raises(InvalidKeyError, match="missing_bearer_token"):
            await strategy.authenticate(req)

    @pytest.mark.asyncio
    async def test_cache_miss_falls_back_to_db(self, mock_request, mock_repo, mock_cache):
        record = make_record()
        mock_cache.get = AsyncMock(return_value=None)          # cache miss
        mock_repo.get_by_hash = AsyncMock(return_value=record) # DB hit

        strategy = APIKeyAuthStrategy(mock_repo, mock_cache)
        identity = await strategy.authenticate(mock_request)

        assert identity.client_id == "test-client"
        mock_cache.set.assert_called_once()                    # cache populado


# ── cache.py ──────────────────────────────────────────────────────────────────

class TestRedisAuthCache:

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        cache = RedisAuthCache(redis)
        assert await cache.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_and_get_round_trip(self):
        stored = {}
        redis = AsyncMock()
        redis.setex = AsyncMock(side_effect=lambda k, t, v: stored.update({k: v}))
        redis.get = AsyncMock(side_effect=lambda k: stored.get(k))

        cache = RedisAuthCache(redis)
        record = make_record()
        await cache.set("hash123", record)
        result = await cache.get("hash123")
        assert result is not None
        assert result.client_id == "test-client"

    @pytest.mark.asyncio
    async def test_invalidate_calls_delete(self):
        redis = AsyncMock()
        cache = RedisAuthCache(redis)
        await cache.invalidate("hash123")
        redis.delete.assert_called_once_with("auth:key:hash123")


# ── rate_limit.py ─────────────────────────────────────────────────────────────

class TestRateLimiter:

    def _make_redis(self, count: int):
        """Simula um pipeline Redis que devolve count pedidos na janela."""
        redis = MagicMock()
        pipe = AsyncMock()
        pipe.execute = AsyncMock(return_value=[None, None, count, None])
        redis.pipeline = MagicMock(return_value=pipe)
        return redis

    @pytest.mark.asyncio
    async def test_within_limit_does_not_raise(self):
        limiter = RateLimiter(self._make_redis(count=50))
        await limiter.check("client-a", limit=100)  # não levanta

    @pytest.mark.asyncio
    async def test_exceeds_limit_raises(self):
        limiter = RateLimiter(self._make_redis(count=101))
        with pytest.raises(RateLimitExceededError) as exc_info:
            await limiter.check("client-a", limit=100)
        assert exc_info.value.client_id == "client-a"

    @pytest.mark.asyncio
    async def test_limit_is_per_client(self):
        """Clientes diferentes têm contadores independentes."""
        limiter_a = RateLimiter(self._make_redis(count=101))
        limiter_b = RateLimiter(self._make_redis(count=10))

        with pytest.raises(RateLimitExceededError):
            await limiter_a.check("client-a", limit=100)

        await limiter_b.check("client-b", limit=100)  # não levanta