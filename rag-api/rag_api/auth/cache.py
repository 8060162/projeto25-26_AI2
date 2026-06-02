import json
from typing import Protocol

from rag_api.schemas.identity import APIKeyRecord
from rag_api.config.settings import get_settings


# ── Interface — permite mock nos testes ──────────────────────────────────────

class AuthCacheProtocol(Protocol):
    async def get(self, key_hash: str) -> APIKeyRecord | None: ...
    async def set(self, key_hash: str, record: APIKeyRecord) -> None: ...
    async def invalidate(self, key_hash: str) -> None: ...


# ── Implementação Redis ───────────────────────────────────────────────────────

class RedisAuthCache:
    """
    Cache de validação de API keys.
    Redis é cache — nunca estado primário.
    A fonte de verdade é sempre o MongoDB (colecção api_keys).
    """

    _PREFIX = "auth:key:"

    def __init__(self, redis):
        # redis é um cliente redis.asyncio injectado — nunca criado aqui
        self._redis = redis
        self._ttl = get_settings().auth_cache_ttl_seconds

    async def get(self, key_hash: str) -> APIKeyRecord | None:
        raw = await self._redis.get(self._cache_key(key_hash))
        if not raw:
            return None
        return APIKeyRecord(**json.loads(raw))

    async def set(self, key_hash: str, record: APIKeyRecord) -> None:
        await self._redis.setex(
            self._cache_key(key_hash),
            self._ttl,
            record.model_dump_json(),
        )

    async def invalidate(self, key_hash: str) -> None:
        """
        Chamado imediatamente após revogação de uma key.
        Garante que a revogação é efectiva antes do TTL expirar.
        """
        await self._redis.delete(self._cache_key(key_hash))

    def _cache_key(self, key_hash: str) -> str:
        return f"{self._PREFIX}{key_hash}"