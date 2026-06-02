from datetime import datetime, timezone
from typing import Protocol

from fastapi import Request

from rag_api.auth.audit import AuditEvent, record
from rag_api.auth.cache import AuthCacheProtocol
from rag_api.auth.keys import hash_key
from rag_api.auth.store import KeyRepositoryProtocol
from rag_api.schemas.identity import APIKeyRecord, ClientIdentity


# ── Interface Strategy — o middleware depende disto, nunca da implementação ───

class AuthStrategyProtocol(Protocol):
    async def authenticate(self, request: Request) -> ClientIdentity: ...


# ── Excepções de autenticação ─────────────────────────────────────────────────

class InvalidKeyError(Exception):
    pass

class InsufficientScopeError(Exception):
    def __init__(self, required: str):
        self.required = required


# ── Implementação: API Key ────────────────────────────────────────────────────

class APIKeyAuthStrategy:
    """
    Autentica via Bearer token no header Authorization.
    Sequência: cache Redis → PostgreSQL → validações de negócio.

    Para adicionar JWT: criar JWTAuthStrategy com a mesma interface.
    O middleware não precisa de mudar.
    """

    def __init__(self, repository: KeyRepositoryProtocol, cache: AuthCacheProtocol):
        self._repo  = repository
        self._cache = cache

    async def authenticate(self, request: Request) -> ClientIdentity:
        raw_key  = self._extract_bearer(request)
        key_hash = hash_key(raw_key)
        trace_id = getattr(request.state, "trace_id", "unknown")

        record_obj = await self._resolve(key_hash)

        if not record_obj:
            record(AuditEvent.AUTH_FAILED, None, trace_id,
                   {"reason": "key_not_found"})
            raise InvalidKeyError("invalid_key")

        if not record_obj.active:
            record(AuditEvent.AUTH_FAILED, record_obj.client_id, trace_id,
                   {"reason": "key_revoked"})
            raise InvalidKeyError("key_revoked")

        if record_obj.expires_at and record_obj.expires_at < datetime.now(timezone.utc):
            record(AuditEvent.AUTH_FAILED, record_obj.client_id, trace_id,
                   {"reason": "key_expired"})
            raise InvalidKeyError("key_expired")

        return ClientIdentity(
            client_id  = record_obj.client_id,
            scopes     = record_obj.scopes,
            rate_limit = record_obj.rate_limit,
        )

    async def _resolve(self, key_hash: str) -> APIKeyRecord | None:
        """Cache-first: Redis → PostgreSQL → preenche cache em caso de miss."""
        cached = await self._cache.get(key_hash)
        if cached:
            return cached

        record_obj = await self._repo.get_by_hash(key_hash)
        if record_obj:
            await self._cache.set(key_hash, record_obj)
        return record_obj

    @staticmethod
    def _extract_bearer(request: Request) -> str:
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise InvalidKeyError("missing_bearer_token")
        token = header.removeprefix("Bearer ").strip()
        if not token:
            raise InvalidKeyError("empty_token")
        return token