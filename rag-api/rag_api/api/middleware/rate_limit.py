import time

from fastapi import Depends, HTTPException, Request

from rag_api.auth.audit import AuditEvent, record
from rag_api.auth.middleware import get_current_client
from rag_api.config.settings import get_settings
from rag_api.schemas.identity import ClientIdentity


class RateLimiter:
    """
    Sliding window rate limiter por client_id.
    Usa Redis sorted sets: chave = rate:{client_id}, score = timestamp.
    """

    _PREFIX = "rate:"

    def __init__(self, redis):
        self._redis    = redis
        self._settings = get_settings()

    async def check(self, client_id: str, limit: int) -> None:
        now    = time.time()
        window = self._settings.rate_limit_window_seconds
        key    = f"{self._PREFIX}{client_id}"

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window)
        results = await pipe.execute()

        count = results[2]
        if count > limit:
            raise RateLimitExceededError(
                client_id=client_id,
                limit=limit,
                window=window,
            )


class RateLimitExceededError(Exception):
    def __init__(self, client_id: str, limit: int, window: int):
        self.client_id   = client_id
        self.limit       = limit
        self.window      = window
        self.retry_after = window


# ── Dependência para injectar o limiter do app.state ─────────────────────────

def get_rate_limiter(request: Request) -> RateLimiter:
    """Lê o rate limiter registado no app.state."""
    return request.app.state.rate_limiter


# ── Dependência FastAPI ───────────────────────────────────────────────────────

def rate_limit():
    """Uso: Depends(rate_limit())"""
    async def _check(
        request:  Request,
        identity: ClientIdentity = Depends(get_current_client),
        limiter:  RateLimiter    = Depends(get_rate_limiter),
    ) -> ClientIdentity:
        trace_id = getattr(request.state, "trace_id", "unknown")
        try:
            await limiter.check(identity.client_id, identity.rate_limit)
        except RateLimitExceededError as exc:
            record(
                AuditEvent.RATE_LIMIT_HIT,
                identity.client_id,
                trace_id,
                {"limit": exc.limit, "window_seconds": exc.window},
            )
            raise HTTPException(
                status_code=429,
                headers={"Retry-After": str(exc.retry_after)},
                detail={
                    "error":    "rate_limit_exceeded",
                    "message":  f"Limit of {exc.limit} requests/min reached.",
                    "trace_id": trace_id,
                },
            )
        return identity

    return _check