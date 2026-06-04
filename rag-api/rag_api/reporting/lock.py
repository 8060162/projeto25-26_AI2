import uuid
import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)


class RedisSchedulerLock:
    """
    Lock distribuído via SET NX PX — garante que apenas uma instância
    do rag-api executa cada job de snapshot, independentemente do número
    de réplicas em execução.
    """

    def __init__(self, redis: Redis, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl   = ttl_seconds * 1000   # Redis aceita milissegundos

    async def acquire(self, job_id: str) -> bool:
        """
        Tenta adquirir o lock. Devolve True se adquirido, False se outra
        instância já está a executar o mesmo job.
        """
        key   = f"scheduler:lock:{job_id}"
        token = str(uuid.uuid4())
        acquired = await self._redis.set(key, token, nx=True, px=self._ttl)
        if not acquired:
            logger.debug("scheduler.lock_skip", job_id=job_id)
        return bool(acquired)