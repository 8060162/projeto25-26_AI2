import asyncio
import structlog
from typing import Protocol, runtime_checkable
from motor.motor_asyncio import AsyncIOMotorDatabase

from .models import QuerySignal

logger = structlog.get_logger(__name__)

COLLECTION = "query_signals"


@runtime_checkable
class SignalStore(Protocol):
    async def emit(self, signal: QuerySignal) -> None: ...
    async def attach_feedback(self, trace_id: str, rating: str, reason: str | None) -> None: ...


class MongoSignalStore:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    async def emit(self, signal: QuerySignal) -> None:
        asyncio.create_task(self._write(signal))

    async def attach_feedback(self, trace_id: str, rating: str, reason: str | None) -> None:
        asyncio.create_task(self._write_feedback(trace_id, rating, reason))

    async def _write(self, signal: QuerySignal) -> None:
        try:
            await self._db[COLLECTION].insert_one(signal.to_document())
        except Exception:
            logger.warning("signal.write_failed", trace_id=signal.trace_id)

    async def _write_feedback(self, trace_id: str, rating: str, reason: str | None) -> None:
        try:
            await self._db[COLLECTION].update_one(
                {"trace_id": trace_id},
                {"$set": {"feedback.rating": rating, "feedback.reason": reason}},
            )
        except Exception:
            logger.warning("signal.feedback_failed", trace_id=trace_id)


class StubSignalStore:
    async def emit(self, signal: QuerySignal) -> None:
        logger.debug("signal.stub_emit", trace_id=signal.trace_id)

    async def attach_feedback(self, trace_id: str, rating: str, reason: str | None) -> None:
        logger.debug("signal.stub_feedback", trace_id=trace_id)