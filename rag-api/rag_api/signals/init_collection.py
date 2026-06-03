import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = structlog.get_logger(__name__)


async def init_signals_collection(db: AsyncIOMotorDatabase) -> None:
    existing = await db.list_collection_names()
    if "query_signals" in existing:
        return

    await db.create_collection(
        "query_signals",
        timeseries={
            "timeField": "timestamp",
            "metaField": "metadata",
            "granularity": "seconds",
        },
    )
    logger.info("signals.collection_created")