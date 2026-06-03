from datetime import datetime
from typing import Protocol, runtime_checkable
from motor.motor_asyncio import AsyncIOMotorDatabase


@runtime_checkable
class MetricsReader(Protocol):
    async def global_summary(self, from_dt: datetime, to_dt: datetime) -> dict: ...
    async def client_summary(self, client_id: str, from_dt: datetime, to_dt: datetime) -> dict: ...


class MongoMetricsReader:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["query_signals"]

    async def global_summary(self, from_dt: datetime, to_dt: datetime) -> dict:
        pipeline = [
            {"$match": {"timestamp": {"$gte": from_dt, "$lte": to_dt}}},
            {
                "$group": {
                    "_id": None,
                    "total_queries":    {"$sum": 1},
                    "avg_score":        {"$avg": "$retrieval_score"},
                    "avg_latency_ms":   {"$avg": "$latency_ms"},
                    "avg_docs":         {"$avg": "$docs_retrieved"},
                    "total_tokens_in":  {"$sum": "$tokens_input"},
                    "total_tokens_out": {"$sum": "$tokens_output"},
                }
            },
        ]
        result = await self._col.aggregate(pipeline).to_list(1)
        return result[0] if result else {}

    async def client_summary(self, client_id: str, from_dt: datetime, to_dt: datetime) -> dict:
        pipeline = [
            {
                "$match": {
                    "metadata.client_id": client_id,
                    "timestamp": {"$gte": from_dt, "$lte": to_dt},
                }
            },
            {
                "$group": {
                    "_id": "$metadata.query_category",
                    "total_queries":     {"$sum": 1},
                    "avg_score":         {"$avg": "$retrieval_score"},
                    "avg_latency_ms":    {"$avg": "$latency_ms"},
                    "positive_feedback": {
                        "$sum": {"$cond": [{"$eq": ["$feedback.rating", "positive"]}, 1, 0]}
                    },
                    "negative_feedback": {
                        "$sum": {"$cond": [{"$eq": ["$feedback.rating", "negative"]}, 1, 0]}
                    },
                }
            },
        ]
        return {"categories": await self._col.aggregate(pipeline).to_list(100)}


class StubMetricsReader:
    async def global_summary(self, from_dt: datetime, to_dt: datetime) -> dict:
        return {}

    async def client_summary(self, client_id: str, from_dt: datetime, to_dt: datetime) -> dict:
        return {"categories": []}