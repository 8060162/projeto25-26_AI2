from datetime import datetime, timezone, timedelta
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from .models import Granularity, SnapshotWindow, period_key

logger = structlog.get_logger(__name__)

SIGNALS             = "query_signals"
SNAPSHOTS           = "metrics_snapshots"
LOW_SCORE_THRESHOLD = 0.6
LISBON              = ZoneInfo("Europe/Lisbon")


@runtime_checkable
class SnapshotAggregator(Protocol):
    async def compute_and_store(self, window: SnapshotWindow) -> dict: ...
    async def get_snapshot(self, granularity: Granularity, period: str) -> dict | None: ...


def _window_for(granularity: Granularity) -> SnapshotWindow:
    now = datetime.now(timezone.utc)
    match granularity:
        case Granularity.REALTIME:
            return SnapshotWindow(granularity, now - timedelta(hours=1),  now, period_key(granularity))
        case Granularity.HOURLY:
            return SnapshotWindow(granularity, now - timedelta(hours=1),  now, period_key(granularity))
        case Granularity.DAILY:
            return SnapshotWindow(granularity, now - timedelta(days=1),   now, period_key(granularity))
        case Granularity.MONTHLY:
            return SnapshotWindow(granularity, now - timedelta(days=30),  now, period_key(granularity))


class MongoSnapshotAggregator:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._signals   = db[SIGNALS]
        self._snapshots = db[SNAPSHOTS]

    async def compute_and_store(self, window: SnapshotWindow) -> dict:
        snapshot = await self._aggregate(window)
        await self._upsert(snapshot)
        return snapshot

    async def get_snapshot(self, granularity: Granularity, period: str) -> dict | None:
        return await self._snapshots.find_one(
            {"granularity": granularity, "period": period},
            {"_id": 0},
        )

    async def purge_before(self, granularity: Granularity, cutoff: datetime) -> None:
        """Interface limpa para purge — sem acesso a atributos internos do exterior."""
        try:
            result = await self._snapshots.delete_many({
                "granularity":  granularity,
                "generated_at": {"$lt": cutoff},
            })
            if result.deleted_count:
                logger.info(
                    "snapshot.purged",
                    granularity=granularity,
                    deleted=result.deleted_count,
                )
        except Exception:
            logger.warning("snapshot.purge_failed", granularity=granularity)

    async def _aggregate(self, window: SnapshotWindow) -> dict:
        base_match = {"timestamp": {"$gte": window.from_dt, "$lte": window.to_dt}}

        main = await self._signals.aggregate([
            {"$match": base_match},
            {"$group": {
                "_id":                 None,
                "total_queries":       {"$sum": 1},
                "successful":          {"$sum": {"$cond": ["$success", 1, 0]}},
                "avg_latency_ms":      {"$avg": "$latency_ms"},
                "avg_retrieval_score": {"$avg": "$retrieval_score"},
                "avg_question_chars":  {"$avg": "$question_chars"},
                "total_tokens_in":     {"$sum": "$tokens_input"},
                "total_tokens_out":    {"$sum": "$tokens_output"},
                "low_score_count":     {"$sum": {
                    "$cond": [{"$lt": ["$retrieval_score", LOW_SCORE_THRESHOLD]}, 1, 0]
                }},
                "unique_clients":      {"$addToSet": "$metadata.client_id"},
            }},
        ]).to_list(1)

        categories = await self._signals.aggregate([
            {"$match": base_match},
            {"$group": {"_id": "$metadata.query_category", "count": {"$sum": 1}}},
            {"$sort":  {"count": -1}},
            {"$limit": 10},
        ]).to_list(10)

        # peak_hour calculado em Europe/Lisbon — relevante para o IPP
        peak = await self._signals.aggregate([
            {"$match": base_match},
            {"$group": {
                "_id": {
                    "$hour": {
                        "date":     "$timestamp",
                        "timezone": "Europe/Lisbon",
                    }
                },
                "count": {"$sum": 1},
            }},
            {"$sort":  {"count": -1}},
            {"$limit": 1},
        ]).to_list(1)

        daily_dist: dict = {}
        if window.granularity in (Granularity.DAILY, Granularity.MONTHLY):
            rows = await self._signals.aggregate([
                {"$match": base_match},
                {"$group": {
                    "_id": {
                        "$dateToString": {
                            "format":   "%Y-%m-%d",
                            "date":     "$timestamp",
                            "timezone": "Europe/Lisbon",
                        }
                    },
                    "count": {"$sum": 1},
                }},
            ]).to_list(31)
            daily_dist = {r["_id"]: r["count"] for r in rows}

        raw        = main[0] if main else {}
        total      = raw.get("total_queries", 0)
        successful = raw.get("successful", 0)
        low_score  = raw.get("low_score_count", 0)

        return {
            "period":              window.period,
            "granularity":         window.granularity,
            "generated_at":        datetime.now(timezone.utc),
            "from_dt":             window.from_dt,
            "to_dt":               window.to_dt,
            "total_queries":       total,
            "success_rate":        round(successful / total, 4) if total else 0.0,
            "avg_latency_ms":      round(raw.get("avg_latency_ms") or 0.0, 2),
            "avg_retrieval_score": round(raw.get("avg_retrieval_score") or 0.0, 4),
            "avg_question_chars":  round(raw.get("avg_question_chars") or 0.0, 1),
            "total_tokens_input":  raw.get("total_tokens_in", 0),
            "total_tokens_output": raw.get("total_tokens_out", 0),
            "low_score_rate":      round(low_score / total, 4) if total else 0.0,
            "active_clients":      len(raw.get("unique_clients") or []),
            "top_categories":      [
                {"category": c["_id"] or "uncategorized", "count": c["count"]}
                for c in categories
            ],
            "peak_hour_lisbon":    peak[0]["_id"] if peak else None,
            "queries_by_day":      daily_dist,
        }

    async def _upsert(self, snapshot: dict) -> None:
        try:
            await self._snapshots.update_one(
                {"period": snapshot["period"], "granularity": snapshot["granularity"]},
                {"$set": snapshot},
                upsert=True,
            )
        except Exception:
            logger.warning(
                "snapshot.upsert_failed",
                period=snapshot["period"],
                granularity=snapshot["granularity"],
            )


class StubSnapshotAggregator:
    async def compute_and_store(self, window: SnapshotWindow) -> dict:
        return {}

    async def get_snapshot(self, granularity: Granularity, period: str) -> dict | None:
        return None

    async def purge_before(self, granularity: Granularity, cutoff: datetime) -> None:
        pass