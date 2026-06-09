"""
Reader — Memory Reporting.

Agrega snapshots diários num intervalo arbitrário pedido pela API.
Três campos novos face à versão anterior:
  clarification_rate por tópico — ponderada pelo volume
  topic_timeline — semanas agregadas dos snapshots diários
  uncategorized_count + uncategorized_rate
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from motor.motor_asyncio import AsyncIOMotorDatabase

SNAPSHOT_COLLECTION = "memory_snapshots"


@runtime_checkable
class MemoryReportReader(Protocol):
    async def get_daily(self, period: str) -> dict | None: ...
    async def get_aggregated(self, from_dt: datetime, to_dt: datetime) -> dict: ...


class MongoMemoryReportReader:

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._snapshots = db[SNAPSHOT_COLLECTION]

    async def get_daily(self, period: str) -> dict | None:
        return await self._snapshots.find_one({"period": period}, {"_id": 0})

    async def get_aggregated(self, from_dt: datetime, to_dt: datetime) -> dict:
        # Filtrar por period (string YYYY-MM-DD) — simples e determinístico
        # Evita problemas de comparação de datetime com timezone no MongoDB
        from_period = from_dt.strftime("%Y-%m-%d")
        to_period   = to_dt.strftime("%Y-%m-%d")
        snapshots = await self._snapshots.find(
            {"period": {"$gte": from_period, "$lte": to_period}},
            {"_id": 0},
        ).sort("period", 1).to_list(length=None)

        if not snapshots:
            return _empty_aggregated(from_dt, to_dt)
        return _merge_snapshots(snapshots, from_dt, to_dt)


def _empty_aggregated(from_dt: datetime, to_dt: datetime) -> dict:
    return {
        "from_dt": from_dt, "to_dt": to_dt, "days_covered": 0,
        "total_interactions": 0, "authenticated_count": 0, "anonymous_count": 0,
        "unique_users": 0, "unique_clients": 0,
        "avg_importance": 0.0, "low_importance_rate": 0.0, "no_sources_rate": 0.0,
        "uncategorized_count": 0, "uncategorized_rate": 0.0,
        "top_topics": [], "weak_topics": [], "top_sources": [],
        "topic_timeline": [], "daily_breakdown": [],
    }


def _merge_snapshots(snapshots: list[dict], from_dt: datetime, to_dt: datetime) -> dict:

    total               = sum(s.get("total_interactions", 0)  for s in snapshots)
    authenticated_count = sum(s.get("authenticated_count", 0) for s in snapshots)
    anonymous_count     = sum(s.get("anonymous_count", 0)     for s in snapshots)
    uncategorized_count = sum(s.get("uncategorized_count", 0) for s in snapshots)
    unique_users        = max((s.get("unique_users", 0)   for s in snapshots), default=0)
    unique_clients      = max((s.get("unique_clients", 0) for s in snapshots), default=0)

    # Média ponderada de importance
    weighted_importance = sum(
        s.get("avg_importance", 0.0) * s.get("total_interactions", 0)
        for s in snapshots
    )
    avg_importance = round(weighted_importance / total, 2) if total else 0.0

    low_importance_count = sum(
        round(s.get("low_importance_rate", 0.0) * s.get("total_interactions", 0))
        for s in snapshots
    )
    no_sources_count = sum(
        round(s.get("no_sources_rate", 0.0) * s.get("total_interactions", 0))
        for s in snapshots
    )

    # ── Top tópicos com clarification_rate ponderada ──────────────────────────
    topic_map: dict[str, dict] = {}
    for s in snapshots:
        for t in s.get("top_topics", []):
            key = t["topic"]
            if key not in topic_map:
                topic_map[key] = {
                    "topic":             key,
                    "count":             0,
                    "importance_sum":    0.0,
                    "clarified_sum":     0.0,
                }
            topic_map[key]["count"]          += t["count"]
            topic_map[key]["importance_sum"] += t["avg_importance"] * t["count"]
            topic_map[key]["clarified_sum"]  += t["clarification_rate"] * t["count"]

    top_topics = sorted(
        [
            {
                "topic":              v["topic"],
                "count":              v["count"],
                "avg_importance":     round(v["importance_sum"] / v["count"], 2) if v["count"] else 0.0,
                "clarification_rate": round(v["clarified_sum"]  / v["count"], 4) if v["count"] else 0.0,
            }
            for v in topic_map.values()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )[:15]

    weak_topics = sorted(
        [t for t in top_topics if t["avg_importance"] < 3.0 and t["topic"] != "uncategorized"],
        key=lambda x: x["avg_importance"],
    )[:10]

    # ── Sources ───────────────────────────────────────────────────────────────
    source_map: dict[str, int] = {}
    for s in snapshots:
        for src in s.get("top_sources", []):
            source_map[src["source"]] = source_map.get(src["source"], 0) + src["count"]

    top_sources = sorted(
        [{"source": k, "count": v} for k, v in source_map.items()],
        key=lambda x: x["count"], reverse=True,
    )[:10]

    # ── Topic timeline — agregar semanas dos snapshots diários ────────────────
    timeline_map: dict[tuple, int] = {}
    for s in snapshots:
        for entry in s.get("topic_timeline", []):
            key = (entry["week"], entry["topic"])
            timeline_map[key] = timeline_map.get(key, 0) + entry["count"]

    topic_timeline = sorted(
        [{"week": k[0], "topic": k[1], "count": v} for k, v in timeline_map.items()],
        key=lambda x: (x["week"], x["topic"]),
    )

    return {
        "from_dt":             from_dt,
        "to_dt":               to_dt,
        "days_covered":        len(snapshots),
        "total_interactions":  total,
        "authenticated_count": authenticated_count,
        "anonymous_count":     anonymous_count,
        "unique_users":        unique_users,
        "unique_clients":      unique_clients,
        "avg_importance":      avg_importance,
        "low_importance_rate": round(low_importance_count / total, 4) if total else 0.0,
        "no_sources_rate":     round(no_sources_count    / total, 4) if total else 0.0,
        "uncategorized_count": uncategorized_count,
        "uncategorized_rate":  round(uncategorized_count / total, 4) if total else 0.0,
        "top_topics":          top_topics,
        "weak_topics":         weak_topics,
        "top_sources":         top_sources,
        "topic_timeline":      topic_timeline,
        "daily_breakdown": [
            {
                "period":              s["period"],
                "total_interactions":  s.get("total_interactions", 0),
                "authenticated_count": s.get("authenticated_count", 0),
                "anonymous_count":     s.get("anonymous_count", 0),
                "avg_importance":      s.get("avg_importance", 0.0),
            }
            for s in snapshots
        ],
    }


class StubMemoryReportReader:
    async def get_daily(self, period: str) -> dict | None:
        return None

    async def get_aggregated(self, from_dt: datetime, to_dt: datetime) -> dict:
        return _empty_aggregated(from_dt, to_dt)