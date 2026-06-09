"""
Aggregator — Memory Reporting.

Fonte:    long_memory
Destino:  memory_snapshots

Três adições face à versão anterior:
  1. clarification_rate por tópico — % interacções com sources E importance >= threshold
     Responde a: "O sistema está a esclarecer os alunos neste tópico?"
  2. topic_timeline — volume por tópico por semana
     Responde a: "Há sazonalidade? Pico de propinas em Setembro?"
  3. uncategorized_count + uncategorized_rate
     Responde a: "Quantas perguntas o pipeline não conseguiu classificar?"
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Protocol, runtime_checkable

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from .models import MemorySnapshotWindow, memory_period_key, LISBON

logger = structlog.get_logger(__name__)

MEMORY_COLLECTION    = "long_memory"
SNAPSHOT_COLLECTION  = "memory_snapshots"
LOW_IMPORTANCE_THRESHOLD  = 3.0
HIGH_IMPORTANCE_THRESHOLD = 6.0   # acima disto considera-se "esclarecido"


@runtime_checkable
class MemorySnapshotAggregator(Protocol):
    async def compute_and_store(self, window: MemorySnapshotWindow) -> dict: ...
    async def get_snapshot(self, period: str) -> dict | None: ...
    async def get_range(self, from_dt: datetime, to_dt: datetime) -> list[dict]: ...
    async def purge_before(self, cutoff: datetime) -> None: ...


def _daily_window(at: datetime | None = None, today: bool = False) -> MemorySnapshotWindow:
    """
    Janela de um dia completo.

    today=False (default) → dia anterior completo — usado pelo scheduler a meia-noite,
                            garante que todos os dados do dia estao disponiveis.
    today=True            → dia actual ate ao momento presente — usado pelo Refresh
                            manual quando ainda nao ha dados do dia anterior.

    Nota: start calculado em Lisboa para o period key ser consistente com o calendario
    academico. end em today=True e o momento actual em UTC puro — evita o problema de
    conversao de timezone fechar a janela na meia-noite UTC em vez do momento actual.
    """
    now_utc  = at or datetime.now(timezone.utc)
    now_lisbon = now_utc.astimezone(LISBON)
    start    = now_lisbon.replace(hour=0, minute=0, second=0, microsecond=0)
    if not today:
        start = start - timedelta(days=1)
    # end em today: momento actual em UTC — nao converter de Lisboa para evitar desfasamento
    end_utc = now_utc if today else (start + timedelta(days=1) - timedelta(microseconds=1)).astimezone(timezone.utc)
    return MemorySnapshotWindow(
        from_dt = start.astimezone(timezone.utc),
        to_dt   = end_utc,
        period  = memory_period_key(start),
    )


class MongoMemoryAggregator:

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._memory    = db[MEMORY_COLLECTION]
        self._snapshots = db[SNAPSHOT_COLLECTION]

    async def compute_and_store(self, window: MemorySnapshotWindow) -> dict:
        snapshot = await self._aggregate(window)
        await self._upsert(snapshot)
        return snapshot

    async def get_snapshot(self, period: str) -> dict | None:
        return await self._snapshots.find_one({"period": period}, {"_id": 0})

    async def get_range(self, from_dt: datetime, to_dt: datetime) -> list[dict]:
        cursor = self._snapshots.find(
            {"from_dt": {"$gte": from_dt}, "to_dt": {"$lte": to_dt}},
            {"_id": 0},
        ).sort("period", 1)
        return await cursor.to_list(length=None)

    async def purge_before(self, cutoff: datetime) -> None:
        try:
            result = await self._snapshots.delete_many({"generated_at": {"$lt": cutoff}})
            if result.deleted_count:
                logger.info("memory_snapshot.purged", deleted=result.deleted_count)
        except Exception:
            logger.warning("memory_snapshot.purge_failed")

    async def _aggregate(self, window: MemorySnapshotWindow) -> dict:
        match = {"created_at": {"$gte": window.from_dt, "$lte": window.to_dt}}

        # ── Totais globais ────────────────────────────────────────────────────
        main = await self._memory.aggregate([
            {"$match": match},
            {"$group": {
                "_id":                  None,
                "total_interactions":   {"$sum": 1},
                "authenticated_count":  {"$sum": {"$cond": [{"$eq": ["$mode", "authenticated"]}, 1, 0]}},
                "anonymous_count":      {"$sum": {"$cond": [{"$eq": ["$mode", "anonymous"]}, 1, 0]}},
                "avg_importance":       {"$avg": "$importance"},
                "low_importance_count": {"$sum": {
                    "$cond": [{"$lt": ["$importance", LOW_IMPORTANCE_THRESHOLD]}, 1, 0]
                }},
                "unique_users":   {"$addToSet": "$user_key"},
                "unique_clients": {"$addToSet": "$client_id"},
                "no_sources_count": {"$sum": {
                    "$cond": [{"$eq": [{"$size": "$interaction.sources"}, 0]}, 1, 0]
                }},
                # 3. Perguntas não classificadas
                "uncategorized_count": {"$sum": {
                    "$cond": [{"$eq": ["$topic", "uncategorized"]}, 1, 0]
                }},
            }},
        ]).to_list(1)

        # ── Top tópicos com clarification_rate ────────────────────────────────
        # clarification_rate = % de interacções com sources E importance >= HIGH_IMPORTANCE_THRESHOLD
        # Responde a: "O sistema está a esclarecer os alunos neste tópico?"
        top_topics = await self._memory.aggregate([
            {"$match": match},
            {"$group": {
                "_id":            "$topic",
                "count":          {"$sum": 1},
                "avg_importance": {"$avg": "$importance"},
                # esclarecido = tem fontes E importance alta
                "clarified_count": {"$sum": {
                    "$cond": [{
                        "$and": [
                            {"$gte": ["$importance", HIGH_IMPORTANCE_THRESHOLD]},
                            {"$gt":  [{"$size": "$interaction.sources"}, 0]},
                        ]
                    }, 1, 0]
                }},
            }},
            {"$addFields": {
                "clarification_rate": {
                    "$cond": [
                        {"$gt": ["$count", 0]},
                        {"$divide": ["$clarified_count", "$count"]},
                        0,
                    ]
                }
            }},
            {"$sort": {"count": -1}},
            {"$limit": 15},
        ]).to_list(15)

        # ── Tópicos fracos ────────────────────────────────────────────────────
        weak_topics = await self._memory.aggregate([
            {"$match": match},
            {"$group": {
                "_id":            "$topic",
                "avg_importance": {"$avg": "$importance"},
                "count":          {"$sum": 1},
            }},
            {"$match": {
                "_id":            {"$ne": "uncategorized"},
                "avg_importance": {"$lt": LOW_IMPORTANCE_THRESHOLD},
                "count":          {"$gte": 3},   # só tópicos com volume relevante
            }},
            {"$sort": {"avg_importance": 1}},
            {"$limit": 10},
        ]).to_list(10)

        # ── Sources mais referenciadas ────────────────────────────────────────
        top_sources = await self._memory.aggregate([
            {"$match": match},
            {"$unwind": "$interaction.sources"},
            {"$group": {"_id": "$interaction.sources", "count": {"$sum": 1}}},
            {"$sort":  {"count": -1}},
            {"$limit": 10},
        ]).to_list(10)

        # ── Timeline semanal por tópico (top 5) ───────────────────────────────
        # Responde a: "Há sazonalidade? Pico de propinas em Setembro?"
        # Agrupamos por semana ISO para reduzir cardinalidade nos snapshots diários.
        # O reader agrega as semanas no intervalo pedido pelo administrador.
        top5_topics = [t["_id"] for t in top_topics[:5] if t["_id"] != "uncategorized"]
        topic_timeline = []
        if top5_topics:
            topic_timeline = await self._memory.aggregate([
                {"$match": {**match, "topic": {"$in": top5_topics}}},
                {"$group": {
                    "_id": {
                        "week": {
                            "$dateToString": {
                                "format":   "%Y-W%V",
                                "date":     "$created_at",
                                "timezone": "Europe/Lisbon",
                            }
                        },
                        "topic": "$topic",
                    },
                    "count": {"$sum": 1},
                }},
                {"$sort": {"_id.week": 1}},
            ]).to_list(None)

        # ── Distribuição diária por modo ──────────────────────────────────────
        daily_by_mode = await self._memory.aggregate([
            {"$match": match},
            {"$group": {
                "_id": {
                    "date": {
                        "$dateToString": {
                            "format":   "%Y-%m-%d",
                            "date":     "$created_at",
                            "timezone": "Europe/Lisbon",
                        }
                    },
                    "mode": "$mode",
                },
                "count": {"$sum": 1},
            }},
        ]).to_list(None)

        # ── Montar snapshot ───────────────────────────────────────────────────
        raw   = main[0] if main else {}
        total = raw.get("total_interactions", 0)
        uncategorized = raw.get("uncategorized_count", 0)

        return {
            "period":       window.period,
            "generated_at": datetime.now(timezone.utc),
            "from_dt":      window.from_dt,
            "to_dt":        window.to_dt,

            # Actividade
            "total_interactions":  total,
            "authenticated_count": raw.get("authenticated_count", 0),
            "anonymous_count":     raw.get("anonymous_count", 0),
            "unique_users":        len(raw.get("unique_users") or []),
            "unique_clients":      len(raw.get("unique_clients") or []),

            # Qualidade global
            "avg_importance":       round(raw.get("avg_importance") or 0.0, 2),
            "low_importance_rate":  round(raw.get("low_importance_count", 0) / total, 4) if total else 0.0,
            "no_sources_rate":      round(raw.get("no_sources_count", 0) / total, 4) if total else 0.0,

            # 3. Não classificadas
            "uncategorized_count": uncategorized,
            "uncategorized_rate":  round(uncategorized / total, 4) if total else 0.0,

            # Conteúdo — tópicos com clarification_rate
            "top_topics": [
                {
                    "topic":              t["_id"] or "uncategorized",
                    "count":              t["count"],
                    "avg_importance":     round(t["avg_importance"], 2),
                    "clarification_rate": round(t["clarification_rate"], 4),
                }
                for t in top_topics
            ],
            "weak_topics": [
                {
                    "topic":          t["_id"] or "uncategorized",
                    "count":          t["count"],
                    "avg_importance": round(t["avg_importance"], 2),
                }
                for t in weak_topics
            ],
            "top_sources": [
                {"source": s["_id"], "count": s["count"]}
                for s in top_sources
            ],

            # 2. Timeline semanal por tópico
            "topic_timeline": [
                {
                    "week":  r["_id"]["week"],
                    "topic": r["_id"]["topic"],
                    "count": r["count"],
                }
                for r in topic_timeline
            ],

            # Distribuição por modo ao longo do dia
            "interactions_by_mode": {
                f"{r['_id']['date']}_{r['_id']['mode']}": r["count"]
                for r in daily_by_mode
            },
        }

    async def _upsert(self, snapshot: dict) -> None:
        try:
            await self._snapshots.update_one(
                {"period": snapshot["period"]},
                {"$set": snapshot},
                upsert=True,
            )
        except Exception:
            logger.warning("memory_snapshot.upsert_failed", period=snapshot["period"])


class StubMemoryAggregator:
    async def compute_and_store(self, window: MemorySnapshotWindow) -> dict:
        return {}

    async def get_snapshot(self, period: str) -> dict | None:
        return None

    async def get_range(self, from_dt: datetime, to_dt: datetime) -> list[dict]:
        return []

    async def purge_before(self, cutoff: datetime) -> None:
        pass