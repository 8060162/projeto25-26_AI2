"""
Repository — Módulo 4 — Memória Longa.

Protocol + MongoMemoryRepository seguem o padrão estabelecido em
MongoKeyRepository e MongoApplicationRepository.

expires_at filtrado na query (não via TTL index MongoDB):
  - Controlo explícito por entrada — possível renovar TTL sem recriar documento
  - Auditoria de entradas expiradas antes da limpeza física
  - TTL index elimina silenciosamente e não é controlável por entrada
  Limpeza física via job periódico no scheduler (ver main.py — dívida técnica).

touch():
  Actualiza last_seen_at quando o pipeline usa esta memória no retrieval.
  Sem touch, memórias continuamente úteis decairiam injustamente por não
  terem sido editadas explicitamente. Chamado pelo RAG Controller — não exposto via API.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from motor.motor_asyncio import AsyncIOMotorDatabase

from .schemas import MemoryCreateRequest, MemoryUpdateRequest


# ── Domain record ─────────────────────────────────────────────────────────────

class MemoryRecord:
    __slots__ = (
        "mem_id", "user_key", "session_id", "client_id", "mode",
        "type", "topic", "importance", "interaction",
        "last_seen_at", "created_at", "expires_at",
    )

    def __init__(self, **kwargs) -> None:
        for attr in self.__slots__:
            setattr(self, attr, kwargs[attr])


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class MemoryRepository(Protocol):

    async def create(
        self,
        user_key:   str,
        client_id:  str,
        mode:       str,
        req:        MemoryCreateRequest,
        expires_at: datetime,
    ) -> MemoryRecord: ...

    async def list_active(
        self,
        user_key:       str,
        type_filter:    str | None,
        min_importance: float | None,
        limit:          int,
        skip:           int,
    ) -> tuple[list[MemoryRecord], int]: ...

    async def get(self, user_key: str, mem_id: str) -> MemoryRecord | None: ...

    async def update(
        self,
        user_key: str,
        mem_id:   str,
        req:      MemoryUpdateRequest,
    ) -> MemoryRecord | None: ...

    async def delete(self, user_key: str, mem_id: str) -> bool: ...

    async def touch(self, user_key: str, mem_id: str) -> None: ...


# ── MongoDB implementation ────────────────────────────────────────────────────

_COLLECTION = "long_memory"


def _doc_to_record(doc: dict) -> MemoryRecord:
    return MemoryRecord(
        mem_id       = doc["_id"],
        user_key     = doc["user_key"],
        session_id   = doc["session_id"],
        client_id    = doc["client_id"],
        mode         = doc["mode"],
        type         = doc["type"],
        topic        = doc["topic"],
        importance   = doc["importance"],
        interaction  = doc["interaction"],
        last_seen_at = doc["last_seen_at"],
        created_at   = doc["created_at"],
        expires_at   = doc["expires_at"],
    )


def _active_filter(user_key: str, now: datetime) -> dict:
    """Filtro base reutilizado em todas as queries — SSOT."""
    return {"user_key": user_key, "expires_at": {"$gt": now}}


class MongoMemoryRepository:

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[_COLLECTION]

    async def create(
        self,
        user_key:   str,
        client_id:  str,
        mode:       str,
        req:        MemoryCreateRequest,
        expires_at: datetime,
    ) -> MemoryRecord:
        now    = datetime.now(timezone.utc)
        mem_id = f"mem_{uuid.uuid4().hex}"
        doc = {
            "_id":          mem_id,
            "user_key":     user_key,
            "session_id":   req.session_id,
            "client_id":    client_id,
            "mode":         mode,
            "type":         req.type,
            "topic":        req.topic,
            "importance":   req.importance,
            "interaction":  req.interaction.model_dump(),
            "last_seen_at": now,
            "created_at":   now,
            "expires_at":   expires_at,
        }
        await self._col.insert_one(doc)
        return _doc_to_record(doc)

    async def list_active(
        self,
        user_key:       str,
        type_filter:    str | None,
        min_importance: float | None,
        limit:          int,
        skip:           int,
    ) -> tuple[list[MemoryRecord], int]:
        now   = datetime.now(timezone.utc)
        query = _active_filter(user_key, now)

        if type_filter    is not None: query["type"]       = type_filter
        if min_importance is not None: query["importance"] = {"$gte": min_importance}

        cursor = (
            self._col.find(query)
            .sort([("importance", -1), ("created_at", -1)])
            .skip(skip)
            .limit(limit)
        )
        docs  = await cursor.to_list(length=limit)
        total = await self._col.count_documents(query)
        return [_doc_to_record(d) for d in docs], total

    async def get(self, user_key: str, mem_id: str) -> MemoryRecord | None:
        now = datetime.now(timezone.utc)
        doc = await self._col.find_one({
            **_active_filter(user_key, now),
            "_id": mem_id,
        })
        return _doc_to_record(doc) if doc else None

    async def update(
        self,
        user_key: str,
        mem_id:   str,
        req:      MemoryUpdateRequest,
    ) -> MemoryRecord | None:
        now    = datetime.now(timezone.utc)
        fields: dict = {"last_seen_at": now}

        if req.topic       is not None: fields["topic"]       = req.topic
        if req.importance  is not None: fields["importance"]  = req.importance
        if req.interaction is not None: fields["interaction"] = req.interaction.model_dump()

        result = await self._col.find_one_and_update(
            {"_id": mem_id, **_active_filter(user_key, now)},
            {"$set": fields},
            return_document=True,
        )
        return _doc_to_record(result) if result else None

    async def delete(self, user_key: str, mem_id: str) -> bool:
        result = await self._col.delete_one({"_id": mem_id, "user_key": user_key})
        return result.deleted_count == 1

    async def touch(self, user_key: str, mem_id: str) -> None:
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"_id": mem_id, "user_key": user_key},
            {"$set": {"last_seen_at": now}},
        )


# ── Index initialisation ──────────────────────────────────────────────────────

async def init_memory_indexes(db: AsyncIOMotorDatabase) -> None:
    """
    Idempotente — seguro chamar em cada startup.

    Índices:
      user_type_importance  — listagem filtrada por tipo, ordenada por relevância
      user_timeline         — linha temporal do aluno (Canal 1)
      client_mode_timeline  — métricas por aplicação/modo (Canal 2)
      expires_at            — filtro de activos em todas as queries
      trace_id (sparse)     — rastreabilidade com o pipeline; anónimos sem trace_id
                              não ocupam entrada no índice
    """
    col = db[_COLLECTION]
    await col.create_index(
        [("user_key", 1), ("type", 1), ("importance", -1)],
        name="user_type_importance",
    )
    await col.create_index(
        [("user_key", 1), ("created_at", -1)],
        name="user_timeline",
    )
    await col.create_index(
        [("client_id", 1), ("mode", 1), ("created_at", -1)],
        name="client_mode_timeline",
    )
    await col.create_index("expires_at", name="expires_at")
    await col.create_index(
        "interaction.trace_id",
        name   = "trace_id",
        sparse = True,
    )