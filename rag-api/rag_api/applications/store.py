import uuid
from datetime import datetime, timezone
from typing import Protocol

from rag_api.schemas.applications import (
    ApplicationRecord,
    CreateApplicationRequest,
    UpdateApplicationRequest,
)

COLLECTION = "applications"


# ── Interface ─────────────────────────────────────────────────────────────────

class ApplicationRepositoryProtocol(Protocol):
    async def get_by_id(self, app_id: str) -> ApplicationRecord | None: ...
    async def get_all(self) -> list[ApplicationRecord]: ...
    async def create(self, request: CreateApplicationRequest, created_by: str) -> ApplicationRecord: ...
    async def update(self, app_id: str, request: UpdateApplicationRequest) -> ApplicationRecord | None: ...
    async def delete(self, app_id: str) -> bool: ...


# ── Implementação MongoDB ─────────────────────────────────────────────────────

class MongoApplicationRepository:
    """
    Repositório de Applications sobre MongoDB.
    Uma Application é a entidade que agrupa um cliente e as suas API Keys.

    Estrutura do documento:
    {
        _id:         str (uuid),
        name:        str,
        description: str | None,
        scopes:      [str],
        rate_limit:  int,
        active:      bool,
        created_at:  datetime,
        created_by:  str
    }
    """

    def __init__(self, db):
        self._col = db[COLLECTION]

    async def get_by_id(self, app_id: str) -> ApplicationRecord | None:
        doc = await self._col.find_one({"_id": app_id, "active": True})
        return _doc_to_record(doc) if doc else None

    async def get_all(self) -> list[ApplicationRecord]:
        cursor = self._col.find({"active": True})
        return [_doc_to_record(doc) async for doc in cursor]

    async def create(
        self,
        request: CreateApplicationRequest,
        created_by: str,
    ) -> ApplicationRecord:
        doc = {
            "_id":         str(uuid.uuid4()),
            "name":        request.name,
            "description": request.description,
            "scopes":      request.scopes,
            "rate_limit":  request.rate_limit,
            "active":      True,
            "created_at":  datetime.now(timezone.utc),
            "created_by":  created_by,
        }
        await self._col.insert_one(doc)
        return _doc_to_record(doc)

    async def update(
        self,
        app_id: str,
        request: UpdateApplicationRequest,
    ) -> ApplicationRecord | None:
        # Constrói o update apenas com os campos fornecidos — SSOT
        updates = {
            k: v for k, v in request.model_dump(exclude_none=True).items()
        }
        if not updates:
            return await self.get_by_id(app_id)

        result = await self._col.find_one_and_update(
            {"_id": app_id, "active": True},
            {"$set": updates},
            return_document=True,
        )
        return _doc_to_record(result) if result else None

    async def delete(self, app_id: str) -> bool:
        # Soft delete — mantém o registo para audit trail
        result = await self._col.update_one(
            {"_id": app_id, "active": True},
            {"$set": {"active": False}},
        )
        return result.modified_count > 0


# ── Helper ────────────────────────────────────────────────────────────────────

def _doc_to_record(doc: dict) -> ApplicationRecord:
    return ApplicationRecord(
        id          = str(doc["_id"]),
        name        = doc["name"],
        description = doc.get("description"),
        scopes      = doc["scopes"],
        rate_limit  = doc["rate_limit"],
        active      = doc["active"],
        created_at  = doc["created_at"],
        created_by  = doc["created_by"],
    )


# ── Índices ───────────────────────────────────────────────────────────────────

async def init_application_indexes(db) -> None:
    col = db[COLLECTION]
    await col.create_index("name")
    await col.create_index("active")