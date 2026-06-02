from datetime import datetime, timezone
from typing import Protocol
import uuid

from rag_api.schemas.identity import APIKeyRecord, CreateKeyRequest

COLLECTION = "api_keys"


# ── Interface (Protocol) — não muda, independente da base de dados ────────────

class KeyRepositoryProtocol(Protocol):
    async def get_by_hash(self, key_hash: str) -> APIKeyRecord | None: ...
    async def create(self, request: CreateKeyRequest, key_hash: str, key_hint: str) -> APIKeyRecord: ...
    async def revoke(self, key_hint: str, client_id: str) -> bool: ...


# ── Implementação MongoDB ─────────────────────────────────────────────────────

class MongoKeyRepository:
    """
    Repositório de API keys sobre MongoDB.
    Recebe a database por injecção — nunca cria a sua própria conexão.

    Estrutura do documento na colecção api_keys:
    {
        _id:        str  (uuid gerado aqui),
        client_id:  str,
        key_hash:   str  (SHA-256 — único, indexado),
        key_hint:   str,
        scopes:     [str],
        rate_limit: int,
        expires_at: datetime | None,
        active:     bool,
        created_at: datetime
    }
    """

    def __init__(self, db):
        # db é um AsyncIOMotorDatabase injectado via app.state
        self._col = db[COLLECTION]

    async def get_by_hash(self, key_hash: str) -> APIKeyRecord | None:
        doc = await self._col.find_one({"key_hash": key_hash})
        if not doc:
            return None
        return _doc_to_record(doc)

    async def create(
        self,
        request: CreateKeyRequest,
        key_hash: str,
        key_hint: str,
    ) -> APIKeyRecord:
        doc = {
            "_id":        str(uuid.uuid4()),
            "client_id":  request.client_id,
            "key_hash":   key_hash,
            "key_hint":   key_hint,
            "scopes":     request.scopes,
            "rate_limit": request.rate_limit,
            "expires_at": request.expires_at,
            "active":     True,
            "created_at": datetime.now(timezone.utc),
        }
        await self._col.insert_one(doc)
        return _doc_to_record(doc)

    async def revoke(self, key_hint: str, client_id: str) -> bool:
        result = await self._col.update_one(
            {"key_hint": key_hint, "client_id": client_id, "active": True},
            {"$set": {"active": False}},
        )
        return result.modified_count > 0


# ── Helper de conversão ───────────────────────────────────────────────────────

def _doc_to_record(doc: dict) -> APIKeyRecord:
    return APIKeyRecord(
        id         = str(doc["_id"]),
        client_id  = doc["client_id"],
        key_hash   = doc["key_hash"],
        key_hint   = doc["key_hint"],
        scopes     = doc["scopes"],
        rate_limit = doc["rate_limit"],
        expires_at = doc.get("expires_at"),
        active     = doc["active"],
    )


# ── Inicialização da colecção — índices ───────────────────────────────────────

async def init_indexes(db) -> None:
    """
    Chamado no startup da aplicação.
    Garante performance equivalente a uma tabela SQL indexada.
    """
    col = db[COLLECTION]
    await col.create_index("key_hash", unique=True)   # lookup por hash — O(1)
    await col.create_index("client_id")               # listagem por cliente