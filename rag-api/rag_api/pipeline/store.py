"""
PdfDocumentStore — persistência de metadados dos PDFs.

Usa a colecção MongoDB `pdf_documents` como source of truth.
O schema foi definido no módulo pipeline e é independente do pipeline
do colega — os PDFs são geridos pelo rag-api, a indexação é delegada
ao embedding/indexer do colega.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from rag_api.pipeline.models import DocumentStatus

logger = structlog.get_logger(__name__)

_COLLECTION = "pdf_documents"


class MongoPdfDocumentStore:
    """
    CRUD de metadados de PDFs.
    Recebe a db já instanciada — sem lógica de conexão aqui.
    """

    def __init__(self, db) -> None:
        self._col = db[_COLLECTION]

    async def create(
        self,
        *,
        doc_id:      str,
        filename:    str,
        file_ref:    str,
        file_size:   int,
        checksum:    str,
        uploaded_by: str,
        page_count:  int = 0,
    ) -> dict:
        now = datetime.now(timezone.utc)
        doc = {
            "doc_id":      doc_id,
            "filename":    filename,
            "file_ref":    file_ref,
            "file_size":   file_size,
            "checksum":    checksum,
            "status":      DocumentStatus.PENDING,
            "error_detail": None,
            "version":     1,
            "chroma_ids":  [],
            "chunk_count": 0,
            "page_count":  page_count,
            "uploaded_by": uploaded_by,
            "uploaded_at": now,
            "indexed_at":  None,
            "updated_at":  now,
        }
        await self._col.insert_one(doc)
        return _clean(doc)

    async def get(self, doc_id: str) -> dict | None:
        doc = await self._col.find_one({"doc_id": doc_id})
        return _clean(doc) if doc else None

    async def find_by_checksum(self, checksum: str) -> dict | None:
        doc = await self._col.find_one({"checksum": checksum})
        return _clean(doc) if doc else None

    async def list(
        self,
        *,
        page:      int = 1,
        page_size: int = 20,
        status:    Optional[str] = None,
    ) -> tuple[list[dict], int]:
        query: dict = {}
        if status:
            query["status"] = status
        else:
            # Por omissão não mostra deleted
            query["status"] = {"$ne": DocumentStatus.DELETED}

        skip  = (page - 1) * page_size
        total = await self._col.count_documents(query)
        cursor = self._col.find(query).sort("uploaded_at", -1).skip(skip).limit(page_size)
        docs   = await cursor.to_list(length=page_size)
        return [_clean(d) for d in docs], total

    async def update_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        error_detail: Optional[str] = None,
    ) -> dict:
        update: dict = {
            "$set": {
                "status":     status,
                "updated_at": datetime.now(timezone.utc),
            }
        }
        if error_detail is not None:
            update["$set"]["error_detail"] = error_detail
        await self._col.update_one({"doc_id": doc_id}, update)
        return await self.get(doc_id)

    async def update_after_index(
        self,
        doc_id:      str,
        chroma_ids:  list[str],
        chunk_count: int,
        status:      DocumentStatus,
        page_count:  int = 0,
    ) -> dict:
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"doc_id": doc_id},
            {"$set": {
                "status":      status,
                "chroma_ids":  chroma_ids,
                "chunk_count": chunk_count,
                "page_count":  page_count,
                "indexed_at":  now,
                "updated_at":  now,
                "error_detail": None,
            }},
        )
        return await self.get(doc_id)

    async def update_file(
        self,
        doc_id:   str,
        file_ref: str,
        checksum: str,
        file_size: int,
        filename: str,
        version:  int,
    ) -> dict:
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"doc_id": doc_id},
            {"$set": {
                "file_ref":  file_ref,
                "checksum":  checksum,
                "file_size": file_size,
                "filename":  filename,
                "version":   version,
                "status":    DocumentStatus.PENDING,
                "updated_at": now,
            }},
        )
        return await self.get(doc_id)

    async def status_counts(self) -> dict:
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        cursor = self._col.aggregate(pipeline)
        rows   = await cursor.to_list(length=20)
        counts = {r["_id"]: r["count"] for r in rows}
        total_chunks = await self._col.aggregate([
            {"$match": {"status": DocumentStatus.INDEXED}},
            {"$group": {"_id": None, "total": {"$sum": "$chunk_count"}}},
        ]).to_list(1)
        return {
            "total_documents": sum(counts.values()),
            "indexed":         counts.get(DocumentStatus.INDEXED, 0),
            "pending":         counts.get(DocumentStatus.PENDING, 0),
            "processing":      counts.get(DocumentStatus.PROCESSING, 0),
            "error":           counts.get(DocumentStatus.ERROR, 0),
            "deleted":         counts.get(DocumentStatus.DELETED, 0),
            "total_chunks":    total_chunks[0]["total"] if total_chunks else 0,
        }


def _clean(doc: dict) -> dict:
    """Remove o _id do MongoDB antes de devolver ao caller."""
    if doc and "_id" in doc:
        doc = {k: v for k, v in doc.items() if k != "_id"}
    return doc


async def init_pdf_document_indexes(db) -> None:
    """Índices criados no startup — idempotente."""
    col = db[_COLLECTION]
    await col.create_index("doc_id",   unique=True)
    await col.create_index("checksum")
    await col.create_index("status")
    await col.create_index("uploaded_at")
    logger.info("pdf_document_indexes_ready")