"""
Backend de storage em disco local.

ACTIVO. Configurado via PDF_STORAGE_PATH no .env.
Para ambientes de produção com múltiplas instâncias considerar migração
para GridFS ou S3 (ver s3.py e gridfs.py comentados).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import aiofiles
import structlog

logger = structlog.get_logger(__name__)


class LocalDiskBackend:
    """
    Persiste PDFs no sistema de ficheiros local.

    O file_ref é o path absoluto do ficheiro — simples e directo
    para um ambiente de instância única.
    """

    def __init__(self, storage_path: str) -> None:
        self._root = Path(storage_path)
        # Garante que a directoria existe no momento de construção,
        # não no primeiro save — falha-rápido na inicialização.
        self._root.mkdir(parents=True, exist_ok=True)

    async def save(self, doc_id: str, filename: str, data: bytes) -> str:
        # Mantém o nome original do ficheiro — o doc_id é a chave no MongoDB,
        # não precisa de estar no nome do ficheiro em disco.
        dest = self._root / filename
        async with aiofiles.open(dest, "wb") as fh:
            await fh.write(data)
        logger.debug("pdf_saved_to_disk", path=str(dest), size=len(data))
        return str(dest)

    async def read(self, file_ref: str) -> bytes:
        async with aiofiles.open(file_ref, "rb") as fh:
            return await fh.read()

    async def delete(self, file_ref: str) -> None:
        path = Path(file_ref)
        if path.exists():
            path.unlink()
            logger.debug("pdf_deleted_from_disk", path=file_ref)

    async def exists(self, file_ref: str) -> bool:
        return Path(file_ref).exists()


# ---------------------------------------------------------------------------
# GridFS backend (MongoDB) — COMENTADO
# Activar: PDF_STORAGE_BACKEND=gridfs + MONGODB_URI no .env
# Vantagens sobre disco: backup automático Atlas, sem dependência de volume,
# pesquisável, funciona com múltiplas instâncias.
# ---------------------------------------------------------------------------
#
# from motor.motor_asyncio import AsyncIOMotorGridFSBucket
#
# class GridFSBackend:
#     def __init__(self, db) -> None:
#         self._bucket = AsyncIOMotorGridFSBucket(db, bucket_name="pdfs")
#
#     async def save(self, doc_id: str, filename: str, data: bytes) -> str:
#         from io import BytesIO
#         grid_id = await self._bucket.upload_from_stream(
#             filename, BytesIO(data), metadata={"doc_id": doc_id}
#         )
#         return str(grid_id)
#
#     async def read(self, file_ref: str) -> bytes:
#         from io import BytesIO
#         from bson import ObjectId
#         buf = BytesIO()
#         await self._bucket.download_to_stream(ObjectId(file_ref), buf)
#         return buf.getvalue()
#
#     async def delete(self, file_ref: str) -> None:
#         from bson import ObjectId
#         await self._bucket.delete(ObjectId(file_ref))
#
#     async def exists(self, file_ref: str) -> bool:
#         from bson import ObjectId
#         cursor = self._bucket.find({"_id": ObjectId(file_ref)})
#         return await cursor.fetch_next


# ---------------------------------------------------------------------------
# S3 backend — COMENTADO
# Activar: PDF_STORAGE_BACKEND=s3 + AWS_* vars no .env
# Recomendado para produção com múltiplas instâncias ou volume elevado.
# ---------------------------------------------------------------------------
#
# import aioboto3
#
# class S3Backend:
#     def __init__(self, bucket: str, region: str) -> None:
#         self._bucket = bucket
#         self._session = aioboto3.Session(region_name=region)
#
#     def _key(self, doc_id: str, filename: str) -> str:
#         return f"pdfs/{doc_id}/{filename}"
#
#     async def save(self, doc_id: str, filename: str, data: bytes) -> str:
#         key = self._key(doc_id, filename)
#         async with self._session.client("s3") as s3:
#             await s3.put_object(Bucket=self._bucket, Key=key, Body=data)
#         return key
#
#     async def read(self, file_ref: str) -> bytes:
#         async with self._session.client("s3") as s3:
#             resp = await s3.get_object(Bucket=self._bucket, Key=file_ref)
#             return await resp["Body"].read()
#
#     async def delete(self, file_ref: str) -> None:
#         async with self._session.client("s3") as s3:
#             await s3.delete_object(Bucket=self._bucket, Key=file_ref)
#
#     async def exists(self, file_ref: str) -> bool:
#         import botocore
#         async with self._session.client("s3") as s3:
#             try:
#                 await s3.head_object(Bucket=self._bucket, Key=file_ref)
#                 return True
#             except botocore.exceptions.ClientError:
#                 return False