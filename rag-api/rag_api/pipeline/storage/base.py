"""
Contrato de storage para PDFs.

Protocol em vez de ABC — permite duck typing e facilita mocks em testes
sem herança forçada. Qualquer classe com os 4 métodos é um backend válido.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PdfStorageBackend(Protocol):
    """
    Interface de storage para ficheiros PDF.

    Cada implementação é responsável por persistir e recuperar bytes —
    a localização física (disco, GridFS, S3) é um detalhe de implementação
    opaco para o resto do módulo.
    """

    async def save(self, doc_id: str, filename: str, data: bytes) -> str:
        """
        Persiste o ficheiro e devolve o file_ref opaco.

        O file_ref é o único identificador que o caller deve guardar —
        o seu formato é específico de cada backend (path, ObjectId, S3 key).
        """
        ...

    async def read(self, file_ref: str) -> bytes:
        """Recupera os bytes do ficheiro dado o file_ref."""
        ...

    async def delete(self, file_ref: str) -> None:
        """Remove o ficheiro. Silencioso se já não existir."""
        ...

    async def exists(self, file_ref: str) -> bool:
        """Verifica se o ficheiro existe sem o carregar."""
        ...