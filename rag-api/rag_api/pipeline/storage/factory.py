"""
Factory de storage — lê configuração via get_settings() para garantir
que o .env é sempre respeitado independentemente do os.environ.

OCP: adicionar um novo backend é registar uma entrada aqui + implementar o Protocol.
Não requer alteração do service.py nem dos callers.
"""
from __future__ import annotations

from rag_api.pipeline.storage.base import PdfStorageBackend
from rag_api.pipeline.storage.local_disk import LocalDiskBackend


def make_storage_backend() -> PdfStorageBackend:
    from rag_api.config.settings import get_settings
    settings = get_settings()

    backend = settings.pdf_storage_backend.lower()

    if backend == "local":
        return LocalDiskBackend(storage_path=settings.pdf_storage_path)

    # Registar backends adicionais aqui quando activados:
    # if backend == "gridfs": ...
    # if backend == "s3": ...

    raise ValueError(
        f"PDF_STORAGE_BACKEND='{backend}' não reconhecido. "
        "Valores suportados: local"
    )