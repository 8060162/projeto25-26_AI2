"""
PipelineService — orquestra storage, chroma, preview e auditoria.

SRP: o service coordena; não implementa lógica de nenhuma dependência.
Todas as dependências são injectadas — testável com mocks simples.

Decisão de design: o service não conhece HTTP nem MongoDB directamente.
A camada de persistência dos metadados dos PDFs é responsabilidade
da estrutura existente do colega — o service recebe o doc_id e o file_ref
já resolvidos, e delega ao indexer do colega a indexação no ChromaDB.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Optional

import structlog

from rag_api.pipeline.audit import PipelineAuditLogger
from rag_api.pipeline.chroma import ChromaPipelineClient
from rag_api.pipeline.models import (
    AuditAction,
    AuditMeta,
    AuditOutcome,
    DocumentStatus,
    PipelineResult,
)
from rag_api.pipeline.preview import PagePreviewRenderer
from rag_api.pipeline.storage.base import PdfStorageBackend

logger = structlog.get_logger(__name__)

_MAX_SIZE_BYTES = int(os.getenv("PDF_MAX_SIZE_MB", "50")) * 1024 * 1024


class PipelineService:
    """
    Coordena o ciclo de vida completo de um PDF:
    upload → indexação → preview → substituição → remoção.
    """

    def __init__(
        self,
        storage:  PdfStorageBackend,
        chroma:   ChromaPipelineClient,
        preview:  PagePreviewRenderer,
        audit:    PipelineAuditLogger,
    ) -> None:
        self._storage = storage
        self._chroma  = chroma
        self._preview = preview
        self._audit   = audit

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def upload(
        self,
        *,
        doc_id:       str,
        filename:     str,
        data:         bytes,
        performed_by: str,
        trace_id:     str,
        ip_address:   Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Persiste o ficheiro e devolve (file_ref, checksum).

        A indexação ChromaDB é disparada separadamente via reindex()
        para permitir que o caller (route) controle o timing e o status.
        A validação de tamanho e duplicados é responsabilidade do route
        antes de chamar este método — fail-fast na fronteira HTTP.
        """
        checksum = hashlib.sha256(data).hexdigest()
        file_ref = await self._storage.save(doc_id, filename, data)

        await self._audit.log_event(
            self._audit.build_event(
                action       = AuditAction.UPLOAD,
                outcome      = AuditOutcome.SUCCESS,
                performed_by = performed_by,
                http_status  = 201,
                doc_id       = doc_id,
                filename     = filename,
                meta         = AuditMeta(trace_id=trace_id, ip_address=ip_address),
            )
        )
        return file_ref, checksum

    # ------------------------------------------------------------------
    # Indexação
    # ------------------------------------------------------------------

    async def reindex(
        self,
        *,
        doc_id:          str,
        filename:        str,
        file_ref:        str,
        old_chroma_ids:  list[str],
        version:         int,
        performed_by:    str,
        trace_id:        str,
        ip_address:      Optional[str] = None,
    ) -> PipelineResult:
        """
        Remove chunks antigos e re-indexa o documento.

        Contrato com o colega: run_embedding_indexer(doc_id) deve
        devolver um objecto com .chroma_ids: list[str] e .chunk_count: int.
        Este contrato está pendente de implementação (dívida técnica).
        """
        # Remove chunks da versão anterior antes de re-indexar.
        if old_chroma_ids:
            await self._chroma.delete_chunks(old_chroma_ids)

        loop = asyncio.get_event_loop()

        def _index():
            from embedding.indexer import run_embedding_indexer
            # Dívida técnica: indexer actual processa todos os ficheiros em data/raw/.
            # Quando o colega implementar run_embedding_indexer(doc_id=...) substituir por:
            # return run_embedding_indexer(doc_id=doc_id)
            return run_embedding_indexer()

        try:
            index_result = await loop.run_in_executor(None, _index)
        except Exception as exc:
            await self._audit.log_event(
                self._audit.build_event(
                    action       = AuditAction.REINDEX,
                    outcome      = AuditOutcome.ERROR,
                    performed_by = performed_by,
                    http_status  = 500,
                    doc_id       = doc_id,
                    filename     = filename,
                    error_detail = repr(exc),
                    meta         = AuditMeta(trace_id=trace_id, ip_address=ip_address),
                )
            )
            raise

        # Dívida técnica: quando indexer devolver chroma_ids por doc_id,
        # substituir chunk_count e chroma_ids_count pelos valores reais.
        result = PipelineResult(
            chunk_count      = getattr(index_result, "embedded_record_count", 0),
            chroma_ids_count = 0,
            status           = DocumentStatus.INDEXED,
        )

        await self._audit.log_event(
            self._audit.build_event(
                action          = AuditAction.REINDEX,
                outcome         = AuditOutcome.SUCCESS,
                performed_by    = performed_by,
                http_status     = 200,
                doc_id          = doc_id,
                filename        = filename,
                pipeline_result = result,
                meta            = AuditMeta(trace_id=trace_id, ip_address=ip_address),
            )
        )

        logger.info(
            "reindex_completed",
            doc_id      = doc_id,
            version     = version,
            chunk_count = result.chunk_count,
            trace_id    = trace_id,
        )
        return result

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete(
        self,
        *,
        doc_id:       str,
        filename:     str,
        file_ref:     str,
        chroma_ids:   list[str],
        performed_by: str,
        trace_id:     str,
        ip_address:   Optional[str] = None,
    ) -> None:
        """
        Remove chunks do ChromaDB e o ficheiro do storage.
        A marcação do status no MongoDB é responsabilidade do route/store.
        """
        await self._chroma.delete_chunks(chroma_ids)
        await self._storage.delete(file_ref)

        await self._audit.log_event(
            self._audit.build_event(
                action       = AuditAction.DELETE,
                outcome      = AuditOutcome.SUCCESS,
                performed_by = performed_by,
                http_status  = 200,
                doc_id       = doc_id,
                filename     = filename,
                meta         = AuditMeta(trace_id=trace_id, ip_address=ip_address),
            )
        )

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    async def render_page(
        self,
        *,
        doc_id:       str,
        filename:     str,
        file_ref:     str,
        page_number:  int,
        page_count:   int,
        dpi:          int,
        performed_by: str,
        trace_id:     str,
        ip_address:   Optional[str] = None,
    ) -> bytes:
        """
        Devolve os bytes PNG da página.
        Validação de page_number feita aqui — fail-fast antes do I/O.
        """
        if page_number < 1 or page_number > page_count:
            raise ValueError(
                f"Página {page_number} inválida. O documento tem {page_count} páginas."
            )

        pdf_bytes = await self._storage.read(file_ref)
        preview   = await self._preview.render(pdf_bytes, page_number, dpi)

        await self._audit.log_event(
            self._audit.build_event(
                action       = AuditAction.PREVIEW,
                outcome      = AuditOutcome.SUCCESS,
                performed_by = performed_by,
                http_status  = 200,
                doc_id       = doc_id,
                filename     = filename,
                meta         = AuditMeta(
                    trace_id   = trace_id,
                    ip_address = ip_address,
                    dpi        = dpi,
                ),
            )
        )
        return preview.png_bytes