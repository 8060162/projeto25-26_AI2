"""
Router do módulo pipeline — /v1/pipeline.

Todos os endpoints requerem scope rag:admin, consistente com /v1/embed existente.
O router não contém lógica de negócio — delega ao PipelineService.
A gestão de metadados MongoDB (create/update/list) é delegada ao store
da estrutura existente do colega para evitar duplicação.

Decisão: os endpoints de auditoria ficam neste router em /v1/pipeline/audit/*
para manter a coesão do módulo em vez de criar um router separado.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from rag_api.auth.middleware import require_scope
from rag_api.pipeline.models import AuditAction, AuditMeta, AuditOutcome, DocumentStatus
from rag_api.pipeline.schemas import (
    AuditLogListResponse,
    AuditLogResponse,
    DocumentListResponse,
    DocumentResponse,
    PipelineStatusResponse,
    PreviewResponse,
    ReindexResponse,
)
from rag_api.pipeline.service import PipelineService
from rag_api.schemas.identity import ClientIdentity

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/pipeline", tags=["pipeline"])

_DEFAULT_DPI = int(os.getenv("PDF_PREVIEW_DEFAULT_DPI", "150"))
_MAX_SIZE_BYTES = int(os.getenv("PDF_MAX_SIZE_MB", "50")) * 1024 * 1024


def _get_service(request: Request) -> PipelineService:
    """
    Recupera o PipelineService do estado da app.
    O service é registado em app.state no startup da aplicação.
    """
    return request.app.state.pipeline_service


def _get_audit_collection(request: Request):
    return request.app.state.audit_collection


def _get_doc_store(request: Request):
    """Store de metadados dos PDFs — estrutura existente do colega."""
    return request.app.state.pdf_doc_store


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

from fastapi import File, UploadFile


@router.post(
    "/documents",
    status_code=201,
    response_model=DocumentResponse,
    summary="Upload de um PDF e disparo de indexação",
)
async def upload_document(
    request:  Request,
    file:     UploadFile = File(...),
    identity: ClientIdentity = Depends(require_scope("rag:admin")),
) -> DocumentResponse:
    trace_id = request.state.trace_id
    service  = _get_service(request)
    store    = _get_doc_store(request)

    data = await file.read()

    if len(data) > _MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Ficheiro excede o limite de {os.getenv('PDF_MAX_SIZE_MB', '50')} MB.",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Apenas ficheiros PDF são aceites.")

    import hashlib
    checksum = hashlib.sha256(data).hexdigest()

    # Duplicado por checksum — evita re-indexação desnecessária.
    existing = await store.find_by_checksum(checksum)
    if existing and existing.get("status") != DocumentStatus.DELETED:
        raise HTTPException(
            status_code=409,
            detail=f"Ficheiro já existe: doc_id={existing['doc_id']}",
        )

    doc_id     = str(uuid.uuid4())
    ip_address = request.client.host if request.client else None

    # Calcular page_count antes de persistir — necessário para o preview.
    try:
        import fitz
        _pdf = fitz.open(stream=data, filetype="pdf")
        page_count = _pdf.page_count
        _pdf.close()
    except Exception:
        page_count = 0

    file_ref, _ = await service.upload(
        doc_id       = doc_id,
        filename     = file.filename,
        data         = data,
        performed_by = identity.client_id,
        trace_id     = trace_id,
        ip_address   = ip_address,
    )

    # Regista metadados — estrutura do colega.
    doc = await store.create(
        doc_id      = doc_id,
        filename    = file.filename,
        file_ref    = file_ref,
        file_size   = len(data),
        checksum    = checksum,
        uploaded_by = identity.client_id,
        page_count  = page_count,
    )

    # Indexação em background — não bloqueia o response.
    import asyncio
    asyncio.create_task(
        _run_reindex_background(
            service      = service,
            store        = store,
            doc          = doc,
            performed_by = identity.client_id,
            trace_id     = trace_id,
            ip_address   = ip_address,
        )
    )

    logger.info("upload_accepted", doc_id=doc_id, filename=file.filename, trace_id=trace_id)
    return DocumentResponse(**doc)


async def _run_reindex_background(
    service, store, doc, performed_by, trace_id, ip_address
) -> None:
    """
    Indexação assíncrona em background task.
    Actualiza o status no store independentemente do resultado.
    """
    doc_id = doc["doc_id"]
    await store.update_status(doc_id, DocumentStatus.PROCESSING)

    try:
        result = await service.reindex(
            doc_id         = doc_id,
            filename       = doc["filename"],
            file_ref       = doc["file_ref"],
            old_chroma_ids = [],
            version        = doc["version"],
            performed_by   = performed_by,
            trace_id       = trace_id,
            ip_address     = ip_address,
        )
        # Calcular page_count a partir do ficheiro real — garante que nunca fica a 0.
        try:
            import fitz
            _pdf       = fitz.open(doc["file_ref"])
            page_count = _pdf.page_count
            _pdf.close()
        except Exception:
            page_count = doc.get("page_count", 0)

        await store.update_after_index(
            doc_id      = doc_id,
            chroma_ids  = [],  # dívida técnica: lista real quando contrato com colega estiver definido
            chunk_count = result.chunk_count,
            status      = DocumentStatus.INDEXED,
            page_count  = page_count,
        )
    except Exception as exc:
        logger.error("background_reindex_failed", doc_id=doc_id, error=repr(exc))
        await store.update_status(doc_id, DocumentStatus.ERROR, error_detail=repr(exc))


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="Lista PDFs com paginação",
)
async def list_documents(
    request:   Request,
    page:      int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status:    Optional[str] = Query(None),
    identity:  ClientIdentity = Depends(require_scope("rag:admin")),
) -> DocumentListResponse:
    trace_id = request.state.trace_id
    store    = _get_doc_store(request)
    audit    = _get_service(request)._audit

    docs, total = await store.list(page=page, page_size=page_size, status=status)

    await audit.log_event(
        audit.build_event(
            action       = AuditAction.LIST,
            outcome      = AuditOutcome.SUCCESS,
            performed_by = identity.client_id,
            http_status  = 200,
            meta         = AuditMeta(
                trace_id   = trace_id,
                ip_address = request.client.host if request.client else None,
            ),
        )
    )

    return DocumentListResponse(
        items     = [DocumentResponse(**d) for d in docs],
        total     = total,
        page      = page,
        page_size = page_size,
    )


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------

@router.get(
    "/documents/{doc_id}",
    response_model=DocumentResponse,
    summary="Detalhe de um PDF",
)
async def get_document(
    doc_id:   str,
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:admin")),
) -> DocumentResponse:
    store = _get_doc_store(request)
    audit = _get_service(request)._audit

    doc = await store.get(doc_id)
    if not doc or doc.get("status") == DocumentStatus.DELETED:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    await audit.log_event(
        audit.build_event(
            action       = AuditAction.GET,
            outcome      = AuditOutcome.SUCCESS,
            performed_by = identity.client_id,
            http_status  = 200,
            doc_id       = doc_id,
            filename     = doc.get("filename"),
            meta         = AuditMeta(
                trace_id   = request.state.trace_id,
                ip_address = request.client.host if request.client else None,
            ),
        )
    )
    return DocumentResponse(**doc)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete(
    "/documents/{doc_id}",
    response_model=DocumentResponse,
    summary="Remove PDF e respectivos chunks do ChromaDB",
)
async def delete_document(
    doc_id:   str,
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:admin")),
) -> DocumentResponse:
    trace_id = request.state.trace_id
    service  = _get_service(request)
    store    = _get_doc_store(request)

    doc = await store.get(doc_id)
    if not doc or doc.get("status") == DocumentStatus.DELETED:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    await service.delete(
        doc_id       = doc_id,
        filename     = doc["filename"],
        file_ref     = doc["file_ref"],
        chroma_ids   = doc.get("chroma_ids", []),
        performed_by = identity.client_id,
        trace_id     = trace_id,
        ip_address   = request.client.host if request.client else None,
    )

    updated = await store.update_status(doc_id, DocumentStatus.DELETED)
    return DocumentResponse(**updated)


# ---------------------------------------------------------------------------
# Replace (PUT /documents/{doc_id}/file)
# ---------------------------------------------------------------------------

@router.put(
    "/documents/{doc_id}/file",
    response_model=DocumentResponse,
    summary="Substitui o ficheiro PDF e re-indexa",
)
async def replace_document(
    doc_id:   str,
    request:  Request,
    file:     UploadFile = File(...),
    identity: ClientIdentity = Depends(require_scope("rag:admin")),
) -> DocumentResponse:
    trace_id = request.state.trace_id
    service  = _get_service(request)
    store    = _get_doc_store(request)

    doc = await store.get(doc_id)
    if not doc or doc.get("status") == DocumentStatus.DELETED:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    data = await file.read()
    if len(data) > _MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Ficheiro excede o limite.")

    import hashlib
    new_checksum = hashlib.sha256(data).hexdigest()

    if new_checksum == doc.get("checksum"):
        raise HTTPException(status_code=409, detail="Ficheiro idêntico à versão actual.")

    ip_address = request.client.host if request.client else None

    # Guarda novo ficheiro.
    file_ref, _ = await service.upload(
        doc_id       = doc_id,
        filename     = file.filename or doc["filename"],
        data         = data,
        performed_by = identity.client_id,
        trace_id     = trace_id,
        ip_address   = ip_address,
    )

    # Remove ficheiro antigo do storage.
    await service._storage.delete(doc["file_ref"])

    new_version = doc.get("version", 1) + 1
    updated_doc = await store.update_file(
        doc_id       = doc_id,
        file_ref     = file_ref,
        checksum     = new_checksum,
        file_size    = len(data),
        filename     = file.filename or doc["filename"],
        version      = new_version,
    )

    import asyncio
    asyncio.create_task(
        _run_reindex_background(
            service      = service,
            store        = store,
            doc          = updated_doc,
            performed_by = identity.client_id,
            trace_id     = trace_id,
            ip_address   = ip_address,
        )
    )

    await service._audit.log_event(
        service._audit.build_event(
            action       = AuditAction.REPLACE,
            outcome      = AuditOutcome.SUCCESS,
            performed_by = identity.client_id,
            http_status  = 200,
            doc_id       = doc_id,
            filename     = updated_doc["filename"],
            meta         = AuditMeta(trace_id=trace_id, ip_address=ip_address),
        )
    )

    return DocumentResponse(**updated_doc)


# ---------------------------------------------------------------------------
# Reindex manual
# ---------------------------------------------------------------------------

@router.post(
    "/documents/{doc_id}/reindex",
    response_model=ReindexResponse,
    summary="Re-indexa manualmente um documento no ChromaDB",
)
async def reindex_document(
    doc_id:   str,
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:admin")),
) -> ReindexResponse:
    trace_id = request.state.trace_id
    service  = _get_service(request)
    store    = _get_doc_store(request)

    doc = await store.get(doc_id)
    if not doc or doc.get("status") == DocumentStatus.DELETED:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    result = await service.reindex(
        doc_id         = doc_id,
        filename       = doc["filename"],
        file_ref       = doc["file_ref"],
        old_chroma_ids = doc.get("chroma_ids", []),
        version        = doc.get("version", 1),
        performed_by   = identity.client_id,
        trace_id       = trace_id,
        ip_address     = request.client.host if request.client else None,
    )

    new_version = doc.get("version", 1) + 1
    await store.update_after_index(
        doc_id      = doc_id,
        chroma_ids  = [],  # dívida técnica: lista real quando contrato com colega estiver definido
        chunk_count = result.chunk_count,
        status      = DocumentStatus.INDEXED,
    )

    return ReindexResponse(
        doc_id          = doc_id,
        version         = new_version,
        pipeline_result = result,
    )


# ---------------------------------------------------------------------------
# Preview de página
# ---------------------------------------------------------------------------

@router.get(
    "/documents/{doc_id}/pages/{page_number}/preview",
    summary="Renderiza uma página do PDF como imagem PNG",
    responses={200: {"content": {"image/png": {}}}},
)
async def preview_page(
    doc_id:      str,
    page_number: int,
    request:     Request,
    dpi:         int = Query(_DEFAULT_DPI, ge=72, le=300),
    identity:    ClientIdentity = Depends(require_scope("rag:admin")),
):
    trace_id = request.state.trace_id
    service  = _get_service(request)
    store    = _get_doc_store(request)

    doc = await store.get(doc_id)
    if not doc or doc.get("status") == DocumentStatus.DELETED:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    if doc.get("status") != DocumentStatus.INDEXED:
        raise HTTPException(
            status_code=409,
            detail=f"Documento com status '{doc['status']}' — preview indisponível.",
        )

    try:
        png_bytes = await service.render_page(
            doc_id       = doc_id,
            filename     = doc["filename"],
            file_ref     = doc["file_ref"],
            page_number  = page_number,
            page_count   = doc.get("page_count", 9999),
            dpi          = dpi,
            performed_by = identity.client_id,
            trace_id     = trace_id,
            ip_address   = request.client.host if request.client else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    import hashlib
    etag = hashlib.md5(
        f"{doc.get('checksum', '')}:{page_number}:{dpi}".encode()
    ).hexdigest()

    return StreamingResponse(
        iter([png_bytes]),
        media_type="image/png",
        headers={
            "ETag":          f'"{etag}"',
            "Cache-Control": "private, max-age=3600",
            "X-Doc-Id":      doc_id,
            "X-Page-Number": str(page_number),
            "X-Page-Count":  str(doc.get("page_count", "?")),
        },
    )


# ---------------------------------------------------------------------------
# Status do índice
# ---------------------------------------------------------------------------

@router.get(
    "/status",
    response_model=PipelineStatusResponse,
    summary="Estado actual do índice de documentos",
)
async def pipeline_status(
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:admin")),
) -> PipelineStatusResponse:
    store = _get_doc_store(request)
    stats = await store.status_counts()
    return PipelineStatusResponse(**stats)


# ---------------------------------------------------------------------------
# Auditoria
# ---------------------------------------------------------------------------

@router.get(
    "/audit",
    response_model=AuditLogListResponse,
    summary="Lista o log de auditoria com paginação",
)
async def list_audit_logs(
    request:   Request,
    page:      int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    doc_id:    Optional[str] = Query(None),
    action:    Optional[str] = Query(None),
    outcome:   Optional[str] = Query(None),
    identity:  ClientIdentity = Depends(require_scope("rag:admin")),
) -> AuditLogListResponse:
    col = _get_audit_collection(request)

    query: dict = {}
    if doc_id:
        query["doc_id"] = doc_id
    if action:
        query["action"] = action
    if outcome:
        query["outcome"] = outcome

    skip  = (page - 1) * page_size
    total = await col.count_documents(query)
    cursor = col.find(query).sort("timestamp", -1).skip(skip).limit(page_size)
    docs   = await cursor.to_list(length=page_size)

    return AuditLogListResponse(
        items     = [AuditLogResponse(**d) for d in docs],
        total     = total,
        page      = page,
        page_size = page_size,
    )