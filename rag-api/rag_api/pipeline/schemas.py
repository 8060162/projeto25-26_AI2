"""
Schemas HTTP do módulo pipeline.

Separados dos modelos de domínio — alterações no contrato da API
não propagam para a camada de persistência e vice-versa.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from rag_api.pipeline.models import DocumentStatus, PipelineResult


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    """Representa um PDF tal como exposto pela API."""
    doc_id:          str
    filename:        str
    file_size:       int
    checksum:        str
    status:          DocumentStatus
    version:         int
    chunk_count:     int
    page_count:      int
    uploaded_by:     str
    uploaded_at:     datetime
    indexed_at:      Optional[datetime]
    updated_at:      datetime
    error_detail:    Optional[str] = None


class DocumentListResponse(BaseModel):
    items:       list[DocumentResponse]
    total:       int
    page:        int
    page_size:   int


class ReindexResponse(BaseModel):
    doc_id:          str
    version:         int
    pipeline_result: Optional[PipelineResult]


class PipelineStatusResponse(BaseModel):
    """Resumo do estado do índice — útil para dashboard admin."""
    total_documents:     int
    indexed:             int
    pending:             int
    processing:          int
    error:               int
    deleted:             int
    total_chunks:        int


class PreviewResponse(BaseModel):
    """
    Metadata do preview — a imagem PNG é devolvida como StreamingResponse,
    não encapsulada aqui. Este schema é usado apenas nos testes e na documentação.
    """
    doc_id:      str
    page_number: int
    page_count:  int
    dpi:         int


class AuditLogResponse(BaseModel):
    event_id:     str
    timestamp:    datetime
    action:       str
    outcome:      str
    doc_id:       Optional[str]
    filename:     Optional[str]
    performed_by: str
    http_status:  int
    error_detail: Optional[str]


class AuditLogListResponse(BaseModel):
    items:     list[AuditLogResponse]
    total:     int
    page:      int
    page_size: int


# ---------------------------------------------------------------------------
# Embed (pipeline.py existente — SSOT centralizado aqui)
# ---------------------------------------------------------------------------

class EmbedResponse(BaseModel):
    run_id:                str
    input_record_count:    int
    embedded_record_count: int
    records_path:          str
    manifest_path:         str