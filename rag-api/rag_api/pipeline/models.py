"""
Modelos de domínio do módulo pipeline.

Separados dos schemas HTTP (schemas.py) — os modelos de domínio representam
o estado interno persistido; os schemas representam o contrato da API.
Manter esta separação evita que alterações na API afectem a camada de persistência.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    INDEXED    = "indexed"
    ERROR      = "error"
    DELETED    = "deleted"


class PipelineResult(BaseModel):
    """Resultado da indexação — preenchido após run_embedding_indexer."""
    chunk_count:      int
    chroma_ids_count: int
    status:           DocumentStatus


class AuditAction(str, Enum):
    UPLOAD  = "upload"
    DELETE  = "delete"
    REPLACE = "replace"
    REINDEX = "reindex"
    PREVIEW = "preview"
    LIST    = "list"
    GET     = "get"


class AuditOutcome(str, Enum):
    SUCCESS = "success"
    ERROR   = "error"


class AuditMeta(BaseModel):
    ip_address: Optional[str] = None
    trace_id:   Optional[str] = None
    dpi:        Optional[int] = None   # preenchido apenas em preview


class AuditEvent(BaseModel):
    """
    Registo imutável de uma acção sobre um ficheiro PDF.

    Nunca actualizado após inserção — append-only por disciplina de código.
    O campo event_id permite correlacionar com trace_id do request.
    """
    event_id:        str
    timestamp:       datetime
    action:          AuditAction
    outcome:         AuditOutcome
    doc_id:          Optional[str]           = None
    filename:        Optional[str]           = None
    performed_by:    str
    http_status:     int
    error_detail:    Optional[str]           = None
    pipeline_result: Optional[PipelineResult] = None
    meta:            AuditMeta               = Field(default_factory=AuditMeta)