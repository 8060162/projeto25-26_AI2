"""
Logger de auditoria append-only para operações sobre PDFs.

Garantia de imutabilidade por disciplina de código:
- Esta classe só expõe log_event() — nunca update nem delete.
- A colecção audit_logs não deve ter índices TTL nem policies de expiração.

Best-effort: falhas de persistência são registadas mas nunca propagadas —
o endpoint principal não deve falhar por uma falha de auditoria.
Padrão consistente com MongoSignalStore nos signals do Módulo 3.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog

from rag_api.pipeline.models import (
    AuditAction,
    AuditEvent,
    AuditMeta,
    AuditOutcome,
    PipelineResult,
)

logger = structlog.get_logger(__name__)


class PipelineAuditLogger:
    """
    Persiste eventos de auditoria na colecção MongoDB audit_logs.

    Recebe a collection via injecção — testável com mock sem MongoDB real.
    """

    def __init__(self, collection) -> None:
        self._col = collection

    async def log_event(self, event: AuditEvent) -> None:
        """Insere um evento. Nunca levanta excepção para o caller."""
        try:
            await self._col.insert_one(event.model_dump())
        except Exception as exc:
            # Auditoria não pode derrubar o endpoint principal.
            logger.error("audit_log_failed", error=repr(exc), event_id=event.event_id)

    def build_event(
        self,
        *,
        action:          AuditAction,
        outcome:         AuditOutcome,
        performed_by:    str,
        http_status:     int,
        doc_id:          str | None          = None,
        filename:        str | None          = None,
        error_detail:    str | None          = None,
        pipeline_result: PipelineResult | None = None,
        meta:            AuditMeta | None    = None,
    ) -> AuditEvent:
        """
        Factory de AuditEvent — centraliza a geração de event_id e timestamp.
        Garante que todos os eventos têm o mesmo formato independentemente do caller.
        """
        return AuditEvent(
            event_id        = str(uuid.uuid4()),
            timestamp       = datetime.now(timezone.utc),
            action          = action,
            outcome         = outcome,
            performed_by    = performed_by,
            http_status     = http_status,
            doc_id          = doc_id,
            filename        = filename,
            error_detail    = error_detail,
            pipeline_result = pipeline_result,
            meta            = meta or AuditMeta(),
        )