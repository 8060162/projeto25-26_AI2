"""
Testes do logger de auditoria.

Verifica o comportamento append-only e a resiliência a falhas de persistência.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_api.pipeline.audit import PipelineAuditLogger
from rag_api.pipeline.models import AuditAction, AuditEvent, AuditMeta, AuditOutcome


@pytest.fixture
def mock_collection():
    col = AsyncMock()
    col.insert_one = AsyncMock(return_value=MagicMock(inserted_id="abc"))
    return col


@pytest.fixture
def audit_logger(mock_collection):
    return PipelineAuditLogger(collection=mock_collection)


class TestPipelineAuditLogger:

    @pytest.mark.asyncio
    async def test_log_event_calls_insert_once(self, audit_logger, mock_collection):
        event = audit_logger.build_event(
            action       = AuditAction.UPLOAD,
            outcome      = AuditOutcome.SUCCESS,
            performed_by = "client-1",
            http_status  = 201,
            doc_id       = "doc-1",
            filename     = "regulamento.pdf",
        )
        await audit_logger.log_event(event)
        mock_collection.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_event_is_best_effort_on_failure(self, audit_logger, mock_collection):
        """Falha de persistência não deve propagar para o caller."""
        mock_collection.insert_one.side_effect = Exception("MongoDB indisponível")
        event = audit_logger.build_event(
            action       = AuditAction.DELETE,
            outcome      = AuditOutcome.SUCCESS,
            performed_by = "client-1",
            http_status  = 200,
        )
        # Não deve levantar excepção
        await audit_logger.log_event(event)

    def test_build_event_generates_unique_ids(self, audit_logger):
        e1 = audit_logger.build_event(
            action=AuditAction.GET, outcome=AuditOutcome.SUCCESS,
            performed_by="c1", http_status=200,
        )
        e2 = audit_logger.build_event(
            action=AuditAction.GET, outcome=AuditOutcome.SUCCESS,
            performed_by="c1", http_status=200,
        )
        assert e1.event_id != e2.event_id

    def test_build_event_timestamp_is_utc(self, audit_logger):
        event = audit_logger.build_event(
            action=AuditAction.UPLOAD, outcome=AuditOutcome.SUCCESS,
            performed_by="c1", http_status=201,
        )
        assert event.timestamp.tzinfo == timezone.utc

    def test_build_event_with_all_fields(self, audit_logger):
        from rag_api.pipeline.models import DocumentStatus, PipelineResult
        result = PipelineResult(
            chunk_count=10, chroma_ids_count=10, status=DocumentStatus.INDEXED
        )
        event = audit_logger.build_event(
            action          = AuditAction.REINDEX,
            outcome         = AuditOutcome.SUCCESS,
            performed_by    = "admin-client",
            http_status     = 200,
            doc_id          = "doc-abc",
            filename        = "regulamento_geral.pdf",
            pipeline_result = result,
            meta            = AuditMeta(trace_id="trace-123", dpi=150),
        )
        assert event.pipeline_result.chunk_count == 10
        assert event.meta.trace_id == "trace-123"
        assert event.meta.dpi == 150

    def test_build_event_error_outcome(self, audit_logger):
        event = audit_logger.build_event(
            action       = AuditAction.REINDEX,
            outcome      = AuditOutcome.ERROR,
            performed_by = "c1",
            http_status  = 500,
            error_detail = "IndexError: out of range",
        )
        assert event.outcome == AuditOutcome.ERROR
        assert "IndexError" in event.error_detail

    @pytest.mark.asyncio
    async def test_insert_payload_contains_required_fields(self, audit_logger, mock_collection):
        event = audit_logger.build_event(
            action       = AuditAction.PREVIEW,
            outcome      = AuditOutcome.SUCCESS,
            performed_by = "c2",
            http_status  = 200,
            doc_id       = "doc-xyz",
            meta         = AuditMeta(dpi=150),
        )
        await audit_logger.log_event(event)
        inserted = mock_collection.insert_one.call_args[0][0]
        assert "event_id" in inserted
        assert "timestamp" in inserted
        assert inserted["action"] == AuditAction.PREVIEW
        assert inserted["meta"]["dpi"] == 150