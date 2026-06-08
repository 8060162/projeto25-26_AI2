"""
Testes do PipelineService.

Todas as dependências são mocks injectados — sem I/O real.
Verifica orquestração, auditoria e comportamento em erro.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_api.pipeline.audit import PipelineAuditLogger
from rag_api.pipeline.chroma import ChromaPipelineClient
from rag_api.pipeline.models import AuditAction, AuditOutcome, DocumentStatus
from rag_api.pipeline.preview import PagePreviewRenderer, PagePreview
from rag_api.pipeline.service import PipelineService


@pytest.fixture
def mock_storage():
    s = AsyncMock()
    s.save   = AsyncMock(return_value="/data/pdfs/doc-1_file.pdf")
    s.read   = AsyncMock(return_value=b"%PDF-1.4 fake")
    s.delete = AsyncMock()
    s.exists = AsyncMock(return_value=True)
    return s


@pytest.fixture
def mock_chroma():
    c = AsyncMock()
    c.delete_chunks = AsyncMock()
    return c


@pytest.fixture
def mock_preview():
    p = AsyncMock()
    p.render = AsyncMock(
        return_value=PagePreview(png_bytes=b"\x89PNG", page_count=5, dpi=150)
    )
    return p


@pytest.fixture
def mock_audit():
    a = MagicMock(spec=PipelineAuditLogger)
    a.log_event  = AsyncMock()
    a.build_event = MagicMock(return_value=MagicMock())
    return a


@pytest.fixture
def service(mock_storage, mock_chroma, mock_preview, mock_audit):
    return PipelineService(
        storage = mock_storage,
        chroma  = mock_chroma,
        preview = mock_preview,
        audit   = mock_audit,
    )


class TestUpload:

    @pytest.mark.asyncio
    async def test_returns_file_ref_and_checksum(self, service, mock_storage):
        file_ref, checksum = await service.upload(
            doc_id="d1", filename="f.pdf", data=b"content",
            performed_by="c1", trace_id="t1",
        )
        assert file_ref == "/data/pdfs/doc-1_file.pdf"
        assert len(checksum) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_audit_logged_on_upload(self, service, mock_audit):
        await service.upload(
            doc_id="d1", filename="f.pdf", data=b"x",
            performed_by="c1", trace_id="t1",
        )
        mock_audit.log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_checksum_is_sha256(self, service):
        import hashlib
        data = b"test content"
        _, checksum = await service.upload(
            doc_id="d1", filename="f.pdf", data=data,
            performed_by="c1", trace_id="t1",
        )
        assert checksum == hashlib.sha256(data).hexdigest()


class TestDelete:

    @pytest.mark.asyncio
    async def test_deletes_chroma_chunks(self, service, mock_chroma):
        await service.delete(
            doc_id="d1", filename="f.pdf", file_ref="/path/f.pdf",
            chroma_ids=["id1", "id2"],
            performed_by="c1", trace_id="t1",
        )
        mock_chroma.delete_chunks.assert_called_once_with(["id1", "id2"])

    @pytest.mark.asyncio
    async def test_deletes_from_storage(self, service, mock_storage):
        await service.delete(
            doc_id="d1", filename="f.pdf", file_ref="/path/f.pdf",
            chroma_ids=[],
            performed_by="c1", trace_id="t1",
        )
        mock_storage.delete.assert_called_once_with("/path/f.pdf")

    @pytest.mark.asyncio
    async def test_audit_logged_on_delete(self, service, mock_audit):
        await service.delete(
            doc_id="d1", filename="f.pdf", file_ref="/path/f.pdf",
            chroma_ids=[],
            performed_by="c1", trace_id="t1",
        )
        mock_audit.log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_chroma_ids_does_not_call_chroma(self, service, mock_chroma):
        await service.delete(
            doc_id="d1", filename="f.pdf", file_ref="/path/f.pdf",
            chroma_ids=[],
            performed_by="c1", trace_id="t1",
        )
        mock_chroma.delete_chunks.assert_called_once_with([])


class TestRenderPage:

    @pytest.mark.asyncio
    async def test_returns_png_bytes(self, service):
        result = await service.render_page(
            doc_id="d1", filename="f.pdf", file_ref="/path/f.pdf",
            page_number=1, page_count=5, dpi=150,
            performed_by="c1", trace_id="t1",
        )
        assert result == b"\x89PNG"

    @pytest.mark.asyncio
    async def test_invalid_page_raises_value_error(self, service):
        with pytest.raises(ValueError, match="inválida"):
            await service.render_page(
                doc_id="d1", filename="f.pdf", file_ref="/path/f.pdf",
                page_number=6, page_count=5, dpi=150,
                performed_by="c1", trace_id="t1",
            )

    @pytest.mark.asyncio
    async def test_page_zero_raises_value_error(self, service):
        with pytest.raises(ValueError):
            await service.render_page(
                doc_id="d1", filename="f.pdf", file_ref="/path/f.pdf",
                page_number=0, page_count=5, dpi=150,
                performed_by="c1", trace_id="t1",
            )

    @pytest.mark.asyncio
    async def test_reads_pdf_from_storage(self, service, mock_storage):
        await service.render_page(
            doc_id="d1", filename="f.pdf", file_ref="/path/f.pdf",
            page_number=1, page_count=5, dpi=150,
            performed_by="c1", trace_id="t1",
        )
        mock_storage.read.assert_called_once_with("/path/f.pdf")

    @pytest.mark.asyncio
    async def test_audit_logged_on_preview(self, service, mock_audit):
        await service.render_page(
            doc_id="d1", filename="f.pdf", file_ref="/path/f.pdf",
            page_number=1, page_count=5, dpi=150,
            performed_by="c1", trace_id="t1",
        )
        mock_audit.log_event.assert_called_once()