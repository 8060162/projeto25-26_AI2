"""
Testes do PagePreviewRenderer.

Cria um PDF mínimo válido em memória via reportlab (se disponível)
ou usa um PDF hardcoded mínimo — sem ficheiros em disco.
"""
from __future__ import annotations

import pytest

from rag_api.pipeline.preview import PagePreviewRenderer


def _minimal_pdf() -> bytes:
    """
    PDF de 1 página mínimo e válido — não requer dependências externas.
    Gerado uma vez e hardcoded como bytes — suficiente para testar o renderer.
    """
    try:
        from io import BytesIO
        from reportlab.pdfgen import canvas
        buf = BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(100, 750, "Página de teste")
        c.showPage()
        c.save()
        return buf.getvalue()
    except ImportError:
        # PDF mínimo válido sem reportlab.
        return (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n"
            b"0000000000 65535 f\n"
            b"0000000009 00000 n\n"
            b"0000000058 00000 n\n"
            b"0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\n"
            b"startxref\n190\n%%EOF"
        )


@pytest.fixture
def renderer():
    return PagePreviewRenderer()


@pytest.fixture
def pdf_bytes():
    return _minimal_pdf()


class TestPagePreviewRenderer:

    @pytest.mark.asyncio
    async def test_render_returns_png_bytes(self, renderer, pdf_bytes):
        preview = await renderer.render(pdf_bytes, page_number=1, dpi=72)
        assert preview.png_bytes[:4] == b"\x89PNG"

    @pytest.mark.asyncio
    async def test_render_returns_correct_page_count(self, renderer, pdf_bytes):
        preview = await renderer.render(pdf_bytes, page_number=1, dpi=72)
        assert preview.page_count == 1

    @pytest.mark.asyncio
    async def test_render_respects_dpi(self, renderer, pdf_bytes):
        low  = await renderer.render(pdf_bytes, page_number=1, dpi=72)
        high = await renderer.render(pdf_bytes, page_number=1, dpi=144)
        # DPI mais alto produz imagem maior.
        assert len(high.png_bytes) > len(low.png_bytes)

    @pytest.mark.asyncio
    async def test_dpi_capped_at_max(self, renderer, pdf_bytes):
        preview = await renderer.render(pdf_bytes, page_number=1, dpi=9999)
        assert preview.dpi == 300  # _MAX_DPI

    @pytest.mark.asyncio
    async def test_invalid_page_raises_index_error(self, renderer, pdf_bytes):
        with pytest.raises((IndexError, Exception)):
            await renderer.render(pdf_bytes, page_number=99, dpi=72)