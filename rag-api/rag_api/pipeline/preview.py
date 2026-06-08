"""
Renderização de páginas PDF para PNG.

PyMuPDF (fitz) escolhido pela performance e zero dependências externas
(não requer Ghostscript ao contrário de pdf2image).
A renderização é síncrona — envolta em executor para não bloquear o event loop.
"""
from __future__ import annotations

import asyncio
import os
from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_DPI = int(os.getenv("PDF_PREVIEW_DEFAULT_DPI", "150"))
_MAX_DPI     = int(os.getenv("PDF_PREVIEW_MAX_DPI", "300"))


class PagePreview(NamedTuple):
    png_bytes:  bytes
    page_count: int
    dpi:        int


class PagePreviewRenderer:
    """
    Renderiza uma página de um PDF para PNG em memória.

    Não persiste — devolve bytes directamente para StreamingResponse.
    O ETag deve ser calculado pelo caller com base em checksum + page_number.
    """

    async def render(
        self,
        pdf_bytes: bytes,
        page_number: int,
        dpi: int = _DEFAULT_DPI,
    ) -> PagePreview:
        """
        page_number é 1-based — mais natural para utilizadores e citações académicas.
        A validação contra page_count é responsabilidade do service.py.
        """
        effective_dpi = min(dpi, _MAX_DPI)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._render_sync,
            pdf_bytes,
            page_number,
            effective_dpi,
        )
        return result

    @staticmethod
    def _render_sync(pdf_bytes: bytes, page_number: int, dpi: int) -> PagePreview:
        import fitz  # PyMuPDF — importação lazy para não falhar se não instalado

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_count = doc.page_count
            # fitz usa índice 0-based internamente
            page = doc[page_number - 1]
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pixmap.tobytes("png")
        finally:
            doc.close()

        logger.debug(
            "page_rendered",
            page_number=page_number,
            dpi=dpi,
            size_kb=len(png_bytes) // 1024,
        )
        return PagePreview(png_bytes=png_bytes, page_count=page_count, dpi=dpi)