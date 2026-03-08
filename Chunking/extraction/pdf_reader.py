from __future__ import annotations

from pathlib import Path
from typing import List

from Chunking.chunking.models import PageText

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyMuPDF is required. Install package 'pymupdf'."
    ) from exc


class PdfReader:
    """
    Lightweight PDF extractor.

    Why page-level extraction first?
    - It preserves page boundaries for metadata.
    - It allows later cleanup of page-specific noise.
    - It avoids losing traceability during normalization.
    """

    def extract_pages(self, pdf_path: Path) -> List[PageText]:
        pages: List[PageText] = []
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                # We intentionally use a simple text extraction mode first.
                # If later you find layout problems in certain PDFs, this can be
                # replaced with block-based extraction or OCR as a fallback.
                text = page.get_text("text")
                pages.append(PageText(page_number=page_index, text=text or ""))
        return pages
