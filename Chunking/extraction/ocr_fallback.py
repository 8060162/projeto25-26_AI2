from __future__ import annotations

import io
from pathlib import Path
from typing import List

from PIL import Image, ImageOps

from Chunking.chunking.models import (
    ExtractedBlock,
    ExtractedDocument,
    ExtractedPage,
    PageText,
)

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyMuPDF is required. Install package 'pymupdf'."
    ) from exc

try:
    import pytesseract
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "pytesseract is required for OCR fallback. Install package 'pytesseract'."
    ) from exc


class OcrFallbackReader:
    """
    OCR-based fallback extractor for problematic PDFs.

    Purpose
    -------
    Some PDFs visually render correctly but contain broken, incomplete, or
    semantically unusable text layers. In those cases, normal text extraction
    may return corrupted content even though the document looks fine to a human.

    This component exists to provide a fallback path:
        PDF page -> rendered image -> OCR -> structured page output

    Design goals
    ------------
    1. Remain page-based for traceability
    2. Integrate with the same intermediate extraction model used by PdfReader
    3. Be explicit and easy to debug
    4. Be used only as a fallback, not as the primary extraction path

    Important architectural note
    ----------------------------
    This class should not interpret legal structure, identify articles, or
    build the final JSON tree. Its job is only to recover readable page text
    when normal extraction is unreliable.

    OCR language
    ------------
    By default this reader uses Portuguese OCR via "por", which is the correct
    default for the target regulation documents.
    """

    def __init__(
        self,
        dpi: int = 300,
        tesseract_lang: str = "por",
    ) -> None:
        """
        Initialize the OCR fallback reader.

        Parameters
        ----------
        dpi : int, default=300
            Rendering DPI used when converting PDF pages into images.

            Why 300 DPI?
            ------------
            This is a practical balance between:
            - OCR readability
            - processing time
            - memory usage

        tesseract_lang : str, default="por"
            Tesseract OCR language code.
        """
        self.dpi = dpi
        self.tesseract_lang = tesseract_lang

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_document(self, pdf_path: Path) -> ExtractedDocument:
        """
        Extract a PDF through OCR into a structured intermediate document model.

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file that should be processed through OCR.

        Returns
        -------
        ExtractedDocument
            OCR-based structured extraction result.

        Why this method matters
        -----------------------
        The project is moving away from "plain text only" extraction and toward
        a richer intermediate representation that will later be transformed into
        the canonical JSON tree. This method aligns OCR fallback with that goal.
        """
        pages: List[ExtractedPage] = []

        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                image = self._render_page_to_image(page)
                text = self._ocr_image(image)

                # OCR does not naturally preserve internal PDF blocks/lines in
                # the same way native PDF extraction does. For now, we expose
                # the full OCR text as a single synthetic block so the rest of
                # the pipeline can still work with a structured page model.
                synthetic_blocks: List[ExtractedBlock] = []
                if text.strip():
                    synthetic_blocks.append(
                        ExtractedBlock(
                            block_index=0,
                            text=text.strip(),
                            bbox=None,
                            lines=[],
                            source_mode="ocr",
                        )
                    )

                pages.append(
                    ExtractedPage(
                        page_number=page_index,
                        text=text.strip(),
                        selected_mode="ocr",
                        quality_score=self._score_ocr_text(text),
                        blocks=synthetic_blocks,
                        corruption_flags=self._detect_ocr_flags(text),
                    )
                )

        return ExtractedDocument(
            source_path=str(pdf_path),
            page_count=len(pages),
            pages=pages,
        )

    def extract_pages(self, pdf_path: Path) -> List[PageText]:
        """
        Legacy compatibility adapter.

        This method preserves the older page-level contract used by earlier
        pipeline stages. New extraction/parsing flows should prefer
        `extract_document()`.

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file.

        Returns
        -------
        List[PageText]
            OCR page text in the legacy lightweight model.
        """
        document = self.extract_document(pdf_path)

        return [
            PageText(
                page_number=page.page_number,
                text=page.text,
            )
            for page in document.pages
        ]

    # ------------------------------------------------------------------
    # OCR internals
    # ------------------------------------------------------------------

    def _render_page_to_image(self, page: fitz.Page) -> Image.Image:
        """
        Render one PDF page into a raster image suitable for OCR.

        Parameters
        ----------
        page : fitz.Page
            PyMuPDF page object.

        Returns
        -------
        Image.Image
            Pillow image ready for OCR.

        Why rendering is needed
        -----------------------
        OCR engines operate on raster images, not on vector PDF content.
        Therefore each page must first be rendered into an image.
        """
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_bytes = pixmap.tobytes("png")

        return Image.open(io.BytesIO(image_bytes))

    def _ocr_image(self, image: Image.Image) -> str:
        """
        Run OCR over one rendered page image.

        Parameters
        ----------
        image : Image.Image
            Pillow image for one page.

        Returns
        -------
        str
            OCR-extracted text.

        Why this helper is isolated
        ---------------------------
        Keeping OCR invocation in a dedicated method makes it easier to:
        - tune OCR configuration
        - add preprocessing later
        - debug OCR-specific issues
        """
        prepared_image = self._prepare_image_for_ocr(image)

        # A conservative OCR configuration is used here. It can be tuned later
        # if the documents reveal recurring layout patterns that benefit from a
        # specific page segmentation mode.
        text = pytesseract.image_to_string(
            prepared_image,
            lang=self.tesseract_lang,
        )

        return text.strip()

    def _prepare_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Apply lightweight preprocessing before OCR.

        Parameters
        ----------
        image : Image.Image
            Original rendered page image.

        Returns
        -------
        Image.Image
            Prepared image for OCR.

        Current preprocessing strategy
        ------------------------------
        This version stays intentionally conservative:
        - convert to grayscale
        - autocontrast

        Why stay conservative?
        ----------------------
        Over-aggressive preprocessing can easily damage diacritics, punctuation,
        or thin characters in legal PDFs, making OCR worse rather than better.
        """
        grayscale = image.convert("L")
        return ImageOps.autocontrast(grayscale)

    # ------------------------------------------------------------------
    # OCR diagnostics
    # ------------------------------------------------------------------

    def _score_ocr_text(self, text: str) -> float:
        """
        Score OCR output heuristically.

        Parameters
        ----------
        text : str
            OCR-extracted text.

        Returns
        -------
        float
            A practical quality score where larger is better.

        Notes
        -----
        OCR text should be evaluated differently from native PDF extraction.
        We still reward:
        - useful length
        - alphabetic content
        - whitespace
        - legal/regulatory markers

        And penalize:
        - excessive symbol noise
        - extremely low alphabetic ratio
        """
        if not text or not text.strip():
            return -1_000_000.0

        stripped = text.strip()
        total_len = len(stripped)

        alpha_count = sum(1 for ch in stripped if ch.isalpha())
        digit_count = sum(1 for ch in stripped if ch.isdigit())
        whitespace_count = sum(1 for ch in stripped if ch.isspace())

        suspicious_chars = {"�", "￾", "", "*", "^", "_", "`", "~", "\\", "|", "<", ">"}
        suspicious_count = sum(1 for ch in stripped if ch in suspicious_chars)

        alpha_ratio = alpha_count / max(total_len, 1)
        whitespace_ratio = whitespace_count / max(total_len, 1)
        suspicious_ratio = suspicious_count / max(total_len, 1)

        score = 0.0
        score += min(total_len, 4000) * 0.01
        score += alpha_ratio * 100.0
        score += whitespace_ratio * 20.0

        if digit_count / max(total_len, 1) > 0.35:
            score -= 20.0

        score -= suspicious_ratio * 180.0

        if alpha_ratio < 0.30:
            score -= 80.0

        lower_text = stripped.lower()
        legal_markers = [
            "artigo",
            "capítulo",
            "capitulo",
            "anexo",
            "regulamento",
            "despacho",
        ]
        marker_hits = sum(1 for marker in legal_markers if marker in lower_text)
        score += marker_hits * 8.0

        return score

    def _detect_ocr_flags(self, text: str) -> List[str]:
        """
        Detect lightweight OCR-specific quality flags.

        Parameters
        ----------
        text : str
            OCR-extracted page text.

        Returns
        -------
        List[str]
            A list of quality/corruption flags.

        Why these flags matter
        ----------------------
        Downstream logic may use them to:
        - trigger warnings
        - mark pages as low-confidence
        - decide whether to attempt extra normalization
        """
        flags: List[str] = []

        if not text or not text.strip():
            flags.append("empty_text")
            return flags

        stripped = text.strip()
        total_len = len(stripped)

        replacement_like_chars = {"�", "￾", ""}
        replacement_count = sum(1 for ch in stripped if ch in replacement_like_chars)
        if replacement_count > 0:
            flags.append("replacement_like_characters")

        suspicious_symbol_chars = {"*", "^", "_", "`", "~", "\\", "|", "<", ">"}
        suspicious_symbol_count = sum(1 for ch in stripped if ch in suspicious_symbol_chars)
        if suspicious_symbol_count / max(total_len, 1) > 0.03:
            flags.append("high_suspicious_symbol_density")

        alpha_ratio = sum(1 for ch in stripped if ch.isalpha()) / max(total_len, 1)
        if alpha_ratio < 0.25:
            flags.append("low_alpha_ratio")

        words = [word for word in stripped.split() if word]
        if words:
            single_char_ratio = sum(1 for word in words if len(word) == 1) / len(words)
            if single_char_ratio > 0.35:
                flags.append("excessive_single_character_tokens")

        return flags