from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

from Chunking.chunking.models import (
    BoundingBox,
    ExtractedBlock,
    ExtractedDocument,
    ExtractedLine,
    ExtractedPage,
    PageExtractionCandidate,
    PageText,
)

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyMuPDF is required. Install package 'pymupdf'."
    ) from exc


class PdfReader:
    """
    Robust PDF extraction component for the first stage of the pipeline.

    Design intent
    -------------
    This class is no longer designed as a plain-text extractor only.
    Instead, it acts as an intermediate structure-preserving PDF reader.

    Core responsibilities
    ---------------------
    1. Extract content page by page
    2. Try multiple extraction modes
    3. Score extraction quality heuristically
    4. Preserve page/block/line structure when possible
    5. Provide a strong intermediate representation for downstream parsing

    Important non-responsibilities
    ------------------------------
    This class does NOT:
    - perform OCR
    - interpret legal structure semantically
    - generate final chunks
    - build the final master JSON tree

    Those concerns belong to later pipeline stages.
    """

    def extract_document(self, pdf_path: Path) -> ExtractedDocument:
        """
        Extract a PDF into a rich intermediate document representation.

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file.

        Returns
        -------
        ExtractedDocument
            A structured extraction result preserving pages, blocks, lines,
            extraction modes, and quality signals.
        """
        pages: List[ExtractedPage] = []

        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                best_candidate = self._extract_best_page_candidate(page)

                pages.append(
                    ExtractedPage(
                        page_number=page_index,
                        text=best_candidate.text or "",
                        selected_mode=best_candidate.source_mode,
                        quality_score=best_candidate.quality_score,
                        blocks=best_candidate.blocks,
                        corruption_flags=best_candidate.corruption_flags,
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

        Returns only page-level plain text for older parts of the pipeline.

        Important note
        --------------
        New parsing logic should prefer `extract_document()`.
        """
        document = self.extract_document(pdf_path)

        return [
            PageText(
                page_number=page.page_number,
                text=page.text,
            )
            for page in document.pages
        ]

    def _extract_best_page_candidate(self, page: fitz.Page) -> PageExtractionCandidate:
        """
        Extract and evaluate several candidates for one page.

        Candidate order
        ---------------
        1. dict-based extraction
        2. block-based extraction
        3. plain text extraction

        Returns
        -------
        PageExtractionCandidate
            The best scoring candidate for the page.
        """
        candidates: List[PageExtractionCandidate] = []

        try:
            dict_text, dict_blocks = self._extract_page_via_dict(page)
            dict_flags = self._detect_corruption_flags(dict_text)
            dict_score = self._score_extracted_text(dict_text)
            candidates.append(
                PageExtractionCandidate(
                    source_mode="dict",
                    text=dict_text,
                    quality_score=dict_score,
                    blocks=dict_blocks,
                    corruption_flags=dict_flags,
                )
            )
        except Exception:
            candidates.append(
                PageExtractionCandidate(
                    source_mode="dict",
                    text="",
                    quality_score=-1_000_000.0,
                    blocks=[],
                    corruption_flags=["dict_extraction_failed"],
                )
            )

        try:
            block_text, block_blocks = self._extract_page_via_blocks(page)
            block_flags = self._detect_corruption_flags(block_text)
            block_score = self._score_extracted_text(block_text)
            candidates.append(
                PageExtractionCandidate(
                    source_mode="blocks",
                    text=block_text,
                    quality_score=block_score,
                    blocks=block_blocks,
                    corruption_flags=block_flags,
                )
            )
        except Exception:
            candidates.append(
                PageExtractionCandidate(
                    source_mode="blocks",
                    text="",
                    quality_score=-1_000_000.0,
                    blocks=[],
                    corruption_flags=["block_extraction_failed"],
                )
            )

        try:
            plain_text = page.get_text("text") or ""
            plain_flags = self._detect_corruption_flags(plain_text)
            plain_score = self._score_extracted_text(plain_text)
            candidates.append(
                PageExtractionCandidate(
                    source_mode="text",
                    text=plain_text,
                    quality_score=plain_score,
                    blocks=[],
                    corruption_flags=plain_flags,
                )
            )
        except Exception:
            candidates.append(
                PageExtractionCandidate(
                    source_mode="text",
                    text="",
                    quality_score=-1_000_000.0,
                    blocks=[],
                    corruption_flags=["plain_text_extraction_failed"],
                )
            )

        return max(candidates, key=lambda item: item.quality_score)

    def _extract_page_via_blocks(
        self,
        page: fitz.Page,
    ) -> Tuple[str, List[ExtractedBlock]]:
        """
        Extract one page using PyMuPDF block tuples.

        Returns
        -------
        Tuple[str, List[ExtractedBlock]]
            - page text reconstructed from blocks
            - structured block list
        """
        raw_blocks = page.get_text("blocks")
        structured_blocks: List[ExtractedBlock] = []
        page_text_parts: List[str] = []

        for block_index, block in enumerate(raw_blocks):
            if len(block) < 5:
                continue

            block_text = (block[4] or "").strip()
            if not block_text:
                continue

            bbox = BoundingBox(
                x0=float(block[0]),
                y0=float(block[1]),
                x1=float(block[2]),
                y1=float(block[3]),
            )

            lines = [
                ExtractedLine(
                    text=line.strip(),
                    bbox=bbox,
                    block_index=block_index,
                    line_index=line_index,
                )
                for line_index, line in enumerate(block_text.splitlines())
                if line.strip()
            ]

            structured_blocks.append(
                ExtractedBlock(
                    block_index=block_index,
                    text=block_text,
                    bbox=bbox,
                    lines=lines,
                    source_mode="blocks",
                )
            )
            page_text_parts.append(block_text)

        return "\n".join(page_text_parts).strip(), structured_blocks

    def _extract_page_via_dict(
        self,
        page: fitz.Page,
    ) -> Tuple[str, List[ExtractedBlock]]:
        """
        Extract one page using dict-based layout reconstruction.

        Returns
        -------
        Tuple[str, List[ExtractedBlock]]
            - page text reconstructed from lines
            - structured block list with line data
        """
        page_dict = page.get_text("dict")
        raw_blocks = page_dict.get("blocks", [])

        structured_blocks: List[ExtractedBlock] = []
        page_text_parts: List[str] = []

        for block_index, block in enumerate(raw_blocks):
            if block.get("type") != 0:
                continue

            block_lines: List[ExtractedLine] = []
            block_text_parts: List[str] = []

            block_bbox = self._bbox_from_sequence(block.get("bbox"))

            for line_index, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans).strip()
                if not line_text:
                    continue

                line_bbox = self._bbox_from_sequence(line.get("bbox"))

                block_lines.append(
                    ExtractedLine(
                        text=line_text,
                        bbox=line_bbox,
                        block_index=block_index,
                        line_index=line_index,
                    )
                )
                block_text_parts.append(line_text)

            block_text = "\n".join(block_text_parts).strip()
            if not block_text:
                continue

            structured_blocks.append(
                ExtractedBlock(
                    block_index=block_index,
                    text=block_text,
                    bbox=block_bbox,
                    lines=block_lines,
                    source_mode="dict",
                )
            )
            page_text_parts.append(block_text)

        return "\n".join(page_text_parts).strip(), structured_blocks

    def _score_extracted_text(self, text: str) -> float:
        """
        Score extracted text heuristically.

        Returns
        -------
        float
            Quality score where larger is better.
        """
        if not text or not text.strip():
            return -1_000_000.0

        stripped = text.strip()
        total_len = len(stripped)

        alpha_count = sum(1 for ch in stripped if ch.isalpha())
        digit_count = sum(1 for ch in stripped if ch.isdigit())
        whitespace_count = sum(1 for ch in stripped if ch.isspace())

        suspicious_chars = {
            "*", "^", "_", "`", "~", "\\", "|", "<", ">", "�", "￾", ""
        }
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

        words = [word for word in stripped.split() if word]
        if words:
            single_char_ratio = sum(1 for word in words if len(word) == 1) / len(words)
            if single_char_ratio > 0.35:
                score -= 35.0

        return score

    def _detect_corruption_flags(self, text: str) -> List[str]:
        """
        Detect lightweight corruption signals in extracted text.
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
        symbol_count = sum(
            1 for ch in stripped
            if not ch.isalnum() and not ch.isspace()
        )

        if suspicious_symbol_count / max(total_len, 1) > 0.03:
            flags.append("high_suspicious_symbol_density")

        if symbol_count / max(total_len, 1) > 0.35:
            flags.append("high_symbol_ratio")

        alpha_ratio = sum(1 for ch in stripped if ch.isalpha()) / max(total_len, 1)
        if alpha_ratio < 0.25:
            flags.append("low_alpha_ratio")

        words = [word for word in stripped.split() if word]
        if words:
            single_char_ratio = sum(1 for word in words if len(word) == 1) / len(words)
            if single_char_ratio > 0.35:
                flags.append("excessive_single_character_tokens")

        return flags

    def _bbox_from_sequence(self, value: Any) -> Optional[BoundingBox]:
        """
        Convert a PyMuPDF bbox-like sequence into a BoundingBox.

        Returns
        -------
        Optional[BoundingBox]
            Parsed bounding box, or None if the input is missing/invalid.
        """
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None

        try:
            return BoundingBox(
                x0=float(value[0]),
                y0=float(value[1]),
                x1=float(value[2]),
                y1=float(value[3]),
            )
        except (TypeError, ValueError):
            return None
