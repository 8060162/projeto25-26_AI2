from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Chunking.chunking.models import PageText

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyMuPDF is required. Install package 'pymupdf'."
    ) from exc


# ============================================================================
# Intermediate extraction models
# ============================================================================
#
# Why define these models here?
# -----------------------------
# At this stage of the project, the pipeline should no longer think in terms
# of "PDF -> plain text only".
#
# Instead, the extraction stage should preserve as much structural information
# as possible so that downstream parsing can build a canonical JSON tree
# resembling the target master dictionary structure.
#
# These models intentionally represent a richer intermediate document view:
# - document
# - pages
# - blocks
# - lines
# - bounding boxes
# - extraction mode
# - quality score
#
# This is not the final domain JSON yet.
# It is an intermediate representation specifically designed to support
# robust structural parsing.
# ============================================================================


@dataclass(slots=True)
class BoundingBox:
    """
    Lightweight bounding box container.

    Coordinates follow the standard PDF-style convention returned by PyMuPDF:
    (x0, y0, x1, y1)

    Why keep this?
    --------------
    Bounding boxes are extremely useful later for:
    - header/footer detection
    - title heuristics
    - identifying isolated structural lines
    - debugging extraction quality
    """

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(slots=True)
class ExtractedLine:
    """
    Represents one reconstructed text line from the PDF.

    Notes
    -----
    A line is a very important intermediate unit for legal/regulatory PDFs
    because many structural markers appear on isolated lines, for example:
    - "CAPÍTULO I"
    - "Artigo 5.º"
    - article titles
    - annex titles
    - index entries
    """

    text: str
    bbox: Optional[BoundingBox] = None
    block_index: Optional[int] = None
    line_index: Optional[int] = None


@dataclass(slots=True)
class ExtractedBlock:
    """
    Represents one extracted text block.

    Why preserve blocks?
    --------------------
    Blocks are often the best compromise between pure text and full layout.
    They help preserve reading order and frequently isolate semantic regions
    such as:
    - title blocks
    - body paragraphs
    - footer/header noise
    - signature areas
    """

    block_index: int
    text: str
    bbox: Optional[BoundingBox] = None
    lines: List[ExtractedLine] = field(default_factory=list)
    source_mode: str = "unknown"


@dataclass(slots=True)
class PageExtractionCandidate:
    """
    Represents one extraction candidate for a single page.

    Multiple extraction modes may produce different quality results for the
    same PDF page. This object allows us to compare them explicitly.

    Example candidate sources:
    - dict
    - blocks
    - text
    """

    source_mode: str
    text: str
    quality_score: float
    blocks: List[ExtractedBlock] = field(default_factory=list)
    corruption_flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedPage:
    """
    Structured representation of one PDF page.

    This model preserves:
    - the final selected text
    - the best extraction mode
    - a quality score
    - reconstructed blocks and lines
    - heuristic corruption signals

    Downstream parsing should consume this object instead of depending only on
    flattened plain text.
    """

    page_number: int
    text: str
    selected_mode: str
    quality_score: float
    blocks: List[ExtractedBlock] = field(default_factory=list)
    corruption_flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedDocument:
    """
    Structured representation of the extracted PDF document.

    Important distinction
    ---------------------
    This is not yet the final canonical regulation JSON tree.
    It is the intermediate extraction output that will feed the structure
    parser.

    The parser should later transform this into a domain-specific JSON
    structure such as:
    - PREAMBULO
    - CAP_I
    - ART_1
    - ART_2
    - ...
    """

    source_path: str
    page_count: int
    pages: List[ExtractedPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """
        Return the document text as a page-joined convenience view.

        Why keep this property?
        -----------------------
        Some legacy or debugging flows may still need a concatenated text
        representation. However, this must be treated as a derived view,
        not as the primary extraction product.
        """
        return "\n\n".join(page.text for page in self.pages if page.text)


class PdfReader:
    """
    Robust PDF extraction component for the first stage of the pipeline.

    Design intent
    -------------
    This class is no longer designed as a "plain text extractor only".
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        Why this method matters
        -----------------------
        This method is the correct API for the current project direction.
        It preserves much more information than a simple page-text list and
        therefore provides a far better substrate for building the target
        JSON tree.
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
        This method is intentionally retained for backward compatibility so
        that existing flows do not break immediately.

        However, new parsing logic should prefer `extract_document()` because
        it preserves the structure needed to build the canonical JSON tree.
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
    # Candidate extraction and selection
    # ------------------------------------------------------------------

    def _extract_best_page_candidate(self, page: fitz.Page) -> PageExtractionCandidate:
        """
        Extract and evaluate several candidates for one page.

        Candidate order
        ---------------
        1. dict-based extraction
        2. block-based extraction
        3. plain text extraction

        Why prefer dict first?
        ----------------------
        Dict mode provides the richest structure:
        - blocks
        - lines
        - spans
        - coordinates

        That makes it the most useful mode for later structural parsing.

        Returns
        -------
        PageExtractionCandidate
            The best scoring candidate for the page.
        """
        candidates: List[PageExtractionCandidate] = []

        # --------------------------------------------------------------
        # Candidate 1: dict-based extraction
        #
        # This is the most valuable mode for structure-aware parsing.
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Candidate 2: block-based extraction
        #
        # This mode is often robust for legal/regulatory PDFs where text
        # blocks preserve reading order reasonably well.
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Candidate 3: plain text extraction
        #
        # This mode preserves the least structure but still serves as a
        # useful fallback because some PDFs behave unexpectedly better with
        # plain text extraction.
        # --------------------------------------------------------------
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

        best_candidate = max(candidates, key=lambda item: item.quality_score)

        # Defensive fallback: always return a candidate object, even if all
        # extraction modes failed or returned empty content.
        return best_candidate

    # ------------------------------------------------------------------
    # Extraction implementations
    # ------------------------------------------------------------------

    def _extract_page_via_blocks(
        self,
        page: fitz.Page,
    ) -> Tuple[str, List[ExtractedBlock]]:
        """
        Extract one page using PyMuPDF block tuples.

        Why keep this method?
        ---------------------
        Block extraction is still useful because some PDFs produce better
        reading order via block tuples than via dict reconstruction.

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
            # PyMuPDF block tuple format is generally:
            # (x0, y0, x1, y1, text, block_no, block_type)
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

            # We may not have reliable line-level detail in this mode, so
            # we store the whole block as text and optionally reconstruct a
            # naive line split for convenience.
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

        Why this method is especially important
        ---------------------------------------
        Dict mode exposes a richer hierarchy:
        - blocks
        - lines
        - spans
        - positions

        This makes it the best extraction source for structural parsing.

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
            # Only process text blocks.
            if block.get("type") != 0:
                continue

            block_lines: List[ExtractedLine] = []
            block_text_parts: List[str] = []

            block_bbox = self._bbox_from_sequence(block.get("bbox"))

            for line_index, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])

                # Rebuild line text from spans in reading order.
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

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def _score_extracted_text(self, text: str) -> float:
        """
        Score extracted text heuristically.

        Goal
        ----
        This scoring function is intentionally heuristic and practical.
        It is not trying to solve language quality academically; it simply
        tries to avoid choosing obviously bad extraction results when a better
        candidate exists.

        Domain-aware heuristics
        -----------------------
        Since the target documents are legal / regulatory PDFs, we reward
        some structural terms often found in healthy extractions:
        - Artigo
        - CAPÍTULO / CAPITULO
        - ANEXO
        - Regulamento
        - Despacho

        We also penalize:
        - suspicious replacement glyphs
        - excessive non-linguistic symbol density
        - extremely low alphabetic ratio
        """
        if not text or not text.strip():
            return -1_000_000.0

        stripped = text.strip()
        total_len = len(stripped)

        alpha_count = sum(1 for ch in stripped if ch.isalpha())
        digit_count = sum(1 for ch in stripped if ch.isdigit())
        whitespace_count = sum(1 for ch in stripped if ch.isspace())

        # Common suspicious characters seen in degraded extraction output.
        suspicious_chars = {
            "*", "^", "_", "`", "~", "\\", "|", "<", ">", "�", "￾", ""
        }
        suspicious_count = sum(1 for ch in stripped if ch in suspicious_chars)

        alpha_ratio = alpha_count / max(total_len, 1)
        whitespace_ratio = whitespace_count / max(total_len, 1)
        suspicious_ratio = suspicious_count / max(total_len, 1)

        score = 0.0

        # Reward non-trivial useful length, but cap the contribution so that
        # huge noisy text does not dominate purely by size.
        score += min(total_len, 4000) * 0.01

        # Reward alphabetic content strongly.
        score += alpha_ratio * 100.0

        # Reward visible word separation slightly.
        score += whitespace_ratio * 20.0

        # Penalize pages that are overly numeric unless justified.
        if digit_count / max(total_len, 1) > 0.35:
            score -= 20.0

        # Penalize suspicious symbol density.
        score -= suspicious_ratio * 180.0

        # Penalize text that looks too non-linguistic.
        if alpha_ratio < 0.30:
            score -= 80.0

        # Reward domain-typical structural language.
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

        # Penalize extreme single-character fragmentation, which is often a
        # symptom of broken extraction or font decoding problems.
        words = [word for word in stripped.split() if word]
        if words:
            single_char_ratio = sum(1 for word in words if len(word) == 1) / len(words)
            if single_char_ratio > 0.35:
                score -= 35.0

        return score

    def _detect_corruption_flags(self, text: str) -> List[str]:
        """
        Detect lightweight corruption signals in extracted text.

        Why flags matter
        ----------------
        Downstream stages may use these flags to:
        - trigger OCR fallback
        - mark a page as suspicious
        - produce extraction diagnostics
        - lower confidence during parsing

        Returns
        -------
        List[str]
            A list of corruption/suspicion signals.
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

        # Heuristic: many isolated one-character tokens may indicate degraded
        # extraction, especially if not explained by list formatting.
        words = [word for word in stripped.split() if word]
        if words:
            single_char_ratio = sum(1 for word in words if len(word) == 1) / len(words)
            if single_char_ratio > 0.35:
                flags.append("excessive_single_character_tokens")

        return flags

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _bbox_from_sequence(self, value: Any) -> Optional[BoundingBox]:
        """
        Convert a PyMuPDF bbox-like sequence into a BoundingBox.

        Parameters
        ----------
        value : Any
            Expected to be a sequence such as [x0, y0, x1, y1].

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