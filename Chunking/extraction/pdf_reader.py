from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from Chunking.chunking.models import PageText

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyMuPDF is required. Install package 'pymupdf'."
    ) from exc


class PdfReader:
    """
    Robust PDF text extractor used as the first stage of the chunking pipeline.

    Design goals
    ------------
    This component is intentionally focused on **reliable text extraction**
    rather than semantic interpretation.

    The responsibilities of this class are:

    1. Extract text page-by-page
    2. Preserve reading order as much as possible
    3. Avoid flattening layout too early
    4. Provide clean page-level text to the normalization stage
    5. Gracefully handle PDFs whose text layer is partially corrupted

    Why page-level extraction?
    --------------------------
    Page boundaries are extremely useful later in the pipeline for:

    - page_start / page_end metadata
    - debugging chunk boundaries
    - identifying repeated headers / footers
    - traceability back to the source document

    Why multi-strategy extraction?
    ------------------------------
    Some PDFs extract well with:
        page.get_text("blocks")

    Others behave better with:
        page.get_text("dict")
    or:
        page.get_text("text")

    Unfortunately, a subset of officially published or institutionally
    generated PDFs may contain font encodings that make one extraction mode
    return garbled text while another still returns usable text.

    Because of that, this reader now:
    - tries multiple extraction modes
    - evaluates their output quality heuristically
    - keeps the best candidate

    Important scope note
    --------------------
    This class does not perform OCR yet.
    OCR fallback should be implemented in a separate extraction fallback layer
    so responsibilities remain clear.
    """

    def extract_pages(self, pdf_path: Path) -> List[PageText]:
        """
        Extract text from a PDF document page by page.

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file to be processed.

        Returns
        -------
        List[PageText]
            A list of PageText objects containing:
            - page_number
            - extracted page text

        Important implementation detail
        -------------------------------
        This method no longer relies on only one PyMuPDF extraction mode.
        Instead, it evaluates several candidates and selects the most promising
        one for each page.

        Why this matters
        ----------------
        The entire downstream pipeline depends on extraction quality.
        If extraction fails, normalization, parsing, and chunking will all
        produce poor results even if their own logic is correct.
        """
        pages: List[PageText] = []

        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                text = self._extract_best_page_text(page)

                pages.append(
                    PageText(
                        page_number=page_index,
                        text=text or "",
                    )
                )

        return pages

    def _extract_best_page_text(self, page: fitz.Page) -> str:
        """
        Extract the best available text candidate for one page.

        Extraction order
        ----------------
        1. blocks
        2. dict-based line reconstruction
        3. plain text

        Why compare multiple candidates?
        --------------------------------
        Different PDFs behave differently across extraction modes.
        A mode that is excellent on one document may produce unreadable text
        on another. Therefore this method generates several candidates and
        keeps the one with the strongest quality score.

        Parameters
        ----------
        page : fitz.Page
            PyMuPDF page object.

        Returns
        -------
        str
            The best available extracted text for the page.
        """
        candidates: List[Tuple[str, str]] = []

        # -------------------------------------------------------------
        # Candidate 1: block-based extraction
        #
        # This often behaves well for legal PDFs because blocks usually
        # preserve reading order more reliably than plain text extraction.
        # -------------------------------------------------------------
        try:
            block_text = self._extract_page_via_blocks(page)
            candidates.append(("blocks", block_text))
        except Exception:
            candidates.append(("blocks", ""))

        # -------------------------------------------------------------
        # Candidate 2: dict-based extraction with manual reconstruction
        #
        # This gives us access to lines and spans. In some documents this
        # produces better results than block tuples.
        # -------------------------------------------------------------
        try:
            dict_text = self._extract_page_via_dict(page)
            candidates.append(("dict", dict_text))
        except Exception:
            candidates.append(("dict", ""))

        # -------------------------------------------------------------
        # Candidate 3: plain text extraction
        #
        # Although simple, this can still outperform more structured modes
        # for some PDFs.
        # -------------------------------------------------------------
        try:
            plain_text = page.get_text("text") or ""
            candidates.append(("text", plain_text))
        except Exception:
            candidates.append(("text", ""))

        # -------------------------------------------------------------
        # Score every candidate and keep the strongest one.
        #
        # We do not try to be academically perfect here. The goal is simply
        # to avoid obviously bad choices such as returning heavily garbled
        # text when a better candidate exists.
        # -------------------------------------------------------------
        scored_candidates = [
            (name, text, self._score_extracted_text(text))
            for name, text in candidates
        ]

        best_name, best_text, _ = max(
            scored_candidates,
            key=lambda item: item[2],
        )

        # -------------------------------------------------------------
        # Defensive final fallback:
        # if all candidates are empty, still return an empty string rather
        # than raising.
        # -------------------------------------------------------------
        if not best_text:
            return ""

        return best_text

    def _extract_page_via_blocks(self, page: fitz.Page) -> str:
        """
        Extract page text using PyMuPDF block tuples.

        Why this method exists
        ----------------------
        Block-based extraction is often the most practical compromise for
        legal/regulatory PDFs because it can preserve reading order better
        than plain text extraction while staying much simpler than a full
        layout engine.

        PyMuPDF block tuple format
        --------------------------
        A text block usually looks like:

            (x0, y0, x1, y1, text, block_no, block_type)

        We only care about:
        - block[4] -> text
        - block[6] -> block type (when available)

        Parameters
        ----------
        page : fitz.Page
            PyMuPDF page object.

        Returns
        -------
        str
            Reconstructed page text from block tuples.
        """
        blocks = page.get_text("blocks")
        text_parts: List[str] = []

        for block in blocks:
            # Defensive validation: malformed tuples should be ignored.
            if len(block) < 5:
                continue

            block_text = (block[4] or "").strip()
            if not block_text:
                continue

            text_parts.append(block_text)

        return "\n".join(text_parts).strip()

    def _extract_page_via_dict(self, page: fitz.Page) -> str:
        """
        Extract page text using dict-based layout reconstruction.

        Why this method exists
        ----------------------
        Some PDFs behave better in dict mode than in block tuple mode.
        Dict mode gives access to a richer structure:

            page
             └── blocks
                 └── lines
                     └── spans

        This can help reconstruct the page in a more controlled way.

        Parameters
        ----------
        page : fitz.Page
            PyMuPDF page object.

        Returns
        -------
        str
            Reconstructed page text.
        """
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
        reconstructed_lines: List[str] = []

        for block in blocks:
            # Ignore non-text blocks such as images.
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])

                # Rebuild the line from its spans.
                line_text = "".join(span.get("text", "") for span in spans).strip()

                if line_text:
                    reconstructed_lines.append(line_text)

        return "\n".join(reconstructed_lines).strip()

    def _score_extracted_text(self, text: str) -> float:
        """
        Score one extracted text candidate heuristically.

        Why this helper exists
        ----------------------
        The pipeline needs a lightweight way to decide which extraction mode
        produced the most usable output for a page.

        Heuristic principles
        --------------------
        A good extracted text candidate usually has:
        - non-trivial length
        - a healthy amount of alphabetic characters
        - visible whitespace / word separation
        - relatively few obviously suspicious symbols

        This is intentionally heuristic, not linguistic perfection.

        Parameters
        ----------
        text : str
            Extracted text candidate.

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

        # Suspicious symbols frequently seen in badly decoded PDFs.
        suspicious_chars = set("*^_`~\\|<>")
        suspicious_count = sum(1 for ch in stripped if ch in suspicious_chars)

        alpha_ratio = alpha_count / max(total_len, 1)
        whitespace_ratio = whitespace_count / max(total_len, 1)
        suspicious_ratio = suspicious_count / max(total_len, 1)

        score = 0.0

        # Reward meaningful length.
        score += min(total_len, 4000) * 0.01

        # Reward alphabetic content strongly.
        score += alpha_ratio * 100.0

        # Reward visible word separation a little.
        score += whitespace_ratio * 20.0

        # Mild penalty for excessive digits.
        if digit_count / max(total_len, 1) > 0.35:
            score -= 20.0

        # Strong penalty for suspicious symbol density.
        score -= suspicious_ratio * 150.0

        # Additional penalty for text that looks too non-linguistic.
        if alpha_ratio < 0.30:
            score -= 80.0

        return score