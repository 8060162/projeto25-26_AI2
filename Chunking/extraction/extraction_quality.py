from __future__ import annotations

from typing import Dict, List

from Chunking.chunking.models import PageText


class ExtractionQualityAnalyzer:
    """
    Analyze the quality of extracted PDF text.

    This component does NOT extract text itself. Instead it evaluates
    the text already produced by the PDF extraction stage.

    Why this module exists
    ----------------------

    Some PDFs visually render correctly but contain unusable text layers
    due to:

    - custom embedded fonts
    - non-Unicode encodings
    - glyph substitution
    - legacy publishing workflows

    In those cases the extractor may produce text like:

        "*+,-*-.-/-1/2*-34+*4/-5/-1/6-/"

    If the pipeline continues with such text, all downstream stages
    (normalization, parsing, chunking) will fail.

    Therefore the pipeline must detect this condition early and trigger
    a fallback extraction method such as OCR.
    """

    def analyze_document(self, pages: List[PageText]) -> Dict:
        """
        Compute extraction quality metrics for an entire document.

        Parameters
        ----------
        pages : List[PageText]
            Extracted page text objects.

        Returns
        -------
        Dict
            Dictionary containing diagnostic metrics.
        """

        page_reports = []
        suspicious_pages = 0

        for page in pages:
            report = self._analyze_text(page.text)
            page_reports.append(report)

            if report["looks_garbled"]:
                suspicious_pages += 1

        total_pages = len(pages)

        suspicious_ratio = (
            suspicious_pages / total_pages if total_pages else 0
        )

        return {
            "total_pages": total_pages,
            "suspicious_pages": suspicious_pages,
            "suspicious_ratio": suspicious_ratio,
            "document_likely_corrupted": suspicious_ratio > 0.4,
            "page_reports": page_reports,
        }

    def _analyze_text(self, text: str) -> Dict:
        """
        Evaluate the quality of one text block.

        Heuristics used
        ---------------
        Good text usually contains:

        - alphabetic characters
        - spaces between words
        - reasonable punctuation

        Corrupted text often contains:

        - excessive symbols
        - extremely low alphabetic ratio
        - long strings without whitespace
        """

        if not text:
            return {
                "length": 0,
                "alpha_ratio": 0,
                "symbol_ratio": 0,
                "looks_garbled": True,
            }

        stripped = text.strip()

        total_len = len(stripped)

        alpha_count = sum(1 for c in stripped if c.isalpha())
        digit_count = sum(1 for c in stripped if c.isdigit())
        whitespace_count = sum(1 for c in stripped if c.isspace())
        symbol_count = sum(
            1 for c in stripped
            if not c.isalnum() and not c.isspace()
        )

        alpha_ratio = alpha_count / max(total_len, 1)
        symbol_ratio = symbol_count / max(total_len, 1)

        looks_garbled = False

        # Extremely low alphabetic content
        if alpha_ratio < 0.25:
            looks_garbled = True

        # Too many symbols
        if symbol_ratio > 0.35:
            looks_garbled = True

        # Almost no whitespace
        if whitespace_count < total_len * 0.02:
            looks_garbled = True

        return {
            "length": total_len,
            "alpha_ratio": alpha_ratio,
            "symbol_ratio": symbol_ratio,
            "digit_count": digit_count,
            "looks_garbled": looks_garbled,
        }