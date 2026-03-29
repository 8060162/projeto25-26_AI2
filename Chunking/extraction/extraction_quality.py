from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Union

from Chunking.chunking.models import ExtractedDocument, ExtractedPage, PageText


class ExtractionQualityAnalyzer:
    """
    Analyze the quality of already extracted PDF text.

    Purpose
    -------
    This component does NOT extract text itself.
    Instead, it evaluates the quality of text produced by a previous
    extraction stage.

    Why this module exists
    ----------------------
    Some PDFs visually render correctly but contain unusable or degraded
    text layers due to:
    - custom embedded fonts
    - non-Unicode encodings
    - glyph substitution
    - corrupted internal text maps
    - legacy or inconsistent publishing workflows

    In such cases, the extraction stage may produce text that looks like:
        "*+,-*-.-/-1/2*-34+*4/-5/-1/6-/"

    If the pipeline continues with such text, all downstream stages will
    degrade severely:
    - normalization
    - structure parsing
    - JSON tree construction
    - later chunking

    Therefore, the pipeline must detect low-quality extraction early and
    decide whether:
    - normal extraction is acceptable
    - OCR fallback should be triggered
    - the document should be marked as suspicious

    Architectural note
    ------------------
    This analyzer now supports both:
    - legacy page lists: List[PageText]
    - richer extraction output: ExtractedDocument

    This is important while the project transitions from:
        PDF -> plain text
    toward:
        PDF -> structured intermediate extraction -> parsed JSON tree
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_document(
        self,
        document_or_pages: Union[ExtractedDocument, Sequence[PageText], Sequence[ExtractedPage]],
    ) -> Dict[str, Any]:
        """
        Compute extraction quality metrics for a full document.

        Parameters
        ----------
        document_or_pages : Union[ExtractedDocument, Sequence[PageText], Sequence[ExtractedPage]]
            Either:
            - a rich ExtractedDocument object
            - a legacy list of PageText
            - a list of ExtractedPage objects

        Returns
        -------
        Dict[str, Any]
            Document-level diagnostic metrics.

        Why the return is dictionary-based
        ----------------------------------
        A dictionary is practical here because:
        - it is easy to serialize to JSON
        - it is easy to inspect in reports
        - it can evolve without breaking a strict schema too often

        High-level decision logic
        -------------------------
        This method does not merely count "bad looking" pages.
        It also combines:
        - page-level garbling signals
        - legal-marker coverage
        - page quality scores
        - suspicious symbol density
        - replacement-like glyph presence

        The final result includes a boolean:
            `document_likely_corrupted`

        This value is intended to help the pipeline decide whether to use
        OCR fallback.
        """
        pages = self._coerce_pages(document_or_pages)

        page_reports: List[Dict[str, Any]] = []
        suspicious_pages = 0
        empty_pages = 0
        pages_with_replacement_chars = 0
        pages_with_legal_markers = 0

        for page in pages:
            page_number, page_text, selected_mode, page_quality_score, page_flags = \
                self._extract_page_fields(page)

            report = self._analyze_text(
                text=page_text,
                page_number=page_number,
                selected_mode=selected_mode,
                upstream_quality_score=page_quality_score,
                upstream_flags=page_flags,
            )
            page_reports.append(report)

            if report["looks_garbled"]:
                suspicious_pages += 1

            if report["is_empty"]:
                empty_pages += 1

            if report["has_replacement_like_characters"]:
                pages_with_replacement_chars += 1

            if report["legal_marker_hits"] > 0:
                pages_with_legal_markers += 1

        total_pages = len(page_reports)

        suspicious_ratio = suspicious_pages / total_pages if total_pages else 0.0
        empty_ratio = empty_pages / total_pages if total_pages else 0.0
        replacement_ratio = (
            pages_with_replacement_chars / total_pages if total_pages else 0.0
        )
        legal_marker_coverage = (
            pages_with_legal_markers / total_pages if total_pages else 0.0
        )

        average_quality_score = (
            sum(report["quality_score"] for report in page_reports) / total_pages
            if total_pages
            else 0.0
        )

        # --------------------------------------------------------------
        # Document-level corruption decision
        #
        # This is intentionally heuristic.
        # We want a practical decision rule that performs well on real
        # institutional/legal PDFs, not a theoretically perfect classifier.
        # --------------------------------------------------------------
        document_likely_corrupted = self._decide_document_corruption(
            total_pages=total_pages,
            suspicious_ratio=suspicious_ratio,
            empty_ratio=empty_ratio,
            replacement_ratio=replacement_ratio,
            average_quality_score=average_quality_score,
            legal_marker_coverage=legal_marker_coverage,
        )

        return {
            "total_pages": total_pages,
            "suspicious_pages": suspicious_pages,
            "suspicious_ratio": suspicious_ratio,
            "empty_pages": empty_pages,
            "empty_ratio": empty_ratio,
            "pages_with_replacement_like_characters": pages_with_replacement_chars,
            "replacement_ratio": replacement_ratio,
            "pages_with_legal_markers": pages_with_legal_markers,
            "legal_marker_coverage": legal_marker_coverage,
            "average_quality_score": average_quality_score,
            "document_likely_corrupted": document_likely_corrupted,
            "recommended_action": (
                "use_ocr_fallback" if document_likely_corrupted else "keep_native_extraction"
            ),
            "page_reports": page_reports,
        }

    # ------------------------------------------------------------------
    # Page-level analysis
    # ------------------------------------------------------------------

    def _analyze_text(
        self,
        text: str,
        page_number: int | None = None,
        selected_mode: str = "",
        upstream_quality_score: float | None = None,
        upstream_flags: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of one extracted page text.

        Parameters
        ----------
        text : str
            Page text to evaluate.

        page_number : int | None, optional
            Page number for traceability.

        selected_mode : str, optional
            Extraction mode used upstream, if known.
            Example values:
            - "dict"
            - "blocks"
            - "text"
            - "ocr"

        upstream_quality_score : float | None, optional
            Precomputed quality score from the extractor, if available.

        upstream_flags : List[str] | None, optional
            Existing corruption/suspicion flags from the extractor.

        Returns
        -------
        Dict[str, Any]
            A rich diagnostic page report.

        Heuristic philosophy
        --------------------
        Good extracted legal/regulatory text usually contains:
        - alphabetic content
        - whitespace between words
        - sentence-like structure
        - recognizable legal markers
        - relatively few suspicious replacement glyphs

        Corrupted extraction often contains:
        - excessive symbol noise
        - extremely low alphabetic ratio
        - many replacement characters
        - too many isolated single-character tokens
        - implausible word structure
        """
        upstream_flags = upstream_flags or []

        if not text or not text.strip():
            return {
                "page_number": page_number,
                "selected_mode": selected_mode,
                "length": 0,
                "alpha_ratio": 0.0,
                "digit_ratio": 0.0,
                "whitespace_ratio": 0.0,
                "symbol_ratio": 0.0,
                "suspicious_symbol_ratio": 0.0,
                "single_character_token_ratio": 0.0,
                "legal_marker_hits": 0,
                "has_replacement_like_characters": False,
                "replacement_like_character_count": 0,
                "quality_score": -1_000_000.0 if upstream_quality_score is None else upstream_quality_score,
                "is_empty": True,
                "looks_garbled": True,
                "is_locally_unreliable": True,
                "garbled_reasons": ["empty_text"],
                "upstream_flags": list(upstream_flags),
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

        suspicious_symbol_chars = {"*", "^", "_", "`", "~", "\\", "|", "<", ">"}
        suspicious_symbol_count = sum(
            1 for c in stripped if c in suspicious_symbol_chars
        )

        replacement_like_chars = {"�", "￾", ""}
        replacement_like_count = sum(
            1 for c in stripped if c in replacement_like_chars
        )

        words = [word for word in stripped.split() if word]
        single_character_token_ratio = (
            sum(1 for word in words if len(word) == 1) / len(words)
            if words else 0.0
        )

        alpha_ratio = alpha_count / max(total_len, 1)
        digit_ratio = digit_count / max(total_len, 1)
        whitespace_ratio = whitespace_count / max(total_len, 1)
        symbol_ratio = symbol_count / max(total_len, 1)
        suspicious_symbol_ratio = suspicious_symbol_count / max(total_len, 1)

        legal_marker_hits = self._count_legal_markers(stripped)

        # --------------------------------------------------------------
        # Practical page quality score
        #
        # This score is intentionally similar in spirit to the one used in
        # extraction, but independent enough to serve as a second opinion.
        # --------------------------------------------------------------
        quality_score = self._compute_quality_score(
            text=stripped,
            alpha_ratio=alpha_ratio,
            digit_ratio=digit_ratio,
            whitespace_ratio=whitespace_ratio,
            suspicious_symbol_ratio=suspicious_symbol_ratio,
            replacement_like_count=replacement_like_count,
            single_character_token_ratio=single_character_token_ratio,
            legal_marker_hits=legal_marker_hits,
        )

        if upstream_quality_score is not None:
            # Blend upstream and local scores to avoid depending on a single
            # stage's judgment. The local analyzer remains the stronger signal
            # because its goal is specifically "fallback decision support".
            quality_score = (quality_score * 0.65) + (upstream_quality_score * 0.35)

        garbled_reasons: List[str] = []

        if alpha_ratio < 0.25:
            garbled_reasons.append("low_alpha_ratio")

        if symbol_ratio > 0.35:
            garbled_reasons.append("high_symbol_ratio")

        if suspicious_symbol_ratio > 0.03:
            garbled_reasons.append("high_suspicious_symbol_density")

        if whitespace_ratio < 0.02:
            garbled_reasons.append("very_low_whitespace_ratio")

        if replacement_like_count > 0:
            garbled_reasons.append("replacement_like_characters_detected")

        if single_character_token_ratio > 0.35:
            garbled_reasons.append("excessive_single_character_tokens")

        # Low legal marker coverage is not by itself proof of corruption,
        # because not every page must contain article/chapter markers.
        # However, on heavily garbled pages it strengthens suspicion.
        if legal_marker_hits == 0 and alpha_ratio < 0.35 and total_len > 120:
            garbled_reasons.append("no_legal_markers_on_low_quality_page")

        looks_garbled = len(garbled_reasons) > 0
        is_locally_unreliable = self._is_locally_unreliable_page(
            quality_score=quality_score,
            garbled_reasons=garbled_reasons,
            upstream_flags=upstream_flags,
        )

        return {
            "page_number": page_number,
            "selected_mode": selected_mode,
            "length": total_len,
            "alpha_ratio": alpha_ratio,
            "digit_ratio": digit_ratio,
            "whitespace_ratio": whitespace_ratio,
            "symbol_ratio": symbol_ratio,
            "suspicious_symbol_ratio": suspicious_symbol_ratio,
            "single_character_token_ratio": single_character_token_ratio,
            "legal_marker_hits": legal_marker_hits,
            "has_replacement_like_characters": replacement_like_count > 0,
            "replacement_like_character_count": replacement_like_count,
            "quality_score": quality_score,
            "is_empty": False,
            "looks_garbled": looks_garbled,
            "is_locally_unreliable": is_locally_unreliable,
            "garbled_reasons": garbled_reasons,
            "upstream_flags": list(upstream_flags),
        }

    # ------------------------------------------------------------------
    # Document-level decision
    # ------------------------------------------------------------------

    def _decide_document_corruption(
        self,
        total_pages: int,
        suspicious_ratio: float,
        empty_ratio: float,
        replacement_ratio: float,
        average_quality_score: float,
        legal_marker_coverage: float,
    ) -> bool:
        """
        Decide whether a document is likely corrupted enough to justify OCR.

        Parameters
        ----------
        total_pages : int
            Total number of analyzed pages.

        suspicious_ratio : float
            Ratio of pages that look garbled.

        empty_ratio : float
            Ratio of empty pages.

        replacement_ratio : float
            Ratio of pages containing replacement-like glyphs.

        average_quality_score : float
            Mean page quality score across the document.

        legal_marker_coverage : float
            Ratio of pages containing at least one legal/regulatory marker.

        Returns
        -------
        bool
            True when OCR fallback is likely justified.

        Decision philosophy
        -------------------
        The fallback should not trigger too eagerly because OCR is slower and
        less structurally faithful than native PDF extraction.

        However, when native extraction is clearly degraded, continuing with it
        would damage every later stage. Therefore the heuristic is intentionally
        conservative but decisive.
        """
        if total_pages == 0:
            return True

        # Strong direct failure cases.
        if suspicious_ratio >= 0.40:
            return True

        if replacement_ratio >= 0.30:
            return True

        if empty_ratio >= 0.30:
            return True

        # Combined weak-signal cases.
        if suspicious_ratio >= 0.25 and average_quality_score < 15.0:
            return True

        if suspicious_ratio >= 0.20 and replacement_ratio >= 0.15:
            return True

        # For legal/regulatory documents, extremely low legal-marker coverage
        # can be suspicious when combined with poor overall quality.
        if legal_marker_coverage < 0.10 and average_quality_score < 10.0:
            return True

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_pages(
        self,
        document_or_pages: Union[ExtractedDocument, Sequence[PageText], Sequence[ExtractedPage]],
    ) -> List[Union[PageText, ExtractedPage]]:
        """
        Normalize the input into a list of page-like objects.

        Parameters
        ----------
        document_or_pages : Union[ExtractedDocument, Sequence[PageText], Sequence[ExtractedPage]]
            Supported input variants.

        Returns
        -------
        List[Union[PageText, ExtractedPage]]
            A list of page-like objects.

        Why this helper exists
        ----------------------
        The codebase is currently transitioning from the legacy PageText model
        to the richer ExtractedDocument/ExtractedPage model. This helper allows
        the analyzer to support both during the migration.
        """
        if isinstance(document_or_pages, ExtractedDocument):
            return list(document_or_pages.pages)

        return list(document_or_pages)

    def _extract_page_fields(
        self,
        page: Union[PageText, ExtractedPage],
    ) -> tuple[int | None, str, str, float | None, List[str]]:
        """
        Extract normalized fields from either PageText or ExtractedPage.

        Parameters
        ----------
        page : Union[PageText, ExtractedPage]
            One page-like object.

        Returns
        -------
        tuple[int | None, str, str, float | None, List[str]]
            Normalized fields:
            - page_number
            - text
            - selected_mode
            - upstream_quality_score
            - upstream_flags
        """
        if isinstance(page, ExtractedPage):
            return (
                page.page_number,
                page.text,
                page.selected_mode,
                page.quality_score,
                list(page.corruption_flags),
            )

        return (
            page.page_number,
            page.text,
            "",
            None,
            [],
        )

    def _is_locally_unreliable_page(
        self,
        quality_score: float,
        garbled_reasons: Sequence[str],
        upstream_flags: Sequence[str],
    ) -> bool:
        """
        Decide whether one page should be treated defensively downstream.

        Why this helper exists
        ----------------------
        Some documents remain globally acceptable while still containing a few
        locally toxic pages. The downstream normalizer/parser can use this
        boolean to avoid treating those pages too optimistically.

        Parameters
        ----------
        quality_score : float
            Blended page quality score.

        garbled_reasons : Sequence[str]
            Local analyzer reasons describing why the page looks degraded.

        upstream_flags : Sequence[str]
            Extraction-stage corruption flags already attached to the page.

        Returns
        -------
        bool
            True when the page looks degraded enough to justify defensive
            downstream handling.
        """
        degrading_flags = {
            "empty_text",
            "replacement_like_characters",
            "high_suspicious_symbol_density",
            "high_symbol_ratio",
            "low_alpha_ratio",
        }

        reason_set = set(garbled_reasons)
        upstream_flag_set = set(upstream_flags)

        if quality_score < 10.0 and (
            reason_set.intersection(
                {
                    "low_alpha_ratio",
                    "high_symbol_ratio",
                    "replacement_like_characters_detected",
                }
            )
            or upstream_flag_set.intersection(degrading_flags)
        ):
            return True

        if quality_score < 0.0:
            return True

        return False

    def _count_legal_markers(self, text: str) -> int:
        """
        Count legal/regulatory structural markers in text.

        Parameters
        ----------
        text : str
            Text to inspect.

        Returns
        -------
        int
            Number of distinct marker hits.

        Why legal markers matter
        ------------------------
        The target documents are institutional regulations. Healthy extraction
        often contains domain-typical markers such as:
        - artigo
        - capítulo / capitulo
        - anexo
        - regulamento
        - despacho
        """
        lower_text = text.lower()

        markers = [
            "artigo",
            "capítulo",
            "capitulo",
            "anexo",
            "regulamento",
            "despacho",
            "secção",
            "seccao",
            "secao",
        ]

        return sum(1 for marker in markers if marker in lower_text)

    def _compute_quality_score(
        self,
        text: str,
        alpha_ratio: float,
        digit_ratio: float,
        whitespace_ratio: float,
        suspicious_symbol_ratio: float,
        replacement_like_count: int,
        single_character_token_ratio: float,
        legal_marker_hits: int,
    ) -> float:
        """
        Compute a practical page quality score.

        Parameters
        ----------
        text : str
            Page text.

        alpha_ratio : float
            Ratio of alphabetic characters.

        digit_ratio : float
            Ratio of numeric characters.

        whitespace_ratio : float
            Ratio of whitespace characters.

        suspicious_symbol_ratio : float
            Ratio of suspicious symbols.

        replacement_like_count : int
            Count of replacement-like glyphs.

        single_character_token_ratio : float
            Ratio of one-character tokens.

        legal_marker_hits : int
            Number of domain-typical legal markers.

        Returns
        -------
        float
            Quality score where larger is better.
        """
        total_len = len(text)
        score = 0.0

        # Reward non-trivial length, with a cap.
        score += min(total_len, 4000) * 0.01

        # Reward alphabetic structure.
        score += alpha_ratio * 100.0

        # Reward visible word separation.
        score += whitespace_ratio * 20.0

        # Penalize suspicious signals.
        score -= suspicious_symbol_ratio * 180.0
        score -= replacement_like_count * 12.0

        if digit_ratio > 0.35:
            score -= 20.0

        if alpha_ratio < 0.30:
            score -= 80.0

        if single_character_token_ratio > 0.35:
            score -= 35.0

        # Reward domain plausibility.
        score += legal_marker_hits * 8.0

        return score
