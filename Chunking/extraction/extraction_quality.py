from __future__ import annotations

from math import isfinite
import re
from typing import Any, Dict, Iterable, List, Sequence, Union

from Chunking.chunking.models import ExtractedDocument, ExtractedPage, PageText
from Chunking.config.settings import PipelineSettings


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

    def __init__(self, settings: PipelineSettings | None = None) -> None:
        """
        Store the runtime settings that control extraction-quality heuristics.

        Parameters
        ----------
        settings : PipelineSettings | None, optional
            Shared pipeline settings. When omitted, default settings are used.
        """
        self.settings = settings or PipelineSettings()

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
        pages_requiring_ocr_comparison: List[int] = []

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

            if report["should_trigger_ocr_comparison"]:
                pages_requiring_ocr_comparison.append(page_number)

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
            "pages_requiring_ocr_comparison": pages_requiring_ocr_comparison,
            "ocr_comparison_candidate_count": len(pages_requiring_ocr_comparison),
            "has_local_pages_requiring_ocr_comparison": bool(pages_requiring_ocr_comparison),
            "average_quality_score": average_quality_score,
            "document_likely_corrupted": document_likely_corrupted,
            "recommended_action": (
                "use_ocr_fallback" if document_likely_corrupted else "keep_native_extraction"
            ),
            "page_reports": page_reports,
        }

    def identify_pages_requiring_ocr_comparison(
        self,
        document_or_pages: Union[ExtractedDocument, Sequence[PageText], Sequence[ExtractedPage]],
    ) -> List[Dict[str, Any]]:
        """
        Return page reports that justify native-versus-OCR comparison.

        Parameters
        ----------
        document_or_pages : Union[ExtractedDocument, Sequence[PageText], Sequence[ExtractedPage]]
            Document or page collection that should be inspected.

        Returns
        -------
        List[Dict[str, Any]]
            Page reports limited to pages whose local signals justify explicit
            OCR comparison, even if the full document remains globally usable.
        """
        analysis = self.analyze_document(document_or_pages)
        page_reports = analysis.get("page_reports", [])
        return [
            page_report
            for page_report in page_reports
            if page_report.get("should_trigger_ocr_comparison", False)
        ]

    def compare_page_versions(
        self,
        native_page: Union[PageText, ExtractedPage],
        ocr_page: Union[PageText, ExtractedPage],
    ) -> Dict[str, Any]:
        """
        Compare native and OCR versions of the same page.

        Parameters
        ----------
        native_page : Union[PageText, ExtractedPage]
            Native extraction candidate for the page.
        ocr_page : Union[PageText, ExtractedPage]
            OCR extraction candidate for the same page.

        Returns
        -------
        Dict[str, Any]
            Explainable comparison result describing:
            - which source is preferred
            - whether OCR is clearly better
            - short reason codes
            - detailed page analysis for both candidates
        """
        native_page_number, native_text, native_mode, native_score, native_flags = (
            self._extract_page_fields(native_page)
        )
        ocr_page_number, ocr_text, ocr_mode, ocr_score, ocr_flags = (
            self._extract_page_fields(ocr_page)
        )

        page_number = native_page_number if native_page_number is not None else ocr_page_number

        native_report = self._analyze_text(
            text=native_text,
            page_number=page_number,
            selected_mode=native_mode or "native",
            upstream_quality_score=native_score,
            upstream_flags=native_flags,
        )
        ocr_report = self._analyze_text(
            text=ocr_text,
            page_number=page_number,
            selected_mode=ocr_mode or "ocr",
            upstream_quality_score=ocr_score,
            upstream_flags=ocr_flags,
        )

        comparison_metrics = self._build_page_comparison_metrics(
            native_report=native_report,
            ocr_report=ocr_report,
        )
        decision = self._decide_preferred_page_source(
            native_report=native_report,
            ocr_report=ocr_report,
            comparison_metrics=comparison_metrics,
        )

        return {
            "page_number": page_number,
            "preferred_source": decision["preferred_source"],
            "preferred_mode": decision["preferred_mode"],
            "decision": decision["decision"],
            "ocr_is_clearly_better": decision["decision"] == "use_ocr",
            "score_gap": decision["score_gap"],
            "reason_codes": decision["reason_codes"],
            "comparison_metrics": comparison_metrics,
            "native_assessment": native_report["semantic_risk_level"],
            "ocr_assessment": ocr_report["semantic_risk_level"],
            "native_report": native_report,
            "ocr_report": ocr_report,
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
        truncation_signals = self._detect_truncation_signals(stripped)
        line_readability_ratio = self._compute_line_readability_ratio(stripped)
        lexical_completeness = self._compute_lexical_completeness(stripped)
        prose_likeness = self._compute_prose_likeness(
            alpha_ratio=alpha_ratio,
            whitespace_ratio=whitespace_ratio,
            symbol_ratio=symbol_ratio,
            suspicious_symbol_ratio=suspicious_symbol_ratio,
            single_character_token_ratio=single_character_token_ratio,
            line_readability_ratio=line_readability_ratio,
            lexical_completeness=lexical_completeness,
        )
        is_structurally_important = self._looks_structurally_important(
            text=stripped,
            legal_marker_hits=legal_marker_hits,
        )
        semantic_fragility_signals = self._detect_semantic_fragility_signals(
            text=stripped,
            is_structurally_important=is_structurally_important,
            legal_marker_hits=legal_marker_hits,
            lexical_completeness=lexical_completeness,
            line_readability_ratio=line_readability_ratio,
            prose_likeness=prose_likeness,
            truncation_signal_count=len(truncation_signals),
        )
        semantic_risk_score = self._compute_semantic_risk_score(
            semantic_fragility_signals=semantic_fragility_signals,
            is_structurally_important=is_structurally_important,
            truncation_signal_count=len(truncation_signals),
        )

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
            symbol_ratio=symbol_ratio,
            suspicious_symbol_ratio=suspicious_symbol_ratio,
            replacement_like_count=replacement_like_count,
            single_character_token_ratio=single_character_token_ratio,
            legal_marker_hits=legal_marker_hits,
            lexical_completeness=lexical_completeness,
            line_readability_ratio=line_readability_ratio,
            truncation_signal_count=len(truncation_signals),
            prose_likeness=prose_likeness,
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

        if truncation_signals:
            garbled_reasons.append("suspicious_truncation_signals")

        if line_readability_ratio < 0.55 and total_len > 120:
            garbled_reasons.append("low_line_readability")

        if lexical_completeness < 0.45 and total_len > 120:
            garbled_reasons.append("low_lexical_completeness")

        if prose_likeness < 0.45 and total_len > 120:
            garbled_reasons.append("low_prose_likeness")

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
        semantic_risk_level = self._classify_semantic_risk_level(
            looks_garbled=looks_garbled,
            is_locally_unreliable=is_locally_unreliable,
            semantic_risk_score=semantic_risk_score,
        )
        comparison_trigger_reasons = self._build_ocr_comparison_trigger_reasons(
            quality_score=quality_score,
            garbled_reasons=garbled_reasons,
            is_locally_unreliable=is_locally_unreliable,
            lexical_completeness=lexical_completeness,
            line_readability_ratio=line_readability_ratio,
            prose_likeness=prose_likeness,
            truncation_signal_count=len(truncation_signals),
            suspicious_symbol_ratio=suspicious_symbol_ratio,
            replacement_like_count=replacement_like_count,
            legal_marker_hits=legal_marker_hits,
            is_structurally_important=is_structurally_important,
            semantic_fragility_signals=semantic_fragility_signals,
            semantic_risk_level=semantic_risk_level,
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
            "lexical_completeness": lexical_completeness,
            "line_readability_ratio": line_readability_ratio,
            "truncation_signals": truncation_signals,
            "truncation_signal_count": len(truncation_signals),
            "prose_likeness": prose_likeness,
            "semantic_fragility_signals": semantic_fragility_signals,
            "semantic_risk_score": semantic_risk_score,
            "semantic_risk_level": semantic_risk_level,
            "has_replacement_like_characters": replacement_like_count > 0,
            "replacement_like_character_count": replacement_like_count,
            "quality_score": quality_score,
            "is_empty": False,
            "looks_garbled": looks_garbled,
            "is_locally_unreliable": is_locally_unreliable,
            "is_structurally_important": is_structurally_important,
            "should_trigger_ocr_comparison": bool(comparison_trigger_reasons),
            "comparison_trigger_reasons": comparison_trigger_reasons,
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
        if suspicious_ratio >= self.settings.suspicious_page_ratio_threshold:
            return True

        if replacement_ratio >= self.settings.document_ocr_replacement_ratio_threshold:
            return True

        if empty_ratio >= self.settings.document_ocr_empty_ratio_threshold:
            return True

        # Combined weak-signal cases.
        if (
            suspicious_ratio
            >= self.settings.document_ocr_low_quality_suspicious_ratio_threshold
            and average_quality_score
            < self.settings.document_ocr_low_quality_average_score_threshold
        ):
            return True

        if (
            suspicious_ratio
            >= self.settings.document_ocr_replacement_mix_suspicious_ratio_threshold
            and replacement_ratio
            >= self.settings.document_ocr_replacement_mix_replacement_ratio_threshold
        ):
            return True

        # For legal/regulatory documents, extremely low legal-marker coverage
        # can be suspicious when combined with poor overall quality.
        if (
            legal_marker_coverage
            < self.settings.document_ocr_low_legal_marker_coverage_threshold
            and average_quality_score
            < self.settings.document_ocr_low_legal_marker_average_score_threshold
        ):
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

        if quality_score < self.settings.local_unreliable_page_min_quality_score and (
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

        if quality_score < self.settings.local_unreliable_page_hard_floor_score:
            return True

        return False

    def _build_ocr_comparison_trigger_reasons(
        self,
        quality_score: float,
        garbled_reasons: Sequence[str],
        is_locally_unreliable: bool,
        lexical_completeness: float,
        line_readability_ratio: float,
        prose_likeness: float,
        truncation_signal_count: int,
        suspicious_symbol_ratio: float,
        replacement_like_count: int,
        legal_marker_hits: int,
        is_structurally_important: bool,
        semantic_fragility_signals: Sequence[str],
        semantic_risk_level: str,
    ) -> List[str]:
        """
        Build explainable reasons for triggering OCR comparison on one page.

        Parameters
        ----------
        quality_score : float
            Page quality score produced by the analyzer.
        garbled_reasons : Sequence[str]
            Existing degradation reason codes for the page.
        is_locally_unreliable : bool
            Whether the page already looks unsafe locally.
        lexical_completeness : float
            Lexical completeness metric for the page.
        line_readability_ratio : float
            Readable-line ratio for the page.
        prose_likeness : float
            Prose-likeness score for the page.
        truncation_signal_count : int
            Number of conservative truncation signals on the page.
        suspicious_symbol_ratio : float
            Ratio of suspicious symbolic characters.
        replacement_like_count : int
            Count of mojibake-like or replacement characters.
        legal_marker_hits : int
            Number of legal markers preserved on the page.
        is_structurally_important : bool
            Whether the page looks structurally important.

        Returns
        -------
        List[str]
            Deterministic reason codes supporting local OCR comparison.
        """
        trigger_reasons: List[str] = []
        garbled_reason_set = set(garbled_reasons)

        if is_locally_unreliable:
            trigger_reasons.append("page_is_locally_unreliable")

        if quality_score < self.settings.local_ocr_trigger_page_quality_score:
            trigger_reasons.append("low_page_quality_score")

        if (
            suspicious_symbol_ratio
            >= self.settings.local_ocr_trigger_suspicious_symbol_ratio
        ):
            trigger_reasons.append("strong_suspicious_symbol_density")

        if replacement_like_count > 0:
            trigger_reasons.append("replacement_like_inline_noise")

        if (
            lexical_completeness
            < self.settings.local_ocr_trigger_min_lexical_completeness
        ):
            trigger_reasons.append("poor_lexical_completeness")

        if (
            line_readability_ratio
            < self.settings.local_ocr_trigger_min_line_readability
        ):
            trigger_reasons.append("poor_line_readability")

        if prose_likeness < self.settings.local_ocr_trigger_min_prose_likeness:
            trigger_reasons.append("poor_prose_likeness")

        if truncation_signal_count > 0:
            trigger_reasons.append("strong_truncation_signals")

        if (
            is_structurally_important
            and legal_marker_hits == 0
            and (
                "low_prose_likeness" in garbled_reason_set
                or "low_line_readability" in garbled_reason_set
                or "low_lexical_completeness" in garbled_reason_set
            )
        ):
            trigger_reasons.append("structurally_important_page_with_low_marker_preservation")

        if semantic_fragility_signals:
            if "interrupted_legal_enumeration" in semantic_fragility_signals:
                trigger_reasons.append("broken_legal_enumeration_continuity")
            if "structural_page_borderline_semantic_quality" in semantic_fragility_signals:
                trigger_reasons.append("borderline_structural_semantic_quality")
            if "structural_page_truncation_risk" in semantic_fragility_signals:
                trigger_reasons.append("structural_truncation_continuity_risk")

        if semantic_risk_level == "semantically_risky":
            trigger_reasons.append("semantically_risky_structural_page")

        return trigger_reasons

    def _looks_structurally_important(
        self,
        text: str,
        legal_marker_hits: int,
    ) -> bool:
        """
        Decide whether a page looks structurally important for legal parsing.

        Parameters
        ----------
        text : str
            Page text under inspection.
        legal_marker_hits : int
            Number of legal-marker hits already found in the page text.

        Returns
        -------
        bool
            True when the page likely contains legal structure or transition
            content whose degradation should trigger OCR comparison.
        """
        if legal_marker_hits > 0:
            return True

        lowered_text = text.lower()
        structural_markers = (
            "art.",
            "artigo",
            "capitulo",
            "capítulo",
            "secção",
            "secao",
            "seccao",
            "anexo",
            "regulamento",
            "despacho",
        )

        if any(marker in lowered_text for marker in structural_markers):
            return True

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        first_lines = lines[:3]
        upper_case_line_count = sum(
            1 for line in first_lines
            if len(line) >= 6 and line == line.upper()
        )
        return upper_case_line_count >= 1

    def _detect_semantic_fragility_signals(
        self,
        text: str,
        is_structurally_important: bool,
        legal_marker_hits: int,
        lexical_completeness: float,
        line_readability_ratio: float,
        prose_likeness: float,
        truncation_signal_count: int,
    ) -> List[str]:
        """
        Detect page defects that are semantically risky even when text is readable.

        Parameters
        ----------
        text : str
            Page text under inspection.
        is_structurally_important : bool
            Whether the page likely carries legal structure or continuity.
        legal_marker_hits : int
            Number of legal markers preserved on the page.
        lexical_completeness : float
            Ratio of plausible lexical tokens.
        line_readability_ratio : float
            Ratio of readable lines.
        prose_likeness : float
            Composite prose-likeness score.
        truncation_signal_count : int
            Number of truncation signals already found on the page.

        Returns
        -------
        List[str]
            Conservative semantic fragility reason codes.
        """
        signals: List[str] = []
        if not is_structurally_important:
            return signals

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if self._has_interrupted_legal_enumeration(lines):
            signals.append("interrupted_legal_enumeration")

        lexical_borderline_threshold = min(
            0.90,
            self.settings.local_ocr_trigger_min_lexical_completeness + 0.20,
        )
        readability_borderline_threshold = min(
            0.95,
            self.settings.local_ocr_trigger_min_line_readability + 0.20,
        )
        prose_borderline_threshold = min(
            0.90,
            self.settings.local_ocr_trigger_min_prose_likeness + 0.18,
        )
        borderline_metric_failures = 0
        if lexical_completeness < lexical_borderline_threshold:
            borderline_metric_failures += 1
        if line_readability_ratio < readability_borderline_threshold:
            borderline_metric_failures += 1
        if prose_likeness < prose_borderline_threshold:
            borderline_metric_failures += 1

        if legal_marker_hits > 0 and borderline_metric_failures >= 2:
            signals.append("structural_page_borderline_semantic_quality")

        if legal_marker_hits > 0 and truncation_signal_count > 0:
            signals.append("structural_page_truncation_risk")

        return signals

    def _has_interrupted_legal_enumeration(self, lines: Sequence[str]) -> bool:
        """
        Detect numbered legal items whose body appears to be interrupted.

        Parameters
        ----------
        lines : Sequence[str]
            Non-empty page lines.

        Returns
        -------
        bool
            True when a numbered legal unit appears to be cut before its body.
        """
        if len(lines) < 2:
            return False

        continuation_cues = (
            "por",
            "para",
            "mediante",
            "nos termos",
            "nas condicoes",
            "nas condições",
            "seguinte",
            "seguintes",
        )

        for current_line, next_line in zip(lines, lines[1:]):
            if not self._looks_like_legal_enumeration_start(current_line):
                continue
            if not self._looks_like_legal_enumeration_start(next_line):
                continue

            normalized_line = current_line.rstrip(".;:").lower()
            if any(normalized_line.endswith(cue) for cue in continuation_cues):
                return True

        return False

    def _looks_like_legal_enumeration_start(self, line: str) -> bool:
        """
        Detect whether one line starts a legal structural or enumerated unit.

        Parameters
        ----------
        line : str
            Page line to inspect.

        Returns
        -------
        bool
            True when the line looks like a legal unit start.
        """
        stripped = line.strip().lower()
        if not stripped:
            return False

        patterns = (
            r"^\d+\s*[-.)]\s+\S",
            r"^[a-z]\)\s+\S",
            r"^art(?:\.|igo)\s+\d",
            r"^n[.ºo]*\s*\d",
        )
        return any(re.match(pattern, stripped) for pattern in patterns)

    def _compute_semantic_risk_score(
        self,
        semantic_fragility_signals: Sequence[str],
        is_structurally_important: bool,
        truncation_signal_count: int,
    ) -> float:
        """
        Aggregate semantic fragility into a comparison-friendly score.

        Parameters
        ----------
        semantic_fragility_signals : Sequence[str]
            Semantic fragility reason codes for the page.
        is_structurally_important : bool
            Whether the page likely carries legal structure.
        truncation_signal_count : int
            Number of truncation signals on the page.

        Returns
        -------
        float
            Semantic risk score where larger means riskier.
        """
        risk_score = float(len(semantic_fragility_signals))
        if is_structurally_important:
            risk_score += 0.5
        if truncation_signal_count > 0:
            risk_score += 0.5
        return risk_score

    def _classify_semantic_risk_level(
        self,
        looks_garbled: bool,
        is_locally_unreliable: bool,
        semantic_risk_score: float,
    ) -> str:
        """
        Classify the practical semantic safety of one page.

        Parameters
        ----------
        looks_garbled : bool
            Whether the page already looks visibly degraded.
        is_locally_unreliable : bool
            Whether the page should be handled defensively downstream.
        semantic_risk_score : float
            Aggregated semantic fragility score.

        Returns
        -------
        str
            One of:
            - "acceptable"
            - "acceptable_borderline"
            - "semantically_risky"
            - "garbled"
        """
        if looks_garbled and is_locally_unreliable:
            return "garbled"
        if is_locally_unreliable or semantic_risk_score >= 2.0:
            return "semantically_risky"
        if semantic_risk_score >= 1.0:
            return "acceptable_borderline"
        return "acceptable"

    def _build_page_comparison_metrics(
        self,
        native_report: Dict[str, Any],
        ocr_report: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Build side-by-side comparison metrics for two page candidates.

        Parameters
        ----------
        native_report : Dict[str, Any]
            Analyzer report for the native candidate.
        ocr_report : Dict[str, Any]
            Analyzer report for the OCR candidate.

        Returns
        -------
        Dict[str, float]
            Normalized metric differences where positive values favor native
            and negative values favor OCR.
        """
        return {
            "quality_score_delta": self._safe_metric_delta(
                native_report["quality_score"],
                ocr_report["quality_score"],
            ),
            "lexical_completeness_delta": self._safe_metric_delta(
                native_report["lexical_completeness"],
                ocr_report["lexical_completeness"],
            ),
            "line_readability_delta": self._safe_metric_delta(
                native_report["line_readability_ratio"],
                ocr_report["line_readability_ratio"],
            ),
            "prose_likeness_delta": self._safe_metric_delta(
                native_report["prose_likeness"],
                ocr_report["prose_likeness"],
            ),
            "legal_marker_delta": self._safe_metric_delta(
                float(native_report["legal_marker_hits"]),
                float(ocr_report["legal_marker_hits"]),
            ),
            "symbol_density_delta": self._safe_metric_delta(
                ocr_report["symbol_ratio"],
                native_report["symbol_ratio"],
            ),
            "suspicious_symbol_density_delta": self._safe_metric_delta(
                ocr_report["suspicious_symbol_ratio"],
                native_report["suspicious_symbol_ratio"],
            ),
            "replacement_penalty_delta": self._safe_metric_delta(
                float(ocr_report["replacement_like_character_count"]),
                float(native_report["replacement_like_character_count"]),
            ),
            "semantic_risk_delta": self._safe_metric_delta(
                ocr_report["semantic_risk_score"],
                native_report["semantic_risk_score"],
            ),
            "truncation_penalty_delta": self._safe_metric_delta(
                float(ocr_report["truncation_signal_count"]),
                float(native_report["truncation_signal_count"]),
            ),
            "whitespace_stability_delta": self._safe_metric_delta(
                self._compute_whitespace_stability(native_report["whitespace_ratio"]),
                self._compute_whitespace_stability(ocr_report["whitespace_ratio"]),
            ),
        }

    def _decide_preferred_page_source(
        self,
        native_report: Dict[str, Any],
        ocr_report: Dict[str, Any],
        comparison_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Decide which page source should win the comparison.

        Parameters
        ----------
        native_report : Dict[str, Any]
            Analyzer report for the native candidate.
        ocr_report : Dict[str, Any]
            Analyzer report for the OCR candidate.
        comparison_metrics : Dict[str, float]
            Side-by-side metric deltas produced by the comparison helper.

        Returns
        -------
        Dict[str, Any]
            Decision payload with the preferred source and reason codes.
        """
        native_score = native_report["quality_score"]
        ocr_score = ocr_report["quality_score"]
        score_gap = ocr_score - native_score
        native_badness = self._compute_badness_score(native_report)
        ocr_badness = self._compute_badness_score(ocr_report)
        badness_gap = native_badness - ocr_badness

        reason_codes: List[str] = []

        if ocr_report["is_empty"] and not native_report["is_empty"]:
            reason_codes.append("ocr_empty_native_has_text")
            return {
                "preferred_source": "native",
                "preferred_mode": native_report["selected_mode"] or "native",
                "decision": "keep_native",
                "score_gap": score_gap,
                "reason_codes": reason_codes,
            }

        if native_report["is_empty"] and not ocr_report["is_empty"]:
            reason_codes.append("native_empty_ocr_has_text")
            return {
                "preferred_source": "ocr",
                "preferred_mode": ocr_report["selected_mode"] or "ocr",
                "decision": "use_ocr",
                "score_gap": score_gap,
                "reason_codes": reason_codes,
            }

        if (
            score_gap >= self.settings.hybrid_ocr_strong_signal_min_score_gap
            and badness_gap >= self.settings.hybrid_ocr_strong_signal_min_badness_gap
        ):
            reason_codes.append("ocr_clearly_higher_quality_score")

        if comparison_metrics["replacement_penalty_delta"] <= -1.0:
            reason_codes.append("native_has_more_replacement_like_characters")

        if comparison_metrics["suspicious_symbol_density_delta"] <= -0.03:
            reason_codes.append("native_has_more_suspicious_symbol_noise")

        if comparison_metrics["truncation_penalty_delta"] <= -1.0:
            reason_codes.append("native_has_more_truncation_signals")

        if comparison_metrics["line_readability_delta"] <= -0.15:
            reason_codes.append("ocr_has_more_readable_lines")

        if comparison_metrics["lexical_completeness_delta"] <= -0.12:
            reason_codes.append("ocr_has_better_lexical_completeness")

        if comparison_metrics["prose_likeness_delta"] <= -0.12:
            reason_codes.append("ocr_looks_more_like_prose")

        if comparison_metrics["legal_marker_delta"] <= -1.0:
            reason_codes.append("ocr_preserves_more_legal_markers")

        if comparison_metrics["semantic_risk_delta"] <= -1.0:
            reason_codes.append("ocr_better_preserves_legal_continuity")

        ocr_clear_win = (
            len(reason_codes) >= self.settings.hybrid_ocr_page_min_reason_count
            and score_gap >= self.settings.hybrid_ocr_page_min_score_gap
            and badness_gap >= self.settings.hybrid_ocr_page_min_badness_gap
        )
        if ocr_clear_win:
            return {
                "preferred_source": "ocr",
                "preferred_mode": ocr_report["selected_mode"] or "ocr",
                "decision": "use_ocr",
                "score_gap": score_gap,
                "reason_codes": reason_codes,
            }

        ocr_continuity_win = (
            comparison_metrics["semantic_risk_delta"] <= -1.0
            and score_gap >= self.settings.hybrid_ocr_page_min_score_gap
            and badness_gap >= self.settings.hybrid_ocr_page_min_badness_gap
            and ocr_report["semantic_risk_level"] == "acceptable"
        )
        if ocr_continuity_win:
            continuity_reasons = list(reason_codes)
            continuity_reasons.append("ocr_restores_structural_continuity")
            return {
                "preferred_source": "ocr",
                "preferred_mode": ocr_report["selected_mode"] or "ocr",
                "decision": "use_ocr",
                "score_gap": score_gap,
                "reason_codes": continuity_reasons,
            }

        ocr_less_harmful_win = (
            score_gap >= self.settings.hybrid_ocr_less_harmful_min_score_gap
            and badness_gap >= self.settings.hybrid_ocr_less_harmful_min_badness_gap
            and comparison_metrics["replacement_penalty_delta"] <= -1.0
            and comparison_metrics["line_readability_delta"] <= 0.0
            and comparison_metrics["lexical_completeness_delta"] <= 0.05
        )
        if ocr_less_harmful_win:
            less_harmful_reasons = list(reason_codes)
            less_harmful_reasons.append("ocr_is_less_harmful_on_weak_page")
            return {
                "preferred_source": "ocr",
                "preferred_mode": ocr_report["selected_mode"] or "ocr",
                "decision": "use_ocr",
                "score_gap": score_gap,
                "reason_codes": less_harmful_reasons,
            }

        keep_native_reasons: List[str] = []
        if native_report["quality_score"] >= ocr_report["quality_score"]:
            keep_native_reasons.append("native_not_worse_by_quality_score")
        if native_report["semantic_risk_level"] == "acceptable_borderline":
            keep_native_reasons.append("native_acceptable_but_borderline")
        if comparison_metrics["whitespace_stability_delta"] >= 0.08:
            keep_native_reasons.append("native_has_more_stable_whitespace")
        if comparison_metrics["line_readability_delta"] >= 0.10:
            keep_native_reasons.append("native_has_more_readable_lines")
        if comparison_metrics["lexical_completeness_delta"] >= 0.10:
            keep_native_reasons.append("native_has_better_lexical_completeness")
        if comparison_metrics["prose_likeness_delta"] >= 0.10:
            keep_native_reasons.append("native_looks_more_like_prose")

        if keep_native_reasons:
            return {
                "preferred_source": "native",
                "preferred_mode": native_report["selected_mode"] or "native",
                "decision": "keep_native",
                "score_gap": score_gap,
                "reason_codes": keep_native_reasons,
            }

        return {
            "preferred_source": "native",
            "preferred_mode": native_report["selected_mode"] or "native",
            "decision": "keep_native_not_clearly_worse",
            "score_gap": score_gap,
            "reason_codes": ["ocr_not_clearly_better"],
        }

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

    def _detect_truncation_signals(self, text: str) -> List[str]:
        """
        Detect conservative page-level truncation signals.

        Parameters
        ----------
        text : str
            Page text to inspect.

        Returns
        -------
        List[str]
            Short truncation reason codes.
        """
        signals: List[str] = []
        stripped = text.strip()
        if not stripped:
            return signals

        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return signals

        last_line = lines[-1]
        if last_line.endswith(("-", "/", "(")):
            signals.append("dangling_last_line_ending")

        if last_line and last_line[-1].isalnum() and len(last_line.split()) >= 4:
            lower_last_line = last_line.lower()
            common_legal_line_endings = (
                "artigo",
                "capítulo",
                "capitulo",
                "anexo",
                "regulamento",
                "despacho",
            )
            if not lower_last_line.endswith(common_legal_line_endings):
                signals.append("unterminated_last_line")

        if stripped.endswith((",", ";", ":")):
            signals.append("dangling_page_terminal_punctuation")

        short_lines = [line for line in lines if len(line) <= 3]
        if len(short_lines) >= 4 and len(short_lines) / len(lines) >= 0.35:
            signals.append("too_many_fragment_lines")

        return signals

    def _compute_line_readability_ratio(self, text: str) -> float:
        """
        Estimate how many lines look readable rather than symbolic noise.

        Parameters
        ----------
        text : str
            Page text to inspect.

        Returns
        -------
        float
            Ratio between 0.0 and 1.0 where larger is better.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return 0.0

        readable_lines = 0
        for line in lines:
            alpha_ratio = sum(1 for char in line if char.isalpha()) / max(len(line), 1)
            suspicious_ratio = sum(
                1 for char in line if char in {"*", "^", "_", "`", "~", "\\", "|", "<", ">"}
            ) / max(len(line), 1)
            has_word_boundary = " " in line
            if alpha_ratio >= 0.45 and suspicious_ratio <= 0.10 and has_word_boundary:
                readable_lines += 1

        return readable_lines / len(lines)

    def _compute_lexical_completeness(self, text: str) -> float:
        """
        Estimate whether page text is composed of plausible word tokens.

        Parameters
        ----------
        text : str
            Page text to inspect.

        Returns
        -------
        float
            Ratio between 0.0 and 1.0 where larger is better.
        """
        words = [word.strip(".,;:()[]{}\"'") for word in text.split() if word.strip()]
        if not words:
            return 0.0

        plausible_words = 0
        vowels = set("aeiouáàâãéêíóôõúü")
        for word in words:
            alpha_chars = [char for char in word if char.isalpha()]
            if len(alpha_chars) < 2:
                continue
            if any(char.lower() in vowels for char in alpha_chars):
                plausible_words += 1

        return plausible_words / len(words)

    def _compute_prose_likeness(
        self,
        alpha_ratio: float,
        whitespace_ratio: float,
        symbol_ratio: float,
        suspicious_symbol_ratio: float,
        single_character_token_ratio: float,
        line_readability_ratio: float,
        lexical_completeness: float,
    ) -> float:
        """
        Estimate how much a page resembles readable prose.

        Parameters
        ----------
        alpha_ratio : float
            Ratio of alphabetic characters.
        whitespace_ratio : float
            Ratio of whitespace characters.
        symbol_ratio : float
            Ratio of symbolic characters.
        suspicious_symbol_ratio : float
            Ratio of suspicious symbolic characters.
        single_character_token_ratio : float
            Ratio of single-character tokens.
        line_readability_ratio : float
            Ratio of readable lines.
        lexical_completeness : float
            Ratio of plausible lexical tokens.

        Returns
        -------
        float
            Prose-likeness score clamped between 0.0 and 1.0.
        """
        prose_score = (
            (alpha_ratio * 0.25)
            + (self._compute_whitespace_stability(whitespace_ratio) * 0.10)
            + (line_readability_ratio * 0.25)
            + (lexical_completeness * 0.25)
            + ((1.0 - min(symbol_ratio, 1.0)) * 0.05)
            + ((1.0 - min(suspicious_symbol_ratio * 5.0, 1.0)) * 0.05)
            + ((1.0 - min(single_character_token_ratio, 1.0)) * 0.05)
        )
        return max(0.0, min(prose_score, 1.0))

    def _compute_whitespace_stability(self, whitespace_ratio: float) -> float:
        """
        Score how close whitespace density is to readable prose.

        Parameters
        ----------
        whitespace_ratio : float
            Ratio of whitespace characters in the page text.

        Returns
        -------
        float
            Score between 0.0 and 1.0 where larger is better.
        """
        distance = abs(whitespace_ratio - 0.16)
        return max(0.0, 1.0 - (distance / 0.16))

    def _compute_badness_score(self, page_report: Dict[str, Any]) -> float:
        """
        Aggregate degradation signals for comparison decisions.

        Parameters
        ----------
        page_report : Dict[str, Any]
            Analyzer report for a single page.

        Returns
        -------
        float
            Practical badness score where larger means worse.
        """
        badness = 0.0
        badness += float(len(page_report["garbled_reasons"]))
        badness += page_report["replacement_like_character_count"] * 0.75
        badness += page_report["truncation_signal_count"] * 0.75
        badness += page_report["semantic_risk_score"] * 1.25
        badness += page_report["suspicious_symbol_ratio"] * 8.0
        badness += max(0.0, 0.55 - page_report["line_readability_ratio"]) * 4.0
        badness += max(0.0, 0.50 - page_report["lexical_completeness"]) * 4.0
        badness += max(0.0, 0.50 - page_report["prose_likeness"]) * 4.0
        if page_report["is_empty"]:
            badness += 6.0
        if page_report["is_locally_unreliable"]:
            badness += 2.0
        return badness

    def _safe_metric_delta(self, left: float, right: float) -> float:
        """
        Compute a deterministic metric delta while guarding invalid numbers.

        Parameters
        ----------
        left : float
            Left-hand metric value.
        right : float
            Right-hand metric value.

        Returns
        -------
        float
            Difference `left - right`, or 0.0 for non-finite inputs.
        """
        if not isfinite(left) or not isfinite(right):
            return 0.0
        return left - right

    def _compute_quality_score(
        self,
        text: str,
        alpha_ratio: float,
        digit_ratio: float,
        whitespace_ratio: float,
        symbol_ratio: float,
        suspicious_symbol_ratio: float,
        replacement_like_count: int,
        single_character_token_ratio: float,
        legal_marker_hits: int,
        lexical_completeness: float,
        line_readability_ratio: float,
        truncation_signal_count: int,
        prose_likeness: float,
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
        symbol_ratio : float
            Ratio of symbolic characters.

        suspicious_symbol_ratio : float
            Ratio of suspicious symbols.

        replacement_like_count : int
            Count of replacement-like glyphs.

        single_character_token_ratio : float
            Ratio of one-character tokens.

        legal_marker_hits : int
            Number of domain-typical legal markers.

        lexical_completeness : float
            Ratio of plausible lexical tokens.

        line_readability_ratio : float
            Ratio of readable lines.

        truncation_signal_count : int
            Number of conservative truncation indicators.

        prose_likeness : float
            Composite prose-likeness score.

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
        score += lexical_completeness * 35.0
        score += line_readability_ratio * 25.0
        score += prose_likeness * 30.0

        # Penalize suspicious signals.
        score -= symbol_ratio * 30.0
        score -= suspicious_symbol_ratio * 180.0
        score -= replacement_like_count * 12.0
        score -= truncation_signal_count * 10.0

        if digit_ratio > 0.35:
            score -= 20.0

        if alpha_ratio < 0.30:
            score -= 80.0

        if single_character_token_ratio > 0.35:
            score -= 35.0

        # Reward domain plausibility.
        score += legal_marker_hits * 8.0

        return score
