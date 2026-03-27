from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from Chunking.chunking.models import Chunk
from Chunking.config.settings import PipelineSettings
from Chunking.utils.text import (
    fold_editorial_text,
    has_suspicious_truncated_ending,
    join_hyphenated_linebreaks,
    repair_broken_enclitic_hyphenation,
    repair_broken_inline_hyphenation,
)


# ============================================================================
# Validator patterns
# ============================================================================
#
# This module validates already generated chunks.
#
# It is intentionally deterministic and explainable:
# - each rule maps to a concrete issue code
# - each issue carries short evidence when available
# - the aggregated output stays JSON-friendly for later pipeline reporting
# ============================================================================

EDITORIAL_DATE_RE = re.compile(r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}\b")
DR_EDITORIAL_LINE_RE = re.compile(
    r"^\s*(N\.?\s*º|PARTE\s+[A-Z]|Di[aá]rio\s+da\s+Rep[úu]blica)\b",
    re.IGNORECASE,
)
PAGE_COUNTER_LINE_RE = re.compile(r"^\s*\d+\s*\|\s*\d+\s*[\-–—.]?\s*$")
LEADING_PAGE_MARKER_RE = re.compile(r"^\s*Pág\.\s*\d+\s*", re.IGNORECASE)
ARTIFICIAL_STRUCTURE_TOKEN_RE = re.compile(
    r"\b(?:CAP|ART|SEC|SUBSEC|TIT|ANX)_[A-Z0-9_]+\b",
    re.IGNORECASE,
)
BROKEN_INLINE_HYPHENATION_RE = re.compile(
    r"\b([A-Za-zÀ-ÿ]{2,})-[ \t]+([a-zà-ÿ]{3,})\b"
)
BROKEN_ENCLITIC_HYPHEN_RE = re.compile(
    r"\b([A-Za-zÀ-ÿ]{3,})(?:\s+-\s*|-\s+)"
    r"(se|lo|la|los|las|lhe|lhes|me|te|nos|vos)\b",
    re.IGNORECASE,
)
HYPHENATED_LINEBREAK_RE = re.compile(r"([A-Za-zÀ-ÿ])-\n([A-Za-zÀ-ÿ])")
NUMBERED_ITEM_PREFIX_RE = re.compile(
    r"^\s*(?:n\.?\s*[ºo]\s*)?\d+(?:\.\d+)*(?:\.\s+|\)\s+|\s+[—–\-]\s+)",
    re.IGNORECASE,
)
LETTERED_ITEM_PREFIX_RE = re.compile(r"^\s*[a-z]\)\s+", re.IGNORECASE)
DASH_CONTINUATION_PREFIX_RE = re.compile(r"^\s*[—–-]\s+")
ACCESS_FOOTNOTE_RE = re.compile(
    r"^\s*\(?\d+\)?\s+"
    r"(?:Acess[íi]vel|Dispon[íi]vel|Publicado|Publicada|Publicados|Publicadas)\b",
    re.IGNORECASE,
)
FOOTNOTE_URL_RE = re.compile(r"^\s*\(?\d+\)?\s+.*(?:https?://|www\.)", re.IGNORECASE)
STANDALONE_URL_LINE_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
SUSPICIOUS_GARBLED_LINE_RE = re.compile(
    r"^[^A-Za-zÀ-ÿ]{0,3}(?:[\*\+\-/=<>\\\[\]\{\}_`~]{2,}|[0-9\W]{12,})$"
)
TITLE_SEPARATOR_RE = re.compile(r"\s*\|\s*")
DOCUMENT_TITLE_RESIDUE_RE = re.compile(
    r"^\s*"
    r"(?:Regulamento|Despacho|Delibera(?:ç|c)ão|Edital|Anexo)"
    r"(?:\s+(?:de\b.*|n\.?\s*[ºo].*|P\.?\s*PORTO.*|[A-Z0-9].*))?"
    r"\s*$",
    re.IGNORECASE,
)

TRACEABILITY_METADATA_KEYS = (
    "article_number",
    "article_label",
    "article_title",
    "section_labels",
    "lettered_labels",
    "document_part",
    "source_node_id",
    "parent_node_id",
)


class ChunkQualityValidator:
    """
    Validate chunk outputs against explicit chunk-quality rules.

    Purpose
    -------
    Manual inspection remains useful, but downstream acceptance should not rely
    only on reading `05_chunks.json` by eye. This validator applies a small and
    deterministic rule set aligned with the repository chunk-quality guide.

    Current validation scope
    ------------------------
    The validator detects at least:
    - editorial noise still present in visible chunk text
    - structural pollution in `text_for_embedding`
    - oversized chunks
    - broken hyphenation patterns still visible in output
    - missing minimum traceability fields
    """

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """
        Initialize validator settings.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Runtime settings used to evaluate chunk-size limits.
        """
        self.settings = settings or PipelineSettings()

    def validate_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Validate one chunk and return a machine-readable report.

        Parameters
        ----------
        chunk : Chunk
            Chunk to validate.

        Returns
        -------
        Dict[str, Any]
            Per-chunk validation report with issue details.
        """
        issues: List[Dict[str, Any]] = []

        editorial_issue = self._validate_editorial_noise_in_text(chunk)
        if editorial_issue is not None:
            issues.append(editorial_issue)

        embedding_issue = self._validate_structural_pollution_in_embedding(chunk)
        if embedding_issue is not None:
            issues.append(embedding_issue)

        oversized_issue = self._validate_chunk_size(chunk)
        if oversized_issue is not None:
            issues.append(oversized_issue)

        broken_hyphenation_issue = self._validate_broken_hyphenation(chunk)
        if broken_hyphenation_issue is not None:
            issues.append(broken_hyphenation_issue)

        footnote_issue = self._validate_note_or_footnote_leakage(chunk)
        if footnote_issue is not None:
            issues.append(footnote_issue)

        document_title_issue = self._validate_document_title_residue(chunk)
        if document_title_issue is not None:
            issues.append(document_title_issue)

        title_leakage_issue = self._validate_title_leakage(chunk)
        if title_leakage_issue is not None:
            issues.append(title_leakage_issue)

        garbled_text_issue = self._validate_garbled_text(chunk)
        if garbled_text_issue is not None:
            issues.append(garbled_text_issue)

        truncated_ending_issue = self._validate_truncated_or_abrupt_ending(chunk)
        if truncated_ending_issue is not None:
            issues.append(truncated_ending_issue)

        undersized_issue = self._validate_problematic_undersized_chunk(chunk)
        if undersized_issue is not None:
            issues.append(undersized_issue)

        autonomy_issue = self._validate_low_semantic_autonomy(chunk)
        if autonomy_issue is not None:
            issues.append(autonomy_issue)

        traceability_issue = self._validate_traceability(chunk)
        if traceability_issue is not None:
            issues.append(traceability_issue)

        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "strategy": chunk.strategy,
            "is_valid": not issues,
            "issue_count": len(issues),
            "issue_codes": [issue["code"] for issue in issues],
            "issues": issues,
        }

    def validate_chunks(self, chunks: Sequence[Chunk]) -> Dict[str, Any]:
        """
        Validate a chunk sequence and aggregate counts by failure type.

        Parameters
        ----------
        chunks : Sequence[Chunk]
            Chunks to validate.

        Returns
        -------
        Dict[str, Any]
            Aggregate summary plus per-chunk reports.
        """
        reports = [self.validate_chunk(chunk) for chunk in chunks]

        issue_type_counts: Dict[str, int] = {}
        issue_examples: Dict[str, List[str]] = {}

        for report in reports:
            for issue in report["issues"]:
                issue_code = issue["code"]
                issue_type_counts[issue_code] = issue_type_counts.get(issue_code, 0) + 1

                example_chunk_ids = issue_examples.setdefault(issue_code, [])
                if (
                    report["chunk_id"]
                    and report["chunk_id"] not in example_chunk_ids
                    and len(example_chunk_ids) < 5
                ):
                    example_chunk_ids.append(report["chunk_id"])

        invalid_chunk_count = sum(1 for report in reports if not report["is_valid"])

        return {
            "chunk_count": len(reports),
            "valid_chunk_count": len(reports) - invalid_chunk_count,
            "invalid_chunk_count": invalid_chunk_count,
            "overall_status": "pass" if invalid_chunk_count == 0 else "fail",
            "issue_type_counts": issue_type_counts,
            "issue_examples": issue_examples,
            "chunk_reports": reports,
        }

    def _validate_editorial_noise_in_text(
        self,
        chunk: Chunk,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect editorial or page-furniture residue still present in `text`.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when editorial residue is detected.
        """
        matched_lines: List[str] = []

        for raw_line in chunk.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if self._looks_like_editorial_line(line):
                matched_lines.append(line)

        if not matched_lines:
            return None

        return {
            "code": "editorial_noise_in_text",
            "severity": "error",
            "message": "Visible chunk text still contains editorial or page-furniture residue.",
            "evidence": matched_lines[:3],
        }

    def _validate_structural_pollution_in_embedding(
        self,
        chunk: Chunk,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect structural pollution inside `text_for_embedding`.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when embedding text carries avoidable pollution.
        """
        embedding_text = getattr(chunk, "text_for_embedding", "") or ""
        if not embedding_text.strip():
            return None

        evidence: List[str] = []

        if " | " in embedding_text:
            evidence.append("contains artificial title separator ' | '")

        if ARTIFICIAL_STRUCTURE_TOKEN_RE.search(embedding_text):
            evidence.append("contains artificial structural token")

        for raw_line in embedding_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if self._looks_like_editorial_line(line):
                evidence.append(f"editorial line: {line}")
                break

        if not evidence:
            return None

        return {
            "code": "structural_pollution_in_embedding",
            "severity": "error",
            "message": "Embedding text still contains structural or editorial pollution.",
            "evidence": evidence[:3],
        }

    def _validate_chunk_size(self, chunk: Chunk) -> Optional[Dict[str, Any]]:
        """
        Detect chunks that exceed the configured hard size limit.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when the chunk is oversized.
        """
        visible_length = len(chunk.text or "")
        configured_length = getattr(chunk, "char_count", 0) or visible_length

        if configured_length <= self.settings.hard_max_chunk_chars:
            return None

        return {
            "code": "oversized_chunk",
            "severity": "error",
            "message": "Chunk exceeds the configured hard maximum size.",
            "evidence": {
                "char_count": configured_length,
                "hard_max_chunk_chars": self.settings.hard_max_chunk_chars,
            },
        }

    def _validate_broken_hyphenation(
        self,
        chunk: Chunk,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect repairable broken hyphenation that should not survive output.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when broken hyphenation patterns are still visible.
        """
        evidence: List[str] = []

        for text_field in (chunk.text, getattr(chunk, "text_for_embedding", "")):
            if not text_field:
                continue

            evidence.extend(self._collect_broken_hyphenation_evidence(text_field))

        unique_evidence = list(dict.fromkeys(evidence))
        if not unique_evidence:
            return None

        return {
            "code": "broken_hyphenation_in_output",
            "severity": "error",
            "message": "Chunk output still contains repairable broken hyphenation.",
            "evidence": unique_evidence[:3],
        }

    def _validate_note_or_footnote_leakage(
        self,
        chunk: Chunk,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect publication or access notes that should not survive in `text`.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when note-like leakage remains visible.
        """
        evidence: List[str] = []

        for raw_line in chunk.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if (
                ACCESS_FOOTNOTE_RE.match(line)
                or FOOTNOTE_URL_RE.match(line)
                or STANDALONE_URL_LINE_RE.match(line)
            ):
                evidence.append(line)

        if not evidence:
            return None

        return {
            "code": "note_or_footnote_in_text",
            "severity": "error",
            "message": "Visible chunk text still contains note or footnote residue.",
            "evidence": evidence[:3],
        }

    def _validate_document_title_residue(
        self,
        chunk: Chunk,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect standalone document-title residue that leaked into chunk text.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when edge lines still look like title residue.
        """
        evidence: List[str] = []

        for line in self._get_edge_lines(chunk.text):
            if self._looks_like_document_title_residue(line):
                evidence.append(line)

        if not evidence:
            return None

        return {
            "code": "document_title_residue_in_text",
            "severity": "error",
            "message": "Visible chunk text still contains leaked document-title residue.",
            "evidence": list(dict.fromkeys(evidence))[:2],
        }

    def _validate_title_leakage(self, chunk: Chunk) -> Optional[Dict[str, Any]]:
        """
        Detect article titles that leaked into visible body text.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when the chunk starts with its own title.
        """
        article_title = self._get_article_title(chunk)
        if not article_title:
            return None

        first_line = self._get_first_non_empty_line(chunk.text)
        if not first_line:
            return None

        normalized_title = self._normalize_title_signature(article_title)
        normalized_first_line = self._normalize_title_signature(first_line)

        if not normalized_title or len(normalized_title) < 6:
            return None

        if normalized_first_line == normalized_title:
            evidence = first_line
        elif normalized_first_line.startswith(normalized_title):
            evidence = first_line
        else:
            return None

        return {
            "code": "title_leakage_in_text",
            "severity": "error",
            "message": "Visible chunk text still starts with leaked title content.",
            "evidence": [evidence],
        }

    def _validate_garbled_text(self, chunk: Chunk) -> Optional[Dict[str, Any]]:
        """
        Detect clearly garbled lines that remain in chunk text.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when visible text still contains corruption.
        """
        evidence: List[str] = []

        for raw_line in chunk.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if SUSPICIOUS_GARBLED_LINE_RE.match(line):
                evidence.append(line)

        if not evidence:
            return None

        return {
            "code": "garbled_text_in_chunk",
            "severity": "error",
            "message": "Visible chunk text still contains garbled or non-semantic residue.",
            "evidence": evidence[:3],
        }

    def _validate_truncated_or_abrupt_ending(
        self,
        chunk: Chunk,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect chunk endings that still look damaged or abruptly cut.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when the final visible ending looks truncated.
        """
        if not has_suspicious_truncated_ending(chunk.text):
            return None

        return {
            "code": "truncated_or_abrupt_chunk_ending",
            "severity": "error",
            "message": "Chunk ends with a suspicious truncated or abrupt fragment.",
            "evidence": [self._get_last_non_empty_line(chunk.text)],
        }

    def _validate_problematic_undersized_chunk(
        self,
        chunk: Chunk,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect short split fragments that are unlikely to stand alone well.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when a short split fragment looks semantically weak.
        """
        if not self._is_split_chunk(chunk):
            return None

        visible_length = len((chunk.text or "").strip())
        short_threshold = min(self.settings.min_chunk_chars, 180)
        if visible_length >= short_threshold:
            return None

        signals = self._collect_semantic_weakness_signals(chunk)
        if not signals:
            return None

        return {
            "code": "problematic_undersized_chunk",
            "severity": "error",
            "message": "Chunk is an undersized split fragment with weak standalone quality.",
            "evidence": {
                "char_count": visible_length,
                "threshold": short_threshold,
                "signals": signals[:3],
            },
        }

    def _validate_low_semantic_autonomy(
        self,
        chunk: Chunk,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect split chunks that still read like semantically orphaned fragments.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when the chunk lacks enough standalone context.
        """
        if not self._is_split_chunk(chunk):
            return None

        signals = self._collect_semantic_weakness_signals(chunk)
        if (
            "starts_with_numbered_item" not in signals
            and "starts_with_lettered_item" not in signals
            and "starts_with_dash_continuation" not in signals
        ):
            return None

        if "starts_with_dash_continuation" in signals:
            return {
                "code": "low_semantic_autonomy_chunk",
                "severity": "error",
                "message": "Split chunk still behaves like a semantically orphaned fragment.",
                "evidence": {
                    "chunk_reason": getattr(chunk, "chunk_reason", ""),
                    "signals": signals[:4],
                    "opening_line": self._get_first_non_empty_line(chunk.text),
                },
            }

        if "too_few_words" not in signals and "missing_terminal_punctuation" not in signals:
            return None

        return {
            "code": "low_semantic_autonomy_chunk",
            "severity": "error",
            "message": "Split chunk still behaves like a semantically orphaned fragment.",
            "evidence": {
                "chunk_reason": getattr(chunk, "chunk_reason", ""),
                "signals": signals[:4],
                "opening_line": self._get_first_non_empty_line(chunk.text),
            },
        }

    def _collect_broken_hyphenation_evidence(self, text: str) -> List[str]:
        """
        Collect only hyphenation patterns that the shared repair helpers change.

        Why compare with repair helpers
        -------------------------------
        The validator should flag only repairable defects, not every string that
        superficially resembles a hyphenated word. Legitimate forms such as
        "inscrever-se" must not be reported as broken hyphenation.

        Parameters
        ----------
        text : str
            Candidate text field from one chunk.

        Returns
        -------
        List[str]
            Concrete matched evidence snippets that remain repairable.
        """
        evidence: List[str] = []

        evidence.extend(
            self._collect_repairable_matches(
                text=text,
                pattern=HYPHENATED_LINEBREAK_RE,
                repair_func=join_hyphenated_linebreaks,
                escape_newlines=True,
            )
        )
        evidence.extend(
            self._collect_repairable_matches(
                text=text,
                pattern=BROKEN_INLINE_HYPHENATION_RE,
                repair_func=repair_broken_inline_hyphenation,
            )
        )
        evidence.extend(
            self._collect_repairable_matches(
                text=text,
                pattern=BROKEN_ENCLITIC_HYPHEN_RE,
                repair_func=repair_broken_enclitic_hyphenation,
            )
        )

        return evidence

    def _collect_repairable_matches(
        self,
        text: str,
        pattern: re.Pattern[str],
        repair_func: Callable[[str], str],
        escape_newlines: bool = False,
    ) -> List[str]:
        """
        Return only pattern matches whose matched snippet changes after repair.

        Parameters
        ----------
        text : str
            Candidate text field.

        pattern : re.Pattern[str]
            Regex used to locate suspicious substrings.

        repair_func : Any
            Shared repair helper applied to the matched substring.

        escape_newlines : bool, default=False
            Whether to render newline evidence in escaped form.

        Returns
        -------
        List[str]
            Evidence snippets that remain truly repairable.
        """
        evidence: List[str] = []

        for match in pattern.finditer(text):
            matched_text = match.group(0)
            repaired_text = repair_func(matched_text)

            if repaired_text == matched_text:
                continue

            if escape_newlines:
                matched_text = matched_text.replace("\n", "\\n")

            evidence.append(matched_text)

        return evidence

    def _collect_semantic_weakness_signals(self, chunk: Chunk) -> List[str]:
        """
        Collect deterministic signals for weak standalone chunk quality.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        List[str]
            Short signal labels explaining why the chunk looks weak.
        """
        text = (chunk.text or "").strip()
        if not text:
            return ["empty_text"]

        first_line = self._get_first_non_empty_line(text)
        word_count = len(re.findall(r"\b[A-Za-zÀ-ÿ0-9]+\b", text))
        signals: List[str] = []

        if NUMBERED_ITEM_PREFIX_RE.match(first_line):
            signals.append("starts_with_numbered_item")

        if LETTERED_ITEM_PREFIX_RE.match(first_line):
            signals.append("starts_with_lettered_item")

        if DASH_CONTINUATION_PREFIX_RE.match(first_line):
            signals.append("starts_with_dash_continuation")

        if word_count < 8:
            signals.append("too_few_words")

        if text[-1] not in ".!?;:":
            signals.append("missing_terminal_punctuation")

        return signals

    def _is_split_chunk(self, chunk: Chunk) -> bool:
        """
        Decide whether the chunk was produced as part of a larger split unit.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        bool
            True when metadata or chunk reason indicate split output.
        """
        metadata = getattr(chunk, "metadata", {}) or {}
        if metadata.get("part_count", 1) and metadata.get("part_count", 1) > 1:
            return True

        chunk_reason = getattr(chunk, "chunk_reason", "") or ""
        return "split" in chunk_reason

    def _get_article_title(self, chunk: Chunk) -> str:
        """
        Return the best available article title candidate from chunk metadata.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        str
            Article title candidate when available.
        """
        metadata = getattr(chunk, "metadata", {}) or {}
        return str(metadata.get("article_title") or "").strip()

    def _normalize_title_signature(self, text: str) -> str:
        """
        Normalize title-like text into a comparison-friendly signature.

        Parameters
        ----------
        text : str
            Candidate title or line.

        Returns
        -------
        str
            Folded signature for conservative prefix comparison.
        """
        normalized = TITLE_SEPARATOR_RE.sub(" ", text or "")
        normalized = normalized.strip(" .;:-")
        return self._fold_editorial_text(normalized)

    def _get_first_non_empty_line(self, text: str) -> str:
        """
        Return the first non-empty line from a visible chunk block.

        Parameters
        ----------
        text : str
            Candidate text block.

        Returns
        -------
        str
            First non-empty line, or an empty string.
        """
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if line:
                return line

        return ""

    def _get_last_non_empty_line(self, text: str) -> str:
        """
        Return the last non-empty line from a visible chunk block.

        Parameters
        ----------
        text : str
            Candidate text block.

        Returns
        -------
        str
            Last non-empty line, or an empty string.
        """
        for raw_line in reversed((text or "").splitlines()):
            line = raw_line.strip()
            if line:
                return line

        return ""

    def _get_edge_lines(self, text: str) -> List[str]:
        """
        Return the first and last visible lines for edge-focused validation.

        Parameters
        ----------
        text : str
            Candidate text block.

        Returns
        -------
        List[str]
            Deduplicated edge lines in display order.
        """
        first_line = self._get_first_non_empty_line(text)
        last_line = self._get_last_non_empty_line(text)

        edge_lines: List[str] = []
        if first_line:
            edge_lines.append(first_line)

        if last_line and last_line != first_line:
            edge_lines.append(last_line)

        return edge_lines

    def _validate_traceability(self, chunk: Chunk) -> Optional[Dict[str, Any]]:
        """
        Detect missing minimum traceability fields.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        Optional[Dict[str, Any]]
            Issue payload when minimum traceability is incomplete.
        """
        missing_fields: List[str] = []

        if not chunk.chunk_id:
            missing_fields.append("chunk_id")

        if not chunk.doc_id:
            missing_fields.append("doc_id")

        if chunk.page_start is None and chunk.page_end is None:
            missing_fields.append("page_mapping")

        if not self._has_structure_context(chunk):
            missing_fields.append("structure_context")

        if not missing_fields:
            return None

        return {
            "code": "missing_traceability_fields",
            "severity": "error",
            "message": "Chunk is missing minimum traceability information.",
            "evidence": {"missing_fields": missing_fields},
        }

    def _has_structure_context(self, chunk: Chunk) -> bool:
        """
        Decide whether a chunk carries minimum structural traceability.

        Parameters
        ----------
        chunk : Chunk
            Chunk under validation.

        Returns
        -------
        bool
            True when the chunk can still be mapped to document structure.
        """
        if getattr(chunk, "hierarchy_path", None):
            return True

        if getattr(chunk, "source_node_type", "") or getattr(chunk, "source_node_label", ""):
            return True

        metadata = getattr(chunk, "metadata", {}) or {}
        return any(metadata.get(key) for key in TRACEABILITY_METADATA_KEYS)

    def _looks_like_editorial_line(self, line: str) -> bool:
        """
        Detect short editorial or page-furniture lines conservatively.

        Parameters
        ----------
        line : str
            Candidate line.

        Returns
        -------
        bool
            True when the line behaves like editorial residue.
        """
        if not line:
            return False

        if PAGE_COUNTER_LINE_RE.match(line):
            return True

        if LEADING_PAGE_MARKER_RE.match(line):
            return True

        if DR_EDITORIAL_LINE_RE.match(line):
            return True

        folded_line = self._fold_editorial_text(line)
        if not folded_line:
            return False

        if "diariodarepublica" in folded_line and EDITORIAL_DATE_RE.search(line):
            return True

        if (
            "diario" in folded_line
            and "republica" in folded_line
            and EDITORIAL_DATE_RE.search(line)
        ):
            return True

        return False

    def _looks_like_document_title_residue(self, line: str) -> bool:
        """
        Detect leaked document-title lines conservatively at chunk edges.

        Parameters
        ----------
        line : str
            Candidate edge line.

        Returns
        -------
        bool
            True when the line behaves like standalone title residue.
        """
        stripped_line = line.strip()
        if not stripped_line or stripped_line[-1] in ".!?;:":
            return False

        word_count = len(re.findall(r"\b[A-Za-zÀ-ÿ0-9]+\b", stripped_line))
        if word_count == 0 or word_count > 16:
            return False

        return DOCUMENT_TITLE_RESIDUE_RE.match(stripped_line) is not None

    def _fold_editorial_text(self, text: str) -> str:
        """
        Fold text into a comparison-friendly editorial signature.

        Parameters
        ----------
        text : str
            Candidate text.

        Returns
        -------
        str
            Lowercased alphanumeric-only signature.
        """
        return fold_editorial_text(text, preserve_word_boundaries=False)
