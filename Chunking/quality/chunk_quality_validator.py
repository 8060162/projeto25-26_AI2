from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence

from Chunking.chunking.models import Chunk
from Chunking.config.settings import PipelineSettings
from Chunking.utils.text import (
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
        normalized = unicodedata.normalize("NFKD", text)
        without_marks = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
        lowered = without_marks.lower()
        return re.sub(r"[^a-z0-9]+", "", lowered)
