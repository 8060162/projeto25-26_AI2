from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

from Chunking.config.settings import PipelineSettings
from retrieval.models import (
    DiagnosticSignal,
    GroundingVerificationResult,
    RetrievalContext,
)


_ARTICLE_REFERENCE_PATTERN = re.compile(
    r"\b(?:artigo|article|art)\.?\s*(?:n[.\u00bao]*\s*)?(\d+[a-z]?)\b",
    re.IGNORECASE,
)
_ARTICLE_EQUALS_PATTERN = re.compile(r"\barticle\s*=\s*(\d+[a-z]?)\b", re.IGNORECASE)
_ARTICLE_NUMBER_EQUALS_PATTERN = re.compile(
    r"\barticle_number\s*=\s*(\d+[a-z]?)\b",
    re.IGNORECASE,
)
_DOCUMENT_TITLE_EQUALS_PATTERN = re.compile(
    r"\bdocument_title\s*=\s*([^\n\]|]+)",
    re.IGNORECASE,
)
_LEGAL_ANCHOR_EQUALS_PATTERN = re.compile(
    r"\blegal_anchor\s*=\s*([^\n\]|]+)",
    re.IGNORECASE,
)
_DOCUMENT_REFERENCE_PATTERN = re.compile(
    r"\b(?:Regulation|Regulamento|Despacho)\s+[^\n.;:,()\[\]|]+",
    re.IGNORECASE,
)
_LEGAL_NUMERIC_CLAIM_PATTERN = re.compile(
    r"\b\d+(?:[,.]\d+)?\s*(?:dias?|days?|horas?|hours?|mes|meses|months?|"
    r"anos?|years?|creditos?|credits?|ects|euros?|prestacoes?|installments?|"
    r"%|por cento)\b"
)
_PERCENT_VALUE_PATTERN = re.compile(
    r"\b(\d+(?:[,.]\d+)?)\s*(?:%|por cento)(?=\s|$)"
)
_INSTALLMENT_CLAIM_PATTERN = re.compile(
    r"^(\d+)\s+(?:prestacoes?|installments?)$"
)
_INSTALLMENT_RANGE_PATTERN = re.compile(
    r"\b(\d+)\s*\.?\s*[ao]?\s*(?:a|ate|-)\s*"
    r"(\d+)\s*\.?\s*[ao]?\s*(?:prestacao|prestacoes|installment|installments)\b"
)
_INSTALLMENT_REFERENCE_PATTERN = re.compile(
    r"\b(\d+)\s*\.?\s*[ao]?\s*(?:prestacao|prestacoes|installment|installments)\b"
)
_INSTALLMENT_TOTAL_PATTERN = re.compile(
    r"\b(?:em|in)\s+(\d+)\s+(?:prestacoes|installments)\b"
)
_ARTICLE_INTENT_PATTERN = re.compile(r"\b(?:artigo|article|art)\b", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_SYMBOL_TEXT_PATTERN = re.compile(r"[a-z0-9%]+")
_LOW_SIGNAL_CLAIM_TOKENS = frozenset(
    {
        "acordo",
        "aplicavel",
        "art",
        "artigo",
        "article",
        "com",
        "contexto",
        "deve",
        "documento",
        "nos",
        "para",
        "pelo",
        "pela",
        "por",
        "regulamento",
        "termos",
    }
)


def _normalize_comparison_text(value: str) -> str:
    """
    Normalize text into an accent-free lowercase comparison form.

    Parameters
    ----------
    value : str
        Raw text value.

    Returns
    -------
    str
        Normalized alphanumeric text suitable for deterministic matching.
    """

    normalized_value = unicodedata.normalize("NFKD", value)
    ascii_value = normalized_value.encode("ascii", "ignore").decode("ascii")
    return " ".join(_TOKEN_PATTERN.findall(ascii_value.lower()))


def _normalize_symbol_text(value: str) -> str:
    """
    Normalize text while preserving percent signs for schedule checks.

    Parameters
    ----------
    value : str
        Raw text value.

    Returns
    -------
    str
        Accent-free lowercase text with alphanumeric tokens and percentages.
    """

    normalized_value = unicodedata.normalize("NFKD", value)
    ascii_value = normalized_value.encode("ascii", "ignore").decode("ascii")
    return " ".join(_SYMBOL_TEXT_PATTERN.findall(ascii_value.lower()))


def _normalize_anchor(value: object) -> str:
    """
    Normalize one citation anchor into a compact comparison key.

    Parameters
    ----------
    value : object
        Candidate anchor value.

    Returns
    -------
    str
        Normalized anchor key, or an empty string when invalid.
    """

    if not isinstance(value, str):
        return ""

    return _normalize_comparison_text(value.strip())


def _normalize_metadata_scalar(candidate: object) -> str:
    """
    Normalize one scalar metadata value into a string.

    Parameters
    ----------
    candidate : object
        Candidate metadata value.

    Returns
    -------
    str
        String representation for string and numeric scalar metadata.
    """

    if isinstance(candidate, str):
        return candidate.strip()
    if isinstance(candidate, bool) or candidate is None:
        return ""
    if isinstance(candidate, int):
        return str(candidate)
    if isinstance(candidate, float) and candidate.is_integer():
        return str(int(candidate))
    return ""


def _append_unique(values: List[str], candidate: object) -> None:
    """
    Append one candidate string to a list when it is non-empty and unique.

    Parameters
    ----------
    values : List[str]
        Mutable destination list.

    candidate : object
        Candidate value to normalize and append.
    """

    normalized_candidate = _normalize_metadata_scalar(candidate)
    comparison_candidate = _normalize_anchor(normalized_candidate)

    if not normalized_candidate or not comparison_candidate:
        return

    if comparison_candidate in {_normalize_anchor(value) for value in values}:
        return

    values.append(normalized_candidate)


def _anchors_overlap(candidate_key: str, available_keys: Set[str]) -> bool:
    """
    Check whether one normalized anchor is covered by available context anchors.

    Parameters
    ----------
    candidate_key : str
        Normalized citation anchor extracted from the answer.

    available_keys : Set[str]
        Normalized grounding anchors extracted from the selected context.

    Returns
    -------
    bool
        `True` when the anchor matches exactly or when a sufficiently specific
        document title is contained in a longer context label.
    """

    if not candidate_key:
        return False
    if candidate_key in available_keys:
        return True

    candidate_tokens = candidate_key.split()
    if len(candidate_tokens) < 2:
        return False

    for available_key in available_keys:
        if candidate_key in available_key or available_key in candidate_key:
            return True

    return False


def _extract_article_numbers(text_values: Iterable[str]) -> List[str]:
    """
    Extract cited article numbers from free text and citation fragments.

    Parameters
    ----------
    text_values : Iterable[str]
        Text fragments to inspect.

    Returns
    -------
    List[str]
        Ordered unique article numbers.
    """

    article_numbers: List[str] = []
    seen_numbers: Set[str] = set()

    for text_value in text_values:
        for pattern in (
            _ARTICLE_REFERENCE_PATTERN,
            _ARTICLE_EQUALS_PATTERN,
            _ARTICLE_NUMBER_EQUALS_PATTERN,
        ):
            for match in pattern.findall(text_value or ""):
                article_number = str(match).strip().lower()
                if article_number and article_number not in seen_numbers:
                    article_numbers.append(article_number)
                    seen_numbers.add(article_number)

    return article_numbers


def _extract_document_references(text_values: Iterable[str]) -> List[str]:
    """
    Extract explicit document-like references from answer text.

    Parameters
    ----------
    text_values : Iterable[str]
        Text fragments to inspect.

    Returns
    -------
    List[str]
        Ordered unique document references.
    """

    document_references: List[str] = []

    for text_value in text_values:
        for match in _DOCUMENT_REFERENCE_PATTERN.findall(text_value or ""):
            _append_unique(document_references, match)

    return document_references


def _extract_context_document_labels(text_values: Iterable[str]) -> List[str]:
    """
    Extract document labels from serialized retrieval-context metadata.

    Parameters
    ----------
    text_values : Iterable[str]
        Context fragments to inspect.

    Returns
    -------
    List[str]
        Ordered unique document labels found in explicit context headers.
    """

    document_labels: List[str] = []

    for text_value in text_values:
        for pattern in (_DOCUMENT_TITLE_EQUALS_PATTERN, _LEGAL_ANCHOR_EQUALS_PATTERN):
            for match in pattern.findall(text_value or ""):
                _append_unique(document_labels, match)

    return document_labels


def _extract_numeric_claims(text: str) -> List[str]:
    """
    Extract compact numeric legal claims from an answer.

    Parameters
    ----------
    text : str
        Generated answer text.

    Returns
    -------
    List[str]
        Ordered unique normalized numeric claims.
    """

    claims: List[str] = []
    seen_claims: Set[str] = set()
    normalized_text = _normalize_comparison_text(text)

    for match in _LEGAL_NUMERIC_CLAIM_PATTERN.findall(normalized_text):
        normalized_claim = _normalize_comparison_text(match)
        if normalized_claim and normalized_claim not in seen_claims:
            claims.append(normalized_claim)
            seen_claims.add(normalized_claim)

    return claims


def _extract_percent_values(text: str) -> List[str]:
    """
    Extract distinct percentage values from one text fragment.

    Parameters
    ----------
    text : str
        Raw legal text to inspect.

    Returns
    -------
    List[str]
        Ordered normalized percentage numbers without the percent marker.
    """

    percent_values: List[str] = []
    seen_values: Set[str] = set()
    normalized_text = _normalize_symbol_text(text)

    for match in _PERCENT_VALUE_PATTERN.findall(normalized_text):
        percent_value = match.replace(",", ".").strip()
        if percent_value and percent_value not in seen_values:
            percent_values.append(percent_value)
            seen_values.add(percent_value)

    return percent_values


def _extract_claim_tokens(text: str) -> Set[str]:
    """
    Extract high-signal lexical tokens from one legal claim text.

    Parameters
    ----------
    text : str
        Raw answer or context text.

    Returns
    -------
    Set[str]
        Distinct normalized tokens useful for conservative claim alignment.
    """

    claim_tokens: Set[str] = set()

    for token in _normalize_comparison_text(text).split():
        if len(token) < 4:
            continue
        if token.isdigit() or token in _LOW_SIGNAL_CLAIM_TOKENS:
            continue
        claim_tokens.add(token)

    return claim_tokens


@dataclass(slots=True)
class _ContextAnchors:
    """
    Internal normalized anchor set derived from selected retrieval context.
    """

    article_numbers: List[str]
    document_labels: List[str]
    context_text: str
    article_text_by_key: dict[str, str]
    primary_article_key: str = ""

    @property
    def article_keys(self) -> Set[str]:
        """
        Return normalized article numbers available in selected context.

        Returns
        -------
        Set[str]
            Normalized article-number keys.
        """

        return {_normalize_anchor(article_number) for article_number in self.article_numbers}

    @property
    def document_keys(self) -> Set[str]:
        """
        Return normalized document labels available in selected context.

        Returns
        -------
        Set[str]
            Normalized document-label keys.
        """

        return {
            _normalize_anchor(document_label)
            for document_label in self.document_labels
            if _normalize_anchor(document_label)
        }


class GroundingValidator:
    """
    Validate post-generation grounding and citation alignment deterministically.

    The validator inspects only the selected context and generated answer. It
    does not retrieve chunks, generate text, or apply safety guardrails.
    """

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """
        Initialize the grounding validator.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared runtime settings kept for parity with other retrieval
            components and later configuration-driven thresholds.
        """

        self.settings = settings or PipelineSettings()

    def validate(
        self,
        *,
        answer_text: str,
        context: Optional[RetrievalContext],
        citations: Optional[Sequence[str]] = None,
    ) -> GroundingVerificationResult:
        """
        Validate one generated answer against the selected grounding context.

        Parameters
        ----------
        answer_text : str
            Generated answer text emitted by the answer adapter.

        context : Optional[RetrievalContext]
            Selected retrieval context used for answer generation.

        citations : Optional[Sequence[str]]
            Optional citation fragments emitted alongside the answer.

        Returns
        -------
        GroundingVerificationResult
            Deterministic grounding and citation verification result.
        """

        normalized_answer_text = answer_text.strip() if isinstance(answer_text, str) else ""
        if context is None or not context.context_text.strip():
            return GroundingVerificationResult(
                status="weak_alignment",
                accepted=False,
                citation_status="missing_context",
                document_alignment="not_evaluated",
                article_alignment="not_evaluated",
                reasons=["grounding.context_missing"],
            )

        context_anchors = self._collect_context_anchors(context)
        answer_reference_texts = self._build_reference_texts(
            answer_text=normalized_answer_text,
            citations=None,
        )
        reference_texts = self._build_reference_texts(
            answer_text=normalized_answer_text,
            citations=citations,
        )
        answer_cited_articles = _extract_article_numbers(answer_reference_texts)
        cited_articles = _extract_article_numbers(reference_texts)
        cited_documents = _extract_document_references(reference_texts)

        mismatched_citations = self._find_citation_mismatches(
            context_anchors=context_anchors,
            cited_articles=cited_articles,
            cited_documents=cited_documents,
        )
        mismatched_citations.extend(
            self._find_article_claim_mismatches(
                answer_text=normalized_answer_text,
                context_anchors=context_anchors,
                cited_articles=answer_cited_articles,
            )
        )
        unsupported_claims, supported_derived_claims = (
            self._classify_numeric_claim_support(
                answer_text=normalized_answer_text,
                context_text=context_anchors.context_text,
            )
        )
        missing_required_facts = self._find_missing_required_anchors(
            answer_text=normalized_answer_text,
            context_anchors=context_anchors,
            cited_articles=cited_articles,
        )

        return self._build_result(
            context_anchors=context_anchors,
            cited_articles=cited_articles,
            cited_documents=cited_documents,
            mismatched_citations=mismatched_citations,
            unsupported_claims=unsupported_claims,
            supported_derived_claims=supported_derived_claims,
            missing_required_facts=missing_required_facts,
        )

    def _build_reference_texts(
        self,
        *,
        answer_text: str,
        citations: Optional[Sequence[str]],
    ) -> List[str]:
        """
        Build one ordered list of answer and citation fragments for anchor parsing.

        Parameters
        ----------
        answer_text : str
            Generated answer text emitted by the answer adapter.

        citations : Optional[Sequence[str]]
            Optional citation fragments emitted alongside the answer.

        Returns
        -------
        List[str]
            Ordered non-empty text fragments used to extract cited anchors.
        """

        reference_texts: List[str] = []

        if answer_text:
            reference_texts.append(answer_text)

        if not citations:
            return reference_texts

        for citation in citations:
            if isinstance(citation, str):
                normalized_citation = citation.strip()
                if normalized_citation:
                    reference_texts.append(normalized_citation)

        return reference_texts

    def _collect_context_anchors(self, context: RetrievalContext) -> _ContextAnchors:
        """
        Collect article and document anchors from selected context chunks.

        Parameters
        ----------
        context : RetrievalContext
            Selected retrieval context.

        Returns
        -------
        _ContextAnchors
            Normalized grounding anchors derived from selected context.
        """

        article_numbers: List[str] = []
        document_labels: List[str] = []
        article_text_parts_by_key: dict[str, List[str]] = {}

        for metadata in context.selected_context_metadata:
            _append_unique(article_numbers, metadata.article_number)
            _append_unique(document_labels, metadata.document_title)
            _append_unique(document_labels, metadata.doc_id)
            _append_unique(document_labels, metadata.source_file)

        for chunk in context.chunks:
            _append_unique(article_numbers, chunk.chunk_metadata.get("article_number"))
            _append_unique(document_labels, chunk.document_metadata.get("document_title"))
            _append_unique(document_labels, chunk.doc_id)
            _append_unique(document_labels, chunk.source_file)
            self._append_article_text(
                article_text_parts_by_key=article_text_parts_by_key,
                article_number=chunk.context_metadata.article_number
                or chunk.chunk_metadata.get("article_number"),
                article_text=chunk.text,
            )

        for article_number in _extract_article_numbers([context.context_text]):
            _append_unique(article_numbers, article_number)

        for document_label in _extract_context_document_labels([context.context_text]):
            _append_unique(document_labels, document_label)

        return _ContextAnchors(
            article_numbers=article_numbers,
            document_labels=document_labels,
            context_text=context.context_text,
            article_text_by_key={
                article_key: "\n".join(text_parts)
                for article_key, text_parts in article_text_parts_by_key.items()
            },
            primary_article_key=self._resolve_primary_article_key_from_metadata(
                context
            ),
        )

    def _resolve_primary_article_key_from_metadata(
        self,
        context: RetrievalContext,
    ) -> str:
        """
        Resolve the context-builder primary article from context metadata.

        Parameters
        ----------
        context : RetrievalContext
            Selected retrieval context emitted by the context builder.

        Returns
        -------
        str
            Normalized article key tied to the primary anchor, otherwise empty.
        """

        article_by_chunk_id: dict[str, str] = {}

        for metadata in context.selected_context_metadata:
            article_key = _normalize_anchor(metadata.article_number)
            if not article_key:
                continue
            for chunk_identifier in (metadata.chunk_id, metadata.record_id):
                normalized_identifier = _normalize_metadata_scalar(chunk_identifier)
                if normalized_identifier:
                    article_by_chunk_id[normalized_identifier] = article_key

        for chunk in context.chunks:
            article_key = _normalize_anchor(
                chunk.context_metadata.article_number
                or chunk.chunk_metadata.get("article_number")
            )
            if not article_key:
                continue
            for chunk_identifier in (chunk.chunk_id, chunk.record_id):
                normalized_identifier = _normalize_metadata_scalar(chunk_identifier)
                if normalized_identifier:
                    article_by_chunk_id[normalized_identifier] = article_key

        for chunk_identifier in self._extract_primary_anchor_chunk_ids(context):
            article_key = article_by_chunk_id.get(chunk_identifier)
            if article_key:
                return article_key

        primary_anchor = _normalize_metadata_scalar(
            context.metadata.get("primary_anchor")
        )
        primary_anchor_articles = _extract_article_numbers([primary_anchor])
        if primary_anchor_articles:
            return _normalize_anchor(primary_anchor_articles[0])

        return ""

    def _extract_primary_anchor_chunk_ids(
        self,
        context: RetrievalContext,
    ) -> List[str]:
        """
        Extract normalized primary-anchor chunk identifiers from context metadata.

        Parameters
        ----------
        context : RetrievalContext
            Selected retrieval context emitted by the context builder.

        Returns
        -------
        List[str]
            Ordered chunk identifiers marked as primary anchor evidence.
        """

        raw_chunk_ids = context.metadata.get("primary_anchor_chunk_ids", [])
        if isinstance(raw_chunk_ids, str):
            raw_values: Sequence[object] = [raw_chunk_ids]
        elif isinstance(raw_chunk_ids, Sequence):
            raw_values = raw_chunk_ids
        else:
            raw_values = []

        chunk_ids: List[str] = []
        for raw_value in raw_values:
            chunk_id = _normalize_metadata_scalar(raw_value)
            if chunk_id and chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)

        return chunk_ids

    def _append_article_text(
        self,
        *,
        article_text_parts_by_key: dict[str, List[str]],
        article_number: object,
        article_text: str,
    ) -> None:
        """
        Attach one selected chunk body to its normalized article anchor.

        Parameters
        ----------
        article_text_parts_by_key : dict[str, List[str]]
            Mutable article-to-text index built from selected context chunks.

        article_number : object
            Candidate article number from chunk metadata.

        article_text : str
            Chunk body text available for grounding checks.
        """

        normalized_article_number = _normalize_metadata_scalar(article_number)
        article_key = _normalize_anchor(normalized_article_number)

        if not article_key or not article_text:
            return

        article_text_parts_by_key.setdefault(article_key, []).append(article_text)

    def _find_citation_mismatches(
        self,
        *,
        context_anchors: _ContextAnchors,
        cited_articles: Sequence[str],
        cited_documents: Sequence[str],
    ) -> List[str]:
        """
        Find explicit article or document citations not present in context.

        Parameters
        ----------
        context_anchors : _ContextAnchors
            Anchors available in selected context.

        cited_articles : Sequence[str]
            Article numbers cited by the answer.

        cited_documents : Sequence[str]
            Document references cited by the answer.

        Returns
        -------
        List[str]
            Explainable citation mismatch labels.
        """

        mismatches: List[str] = []

        for article_number in cited_articles:
            article_key = _normalize_anchor(article_number)
            if context_anchors.article_keys and article_key not in context_anchors.article_keys:
                mismatches.append(f"article={article_number}")

        for document_reference in cited_documents:
            document_key = _normalize_anchor(document_reference)
            if context_anchors.document_keys and not _anchors_overlap(
                document_key,
                context_anchors.document_keys,
            ):
                mismatches.append(f"document={document_reference}")

        return mismatches

    def _classify_numeric_claim_support(
        self,
        *,
        answer_text: str,
        context_text: str,
    ) -> tuple[List[str], List[str]]:
        """
        Classify answer numeric claims into unsupported and derived-supported sets.

        Parameters
        ----------
        answer_text : str
            Generated answer text.

        context_text : str
            Selected grounding context text.

        Returns
        -------
        tuple[List[str], List[str]]
            Unsupported numeric claims and supported derived claims.
        """

        context_comparison_text = _normalize_comparison_text(context_text)
        unsupported_claims: List[str] = []
        supported_derived_claims: List[str] = []

        for claim in _extract_numeric_claims(answer_text):
            is_supported, is_derived = self._classify_numeric_claim(
                claim=claim,
                answer_text=answer_text,
                grounding_text=context_text,
                grounding_comparison_text=context_comparison_text,
            )
            if not is_supported:
                unsupported_claims.append(claim)
                continue
            if is_derived:
                supported_derived_claims.append(claim)

        return unsupported_claims, supported_derived_claims

    def _classify_numeric_claim(
        self,
        *,
        claim: str,
        answer_text: str,
        grounding_text: str,
        grounding_comparison_text: str,
    ) -> tuple[bool, bool]:
        """
        Classify one numeric claim as supported and whether that support is derived.

        Parameters
        ----------
        claim : str
            Normalized numeric claim extracted from the generated answer.

        answer_text : str
            Full generated answer text containing the claim.

        grounding_text : str
            Context text available for grounding checks.

        grounding_comparison_text : str
            Normalized comparison form for the grounding text.

        Returns
        -------
        tuple[bool, bool]
            Support flag and derived-support flag.
        """

        if claim in grounding_comparison_text:
            return True, False

        if not self.settings.retrieval_grounding_derived_claims_enabled:
            return False, False

        is_supported_derived = self._is_supported_derived_numeric_claim(
            claim=claim,
            answer_text=answer_text,
            grounding_text=grounding_text,
        )
        return is_supported_derived, is_supported_derived

    def _is_supported_derived_numeric_claim(
        self,
        *,
        claim: str,
        answer_text: str,
        grounding_text: str,
    ) -> bool:
        """
        Decide whether a numeric claim compresses enumerated legal evidence.

        Parameters
        ----------
        claim : str
            Normalized numeric claim extracted from the generated answer.

        answer_text : str
            Full generated answer text containing the claim.

        grounding_text : str
            Context text available for grounding checks.

        Returns
        -------
        bool
            True when the claim is a conservative derived installment summary.
        """

        installment_match = _INSTALLMENT_CLAIM_PATTERN.match(claim)
        if not installment_match:
            return False

        claimed_count = int(installment_match.group(1))
        tolerance = max(
            0,
            int(self.settings.retrieval_grounding_derived_numeric_claim_tolerance),
        )
        coverage_threshold = float(
            self.settings.retrieval_grounding_enumeration_coverage_threshold
        )

        for percent_value in _extract_percent_values(answer_text):
            supported_count = self._count_installments_with_percent(
                grounding_text=grounding_text,
                percent_value=percent_value,
            )
            if self._claim_count_within_tolerance(
                claimed_count=claimed_count,
                supported_count=supported_count,
                tolerance=tolerance,
                coverage_threshold=coverage_threshold,
            ):
                return True

        if "restantes" not in _normalize_comparison_text(answer_text):
            return False

        remaining_count = self._derive_remaining_installment_count(grounding_text)
        return self._claim_count_within_tolerance(
            claimed_count=claimed_count,
            supported_count=remaining_count,
            tolerance=tolerance,
            coverage_threshold=coverage_threshold,
        )

    def _count_installments_with_percent(
        self,
        *,
        grounding_text: str,
        percent_value: str,
    ) -> int:
        """
        Count installment entries in context tied to one percentage value.

        Parameters
        ----------
        grounding_text : str
            Context text available for grounding checks.

        percent_value : str
            Percentage number extracted from the answer.

        Returns
        -------
        int
            Number of distinct installment entries or range positions found.
        """

        normalized_text = _normalize_symbol_text(grounding_text)
        supported_installments: Set[int] = set()

        for match in _INSTALLMENT_RANGE_PATTERN.finditer(normalized_text):
            if not self._text_window_has_percent(
                normalized_text=normalized_text,
                start=match.start(),
                end=match.end(),
                percent_value=percent_value,
                window_size=120,
            ):
                continue
            first_installment = int(match.group(1))
            last_installment = int(match.group(2))
            lower_bound = min(first_installment, last_installment)
            upper_bound = max(first_installment, last_installment)
            supported_installments.update(range(lower_bound, upper_bound + 1))

        for match in _INSTALLMENT_REFERENCE_PATTERN.finditer(normalized_text):
            if not self._text_window_has_percent(
                normalized_text=normalized_text,
                start=match.start(),
                end=match.end(),
                percent_value=percent_value,
                window_size=24,
            ):
                continue
            supported_installments.add(int(match.group(1)))

        return len(supported_installments)

    def _text_window_has_percent(
        self,
        *,
        normalized_text: str,
        start: int,
        end: int,
        percent_value: str,
        window_size: int,
    ) -> bool:
        """
        Check whether a local text window contains the expected percentage.

        Parameters
        ----------
        normalized_text : str
            Symbol-preserving normalized grounding text.

        start : int
            Start offset of an installment mention.

        end : int
            End offset of an installment mention.

        percent_value : str
            Percentage number expected near the installment mention.

        window_size : int
            Number of characters inspected around the installment mention.

        Returns
        -------
        bool
            True when the percentage appears close to the installment mention.
        """

        window_start = max(0, start - window_size)
        window_end = min(len(normalized_text), end + window_size)
        window_text = normalized_text[window_start:window_end]
        return any(
            percent_value == value for value in _extract_percent_values(window_text)
        )

    def _derive_remaining_installment_count(self, grounding_text: str) -> int:
        """
        Derive a remaining-installment count from total and first installment.

        Parameters
        ----------
        grounding_text : str
            Context text available for grounding checks.

        Returns
        -------
        int
            Remaining installment count, or zero when not safely derivable.
        """

        normalized_text = _normalize_comparison_text(grounding_text)
        total_counts = [
            int(match)
            for match in _INSTALLMENT_TOTAL_PATTERN.findall(normalized_text)
            if match
        ]
        if not total_counts:
            return 0

        first_installment_present = any(
            installment == "1"
            for installment in _INSTALLMENT_REFERENCE_PATTERN.findall(normalized_text)
        ) or "primeira prestacao" in normalized_text
        if not first_installment_present:
            return 0

        return max(total_counts) - 1

    def _claim_count_within_tolerance(
        self,
        *,
        claimed_count: int,
        supported_count: int,
        tolerance: int,
        coverage_threshold: float,
    ) -> bool:
        """
        Compare a derived numeric claim with an enumerated support count.

        Parameters
        ----------
        claimed_count : int
            Count stated in the generated answer.

        supported_count : int
            Count derived from the selected grounding context.

        tolerance : int
            Absolute numeric tolerance configured for derived claims.

        coverage_threshold : float
            Minimum supported coverage required for conservative acceptance.

        Returns
        -------
        bool
            True when the support count covers the claim within thresholds.
        """

        if claimed_count <= 0 or supported_count <= 0:
            return False

        required_support = max(1, int(claimed_count * coverage_threshold))
        if supported_count < required_support:
            return False

        return abs(supported_count - claimed_count) <= tolerance

    def _find_article_claim_mismatches(
        self,
        *,
        answer_text: str,
        context_anchors: _ContextAnchors,
        cited_articles: Sequence[str],
    ) -> List[str]:
        """
        Find answer claims assigned to an article that does not support them.

        Parameters
        ----------
        answer_text : str
            Generated answer text.

        context_anchors : _ContextAnchors
            Anchors and article-local text available in selected context.

        cited_articles : Sequence[str]
            Article numbers explicitly cited by the generated answer.

        Returns
        -------
        List[str]
            Citation mismatch labels for article-local claim conflicts.
        """

        if not cited_articles or len(context_anchors.article_text_by_key) <= 1:
            return []

        mismatches: List[str] = []
        mismatches.extend(
            self._find_article_numeric_claim_mismatches(
                answer_text=answer_text,
                context_anchors=context_anchors,
                cited_articles=cited_articles,
            )
        )
        mismatches.extend(
            self._find_primary_article_selection_mismatches(
                answer_text=answer_text,
                context_anchors=context_anchors,
                cited_articles=cited_articles,
            )
        )

        return list(dict.fromkeys(mismatches))

    def _find_article_numeric_claim_mismatches(
        self,
        *,
        answer_text: str,
        context_anchors: _ContextAnchors,
        cited_articles: Sequence[str],
    ) -> List[str]:
        """
        Find numeric claims that are present in context but absent from cited articles.

        Parameters
        ----------
        answer_text : str
            Generated answer text.

        context_anchors : _ContextAnchors
            Anchors and article-local text available in selected context.

        cited_articles : Sequence[str]
            Article numbers explicitly cited by the generated answer.

        Returns
        -------
        List[str]
            Citation mismatch labels for article-local numeric claim conflicts.
        """

        context_comparison_text = _normalize_comparison_text(
            context_anchors.context_text
        )
        mismatches: List[str] = []

        for article_number in cited_articles:
            article_key = _normalize_anchor(article_number)
            article_text = context_anchors.article_text_by_key.get(article_key, "")
            if not article_text:
                continue

            article_comparison_text = _normalize_comparison_text(article_text)
            for numeric_claim in _extract_numeric_claims(answer_text):
                is_supported_in_context, _ = self._classify_numeric_claim(
                    claim=numeric_claim,
                    answer_text=answer_text,
                    grounding_text=context_anchors.context_text,
                    grounding_comparison_text=context_comparison_text,
                )
                if not is_supported_in_context:
                    continue
                is_supported_in_article, _ = self._classify_numeric_claim(
                    claim=numeric_claim,
                    answer_text=answer_text,
                    grounding_text=article_text,
                    grounding_comparison_text=article_comparison_text,
                )
                if not is_supported_in_article:
                    mismatches.append(f"article_claim={article_number}:{numeric_claim}")

        return mismatches

    def _find_primary_article_selection_mismatches(
        self,
        *,
        answer_text: str,
        context_anchors: _ContextAnchors,
        cited_articles: Sequence[str],
    ) -> List[str]:
        """
        Detect answers that omit the strongest selected article in close context.

        Parameters
        ----------
        answer_text : str
            Generated answer text.

        context_anchors : _ContextAnchors
            Anchors and article-local text available in selected context.

        cited_articles : Sequence[str]
            Article numbers explicitly cited by the generated answer.

        Returns
        -------
        List[str]
            Citation mismatch labels for likely wrong article selection.
        """

        primary_article_key = self._resolve_primary_article_key(context_anchors)
        if not primary_article_key:
            return []

        cited_article_keys = {_normalize_anchor(article) for article in cited_articles}
        if primary_article_key in cited_article_keys:
            return []

        answer_claim_tokens = _extract_claim_tokens(answer_text)
        if len(answer_claim_tokens) < 2:
            return []

        primary_text = context_anchors.article_text_by_key.get(primary_article_key, "")
        primary_overlap = self._count_claim_overlap(answer_claim_tokens, primary_text)
        cited_overlap = max(
            (
                self._count_claim_overlap(
                    answer_claim_tokens,
                    context_anchors.article_text_by_key.get(cited_article_key, ""),
                )
                for cited_article_key in cited_article_keys
            ),
            default=0,
        )

        if primary_overlap >= cited_overlap + 2:
            return [f"primary_article={primary_article_key}"]

        return []

    def _resolve_primary_article_key(
        self,
        context_anchors: _ContextAnchors,
    ) -> str:
        """
        Resolve the first selected article with article-local text.

        Parameters
        ----------
        context_anchors : _ContextAnchors
            Anchors and article-local text available in selected context.

        Returns
        -------
        str
            Normalized primary article key, otherwise an empty string.
        """

        if (
            context_anchors.primary_article_key
            and context_anchors.primary_article_key in context_anchors.article_text_by_key
        ):
            return context_anchors.primary_article_key

        for article_number in context_anchors.article_numbers:
            article_key = _normalize_anchor(article_number)
            if article_key in context_anchors.article_text_by_key:
                return article_key

        return ""

    def _count_claim_overlap(
        self,
        answer_claim_tokens: Set[str],
        article_text: str,
    ) -> int:
        """
        Count high-signal answer tokens supported by one article text.

        Parameters
        ----------
        answer_claim_tokens : Set[str]
            High-signal tokens extracted from the generated answer.

        article_text : str
            Article-local context text.

        Returns
        -------
        int
            Number of claim tokens present in the article-local text.
        """

        if not answer_claim_tokens or not article_text:
            return 0

        return len(answer_claim_tokens & _extract_claim_tokens(article_text))

    def _find_missing_required_anchors(
        self,
        *,
        answer_text: str,
        context_anchors: _ContextAnchors,
        cited_articles: Sequence[str],
    ) -> List[str]:
        """
        Find article references that are claimed without a concrete article.

        Parameters
        ----------
        answer_text : str
            Generated answer text.

        context_anchors : _ContextAnchors
            Anchors available in selected context.

        cited_articles : Sequence[str]
            Explicit article numbers cited by the answer.

        Returns
        -------
        List[str]
            Missing anchor labels.
        """

        if (
            context_anchors.article_numbers
            and not cited_articles
            and _ARTICLE_INTENT_PATTERN.search(answer_text)
        ):
            return ["article_number"]

        return []

    def _build_result(
        self,
        *,
        context_anchors: _ContextAnchors,
        cited_articles: Sequence[str],
        cited_documents: Sequence[str],
        mismatched_citations: Sequence[str],
        unsupported_claims: Sequence[str],
        supported_derived_claims: Sequence[str],
        missing_required_facts: Sequence[str],
    ) -> GroundingVerificationResult:
        """
        Build the final grounding-verification result from validation signals.

        Parameters
        ----------
        context_anchors : _ContextAnchors
            Anchors available in selected context.

        cited_articles : Sequence[str]
            Article numbers cited by the answer.

        cited_documents : Sequence[str]
            Document references cited by the answer.

        mismatched_citations : Sequence[str]
            Citation mismatch labels.

        unsupported_claims : Sequence[str]
            Unsupported numeric legal claims.

        supported_derived_claims : Sequence[str]
            Conservative derived claims supported by enumerated context.

        missing_required_facts : Sequence[str]
            Missing required anchor labels.

        Returns
        -------
        GroundingVerificationResult
            Normalized verification result.
        """

        reasons: List[str] = []
        status = "strong_alignment"
        accepted = True

        if mismatched_citations:
            status = "citation_mismatch"
            accepted = False
            reasons.append("grounding.citation_mismatch")
        if unsupported_claims:
            status = "unsupported_claim" if accepted else status
            accepted = False
            reasons.append("grounding.unsupported_numeric_claim")
        if missing_required_facts:
            status = "missing_required_anchor" if accepted else status
            accepted = False
            reasons.append("grounding.missing_required_anchor")

        if accepted and context_anchors.article_numbers and not cited_articles:
            status = "weak_alignment"
            accepted = False
            reasons.append("grounding.legal_anchor_not_cited")

        citation_status = "aligned" if not mismatched_citations else "mismatch"
        document_alignment = self._resolve_document_alignment(
            cited_documents=cited_documents,
            mismatched_citations=mismatched_citations,
        )
        article_alignment = self._resolve_article_alignment(
            cited_articles=cited_articles,
            mismatched_citations=mismatched_citations,
            missing_required_facts=missing_required_facts,
        )

        if not reasons:
            reasons.append("grounding.aligned")

        diagnostic_stage, diagnostic_category = self._resolve_diagnostic_taxonomy(
            accepted=accepted,
            mismatched_citations=mismatched_citations,
            unsupported_claims=unsupported_claims,
            missing_required_facts=missing_required_facts,
            supported_derived_claims=supported_derived_claims,
        )
        diagnostic_signals = self._build_diagnostic_signals(
            mismatched_citations=mismatched_citations,
            unsupported_claims=unsupported_claims,
            supported_derived_claims=supported_derived_claims,
            missing_required_facts=missing_required_facts,
            primary_article_key=context_anchors.primary_article_key,
        )

        return GroundingVerificationResult(
            status=status,
            accepted=accepted,
            citation_status=citation_status,
            document_alignment=document_alignment,
            article_alignment=article_alignment,
            diagnostic_stage=diagnostic_stage,
            diagnostic_category=diagnostic_category,
            cited_documents=list(cited_documents),
            cited_article_numbers=list(cited_articles),
            mismatched_citations=list(mismatched_citations),
            unsupported_claims=list(unsupported_claims),
            supported_derived_claims=list(supported_derived_claims),
            missing_required_facts=list(missing_required_facts),
            diagnostic_signals=diagnostic_signals,
            reasons=reasons,
            metadata={
                "context_article_numbers": list(context_anchors.article_numbers),
                "context_document_labels": list(context_anchors.document_labels),
                "primary_article_key": context_anchors.primary_article_key,
            },
        )

    def _resolve_diagnostic_taxonomy(
        self,
        *,
        accepted: bool,
        mismatched_citations: Sequence[str],
        unsupported_claims: Sequence[str],
        missing_required_facts: Sequence[str],
        supported_derived_claims: Sequence[str],
    ) -> tuple[str, str]:
        """
        Resolve the shared grounding diagnostic taxonomy for one result.

        Parameters
        ----------
        accepted : bool
            Final acceptance decision for the grounding check.

        mismatched_citations : Sequence[str]
            Citation mismatch labels gathered during validation.

        unsupported_claims : Sequence[str]
            Unsupported numeric legal claims.

        missing_required_facts : Sequence[str]
            Missing required anchor labels.

        supported_derived_claims : Sequence[str]
            Derived claims conservatively supported by the selected context.

        Returns
        -------
        tuple[str, str]
            Diagnostic stage and category.
        """

        diagnostic_stage = "grounding_validation"

        if mismatched_citations or unsupported_claims or missing_required_facts:
            return diagnostic_stage, "grounding_failure"
        if accepted and supported_derived_claims:
            return diagnostic_stage, "grounding_success"
        if accepted:
            return diagnostic_stage, "grounding_success"
        return diagnostic_stage, "grounding_warning"

    def _build_diagnostic_signals(
        self,
        *,
        mismatched_citations: Sequence[str],
        unsupported_claims: Sequence[str],
        supported_derived_claims: Sequence[str],
        missing_required_facts: Sequence[str],
        primary_article_key: str,
    ) -> List[DiagnosticSignal]:
        """
        Build explicit shared diagnostic signals for one grounding result.

        Parameters
        ----------
        mismatched_citations : Sequence[str]
            Citation mismatch labels gathered during validation.

        unsupported_claims : Sequence[str]
            Unsupported numeric legal claims.

        supported_derived_claims : Sequence[str]
            Derived claims conservatively supported by the selected context.

        missing_required_facts : Sequence[str]
            Missing required anchor labels.

        primary_article_key : str
            Normalized primary-article anchor selected upstream.

        Returns
        -------
        List[DiagnosticSignal]
            Ordered diagnostic signals describing the grounding outcome.
        """

        diagnostic_signals: List[DiagnosticSignal] = []

        for mismatch in mismatched_citations:
            signal_code = "answer_citation_mismatch"
            metadata = {"mismatch": mismatch}

            if mismatch.startswith("primary_article="):
                signal_code = "wrong_primary_anchor_selected"
                metadata["primary_article_key"] = primary_article_key or mismatch.split(
                    "=", 1
                )[1]

            diagnostic_signals.append(
                DiagnosticSignal(
                    stage="grounding_validation",
                    category="grounding_failure",
                    code=signal_code,
                    detail=mismatch,
                    metadata=metadata,
                )
            )

        for unsupported_claim in unsupported_claims:
            diagnostic_signals.append(
                DiagnosticSignal(
                    stage="grounding_validation",
                    category="grounding_failure",
                    code="unsupported_legal_claim",
                    detail=unsupported_claim,
                )
            )

        for missing_required_fact in missing_required_facts:
            diagnostic_signals.append(
                DiagnosticSignal(
                    stage="grounding_validation",
                    category="grounding_failure",
                    code="missing_required_anchor",
                    detail=missing_required_fact,
                )
            )

        for supported_derived_claim in supported_derived_claims:
            diagnostic_signals.append(
                DiagnosticSignal(
                    stage="grounding_validation",
                    category="grounding_success",
                    code="supported_derived_claim",
                    detail=supported_derived_claim,
                )
            )

        return diagnostic_signals

    def _resolve_document_alignment(
        self,
        *,
        cited_documents: Sequence[str],
        mismatched_citations: Sequence[str],
    ) -> str:
        """
        Resolve the document-alignment status.

        Parameters
        ----------
        cited_documents : Sequence[str]
            Documents cited by the answer.

        mismatched_citations : Sequence[str]
            Citation mismatch labels.

        Returns
        -------
        str
            Document-alignment status.
        """

        if any(mismatch.startswith("document=") for mismatch in mismatched_citations):
            return "mismatch"
        if cited_documents:
            return "aligned"
        return "not_cited"

    def _resolve_article_alignment(
        self,
        *,
        cited_articles: Sequence[str],
        mismatched_citations: Sequence[str],
        missing_required_facts: Sequence[str],
    ) -> str:
        """
        Resolve the article-alignment status.

        Parameters
        ----------
        cited_articles : Sequence[str]
            Article numbers cited by the answer.

        mismatched_citations : Sequence[str]
            Citation mismatch labels.

        missing_required_facts : Sequence[str]
            Missing anchor labels.

        Returns
        -------
        str
            Article-alignment status.
        """

        if any(
            mismatch.startswith(("article=", "article_claim=", "primary_article="))
            for mismatch in mismatched_citations
        ):
            return "mismatch"
        if missing_required_facts:
            return "missing_required_anchor"
        if cited_articles:
            return "aligned"
        return "not_cited"
