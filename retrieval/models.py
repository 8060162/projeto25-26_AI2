from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _normalize_mapping(value: Any) -> Dict[str, Any]:
    """
    Normalize one optional mapping payload into a detached dictionary.

    Parameters
    ----------
    value : Any
        Candidate mapping payload.

    Returns
    -------
    Dict[str, Any]
        Detached dictionary when the payload is mapping-like, otherwise an
        empty dictionary.
    """

    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_string(value: Any) -> str:
    """
    Normalize one optional string payload into a stripped string.

    Parameters
    ----------
    value : Any
        Candidate string payload.

    Returns
    -------
    str
        Stripped string when the payload is string-like, otherwise an empty
        string.
    """

    if isinstance(value, str):
        return value.strip()
    return ""


def _normalize_string_list(value: Any) -> List[str]:
    """
    Normalize one optional list payload into a clean string list.

    Parameters
    ----------
    value : Any
        Candidate list payload.

    Returns
    -------
    List[str]
        Ordered non-empty string values.
    """

    if not isinstance(value, list):
        return []

    normalized_values: List[str] = []

    for item in value:
        normalized_item = _normalize_string(item)
        if normalized_item:
            normalized_values.append(normalized_item)

    return normalized_values


def _normalize_optional_int(value: Any) -> Optional[int]:
    """
    Normalize one optional integer payload.

    Parameters
    ----------
    value : Any
        Candidate integer payload.

    Returns
    -------
    Optional[int]
        Integer value when conversion is safe, otherwise `None`.
    """

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_float(value: Any) -> Optional[float]:
    """
    Normalize one optional float payload.

    Parameters
    ----------
    value : Any
        Candidate float payload.

    Returns
    -------
    Optional[float]
        Float value when conversion is safe, otherwise `None`.
    """

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_non_negative_int(value: Any) -> int:
    """
    Normalize one optional integer payload into a non-negative integer.

    Parameters
    ----------
    value : Any
        Candidate integer payload.

    Returns
    -------
    int
        Non-negative integer value, or zero when conversion is unsafe.
    """

    normalized_value = _normalize_optional_int(value)
    if normalized_value is None:
        return 0
    return max(0, normalized_value)


def _normalize_float_mapping(value: Any) -> Dict[str, float]:
    """
    Normalize one optional stage-latency mapping into float values.

    Parameters
    ----------
    value : Any
        Candidate mapping payload.

    Returns
    -------
    Dict[str, float]
        Detached mapping containing only non-empty string keys and numeric
        values converted to floats.
    """

    if not isinstance(value, dict):
        return {}

    normalized_mapping: Dict[str, float] = {}

    for raw_key, raw_value in value.items():
        normalized_key = _normalize_string(raw_key)
        normalized_value = _normalize_optional_float(raw_value)

        if normalized_key and normalized_value is not None:
            normalized_mapping[normalized_key] = normalized_value

    return normalized_mapping


def _normalize_diagnostic_signal_list(value: Any) -> List["DiagnosticSignal"]:
    """
    Normalize one optional diagnostic-signal list payload.

    Parameters
    ----------
    value : Any
        Candidate list payload.

    Returns
    -------
    List[DiagnosticSignal]
        Ordered diagnostic signals coerced into the shared contract.
    """

    if not isinstance(value, list):
        return []

    normalized_signals: List[DiagnosticSignal] = []

    for item in value:
        if isinstance(item, DiagnosticSignal):
            normalized_signals.append(item)
            continue

        if isinstance(item, dict):
            normalized_signals.append(DiagnosticSignal(**item))

    return normalized_signals


class RetrievalModelBase:
    """
    Provide a shared serialization helper for retrieval data contracts.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the model into a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            Standard Python dictionary representation of the model.
        """

        return asdict(self)


@dataclass(slots=True)
class DiagnosticSignal(RetrievalModelBase):
    """
    Shared typed diagnostic signal emitted by retrieval runtime components.

    Design goals
    ------------
    This model provides one stable diagnostic vocabulary by storing:
    - the stage where the issue or confirmation was detected
    - the coarse diagnostic category for aggregation and routing
    - the specific code describing the concrete retrieval or grounding outcome
    - optional chunk-level scope and extension metadata
    """

    stage: str = ""
    category: str = ""
    code: str = ""
    detail: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the diagnostic-signal payload after initialization.
        """

        self.stage = _normalize_string(self.stage)
        self.category = _normalize_string(self.category)
        self.code = _normalize_string(self.code)
        self.detail = _normalize_string(self.detail)
        self.chunk_ids = _normalize_string_list(self.chunk_ids)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class ContextChunkMetadata(RetrievalModelBase):
    """
    Explicit structural metadata attached to one selected retrieval chunk.

    Design goals
    ------------
    This model exposes high-value context fields used for legal grounding:
    - article identifiers and titles
    - parent structure and document labels
    - page boundaries and source identifiers
    - one metadata bucket for additional chunk-level signals
    """

    chunk_id: str = ""
    record_id: str = ""
    doc_id: str = ""
    source_file: str = ""
    article_number: str = ""
    article_title: str = ""
    section_title: str = ""
    parent_structure: List[str] = field(default_factory=list)
    document_title: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the context-chunk metadata payload after initialization.
        """

        self.chunk_id = _normalize_string(self.chunk_id)
        self.record_id = _normalize_string(self.record_id)
        self.doc_id = _normalize_string(self.doc_id)
        self.source_file = _normalize_string(self.source_file)
        self.article_number = _normalize_string(self.article_number)
        self.article_title = _normalize_string(self.article_title)
        self.section_title = _normalize_string(self.section_title)
        self.parent_structure = _normalize_string_list(self.parent_structure)
        self.document_title = _normalize_string(self.document_title)
        self.page_start = _normalize_optional_int(self.page_start)
        self.page_end = _normalize_optional_int(self.page_end)
        self.metadata = _normalize_mapping(self.metadata)

    @classmethod
    def from_retrieved_chunk(
        cls,
        chunk: "RetrievedChunkResult",
    ) -> "ContextChunkMetadata":
        """
        Build explicit context metadata from one retrieved chunk record.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Retrieved chunk carrying raw metadata scopes.

        Returns
        -------
        ContextChunkMetadata
            Explicit structural metadata derived from the chunk payload.
        """

        chunk_metadata = _normalize_mapping(chunk.chunk_metadata)
        document_metadata = _normalize_mapping(chunk.document_metadata)
        combined_metadata = _normalize_mapping(chunk.metadata)

        article_title = _normalize_string(
            chunk_metadata.get("article_title") or chunk_metadata.get("section_title")
        )
        parent_structure = _normalize_string_list(
            chunk_metadata.get("parent_structure")
            or chunk_metadata.get("parent_titles")
            or []
        )

        return cls(
            chunk_id=chunk.chunk_id,
            record_id=chunk.record_id,
            doc_id=chunk.doc_id,
            source_file=chunk.source_file,
            article_number=chunk_metadata.get("article_number", ""),
            article_title=article_title,
            section_title=chunk_metadata.get("section_title", ""),
            parent_structure=parent_structure,
            document_title=document_metadata.get("document_title", ""),
            page_start=chunk_metadata.get("page_start"),
            page_end=chunk_metadata.get("page_end"),
            metadata={
                **combined_metadata,
                **document_metadata,
                **chunk_metadata,
            },
        )

    def has_structural_content(self) -> bool:
        """
        Indicate whether the metadata contains grounding-relevant structure.

        Returns
        -------
        bool
            `True` when at least one structural grounding field is present.
        """

        return any(
            [
                self.article_number,
                self.article_title,
                self.section_title,
                self.parent_structure,
                self.document_title,
                self.page_start is not None,
                self.page_end is not None,
            ]
        )


@dataclass(slots=True)
class RetrievalQualitySignals(RetrievalModelBase):
    """
    Lightweight retrieval-quality summary attached to one built context.

    Design goals
    ------------
    This model keeps retrieval-quality signals explicit by storing:
    - input, candidate, and selected chunk counts
    - truncation and omission signals
    - structural richness of the final selected context
    - optional extra metadata for later metrics expansion
    """

    total_input_chunks: int = 0
    candidate_chunk_count: int = 0
    selected_chunk_count: int = 0
    duplicate_count: int = 0
    score_filtered_count: int = 0
    omitted_by_rank_limit_count: int = 0
    omitted_by_budget_count: int = 0
    structural_metadata_chunk_count: int = 0
    truncated: bool = False
    selected_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the retrieval-quality payload after initialization.
        """

        self.total_input_chunks = _normalize_non_negative_int(self.total_input_chunks)
        self.candidate_chunk_count = _normalize_non_negative_int(
            self.candidate_chunk_count
        )
        self.selected_chunk_count = _normalize_non_negative_int(self.selected_chunk_count)
        self.duplicate_count = _normalize_non_negative_int(self.duplicate_count)
        self.score_filtered_count = _normalize_non_negative_int(
            self.score_filtered_count
        )
        self.omitted_by_rank_limit_count = _normalize_non_negative_int(
            self.omitted_by_rank_limit_count
        )
        self.omitted_by_budget_count = _normalize_non_negative_int(
            self.omitted_by_budget_count
        )
        self.structural_metadata_chunk_count = _normalize_non_negative_int(
            self.structural_metadata_chunk_count
        )
        self.truncated = bool(self.truncated)
        self.selected_chunk_ids = _normalize_string_list(self.selected_chunk_ids)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class RetrievalRouteDecision(RetrievalModelBase):
    """
    Deterministic retrieval-route contract emitted before vector retrieval.

    Design goals
    ------------
    This model keeps routing decisions explicit by storing:
    - the selected retrieval profile and scope
    - legal targets inferred from query metadata
    - whether comparative or fallback retrieval behavior is expected
    - explainable reasons and extension metadata for later router iterations
    """

    route_name: str = "default"
    retrieval_profile: str = "standard"
    retrieval_scope: str = "broad"
    target_doc_ids: List[str] = field(default_factory=list)
    target_document_titles: List[str] = field(default_factory=list)
    target_article_numbers: List[str] = field(default_factory=list)
    target_article_titles: List[str] = field(default_factory=list)
    comparative: bool = False
    allow_second_pass: bool = False
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the retrieval-route payload after initialization.
        """

        self.route_name = _normalize_string(self.route_name) or "default"
        self.retrieval_profile = (
            _normalize_string(self.retrieval_profile) or "standard"
        )
        self.retrieval_scope = _normalize_string(self.retrieval_scope) or "broad"
        self.target_doc_ids = _normalize_string_list(self.target_doc_ids)
        self.target_document_titles = _normalize_string_list(
            self.target_document_titles
        )
        self.target_article_numbers = _normalize_string_list(
            self.target_article_numbers
        )
        self.target_article_titles = _normalize_string_list(self.target_article_titles)
        self.comparative = bool(self.comparative)
        self.allow_second_pass = bool(self.allow_second_pass)
        self.reasons = _normalize_string_list(self.reasons)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class EvidenceQualityClassification(RetrievalModelBase):
    """
    Typed evidence-quality contract emitted after context selection.

    Design goals
    ------------
    This model separates evidence assessment from context assembly by storing:
    - strength, ambiguity, and conflict classifications
    - whether the selected evidence is sufficient for answer generation
    - close or conflicting chunk identifiers for metrics and routing
    - explainable reasons and extension metadata
    """

    strength: str = "unknown"
    ambiguity: str = "unknown"
    conflict: str = "unknown"
    sufficient_for_answer: bool = True
    confidence_score: Optional[float] = None
    diagnostic_stage: str = ""
    diagnostic_category: str = ""
    close_competitor_chunk_ids: List[str] = field(default_factory=list)
    conflicting_chunk_ids: List[str] = field(default_factory=list)
    missing_expected_chunk_ids: List[str] = field(default_factory=list)
    diagnostic_signals: List[DiagnosticSignal] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the evidence-quality payload after initialization.
        """

        self.strength = _normalize_string(self.strength) or "unknown"
        self.ambiguity = _normalize_string(self.ambiguity) or "unknown"
        self.conflict = _normalize_string(self.conflict) or "unknown"
        self.sufficient_for_answer = bool(self.sufficient_for_answer)
        self.confidence_score = _normalize_optional_float(self.confidence_score)
        self.diagnostic_stage = _normalize_string(self.diagnostic_stage)
        self.diagnostic_category = _normalize_string(self.diagnostic_category)
        self.close_competitor_chunk_ids = _normalize_string_list(
            self.close_competitor_chunk_ids
        )
        self.conflicting_chunk_ids = _normalize_string_list(
            self.conflicting_chunk_ids
        )
        self.missing_expected_chunk_ids = _normalize_string_list(
            self.missing_expected_chunk_ids
        )
        self.diagnostic_signals = _normalize_diagnostic_signal_list(
            self.diagnostic_signals
        )
        self.reasons = _normalize_string_list(self.reasons)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class GroundingVerificationResult(RetrievalModelBase):
    """
    Post-generation grounding and citation verification contract.

    Design goals
    ------------
    This model keeps grounding validation explicit by storing:
    - the overall grounding status and acceptance flag
    - citation, document, and article alignment results
    - unsupported or missing-answer signals
    - explainable reasons and extension metadata
    """

    status: str = "not_evaluated"
    accepted: bool = True
    citation_status: str = "not_evaluated"
    document_alignment: str = "not_evaluated"
    article_alignment: str = "not_evaluated"
    diagnostic_stage: str = ""
    diagnostic_category: str = ""
    cited_documents: List[str] = field(default_factory=list)
    cited_article_numbers: List[str] = field(default_factory=list)
    mismatched_citations: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    supported_derived_claims: List[str] = field(default_factory=list)
    missing_required_facts: List[str] = field(default_factory=list)
    diagnostic_signals: List[DiagnosticSignal] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the grounding-verification payload after initialization.
        """

        self.status = _normalize_string(self.status) or "not_evaluated"
        self.accepted = bool(self.accepted)
        self.citation_status = (
            _normalize_string(self.citation_status) or "not_evaluated"
        )
        self.document_alignment = (
            _normalize_string(self.document_alignment) or "not_evaluated"
        )
        self.article_alignment = (
            _normalize_string(self.article_alignment) or "not_evaluated"
        )
        self.diagnostic_stage = _normalize_string(self.diagnostic_stage)
        self.diagnostic_category = _normalize_string(self.diagnostic_category)
        self.cited_documents = _normalize_string_list(self.cited_documents)
        self.cited_article_numbers = _normalize_string_list(
            self.cited_article_numbers
        )
        self.mismatched_citations = _normalize_string_list(
            self.mismatched_citations
        )
        self.unsupported_claims = _normalize_string_list(self.unsupported_claims)
        self.supported_derived_claims = _normalize_string_list(
            self.supported_derived_claims
        )
        self.missing_required_facts = _normalize_string_list(
            self.missing_required_facts
        )
        self.diagnostic_signals = _normalize_diagnostic_signal_list(
            self.diagnostic_signals
        )
        self.reasons = _normalize_string_list(self.reasons)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class UserQuestionInput(RetrievalModelBase):
    """
    Input contract representing one user question submitted to retrieval.

    Design goals
    ------------
    This model keeps the request explicit and traceable by storing:
    - the question text
    - stable request identifiers when available
    - optional caller metadata used by higher-level orchestration
    """

    question_text: str
    request_id: str = ""
    conversation_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    normalized_query_text: str = ""
    formatting_instructions: List[str] = field(default_factory=list)
    query_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the user-question payload after initialization.
        """

        self.question_text = _normalize_string(self.question_text)
        self.request_id = _normalize_string(self.request_id)
        self.conversation_id = _normalize_string(self.conversation_id)
        self.metadata = _normalize_mapping(self.metadata)
        self.normalized_query_text = (
            _normalize_string(self.normalized_query_text) or self.question_text
        )
        self.formatting_instructions = _normalize_string_list(
            self.formatting_instructions
        )
        self.query_metadata = _normalize_mapping(self.query_metadata)


@dataclass(slots=True)
class RetrievedChunkResult(RetrievalModelBase):
    """
    Normalized retrieval record returned from vector search.

    Design goals
    ------------
    This model preserves the retrieved chunk identity together with:
    - the chunk text used for grounding
    - the source document identity
    - optional ranking and distance information
    - normalized metadata scopes for filtering and citation building
    """

    chunk_id: str
    doc_id: str
    text: str
    record_id: str = ""
    rank: Optional[int] = None
    distance: Optional[float] = None
    similarity_score: Optional[float] = None
    source_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    context_metadata: ContextChunkMetadata = field(default_factory=ContextChunkMetadata)

    def __post_init__(self) -> None:
        """
        Normalize the retrieval-result payload after initialization.
        """

        self.chunk_id = _normalize_string(self.chunk_id)
        self.doc_id = _normalize_string(self.doc_id)
        self.text = _normalize_string(self.text)
        self.record_id = _normalize_string(self.record_id) or self.chunk_id
        self.rank = _normalize_optional_int(self.rank)
        self.distance = _normalize_optional_float(self.distance)
        self.similarity_score = _normalize_optional_float(self.similarity_score)
        self.source_file = _normalize_string(self.source_file)
        self.metadata = _normalize_mapping(self.metadata)
        self.chunk_metadata = _normalize_mapping(self.chunk_metadata)
        self.document_metadata = _normalize_mapping(self.document_metadata)
        if not isinstance(self.context_metadata, ContextChunkMetadata):
            self.context_metadata = ContextChunkMetadata()
        self.context_metadata = self._resolve_context_metadata()

    def _resolve_context_metadata(self) -> ContextChunkMetadata:
        """
        Resolve the explicit context metadata attached to the retrieved chunk.

        Returns
        -------
        ContextChunkMetadata
            Explicit context metadata carrying normalized structural fields.
        """

        if self.context_metadata.has_structural_content():
            return self.context_metadata

        return ContextChunkMetadata.from_retrieved_chunk(self)


@dataclass(slots=True)
class RetrievalContext(RetrievalModelBase):
    """
    Compact grounded context assembled from retrieved chunks.

    Design goals
    ------------
    This model keeps context-building output explicit by storing:
    - the selected chunk records
    - the final packed context text
    - bounded size counters
    - context assembly metadata such as truncation or filtering notes
    """

    chunks: List[RetrievedChunkResult] = field(default_factory=list)
    context_text: str = ""
    chunk_count: int = 0
    character_count: int = 0
    truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    selected_context_metadata: List[ContextChunkMetadata] = field(default_factory=list)
    retrieval_quality: RetrievalQualitySignals = field(
        default_factory=RetrievalQualitySignals
    )
    route_decision: Optional[RetrievalRouteDecision] = None
    evidence_quality: Optional[EvidenceQualityClassification] = None

    def __post_init__(self) -> None:
        """
        Normalize the retrieval-context payload after initialization.
        """

        self.chunks = list(self.chunks)
        self.context_text = _normalize_string(self.context_text)
        self.chunk_count = self.chunk_count or len(self.chunks)
        self.character_count = self.character_count or len(self.context_text)
        self.metadata = _normalize_mapping(self.metadata)
        self.selected_context_metadata = self._resolve_selected_context_metadata()
        self.retrieval_quality = self._resolve_retrieval_quality()
        self.route_decision = self._resolve_route_decision()
        self.evidence_quality = self._resolve_evidence_quality()

    def _resolve_selected_context_metadata(self) -> List[ContextChunkMetadata]:
        """
        Resolve explicit structural metadata for the selected context chunks.

        Returns
        -------
        List[ContextChunkMetadata]
            One explicit metadata record per selected chunk.
        """

        if self.selected_context_metadata:
            return [
                metadata
                if isinstance(metadata, ContextChunkMetadata)
                else ContextChunkMetadata()
                for metadata in self.selected_context_metadata
            ]

        return [chunk.context_metadata for chunk in self.chunks]

    def _resolve_retrieval_quality(self) -> RetrievalQualitySignals:
        """
        Resolve retrieval-quality signals from explicit fields or context metadata.

        Returns
        -------
        RetrievalQualitySignals
            Normalized retrieval-quality summary aligned with the selected context.
        """

        if isinstance(self.retrieval_quality, RetrievalQualitySignals):
            existing_signals = self.retrieval_quality
        else:
            existing_signals = RetrievalQualitySignals()

        if any(
            [
                existing_signals.total_input_chunks,
                existing_signals.candidate_chunk_count,
                existing_signals.selected_chunk_count,
                existing_signals.duplicate_count,
                existing_signals.score_filtered_count,
                existing_signals.omitted_by_rank_limit_count,
                existing_signals.omitted_by_budget_count,
                existing_signals.structural_metadata_chunk_count,
                existing_signals.truncated,
                existing_signals.selected_chunk_ids,
                existing_signals.metadata,
            ]
        ):
            return existing_signals

        structural_metadata_chunk_count = sum(
            1
            for metadata in self.selected_context_metadata
            if metadata.has_structural_content()
        )

        return RetrievalQualitySignals(
            total_input_chunks=self.metadata.get("total_input_chunks", self.chunk_count),
            candidate_chunk_count=self.metadata.get(
                "candidate_chunk_count",
                self.chunk_count,
            ),
            selected_chunk_count=self.chunk_count,
            duplicate_count=self.metadata.get("duplicate_count", 0),
            score_filtered_count=self.metadata.get("score_filtered_count", 0),
            omitted_by_rank_limit_count=self.metadata.get(
                "omitted_by_rank_limit_count",
                0,
            ),
            omitted_by_budget_count=self.metadata.get("omitted_by_budget_count", 0),
            structural_metadata_chunk_count=structural_metadata_chunk_count,
            truncated=self.truncated,
            selected_chunk_ids=self.metadata.get(
                "selected_chunk_ids",
                [chunk.chunk_id for chunk in self.chunks],
            ),
            metadata=self.metadata,
        )

    def _resolve_route_decision(self) -> Optional[RetrievalRouteDecision]:
        """
        Resolve the optional route decision attached to the retrieval context.

        Returns
        -------
        Optional[RetrievalRouteDecision]
            Normalized route decision when provided, otherwise `None`.
        """

        if isinstance(self.route_decision, RetrievalRouteDecision):
            return self.route_decision
        return None

    def _resolve_evidence_quality(self) -> Optional[EvidenceQualityClassification]:
        """
        Resolve the optional evidence-quality contract attached to the context.

        Returns
        -------
        Optional[EvidenceQualityClassification]
            Normalized evidence classification when provided, otherwise `None`.
        """

        if isinstance(self.evidence_quality, EvidenceQualityClassification):
            return self.evidence_quality
        return None


@dataclass(slots=True)
class RetrievalRouteMetadata(RetrievalModelBase):
    """
    Shared metadata bundle carrying route, evidence, and grounding contracts.

    Design goals
    ------------
    This model provides one typed envelope for service, metrics, grounding, and
    benchmark-facing code that needs the routing decisions attached to a result.
    """

    route_decision: Optional[RetrievalRouteDecision] = None
    evidence_quality: Optional[EvidenceQualityClassification] = None
    grounding_verification: Optional[GroundingVerificationResult] = None
    diagnostic_stage: str = ""
    diagnostic_category: str = ""
    diagnostic_signals: List[DiagnosticSignal] = field(default_factory=list)
    benchmark_case_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the route-metadata payload after initialization.
        """

        if not isinstance(self.route_decision, RetrievalRouteDecision):
            self.route_decision = None
        if not isinstance(self.evidence_quality, EvidenceQualityClassification):
            self.evidence_quality = None
        if not isinstance(self.grounding_verification, GroundingVerificationResult):
            self.grounding_verification = None
        self.diagnostic_stage = _normalize_string(self.diagnostic_stage)
        self.diagnostic_category = _normalize_string(self.diagnostic_category)
        self.diagnostic_signals = _normalize_diagnostic_signal_list(
            self.diagnostic_signals
        )
        self.benchmark_case_id = _normalize_string(self.benchmark_case_id)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class GuardrailDecision(RetrievalModelBase):
    """
    Deterministic decision emitted by pre-request or post-response guardrails.

    Design goals
    ------------
    This model keeps guardrail evaluation explainable by storing:
    - the stage that produced the decision
    - whether the request or response is allowed
    - the triggered category and action
    - matched rules or patterns supporting the decision
    """

    stage: str
    allowed: bool
    category: str = ""
    action: str = "allow"
    reason: str = ""
    matched_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the guardrail-decision payload after initialization.
        """

        self.stage = _normalize_string(self.stage)
        self.category = _normalize_string(self.category)
        self.action = _normalize_string(self.action) or "allow"
        self.reason = _normalize_string(self.reason)
        self.matched_rules = _normalize_string_list(self.matched_rules)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class AnswerGenerationInput(RetrievalModelBase):
    """
    Grounded input contract consumed by the answer-generation adapter.

    Design goals
    ------------
    This model isolates answer-generation dependencies by storing:
    - the normalized user question
    - the selected retrieval context
    - optional system or grounding instructions
    - auxiliary metadata needed by the adapter layer
    """

    question: UserQuestionInput
    context: RetrievalContext
    system_instruction: str = ""
    grounding_instruction: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    route_metadata: Optional[RetrievalRouteMetadata] = None

    def __post_init__(self) -> None:
        """
        Normalize the answer-generation payload after initialization.
        """

        self.system_instruction = _normalize_string(self.system_instruction)
        self.grounding_instruction = _normalize_string(self.grounding_instruction)
        self.metadata = _normalize_mapping(self.metadata)
        self.route_metadata = self._resolve_route_metadata()

    def _resolve_route_metadata(self) -> Optional[RetrievalRouteMetadata]:
        """
        Resolve optional routing metadata attached to answer generation.

        Returns
        -------
        Optional[RetrievalRouteMetadata]
            Normalized route metadata when provided, otherwise `None`.
        """

        if isinstance(self.route_metadata, RetrievalRouteMetadata):
            return self.route_metadata
        return None


@dataclass(slots=True)
class MetricsSnapshot(RetrievalModelBase):
    """
    Lightweight runtime metrics snapshot for the retrieval flow.

    Design goals
    ------------
    This model keeps safety, latency, and retrieval-quality metrics explicit by storing:
    - request outcome counters
    - false-positive and jailbreak tracking counters
    - retrieval/context quality counters and labeled recovery signals
    - per-stage timing measurements
    - one total-latency aggregate for the full flow
    """

    total_requests: int = 0
    successful_requests: int = 0
    blocked_requests: int = 0
    deflected_requests: int = 0
    false_positive_count: int = 0
    jailbreak_attempt_count: int = 0
    blocked_jailbreak_attempt_count: int = 0
    retrieval_quality_sample_count: int = 0
    requested_retrieval_count: int = 0
    returned_retrieval_count: int = 0
    retrieved_candidate_count: int = 0
    selected_context_count: int = 0
    truncated_context_count: int = 0
    contexts_with_structural_metadata_count: int = 0
    structural_metadata_chunk_count: int = 0
    labeled_recovery_sample_count: int = 0
    recovered_labeled_chunk_count: int = 0
    stage_latency_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        """
        Normalize the metrics snapshot after initialization.
        """

        self.total_requests = _normalize_non_negative_int(self.total_requests)
        self.successful_requests = _normalize_non_negative_int(self.successful_requests)
        self.blocked_requests = _normalize_non_negative_int(self.blocked_requests)
        self.deflected_requests = _normalize_non_negative_int(self.deflected_requests)
        self.false_positive_count = _normalize_non_negative_int(self.false_positive_count)
        self.jailbreak_attempt_count = _normalize_non_negative_int(
            self.jailbreak_attempt_count
        )
        self.blocked_jailbreak_attempt_count = _normalize_non_negative_int(
            self.blocked_jailbreak_attempt_count
        )
        self.retrieval_quality_sample_count = _normalize_non_negative_int(
            self.retrieval_quality_sample_count
        )
        self.requested_retrieval_count = _normalize_non_negative_int(
            self.requested_retrieval_count
        )
        self.returned_retrieval_count = _normalize_non_negative_int(
            self.returned_retrieval_count
        )
        self.retrieved_candidate_count = _normalize_non_negative_int(
            self.retrieved_candidate_count
        )
        self.selected_context_count = _normalize_non_negative_int(
            self.selected_context_count
        )
        self.truncated_context_count = _normalize_non_negative_int(
            self.truncated_context_count
        )
        self.contexts_with_structural_metadata_count = _normalize_non_negative_int(
            self.contexts_with_structural_metadata_count
        )
        self.structural_metadata_chunk_count = _normalize_non_negative_int(
            self.structural_metadata_chunk_count
        )
        self.labeled_recovery_sample_count = _normalize_non_negative_int(
            self.labeled_recovery_sample_count
        )
        self.recovered_labeled_chunk_count = _normalize_non_negative_int(
            self.recovered_labeled_chunk_count
        )
        self.stage_latency_ms = _normalize_float_mapping(self.stage_latency_ms)
        self.total_latency_ms = float(self.total_latency_ms)


@dataclass(slots=True)
class FinalAnswerResult(RetrievalModelBase):
    """
    Final contract returned by the retrieval service.

    Design goals
    ------------
    This model keeps end-to-end outcomes explicit by storing:
    - the original question contract
    - the final answer text or deflection text
    - answer status and grounding flag
    - optional context, guardrail decisions, and metrics snapshot
    """

    question: UserQuestionInput
    status: str
    answer_text: str = ""
    grounded: bool = False
    retrieval_context: Optional[RetrievalContext] = None
    pre_guardrail: Optional[GuardrailDecision] = None
    post_guardrail: Optional[GuardrailDecision] = None
    citations: List[str] = field(default_factory=list)
    diagnostic_stage: str = ""
    diagnostic_category: str = ""
    diagnostic_signals: List[DiagnosticSignal] = field(default_factory=list)
    answer_metadata: Dict[str, Any] = field(default_factory=dict)
    route_metadata: Optional[RetrievalRouteMetadata] = None
    metrics_snapshot: Optional[MetricsSnapshot] = None

    def __post_init__(self) -> None:
        """
        Normalize the final answer payload after initialization.
        """

        self.status = _normalize_string(self.status)
        self.answer_text = _normalize_string(self.answer_text)
        self.citations = _normalize_string_list(self.citations)
        self.diagnostic_stage = _normalize_string(self.diagnostic_stage)
        self.diagnostic_category = _normalize_string(self.diagnostic_category)
        self.diagnostic_signals = _normalize_diagnostic_signal_list(
            self.diagnostic_signals
        )
        self.answer_metadata = _normalize_mapping(self.answer_metadata)
        self.route_metadata = self._resolve_route_metadata()

    def _resolve_route_metadata(self) -> Optional[RetrievalRouteMetadata]:
        """
        Resolve optional routing metadata attached to the final answer result.

        Returns
        -------
        Optional[RetrievalRouteMetadata]
            Normalized route metadata when provided, otherwise `None`.
        """

        if isinstance(self.route_metadata, RetrievalRouteMetadata):
            return self.route_metadata
        return None
