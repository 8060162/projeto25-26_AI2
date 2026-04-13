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

    def __post_init__(self) -> None:
        """
        Normalize the user-question payload after initialization.
        """

        self.question_text = _normalize_string(self.question_text)
        self.request_id = _normalize_string(self.request_id)
        self.conversation_id = _normalize_string(self.conversation_id)
        self.metadata = _normalize_mapping(self.metadata)


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

    def __post_init__(self) -> None:
        """
        Normalize the retrieval-context payload after initialization.
        """

        self.chunks = list(self.chunks)
        self.context_text = _normalize_string(self.context_text)
        self.chunk_count = self.chunk_count or len(self.chunks)
        self.character_count = self.character_count or len(self.context_text)
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

    def __post_init__(self) -> None:
        """
        Normalize the answer-generation payload after initialization.
        """

        self.system_instruction = _normalize_string(self.system_instruction)
        self.grounding_instruction = _normalize_string(self.grounding_instruction)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class MetricsSnapshot(RetrievalModelBase):
    """
    Lightweight runtime metrics snapshot for the retrieval flow.

    Design goals
    ------------
    This model keeps safety and latency metrics explicit by storing:
    - request outcome counters
    - false-positive and jailbreak tracking counters
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
    stage_latency_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        """
        Normalize the metrics snapshot after initialization.
        """

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
    answer_metadata: Dict[str, Any] = field(default_factory=dict)
    metrics_snapshot: Optional[MetricsSnapshot] = None

    def __post_init__(self) -> None:
        """
        Normalize the final answer payload after initialization.
        """

        self.status = _normalize_string(self.status)
        self.answer_text = _normalize_string(self.answer_text)
        self.citations = _normalize_string_list(self.citations)
        self.answer_metadata = _normalize_mapping(self.answer_metadata)
