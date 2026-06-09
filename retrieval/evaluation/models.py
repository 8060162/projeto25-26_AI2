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


def _normalize_optional_string(value: Any) -> Optional[str]:
    """
    Normalize one optional string payload while preserving missing values.

    Parameters
    ----------
    value : Any
        Candidate string payload.

    Returns
    -------
    Optional[str]
        Stripped string when present, otherwise `None`.
    """

    normalized_value = _normalize_string(value)
    if normalized_value:
        return normalized_value
    return None


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


def _normalize_optional_float(value: Any) -> Optional[float]:
    """
    Normalize one optional numeric payload into a float.

    Parameters
    ----------
    value : Any
        Candidate numeric payload.

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

    if value is None or isinstance(value, bool):
        return 0

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _normalize_float_mapping(value: Any) -> Dict[str, float]:
    """
    Normalize one optional mapping payload into string-float metric values.

    Parameters
    ----------
    value : Any
        Candidate metric mapping payload.

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


class EvaluationModelBase:
    """
    Provide a shared serialization helper for evaluation data contracts.
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
class BenchmarkRouteExpectation(EvaluationModelBase):
    """
    Expected deterministic retrieval route for one benchmark question case.
    """

    route_name: str = ""
    retrieval_scope: str = ""
    retrieval_profile: str = ""
    target_document_titles: List[str] = field(default_factory=list)
    target_doc_ids: List[str] = field(default_factory=list)
    target_article_numbers: List[str] = field(default_factory=list)
    comparative: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the route expectation after initialization.
        """

        self.route_name = _normalize_string(self.route_name)
        self.retrieval_scope = _normalize_string(self.retrieval_scope)
        self.retrieval_profile = _normalize_string(self.retrieval_profile)
        self.target_document_titles = _normalize_string_list(
            self.target_document_titles
        )
        self.target_doc_ids = _normalize_string_list(self.target_doc_ids)
        self.target_article_numbers = _normalize_string_list(
            self.target_article_numbers
        )
        self.comparative = bool(self.comparative)
        self.metadata = _normalize_mapping(self.metadata)

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkRouteExpectation":
        """
        Build a route expectation from one raw benchmark mapping.

        Parameters
        ----------
        value : Any
            Raw route expectation payload.

        Returns
        -------
        BenchmarkRouteExpectation
            Normalized route expectation.
        """

        payload = _normalize_mapping(value)
        known_keys = {
            "route_name",
            "retrieval_scope",
            "retrieval_profile",
            "target_document_titles",
            "target_doc_ids",
            "target_article_numbers",
            "comparative",
        }

        return cls(
            route_name=payload.get("route_name", ""),
            retrieval_scope=payload.get("retrieval_scope", ""),
            retrieval_profile=payload.get("retrieval_profile", ""),
            target_document_titles=payload.get("target_document_titles", []),
            target_doc_ids=payload.get("target_doc_ids", []),
            target_article_numbers=payload.get("target_article_numbers", []),
            comparative=payload.get("comparative", False),
            metadata={
                key: value for key, value in payload.items() if key not in known_keys
            },
        )


@dataclass(slots=True)
class BenchmarkGroundingLabels(EvaluationModelBase):
    """
    Expected grounding labels used by answer and citation evaluators.
    """

    expected_citation_doc_ids: List[str] = field(default_factory=list)
    expected_citation_article_numbers: List[str] = field(default_factory=list)
    ambiguity: str = ""
    article_misattribution_risk: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the grounding labels after initialization.
        """

        self.expected_citation_doc_ids = _normalize_string_list(
            self.expected_citation_doc_ids
        )
        self.expected_citation_article_numbers = _normalize_string_list(
            self.expected_citation_article_numbers
        )
        self.ambiguity = _normalize_string(self.ambiguity)
        self.article_misattribution_risk = _normalize_string(
            self.article_misattribution_risk
        )
        self.metadata = _normalize_mapping(self.metadata)

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkGroundingLabels":
        """
        Build grounding labels from one raw benchmark mapping.

        Parameters
        ----------
        value : Any
            Raw grounding-label payload.

        Returns
        -------
        BenchmarkGroundingLabels
            Normalized grounding labels.
        """

        payload = _normalize_mapping(value)
        known_keys = {
            "expected_citation_doc_ids",
            "expected_citation_article_numbers",
            "ambiguity",
            "article_misattribution_risk",
        }

        return cls(
            expected_citation_doc_ids=payload.get("expected_citation_doc_ids", []),
            expected_citation_article_numbers=payload.get(
                "expected_citation_article_numbers",
                [],
            ),
            ambiguity=payload.get("ambiguity", ""),
            article_misattribution_risk=payload.get("article_misattribution_risk", ""),
            metadata={
                key: value for key, value in payload.items() if key not in known_keys
            },
        )


@dataclass(slots=True)
class BenchmarkQuestionCase(EvaluationModelBase):
    """
    Typed benchmark case for factual legal QA and retrieval evaluation.
    """

    case_id: str
    question: str
    case_type: str = ""
    expected_route: BenchmarkRouteExpectation = field(
        default_factory=BenchmarkRouteExpectation
    )
    expected_doc_id: Optional[str] = None
    expected_article_numbers: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    required_facts: List[str] = field(default_factory=list)
    forbidden_facts: List[str] = field(default_factory=list)
    expected_answer_behavior: str = ""
    grounding_labels: BenchmarkGroundingLabels = field(
        default_factory=BenchmarkGroundingLabels
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the benchmark question case after initialization.
        """

        self.case_id = _normalize_string(self.case_id)
        self.question = _normalize_string(self.question)
        self.case_type = _normalize_string(self.case_type)
        self.expected_doc_id = _normalize_optional_string(self.expected_doc_id)
        self.expected_article_numbers = _normalize_string_list(
            self.expected_article_numbers
        )
        self.expected_chunk_ids = _normalize_string_list(self.expected_chunk_ids)
        self.required_facts = _normalize_string_list(self.required_facts)
        self.forbidden_facts = _normalize_string_list(self.forbidden_facts)
        self.expected_answer_behavior = _normalize_string(
            self.expected_answer_behavior
        )
        if not isinstance(self.expected_route, BenchmarkRouteExpectation):
            self.expected_route = BenchmarkRouteExpectation.from_mapping(
                self.expected_route
            )
        if not isinstance(self.grounding_labels, BenchmarkGroundingLabels):
            self.grounding_labels = BenchmarkGroundingLabels.from_mapping(
                self.grounding_labels
            )
        self.metadata = _normalize_mapping(self.metadata)

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkQuestionCase":
        """
        Build a benchmark question case from one raw JSONL record.

        Parameters
        ----------
        value : Any
            Raw benchmark question payload.

        Returns
        -------
        BenchmarkQuestionCase
            Normalized benchmark question case.
        """

        payload = _normalize_mapping(value)
        known_keys = {
            "case_id",
            "question",
            "case_type",
            "expected_route",
            "expected_doc_id",
            "expected_article_numbers",
            "expected_chunk_ids",
            "required_facts",
            "forbidden_facts",
            "expected_answer_behavior",
            "grounding_labels",
        }

        return cls(
            case_id=payload.get("case_id", ""),
            question=payload.get("question", ""),
            case_type=payload.get("case_type", ""),
            expected_route=BenchmarkRouteExpectation.from_mapping(
                payload.get("expected_route", {})
            ),
            expected_doc_id=payload.get("expected_doc_id"),
            expected_article_numbers=payload.get("expected_article_numbers", []),
            expected_chunk_ids=payload.get("expected_chunk_ids", []),
            required_facts=payload.get("required_facts", []),
            forbidden_facts=payload.get("forbidden_facts", []),
            expected_answer_behavior=payload.get("expected_answer_behavior", ""),
            grounding_labels=BenchmarkGroundingLabels.from_mapping(
                payload.get("grounding_labels", {})
            ),
            metadata={
                key: value for key, value in payload.items() if key not in known_keys
            },
        )


@dataclass(slots=True)
class BenchmarkGuardrailCase(EvaluationModelBase):
    """
    Typed benchmark case for deterministic guardrail evaluation.
    """

    case_id: str
    question: str
    category: str = ""
    expected_action: str = ""
    expected_safe: bool = True
    expected_route: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the benchmark guardrail case after initialization.
        """

        self.case_id = _normalize_string(self.case_id)
        self.question = _normalize_string(self.question)
        self.category = _normalize_string(self.category)
        self.expected_action = _normalize_string(self.expected_action)
        self.expected_safe = bool(self.expected_safe)
        self.expected_route = _normalize_string(self.expected_route)
        self.notes = _normalize_mapping(self.notes)
        self.metadata = _normalize_mapping(self.metadata)

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkGuardrailCase":
        """
        Build a benchmark guardrail case from one raw JSONL record.

        Parameters
        ----------
        value : Any
            Raw benchmark guardrail payload.

        Returns
        -------
        BenchmarkGuardrailCase
            Normalized benchmark guardrail case.
        """

        payload = _normalize_mapping(value)
        known_keys = {
            "case_id",
            "question",
            "category",
            "expected_action",
            "expected_safe",
            "expected_route",
            "notes",
        }

        return cls(
            case_id=payload.get("case_id", ""),
            question=payload.get("question", ""),
            category=payload.get("category", ""),
            expected_action=payload.get("expected_action", ""),
            expected_safe=payload.get("expected_safe", True),
            expected_route=payload.get("expected_route", ""),
            notes=payload.get("notes", {}),
            metadata={
                key: value for key, value in payload.items() if key not in known_keys
            },
        )


@dataclass(slots=True)
class RetrievalEvaluationResult(EvaluationModelBase):
    """
    Result contract for one retrieval benchmark evaluation case.
    """

    case_id: str
    expected_doc_id: Optional[str] = None
    expected_article_numbers: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    selected_chunk_ids: List[str] = field(default_factory=list)
    retrieved_doc_ids: List[str] = field(default_factory=list)
    retrieved_article_numbers: List[str] = field(default_factory=list)
    expected_doc_recovered: bool = False
    expected_article_recovered: bool = False
    expected_chunk_recovered: bool = False
    selected_context_hit: bool = False
    reciprocal_rank: Optional[float] = None
    conflict_present: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the retrieval evaluation result after initialization.
        """

        self.case_id = _normalize_string(self.case_id)
        self.expected_doc_id = _normalize_optional_string(self.expected_doc_id)
        self.expected_article_numbers = _normalize_string_list(
            self.expected_article_numbers
        )
        self.expected_chunk_ids = _normalize_string_list(self.expected_chunk_ids)
        self.retrieved_chunk_ids = _normalize_string_list(self.retrieved_chunk_ids)
        self.selected_chunk_ids = _normalize_string_list(self.selected_chunk_ids)
        self.retrieved_doc_ids = _normalize_string_list(self.retrieved_doc_ids)
        self.retrieved_article_numbers = _normalize_string_list(
            self.retrieved_article_numbers
        )
        self.expected_doc_recovered = bool(self.expected_doc_recovered)
        self.expected_article_recovered = bool(self.expected_article_recovered)
        self.expected_chunk_recovered = bool(self.expected_chunk_recovered)
        self.selected_context_hit = bool(self.selected_context_hit)
        self.reciprocal_rank = _normalize_optional_float(self.reciprocal_rank)
        self.conflict_present = bool(self.conflict_present)
        self.metrics = _normalize_float_mapping(self.metrics)
        self.reasons = _normalize_string_list(self.reasons)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class AnswerEvaluationResult(EvaluationModelBase):
    """
    Result contract for one answer and grounding benchmark evaluation case.
    """

    case_id: str
    expected_behavior: str = ""
    observed_behavior: str = ""
    document_citation_correct: bool = False
    article_citation_correct: bool = False
    required_fact_matches: List[str] = field(default_factory=list)
    missing_required_facts: List[str] = field(default_factory=list)
    forbidden_fact_violations: List[str] = field(default_factory=list)
    deflection_correct: bool = False
    caution_correct: bool = False
    passed: bool = False
    score: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the answer evaluation result after initialization.
        """

        self.case_id = _normalize_string(self.case_id)
        self.expected_behavior = _normalize_string(self.expected_behavior)
        self.observed_behavior = _normalize_string(self.observed_behavior)
        self.document_citation_correct = bool(self.document_citation_correct)
        self.article_citation_correct = bool(self.article_citation_correct)
        self.required_fact_matches = _normalize_string_list(
            self.required_fact_matches
        )
        self.missing_required_facts = _normalize_string_list(
            self.missing_required_facts
        )
        self.forbidden_fact_violations = _normalize_string_list(
            self.forbidden_fact_violations
        )
        self.deflection_correct = bool(self.deflection_correct)
        self.caution_correct = bool(self.caution_correct)
        self.passed = bool(self.passed)
        self.score = _normalize_optional_float(self.score)
        self.reasons = _normalize_string_list(self.reasons)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class GuardrailEvaluationResult(EvaluationModelBase):
    """
    Result contract for one deterministic guardrail benchmark case.
    """

    case_id: str
    category: str = ""
    expected_action: str = ""
    observed_action: str = ""
    expected_safe: bool = True
    observed_safe: bool = True
    expected_route: str = ""
    observed_route: str = ""
    matched_rules: List[str] = field(default_factory=list)
    passed: bool = False
    false_positive: bool = False
    false_negative: bool = False
    jailbreak_attempt: bool = False
    blocked_jailbreak: bool = False
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the guardrail evaluation result after initialization.
        """

        self.case_id = _normalize_string(self.case_id)
        self.category = _normalize_string(self.category)
        self.expected_action = _normalize_string(self.expected_action)
        self.observed_action = _normalize_string(self.observed_action)
        self.expected_safe = bool(self.expected_safe)
        self.observed_safe = bool(self.observed_safe)
        self.expected_route = _normalize_string(self.expected_route)
        self.observed_route = _normalize_string(self.observed_route)
        self.matched_rules = _normalize_string_list(self.matched_rules)
        self.passed = bool(self.passed)
        self.false_positive = bool(self.false_positive)
        self.false_negative = bool(self.false_negative)
        self.jailbreak_attempt = bool(self.jailbreak_attempt)
        self.blocked_jailbreak = bool(self.blocked_jailbreak)
        self.reasons = _normalize_string_list(self.reasons)
        self.metadata = _normalize_mapping(self.metadata)


@dataclass(slots=True)
class BenchmarkRunSummary(EvaluationModelBase):
    """
    Aggregate result contract for one benchmark execution.
    """

    run_id: str = ""
    mode: str = ""
    question_case_count: int = 0
    guardrail_case_count: int = 0
    retrieval_results: List[RetrievalEvaluationResult] = field(default_factory=list)
    answer_results: List[AnswerEvaluationResult] = field(default_factory=list)
    guardrail_results: List[GuardrailEvaluationResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the benchmark run summary after initialization.
        """

        self.run_id = _normalize_string(self.run_id)
        self.mode = _normalize_string(self.mode)
        self.question_case_count = _normalize_non_negative_int(
            self.question_case_count
        )
        self.guardrail_case_count = _normalize_non_negative_int(
            self.guardrail_case_count
        )
        self.retrieval_results = [
            result
            if isinstance(result, RetrievalEvaluationResult)
            else RetrievalEvaluationResult(case_id="")
            for result in self.retrieval_results
        ]
        self.answer_results = [
            result
            if isinstance(result, AnswerEvaluationResult)
            else AnswerEvaluationResult(case_id="")
            for result in self.answer_results
        ]
        self.guardrail_results = [
            result
            if isinstance(result, GuardrailEvaluationResult)
            else GuardrailEvaluationResult(case_id="")
            for result in self.guardrail_results
        ]
        self.metrics = _normalize_float_mapping(self.metrics)
        self.errors = _normalize_string_list(self.errors)
        self.metadata = _normalize_mapping(self.metadata)
