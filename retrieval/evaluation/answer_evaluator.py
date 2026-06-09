from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Set

from retrieval.evaluation.models import (
    AnswerEvaluationResult,
    BenchmarkQuestionCase,
    BenchmarkRunSummary,
)
from retrieval.models import GroundingVerificationResult


_ARTICLE_REFERENCE_PATTERN = re.compile(
    r"\b(?:artigo|article|art)\.?\s*(?:n[.\u00bao]*\s*)?(\d+[a-z]?)\b",
    re.IGNORECASE,
)
_ARTICLE_EQUALS_PATTERN = re.compile(r"\barticle\s*=\s*(\d+[a-z]?)\b", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_DEFLECTION_MARKERS = (
    "nao ha evidencia",
    "sem evidencia",
    "nao existe evidencia",
    "nao tenho contexto",
    "contexto insuficiente",
    "informacao insuficiente",
    "insufficient context",
    "not enough evidence",
    "cannot answer",
)
_CAUTION_MARKERS = (
    "com base no contexto disponivel",
    "o contexto disponivel",
    "nao fornece",
    "nao permite confirmar",
    "deve confirmar",
    "pode depender",
    "cautela",
    "available context",
)


class AnswerBenchmarkEvaluator:
    """
    Evaluate offline answer and grounding quality against benchmark labels.
    """

    def evaluate_case(
        self,
        benchmark_case: BenchmarkQuestionCase,
        answer_text: str,
        *,
        observed_behavior: str = "",
        citations: Optional[Sequence[str]] = None,
        grounding_verification: Optional[GroundingVerificationResult] = None,
        observed_route: str = "",
    ) -> AnswerEvaluationResult:
        """
        Evaluate one generated answer against one benchmark question case.

        Parameters
        ----------
        benchmark_case : BenchmarkQuestionCase
            Benchmark case carrying expected facts, citations, and behavior.

        answer_text : str
            Generated answer text to evaluate.

        observed_behavior : str
            Optional observed behavior emitted by the runtime flow.

        citations : Optional[Sequence[str]]
            Optional citation fragments returned with the answer.

        grounding_verification : Optional[GroundingVerificationResult]
            Optional runtime grounding result to include in citation checks.

        observed_route : str
            Optional route name emitted by the runtime flow.

        Returns
        -------
        AnswerEvaluationResult
            Per-case answer-quality result with deterministic metrics.
        """

        normalized_answer_text = (
            answer_text.strip() if isinstance(answer_text, str) else ""
        )
        citation_fragments = self._collect_citation_fragments(
            answer_text=normalized_answer_text,
            citations=citations,
            grounding_verification=grounding_verification,
        )
        expected_doc_ids = self._resolve_expected_doc_ids(benchmark_case)
        expected_article_numbers = self._resolve_expected_article_numbers(
            benchmark_case
        )
        cited_doc_ids = self._extract_expected_doc_citations(
            expected_doc_ids=expected_doc_ids,
            text_fragments=citation_fragments,
            grounding_verification=grounding_verification,
        )
        cited_article_numbers = self._extract_article_numbers(
            text_fragments=citation_fragments,
            grounding_verification=grounding_verification,
        )
        required_fact_matches = self._find_required_fact_matches(
            benchmark_case.required_facts,
            normalized_answer_text,
        )
        missing_required_facts = [
            fact
            for fact in benchmark_case.required_facts
            if fact not in required_fact_matches
        ]
        forbidden_fact_violations = self._find_forbidden_fact_violations(
            benchmark_case.forbidden_facts,
            normalized_answer_text,
        )
        resolved_observed_behavior = self._resolve_observed_behavior(
            observed_behavior=observed_behavior,
            answer_text=normalized_answer_text,
        )
        document_citation_correct = self._has_expected_citations(
            expected_values=expected_doc_ids,
            observed_values=cited_doc_ids,
        )
        article_citation_correct = self._has_expected_citations(
            expected_values=expected_article_numbers,
            observed_values=cited_article_numbers,
        )
        deflection_correct = self._is_expected_behavior(
            expected_behavior=benchmark_case.expected_answer_behavior,
            observed_behavior=resolved_observed_behavior,
            expected_values={"deflect", "deflected"},
        )
        caution_correct = self._is_expected_behavior(
            expected_behavior=benchmark_case.expected_answer_behavior,
            observed_behavior=resolved_observed_behavior,
            expected_values={"cautious_answer", "caution", "cautious"},
        )
        behavior_correct = self._behavior_matches_expectation(
            expected_behavior=benchmark_case.expected_answer_behavior,
            observed_behavior=resolved_observed_behavior,
        )
        route_correct = self._route_matches_expectation(
            expected_route=benchmark_case.expected_route.route_name,
            observed_route=observed_route,
        )
        passed = self._resolve_passed(
            document_citation_correct=document_citation_correct,
            article_citation_correct=article_citation_correct,
            missing_required_facts=missing_required_facts,
            forbidden_fact_violations=forbidden_fact_violations,
            behavior_correct=behavior_correct,
            route_correct=route_correct,
        )
        metrics = self._build_metrics(
            document_citation_correct=document_citation_correct,
            article_citation_correct=article_citation_correct,
            required_fact_count=len(benchmark_case.required_facts),
            required_fact_match_count=len(required_fact_matches),
            forbidden_fact_violations=forbidden_fact_violations,
            behavior_correct=behavior_correct,
            route_correct=route_correct,
            passed=passed,
        )

        return AnswerEvaluationResult(
            case_id=benchmark_case.case_id,
            expected_behavior=benchmark_case.expected_answer_behavior,
            observed_behavior=resolved_observed_behavior,
            document_citation_correct=document_citation_correct,
            article_citation_correct=article_citation_correct,
            required_fact_matches=required_fact_matches,
            missing_required_facts=missing_required_facts,
            forbidden_fact_violations=forbidden_fact_violations,
            deflection_correct=deflection_correct,
            caution_correct=caution_correct,
            passed=passed,
            score=metrics["score"],
            reasons=self._build_reasons(
                document_citation_correct=document_citation_correct,
                article_citation_correct=article_citation_correct,
                missing_required_facts=missing_required_facts,
                forbidden_fact_violations=forbidden_fact_violations,
                behavior_correct=behavior_correct,
                route_correct=route_correct,
            ),
            metadata={
                "expected_doc_ids": expected_doc_ids,
                "expected_article_numbers": expected_article_numbers,
                "cited_doc_ids": cited_doc_ids,
                "cited_article_numbers": cited_article_numbers,
                "expected_route": benchmark_case.expected_route.route_name,
                "observed_route": observed_route,
                "metrics": metrics,
            },
        )

    def evaluate_cases(
        self,
        benchmark_cases: Sequence[BenchmarkQuestionCase],
        answers_by_case_id: Dict[str, str],
        *,
        observed_behavior_by_case_id: Optional[Dict[str, str]] = None,
        citations_by_case_id: Optional[Dict[str, Sequence[str]]] = None,
        grounding_by_case_id: Optional[Dict[str, GroundingVerificationResult]] = None,
        observed_route_by_case_id: Optional[Dict[str, str]] = None,
    ) -> BenchmarkRunSummary:
        """
        Evaluate multiple answer benchmark cases and aggregate their metrics.

        Parameters
        ----------
        benchmark_cases : Sequence[BenchmarkQuestionCase]
            Ordered benchmark cases to evaluate.

        answers_by_case_id : Dict[str, str]
            Generated answers keyed by benchmark case identifier.

        observed_behavior_by_case_id : Optional[Dict[str, str]]
            Optional observed answer behaviors keyed by case identifier.

        citations_by_case_id : Optional[Dict[str, Sequence[str]]]
            Optional answer citations keyed by case identifier.

        grounding_by_case_id : Optional[Dict[str, GroundingVerificationResult]]
            Optional grounding results keyed by case identifier.

        observed_route_by_case_id : Optional[Dict[str, str]]
            Optional observed route names keyed by case identifier.

        Returns
        -------
        BenchmarkRunSummary
            Aggregate answer benchmark summary.
        """

        observed_behaviors = observed_behavior_by_case_id or {}
        citations = citations_by_case_id or {}
        grounding_results = grounding_by_case_id or {}
        observed_routes = observed_route_by_case_id or {}
        results = [
            self.evaluate_case(
                benchmark_case=benchmark_case,
                answer_text=answers_by_case_id.get(benchmark_case.case_id, ""),
                observed_behavior=observed_behaviors.get(benchmark_case.case_id, ""),
                citations=citations.get(benchmark_case.case_id),
                grounding_verification=grounding_results.get(benchmark_case.case_id),
                observed_route=observed_routes.get(benchmark_case.case_id, ""),
            )
            for benchmark_case in benchmark_cases
        ]

        return BenchmarkRunSummary(
            mode="answer",
            question_case_count=len(benchmark_cases),
            answer_results=results,
            metrics=summarize_answer_results(results),
        )

    def _collect_citation_fragments(
        self,
        *,
        answer_text: str,
        citations: Optional[Sequence[str]],
        grounding_verification: Optional[GroundingVerificationResult],
    ) -> List[str]:
        """
        Collect answer and citation text fragments used for citation checks.
        """

        fragments = [answer_text]
        fragments.extend(
            citation for citation in citations or [] if isinstance(citation, str)
        )

        if isinstance(grounding_verification, GroundingVerificationResult):
            fragments.extend(grounding_verification.cited_documents)
            fragments.extend(grounding_verification.cited_article_numbers)

        return fragments

    def _resolve_expected_doc_ids(
        self,
        benchmark_case: BenchmarkQuestionCase,
    ) -> List[str]:
        """
        Resolve expected document citation identifiers from benchmark labels.
        """

        doc_ids = list(benchmark_case.grounding_labels.expected_citation_doc_ids)
        if (
            benchmark_case.expected_doc_id
            and benchmark_case.expected_doc_id not in doc_ids
        ):
            doc_ids.append(benchmark_case.expected_doc_id)
        return doc_ids

    def _resolve_expected_article_numbers(
        self,
        benchmark_case: BenchmarkQuestionCase,
    ) -> List[str]:
        """
        Resolve expected article citation numbers from benchmark labels.
        """

        article_numbers = list(
            benchmark_case.grounding_labels.expected_citation_article_numbers
        )
        for article_number in benchmark_case.expected_article_numbers:
            if article_number not in article_numbers:
                article_numbers.append(article_number)
        return article_numbers

    def _extract_expected_doc_citations(
        self,
        *,
        expected_doc_ids: Sequence[str],
        text_fragments: Sequence[str],
        grounding_verification: Optional[GroundingVerificationResult],
    ) -> List[str]:
        """
        Extract expected document identifiers that appear in answer evidence.
        """

        observed_doc_ids: List[str] = []
        normalized_fragments = " ".join(
            _normalize_comparison_text(fragment) for fragment in text_fragments
        )

        for expected_doc_id in expected_doc_ids:
            if _normalize_comparison_text(expected_doc_id) in normalized_fragments:
                observed_doc_ids.append(expected_doc_id)

        if isinstance(grounding_verification, GroundingVerificationResult):
            for document in grounding_verification.cited_documents:
                for expected_doc_id in expected_doc_ids:
                    if (
                        expected_doc_id not in observed_doc_ids
                        and _normalize_comparison_text(expected_doc_id)
                        == _normalize_comparison_text(document)
                    ):
                        observed_doc_ids.append(expected_doc_id)

        return observed_doc_ids

    def _extract_article_numbers(
        self,
        *,
        text_fragments: Sequence[str],
        grounding_verification: Optional[GroundingVerificationResult],
    ) -> List[str]:
        """
        Extract cited article numbers from answers, citations, and grounding data.
        """

        article_numbers: List[str] = []
        seen_numbers: Set[str] = set()

        for text_fragment in text_fragments:
            for pattern in (_ARTICLE_REFERENCE_PATTERN, _ARTICLE_EQUALS_PATTERN):
                for match in pattern.findall(text_fragment or ""):
                    article_number = str(match).strip().lower()
                    if article_number and article_number not in seen_numbers:
                        article_numbers.append(article_number)
                        seen_numbers.add(article_number)

        if isinstance(grounding_verification, GroundingVerificationResult):
            for article_number in grounding_verification.cited_article_numbers:
                normalized_number = str(article_number).strip().lower()
                if normalized_number and normalized_number not in seen_numbers:
                    article_numbers.append(normalized_number)
                    seen_numbers.add(normalized_number)

        return article_numbers

    def _find_required_fact_matches(
        self,
        required_facts: Sequence[str],
        answer_text: str,
    ) -> List[str]:
        """
        Find required benchmark facts covered by the answer text.
        """

        normalized_answer_text = _normalize_comparison_text(answer_text)
        return [
            fact
            for fact in required_facts
            if _normalize_comparison_text(fact) in normalized_answer_text
        ]

    def _find_forbidden_fact_violations(
        self,
        forbidden_facts: Sequence[str],
        answer_text: str,
    ) -> List[str]:
        """
        Find forbidden benchmark facts present in the answer text.
        """

        normalized_answer_text = _normalize_comparison_text(answer_text)
        return [
            fact
            for fact in forbidden_facts
            if _normalize_comparison_text(fact) in normalized_answer_text
        ]

    def _resolve_observed_behavior(
        self,
        *,
        observed_behavior: str,
        answer_text: str,
    ) -> str:
        """
        Resolve observed answer behavior from explicit input or answer markers.
        """

        normalized_behavior = _normalize_behavior(observed_behavior)
        if normalized_behavior:
            return normalized_behavior

        normalized_answer_text = _normalize_comparison_text(answer_text)
        if any(marker in normalized_answer_text for marker in _DEFLECTION_MARKERS):
            return "deflect"
        if any(marker in normalized_answer_text for marker in _CAUTION_MARKERS):
            return "cautious_answer"
        return "answer"

    def _has_expected_citations(
        self,
        *,
        expected_values: Sequence[str],
        observed_values: Sequence[str],
    ) -> bool:
        """
        Check whether all labeled expected citations were observed.
        """

        if not expected_values:
            return True

        observed_keys = {_normalize_comparison_text(value) for value in observed_values}
        return all(
            _normalize_comparison_text(expected_value) in observed_keys
            for expected_value in expected_values
        )

    def _is_expected_behavior(
        self,
        *,
        expected_behavior: str,
        observed_behavior: str,
        expected_values: Set[str],
    ) -> bool:
        """
        Check whether expected and observed behavior match a behavior family.
        """

        expected_key = _normalize_behavior(expected_behavior)
        observed_key = _normalize_behavior(observed_behavior)
        return expected_key in expected_values and observed_key in expected_values

    def _behavior_matches_expectation(
        self,
        *,
        expected_behavior: str,
        observed_behavior: str,
    ) -> bool:
        """
        Check whether observed answer behavior matches the benchmark label.
        """

        expected_key = _normalize_behavior(expected_behavior)
        observed_key = _normalize_behavior(observed_behavior)
        return expected_key == observed_key

    def _route_matches_expectation(
        self,
        *,
        expected_route: str,
        observed_route: str,
    ) -> bool:
        """
        Check optional route alignment when an observed route is provided.
        """

        if not observed_route:
            return True
        if not expected_route:
            return True
        return _normalize_comparison_text(expected_route) == _normalize_comparison_text(
            observed_route
        )

    def _resolve_passed(
        self,
        *,
        document_citation_correct: bool,
        article_citation_correct: bool,
        missing_required_facts: Sequence[str],
        forbidden_fact_violations: Sequence[str],
        behavior_correct: bool,
        route_correct: bool,
    ) -> bool:
        """
        Resolve the final pass flag from all deterministic evaluation signals.
        """

        return all(
            [
                document_citation_correct,
                article_citation_correct,
                not missing_required_facts,
                not forbidden_fact_violations,
                behavior_correct,
                route_correct,
            ]
        )

    def _build_metrics(
        self,
        *,
        document_citation_correct: bool,
        article_citation_correct: bool,
        required_fact_count: int,
        required_fact_match_count: int,
        forbidden_fact_violations: Sequence[str],
        behavior_correct: bool,
        route_correct: bool,
        passed: bool,
    ) -> Dict[str, float]:
        """
        Build stable per-case answer metrics.
        """

        required_fact_coverage = (
            required_fact_match_count / required_fact_count
            if required_fact_count
            else 1.0
        )
        clean_forbidden_facts = 0.0 if forbidden_fact_violations else 1.0
        component_scores = [
            1.0 if document_citation_correct else 0.0,
            1.0 if article_citation_correct else 0.0,
            required_fact_coverage,
            clean_forbidden_facts,
            1.0 if behavior_correct else 0.0,
            1.0 if route_correct else 0.0,
        ]

        return {
            "document_citation_correct": (
                1.0 if document_citation_correct else 0.0
            ),
            "article_citation_correct": 1.0 if article_citation_correct else 0.0,
            "required_fact_coverage": required_fact_coverage,
            "forbidden_fact_violation_rate": (
                1.0 if forbidden_fact_violations else 0.0
            ),
            "behavior_correct": 1.0 if behavior_correct else 0.0,
            "route_correct": 1.0 if route_correct else 0.0,
            "passed": 1.0 if passed else 0.0,
            "score": sum(component_scores) / len(component_scores),
        }

    def _build_reasons(
        self,
        *,
        document_citation_correct: bool,
        article_citation_correct: bool,
        missing_required_facts: Sequence[str],
        forbidden_fact_violations: Sequence[str],
        behavior_correct: bool,
        route_correct: bool,
    ) -> List[str]:
        """
        Build deterministic diagnostic reason codes for answer evaluation.
        """

        reasons: List[str] = []

        if not document_citation_correct:
            reasons.append("answer.document_citation_mismatch")
        if not article_citation_correct:
            reasons.append("answer.article_citation_mismatch")
        if missing_required_facts:
            reasons.append("answer.required_facts_missing")
        if forbidden_fact_violations:
            reasons.append("answer.forbidden_facts_present")
        if not behavior_correct:
            reasons.append("answer.behavior_mismatch")
        if not route_correct:
            reasons.append("answer.route_mismatch")

        if not reasons:
            reasons.append("answer.labels_satisfied")

        return reasons


def evaluate_answer_case(
    benchmark_case: BenchmarkQuestionCase,
    answer_text: str,
    *,
    observed_behavior: str = "",
    citations: Optional[Sequence[str]] = None,
    grounding_verification: Optional[GroundingVerificationResult] = None,
    observed_route: str = "",
) -> AnswerEvaluationResult:
    """
    Evaluate one answer benchmark case with the shared evaluator.
    """

    return AnswerBenchmarkEvaluator().evaluate_case(
        benchmark_case=benchmark_case,
        answer_text=answer_text,
        observed_behavior=observed_behavior,
        citations=citations,
        grounding_verification=grounding_verification,
        observed_route=observed_route,
    )


def summarize_answer_results(
    results: Sequence[AnswerEvaluationResult],
) -> Dict[str, float]:
    """
    Aggregate answer benchmark results into stable numeric metrics.
    """

    if not results:
        return {
            "case_count": 0.0,
            "pass_rate": 0.0,
            "score": 0.0,
            "document_citation_accuracy": 0.0,
            "article_citation_accuracy": 0.0,
            "required_fact_coverage": 0.0,
            "forbidden_fact_violation_rate": 0.0,
            "behavior_accuracy": 0.0,
            "deflection_accuracy": 0.0,
            "caution_accuracy": 0.0,
        }

    return {
        "case_count": float(len(results)),
        "pass_rate": _mean_bool(results, "passed"),
        "score": sum(result.score or 0.0 for result in results) / len(results),
        "document_citation_accuracy": _mean_bool(
            results,
            "document_citation_correct",
        ),
        "article_citation_accuracy": _mean_bool(
            results,
            "article_citation_correct",
        ),
        "required_fact_coverage": _mean_nested_metric(
            results,
            "required_fact_coverage",
        ),
        "forbidden_fact_violation_rate": _mean_nested_metric(
            results,
            "forbidden_fact_violation_rate",
        ),
        "behavior_accuracy": _mean_nested_metric(results, "behavior_correct"),
        "deflection_accuracy": _mean_behavior_bool(
            results,
            expected_behavior="deflect",
            attribute_name="deflection_correct",
        ),
        "caution_accuracy": _mean_behavior_bool(
            results,
            expected_behavior="cautious_answer",
            attribute_name="caution_correct",
        ),
    }


def _normalize_comparison_text(value: object) -> str:
    """
    Normalize text into an accent-free lowercase comparison form.
    """

    if not isinstance(value, str):
        return ""

    normalized_value = unicodedata.normalize("NFKD", value)
    ascii_value = normalized_value.encode("ascii", "ignore").decode("ascii")
    return " ".join(_TOKEN_PATTERN.findall(ascii_value.lower()))


def _normalize_behavior(value: object) -> str:
    """
    Normalize answer-behavior labels into benchmark behavior keys.
    """

    normalized_value = _normalize_comparison_text(value).replace(" ", "_")
    if normalized_value in {"deflected", "deflection"}:
        return "deflect"
    if normalized_value in {"caution", "cautious"}:
        return "cautious_answer"
    if normalized_value in {"completed", "grounded", "normal_answer"}:
        return "answer"
    return normalized_value


def _mean_bool(
    results: Sequence[AnswerEvaluationResult],
    attribute_name: str,
) -> float:
    """
    Average one boolean answer-result attribute across all results.
    """

    matching_count = sum(
        1.0 for result in results if bool(getattr(result, attribute_name))
    )
    return matching_count / len(results)


def _mean_nested_metric(
    results: Sequence[AnswerEvaluationResult],
    metric_name: str,
) -> float:
    """
    Average one numeric per-case metric stored in result metadata.
    """

    return sum(
        float(result.metadata.get("metrics", {}).get(metric_name, 0.0))
        for result in results
    ) / len(results)


def _mean_behavior_bool(
    results: Sequence[AnswerEvaluationResult],
    *,
    expected_behavior: str,
    attribute_name: str,
) -> float:
    """
    Average behavior-specific correctness for matching expected cases only.
    """

    matching_results = [
        result
        for result in results
        if _normalize_behavior(result.expected_behavior) == expected_behavior
    ]
    if not matching_results:
        return 0.0

    return sum(
        1.0 for result in matching_results if bool(getattr(result, attribute_name))
    ) / len(matching_results)
