"""Regression tests for answer and grounding benchmark evaluation logic."""

from __future__ import annotations

import unittest

from retrieval.evaluation.answer_evaluator import (
    AnswerBenchmarkEvaluator,
    evaluate_answer_case,
    summarize_answer_results,
)
from retrieval.evaluation.models import (
    BenchmarkGroundingLabels,
    BenchmarkQuestionCase,
    BenchmarkRouteExpectation,
)
from retrieval.models import GroundingVerificationResult


class AnswerBenchmarkEvaluatorTests(unittest.TestCase):
    """Protect deterministic answer benchmark scoring."""

    def test_evaluator_passes_correct_grounded_answer(self) -> None:
        """Ensure correct citations, facts, behavior, and route pass."""

        benchmark_case = _build_case()
        answer_text = (
            "According to article 5 of doc_expected, the filing deadline is "
            "10 working days and the request must be submitted in DOMUS."
        )

        result = evaluate_answer_case(
            benchmark_case,
            answer_text,
            observed_behavior="answer",
            observed_route="document_scoped",
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.document_citation_correct)
        self.assertTrue(result.article_citation_correct)
        self.assertEqual(result.missing_required_facts, [])
        self.assertEqual(result.forbidden_fact_violations, [])
        self.assertEqual(result.score, 1.0)
        self.assertIn("answer.labels_satisfied", result.reasons)

    def test_evaluator_fails_article_swapped_answer(self) -> None:
        """Ensure wrong article citations are detected deterministically."""

        benchmark_case = _build_case()
        answer_text = (
            "According to article 6 of doc_expected, the filing deadline is "
            "10 working days and the request must be submitted in DOMUS."
        )

        result = AnswerBenchmarkEvaluator().evaluate_case(
            benchmark_case,
            answer_text,
            observed_behavior="answer",
            observed_route="document_scoped",
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.document_citation_correct)
        self.assertFalse(result.article_citation_correct)
        self.assertIn("answer.article_citation_mismatch", result.reasons)

    def test_evaluator_detects_missing_required_fact(self) -> None:
        """Ensure required facts absent from the answer are reported."""

        benchmark_case = _build_case()
        answer_text = (
            "According to article 5 of doc_expected, the filing deadline is "
            "10 working days."
        )

        result = AnswerBenchmarkEvaluator().evaluate_case(
            benchmark_case,
            answer_text,
            observed_behavior="answer",
            observed_route="document_scoped",
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            result.missing_required_facts,
            ["the request must be submitted in DOMUS"],
        )
        self.assertIn("answer.required_facts_missing", result.reasons)

    def test_evaluator_detects_forbidden_fact_contamination(self) -> None:
        """Ensure forbidden facts present in the answer fail the case."""

        benchmark_case = _build_case()
        answer_text = (
            "According to article 5 of doc_expected, the filing deadline is "
            "10 working days and the request must be submitted in DOMUS. "
            "The answer must not use article 6."
        )

        result = AnswerBenchmarkEvaluator().evaluate_case(
            benchmark_case,
            answer_text,
            observed_behavior="answer",
            observed_route="document_scoped",
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            result.forbidden_fact_violations,
            ["the answer must not use article 6"],
        )
        self.assertIn("answer.forbidden_facts_present", result.reasons)

    def test_evaluator_scores_expected_deflection_behavior(self) -> None:
        """Ensure expected deflections are evaluated as answer behavior."""

        benchmark_case = _build_case(
            expected_answer_behavior="deflect",
            expected_doc_id=None,
            expected_article_numbers=[],
            expected_citation_doc_ids=[],
            expected_citation_article_numbers=[],
            required_facts=["there is not enough evidence"],
        )
        answer_text = "There is not enough evidence in the available context to answer."

        result = AnswerBenchmarkEvaluator().evaluate_case(
            benchmark_case,
            answer_text,
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.deflection_correct)
        self.assertEqual(result.observed_behavior, "deflect")

    def test_evaluator_uses_grounding_verification_citations(self) -> None:
        """Ensure runtime grounding citations can satisfy citation checks."""

        benchmark_case = _build_case()
        grounding = GroundingVerificationResult(
            cited_documents=["doc_expected"],
            cited_article_numbers=["5"],
        )
        answer_text = (
            "The filing deadline is 10 working days and the request must be "
            "submitted in DOMUS."
        )

        result = AnswerBenchmarkEvaluator().evaluate_case(
            benchmark_case,
            answer_text,
            observed_behavior="answer",
            grounding_verification=grounding,
            observed_route="document_scoped",
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.document_citation_correct)
        self.assertTrue(result.article_citation_correct)

    def test_evaluator_aggregates_answer_metrics(self) -> None:
        """Ensure answer result summaries expose stable benchmark metrics."""

        passed_result = AnswerBenchmarkEvaluator().evaluate_case(
            _build_case(case_id="case_one"),
            (
                "According to article 5 of doc_expected, the filing deadline is "
                "10 working days and the request must be submitted in DOMUS."
            ),
            observed_behavior="answer",
            observed_route="document_scoped",
        )
        failed_result = AnswerBenchmarkEvaluator().evaluate_case(
            _build_case(case_id="case_two"),
            (
                "According to article 6 of doc_expected, the filing deadline is "
                "10 working days."
            ),
            observed_behavior="answer",
            observed_route="document_scoped",
        )

        metrics = summarize_answer_results([passed_result, failed_result])

        self.assertEqual(metrics["case_count"], 2.0)
        self.assertEqual(metrics["pass_rate"], 0.5)
        self.assertEqual(metrics["document_citation_accuracy"], 1.0)
        self.assertEqual(metrics["article_citation_accuracy"], 0.5)

    def test_evaluator_returns_benchmark_run_summary_for_multiple_cases(self) -> None:
        """Ensure multiple answers are evaluated into a benchmark summary."""

        benchmark_case = _build_case(case_id="case_one")
        summary = AnswerBenchmarkEvaluator().evaluate_cases(
            [benchmark_case],
            {
                "case_one": (
                    "According to article 5 of doc_expected, the filing deadline is "
                    "10 working days and the request must be submitted in DOMUS."
                )
            },
            observed_behavior_by_case_id={"case_one": "answer"},
            observed_route_by_case_id={"case_one": "document_scoped"},
        )

        self.assertEqual(summary.mode, "answer")
        self.assertEqual(summary.question_case_count, 1)
        self.assertEqual(len(summary.answer_results), 1)
        self.assertEqual(summary.metrics["pass_rate"], 1.0)


def _build_case(
    case_id: str = "case_one",
    expected_answer_behavior: str = "answer",
    expected_doc_id: str | None = "doc_expected",
    expected_article_numbers: list[str] | None = None,
    expected_citation_doc_ids: list[str] | None = None,
    expected_citation_article_numbers: list[str] | None = None,
    required_facts: list[str] | None = None,
) -> BenchmarkQuestionCase:
    """Build one answer benchmark case with expected grounding labels."""

    article_numbers = expected_article_numbers
    if article_numbers is None:
        article_numbers = ["5"]

    citation_doc_ids = expected_citation_doc_ids
    if citation_doc_ids is None:
        citation_doc_ids = ["doc_expected"] if expected_doc_id else []

    citation_article_numbers = expected_citation_article_numbers
    if citation_article_numbers is None:
        citation_article_numbers = list(article_numbers)

    facts = required_facts
    if facts is None:
        facts = [
            "the filing deadline is 10 working days",
            "the request must be submitted in DOMUS",
        ]

    return BenchmarkQuestionCase(
        case_id=case_id,
        question="What is the filing deadline?",
        case_type="legal_anchor",
        expected_route=BenchmarkRouteExpectation(route_name="document_scoped"),
        expected_doc_id=expected_doc_id,
        expected_article_numbers=article_numbers,
        expected_chunk_ids=["expected_chunk"],
        required_facts=facts,
        forbidden_facts=["the answer must not use article 6"],
        expected_answer_behavior=expected_answer_behavior,
        grounding_labels=BenchmarkGroundingLabels(
            expected_citation_doc_ids=citation_doc_ids,
            expected_citation_article_numbers=citation_article_numbers,
        ),
    )


if __name__ == "__main__":
    unittest.main()
