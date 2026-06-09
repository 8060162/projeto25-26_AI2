"""Regression tests for retrieval benchmark evaluation logic."""

from __future__ import annotations

import unittest

from retrieval.evaluation.models import BenchmarkQuestionCase
from retrieval.evaluation.retrieval_evaluator import (
    RetrievalBenchmarkEvaluator,
    evaluate_retrieval_case,
    summarize_retrieval_results,
)
from retrieval.models import RetrievalContext, RetrievedChunkResult


class RetrievalBenchmarkEvaluatorTests(unittest.TestCase):
    """Protect deterministic retrieval benchmark scoring."""

    def test_evaluator_scores_recovered_expected_labels_and_selected_context(
        self,
    ) -> None:
        """Ensure expected document, article, chunk, recall, and MRR are scored."""

        benchmark_case = _build_case()
        ranking = [
            _build_chunk("competing_chunk", "doc_expected", "4"),
            _build_chunk("expected_chunk", "doc_expected", "5"),
            _build_chunk("other_chunk", "other_doc", "9"),
        ]
        selected_context = RetrievalContext(chunks=[ranking[1]])

        result = evaluate_retrieval_case(
            benchmark_case,
            ranking,
            selected_context=selected_context,
            top_k=2,
        )

        self.assertTrue(result.expected_doc_recovered)
        self.assertTrue(result.expected_article_recovered)
        self.assertTrue(result.expected_chunk_recovered)
        self.assertTrue(result.selected_context_hit)
        self.assertAlmostEqual(result.reciprocal_rank or 0.0, 0.5)
        self.assertEqual(result.metrics["recall_at_k"], 1.0)
        self.assertEqual(result.retrieved_chunk_ids, ["competing_chunk", "expected_chunk"])

    def test_evaluator_detects_conflicting_candidate_ahead_of_expected_chunk(
        self,
    ) -> None:
        """Ensure legal competitors that outrank expected chunks are surfaced."""

        benchmark_case = _build_case()
        ranking = [
            _build_chunk("wrong_article_chunk", "doc_expected", "6"),
            _build_chunk("expected_chunk", "doc_expected", "5"),
        ]

        result = RetrievalBenchmarkEvaluator().evaluate_case(
            benchmark_case,
            ranking,
            top_k=2,
        )

        self.assertTrue(result.conflict_present)
        self.assertIn("conflicting_candidate_present", result.reasons)

    def test_evaluator_handles_missing_expected_chunk_inside_cutoff(self) -> None:
        """Ensure unrecovered labels fail without fabricating positive metrics."""

        benchmark_case = _build_case()
        ranking = [
            _build_chunk("wrong_doc_chunk", "wrong_doc", "2"),
            _build_chunk("expected_chunk", "doc_expected", "5"),
        ]

        result = RetrievalBenchmarkEvaluator().evaluate_case(
            benchmark_case,
            ranking,
            top_k=1,
        )

        self.assertFalse(result.expected_doc_recovered)
        self.assertFalse(result.expected_article_recovered)
        self.assertFalse(result.expected_chunk_recovered)
        self.assertFalse(result.selected_context_hit)
        self.assertTrue(result.conflict_present)
        self.assertEqual(result.metrics["recall_at_k"], 0.0)
        self.assertEqual(result.reciprocal_rank, 0.5)
        self.assertIn("expected_chunk_not_recovered", result.reasons)

    def test_evaluator_aggregates_labeled_case_metrics(self) -> None:
        """Ensure aggregate retrieval rates use stable labeled denominators."""

        benchmark_case = _build_case()
        evaluator = RetrievalBenchmarkEvaluator()
        successful_result = evaluator.evaluate_case(
            benchmark_case,
            [_build_chunk("expected_chunk", "doc_expected", "5")],
            selected_context=RetrievalContext(
                chunks=[_build_chunk("expected_chunk", "doc_expected", "5")]
            ),
            top_k=1,
        )
        failed_result = evaluator.evaluate_case(
            benchmark_case,
            [_build_chunk("wrong_doc_chunk", "wrong_doc", "2")],
            top_k=1,
        )

        metrics = summarize_retrieval_results([successful_result, failed_result])

        self.assertEqual(metrics["case_count"], 2.0)
        self.assertEqual(metrics["recall_at_k"], 0.5)
        self.assertEqual(metrics["expected_chunk_recovery_rate"], 0.5)
        self.assertEqual(metrics["selected_context_hit_rate"], 0.5)
        self.assertEqual(metrics["conflict_presence_rate"], 0.5)

    def test_evaluator_returns_benchmark_run_summary_for_multiple_cases(self) -> None:
        """Ensure multiple rankings are evaluated into a benchmark summary."""

        benchmark_case = _build_case(case_id="case_one")
        summary = RetrievalBenchmarkEvaluator().evaluate_cases(
            [benchmark_case],
            {
                "case_one": [
                    _build_chunk("expected_chunk", "doc_expected", "5"),
                ]
            },
            selected_context_by_case_id={
                "case_one": RetrievalContext(
                    chunks=[_build_chunk("expected_chunk", "doc_expected", "5")]
                )
            },
            top_k=1,
        )

        self.assertEqual(summary.mode, "retrieval")
        self.assertEqual(summary.question_case_count, 1)
        self.assertEqual(len(summary.retrieval_results), 1)
        self.assertEqual(summary.metrics["recall_at_k"], 1.0)


def _build_case(case_id: str = "case_one") -> BenchmarkQuestionCase:
    """Build one retrieval benchmark case with expected legal labels."""

    return BenchmarkQuestionCase(
        case_id=case_id,
        question="Qual e o artigo correto?",
        case_type="legal_anchor",
        expected_doc_id="doc_expected",
        expected_article_numbers=["5"],
        expected_chunk_ids=["expected_chunk"],
        expected_answer_behavior="answer",
    )


def _build_chunk(
    chunk_id: str,
    doc_id: str,
    article_number: str,
) -> RetrievedChunkResult:
    """Build one retrieved chunk with article metadata for evaluator tests."""

    return RetrievedChunkResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"Conteudo do artigo {article_number}.",
        similarity_score=0.9,
        chunk_metadata={"article_number": article_number},
        document_metadata={"document_title": doc_id},
    )


if __name__ == "__main__":
    unittest.main()
