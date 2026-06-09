"""Regression tests for retrieval data contracts."""

from __future__ import annotations

import unittest

from retrieval.models import (
    DiagnosticSignal,
    GroundingVerificationResult,
    RetrievalContext,
    RetrievalRouteMetadata,
    RetrievedChunkResult,
    UserQuestionInput,
)


class RetrievalModelsTests(unittest.TestCase):
    """Protect refined retrieval contracts used by later pipeline tasks."""

    def test_user_question_input_separates_original_query_and_formatting(self) -> None:
        """Ensure the question model keeps raw wording and retrieval query distinct."""

        question = UserQuestionInput(
            question_text="Respond in PT-PT and cite the article about deadlines.",
            normalized_query_text="article about deadlines",
            formatting_instructions=[" Respond in PT-PT ", "", "cite the article"],
            query_metadata={"language": "pt-pt"},
        )

        self.assertEqual(
            question.question_text,
            "Respond in PT-PT and cite the article about deadlines.",
        )
        self.assertEqual(question.normalized_query_text, "article about deadlines")
        self.assertEqual(
            question.formatting_instructions,
            ["Respond in PT-PT", "cite the article"],
        )
        self.assertEqual(question.query_metadata["language"], "pt-pt")

    def test_user_question_input_defaults_normalized_query_to_original_question(self) -> None:
        """Ensure legacy callers keep working without explicit normalization output."""

        question = UserQuestionInput(question_text="What deadline applies?")

        self.assertEqual(question.question_text, "What deadline applies?")
        self.assertEqual(question.normalized_query_text, "What deadline applies?")
        self.assertEqual(question.formatting_instructions, [])

    def test_retrieval_context_exposes_structural_metadata_and_quality_signals(self) -> None:
        """Ensure selected context exposes richer chunk metadata without extra service logic."""

        chunk = RetrievedChunkResult(
            chunk_id="chunk_5",
            record_id="record_5",
            doc_id="doc_reg_a",
            text="Article 5 states the filing deadline is 10 working days.",
            source_file="data/chunks/reg_a.json",
            chunk_metadata={
                "article_number": "5",
                "article_title": "Deadlines",
                "section_title": "Article 5",
                "parent_structure": ["Chapter II", "Applications"],
                "page_start": 3,
            },
            document_metadata={"document_title": "Regulation A"},
        )

        context = RetrievalContext(
            chunks=[chunk],
            context_text="Source 1: Article 5 states the filing deadline is 10 working days.",
            metadata={
                "total_input_chunks": 4,
                "candidate_chunk_count": 3,
                "duplicate_count": 1,
                "omitted_by_rank_limit_count": 1,
                "selected_chunk_ids": ["chunk_5"],
            },
        )

        self.assertEqual(context.selected_context_metadata[0].article_number, "5")
        self.assertEqual(context.selected_context_metadata[0].article_title, "Deadlines")
        self.assertEqual(
            context.selected_context_metadata[0].parent_structure,
            ["Chapter II", "Applications"],
        )
        self.assertEqual(context.selected_context_metadata[0].document_title, "Regulation A")
        self.assertEqual(context.retrieval_quality.total_input_chunks, 4)
        self.assertEqual(context.retrieval_quality.candidate_chunk_count, 3)
        self.assertEqual(context.retrieval_quality.selected_chunk_count, 1)
        self.assertEqual(context.retrieval_quality.structural_metadata_chunk_count, 1)
        self.assertEqual(context.retrieval_quality.selected_chunk_ids, ["chunk_5"])

    def test_grounding_verification_result_normalizes_diagnostic_taxonomy(self) -> None:
        """Ensure grounding contracts carry explicit diagnostic vocabulary."""

        result = GroundingVerificationResult(
            status="unsupported_claim",
            accepted=False,
            diagnostic_stage=" grounding_validation ",
            diagnostic_category=" grounding_failure ",
            supported_derived_claims=[" 7 prestacoes de 10% ", ""],
            diagnostic_signals=[
                {
                    "stage": " grounding_validation ",
                    "category": " grounding_failure ",
                    "code": " unsupported_legal_claim ",
                    "detail": " inflated payment schedule ",
                    "chunk_ids": [" chunk_5 ", ""],
                }
            ],
        )

        self.assertEqual(result.diagnostic_stage, "grounding_validation")
        self.assertEqual(result.diagnostic_category, "grounding_failure")
        self.assertEqual(result.supported_derived_claims, ["7 prestacoes de 10%"])
        self.assertEqual(len(result.diagnostic_signals), 1)
        self.assertEqual(
            result.diagnostic_signals[0].code,
            "unsupported_legal_claim",
        )
        self.assertEqual(result.diagnostic_signals[0].chunk_ids, ["chunk_5"])

    def test_route_metadata_keeps_shared_diagnostic_signal_contract(self) -> None:
        """Ensure route metadata can transport cross-stage diagnostic signals."""

        route_metadata = RetrievalRouteMetadata(
            diagnostic_stage="context_builder",
            diagnostic_category="retrieval_failure",
            diagnostic_signals=[
                DiagnosticSignal(
                    stage="context_builder",
                    category="retrieval_failure",
                    code="wrong_primary_anchor_selected",
                    chunk_ids=["chunk_14"],
                ),
                {
                    "stage": "grounding_validation",
                    "category": "grounding_warning",
                    "code": "supported_derived_claim",
                },
            ],
        )

        self.assertEqual(route_metadata.diagnostic_stage, "context_builder")
        self.assertEqual(route_metadata.diagnostic_category, "retrieval_failure")
        self.assertEqual(
            [signal.code for signal in route_metadata.diagnostic_signals],
            ["wrong_primary_anchor_selected", "supported_derived_claim"],
        )


if __name__ == "__main__":
    unittest.main()
