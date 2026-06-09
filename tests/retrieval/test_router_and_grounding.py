"""Focused tests for retrieval routing and grounding validation."""

from __future__ import annotations

import unittest

from Chunking.config.settings import PipelineSettings
from retrieval.grounding_validator import GroundingValidator
from retrieval.models import (
    EvidenceQualityClassification,
    RetrievalContext,
    RetrievedChunkResult,
    UserQuestionInput,
)
from retrieval.retrieval_router import RetrievalRouter
from retrieval.service import RetrievalService


class RouterAndGroundingTests(unittest.TestCase):
    """Protect routing and grounding decisions introduced in the retrieval flow."""

    def _build_settings(self) -> PipelineSettings:
        """
        Build deterministic routing settings for focused regression tests.

        Returns
        -------
        PipelineSettings
            Pipeline settings with explicit retrieval routing enabled.
        """

        settings = PipelineSettings()
        settings.retrieval_routing_enabled = True
        settings.retrieval_routing_article_scoping_enabled = True
        settings.retrieval_routing_document_scoping_enabled = True
        settings.retrieval_routing_comparative_retrieval_enabled = True
        settings.retrieval_routing_weak_evidence_retry_enabled = True
        settings.retrieval_candidate_pool_size = 5
        settings.retrieval_routing_scoped_candidate_pool_size = 7
        settings.retrieval_routing_broad_candidate_pool_size = 19
        return settings

    def _build_legal_context(self) -> RetrievalContext:
        """
        Build a minimal selected context with explicit legal anchors.

        Returns
        -------
        RetrievalContext
            Retrieval context suitable for grounding-validation tests.
        """

        chunk = RetrievedChunkResult(
            chunk_id="propinas_article_5",
            doc_id="reg_propinas",
            text="O artigo 5 estabelece o pagamento de propinas em 10 dias.",
            source_file="data/chunks/regulamento_propinas.json",
            chunk_metadata={
                "article_number": "5",
                "article_title": "Pagamento",
                "section_title": "Artigo 5",
            },
            document_metadata={
                "document_title": (
                    "Despacho P.PORTO-P-043-2025 Regulamento de Propinas"
                )
            },
        )

        return RetrievalContext(
            chunks=[chunk],
            context_text=(
                "[Source 1 | doc_id=reg_propinas | article_number=5 | "
                "document_title=Despacho P.PORTO-P-043-2025 Regulamento de "
                "Propinas]\n"
                "O artigo 5 estabelece o pagamento de propinas em 10 dias."
            ),
        )

    def _build_evidence_context(
        self,
        evidence_quality: EvidenceQualityClassification,
    ) -> RetrievalContext:
        """
        Build a context carrying one explicit evidence-quality classification.

        Parameters
        ----------
        evidence_quality : EvidenceQualityClassification
            Evidence classification to attach to the context.

        Returns
        -------
        RetrievalContext
            Context used to exercise service evidence-routing decisions.
        """

        return RetrievalContext(
            context_text="Selected legal context.",
            evidence_quality=evidence_quality,
        )

    def test_scoped_broad_and_article_routing_signals_are_explicit(self) -> None:
        """
        Ensure scoped, broad comparative, and article-biased routes are stable.
        """

        router = RetrievalRouter(settings=self._build_settings())

        scoped_decision = router.route(
            UserQuestionInput(
                question_text="Qual e o prazo no Regulamento de Propinas?",
                normalized_query_text="Qual e o prazo no Regulamento de Propinas?",
                query_metadata={"document_title": "Regulamento de Propinas"},
            )
        )
        comparative_decision = router.route(
            UserQuestionInput(
                question_text=(
                    "Compara o Regulamento de Propinas e o Regulamento Academico."
                ),
                normalized_query_text=(
                    "Compara o Regulamento de Propinas e o Regulamento Academico."
                ),
                query_metadata={
                    "document_titles": [
                        "Regulamento de Propinas",
                        "Regulamento Academico",
                    ],
                },
            )
        )
        article_decision = router.route(
            UserQuestionInput(
                question_text="O que diz o artigo 12 sobre prazos?",
                normalized_query_text="O que diz o artigo 12 sobre prazos?",
            )
        )

        self.assertEqual(scoped_decision.retrieval_scope, "scoped")
        self.assertEqual(scoped_decision.retrieval_profile, "document_scoped")
        self.assertEqual(scoped_decision.metadata["candidate_pool_size"], 7)
        self.assertIn("document_target_detected", scoped_decision.reasons)

        self.assertEqual(comparative_decision.retrieval_scope, "broad")
        self.assertEqual(comparative_decision.retrieval_profile, "comparative")
        self.assertTrue(comparative_decision.comparative)
        self.assertEqual(comparative_decision.metadata["candidate_pool_size"], 19)
        self.assertIn(
            "multi_document_retrieval_preserved",
            comparative_decision.reasons,
        )

        self.assertEqual(article_decision.retrieval_scope, "broad")
        self.assertEqual(article_decision.retrieval_profile, "article_biased")
        self.assertEqual(article_decision.target_article_numbers, ["12"])
        self.assertIn("article_bias_selected", article_decision.reasons)

    def test_evidence_routing_blocks_empty_weak_and_conflicting_context(self) -> None:
        """
        Ensure evidence routing blocks contexts that should not reach generation.
        """

        service = RetrievalService.__new__(RetrievalService)
        empty_context = self._build_evidence_context(
            EvidenceQualityClassification(
                strength="empty",
                sufficient_for_answer=False,
            )
        )
        weak_context = self._build_evidence_context(
            EvidenceQualityClassification(
                strength="weak",
                sufficient_for_answer=False,
                reasons=["weak_retrieval_score"],
            )
        )
        conflicting_context = self._build_evidence_context(
            EvidenceQualityClassification(
                strength="strong",
                conflict="conflicting",
                sufficient_for_answer=True,
                conflicting_chunk_ids=["chunk_standard"],
            )
        )

        self.assertFalse(service._evidence_allows_generation(empty_context))
        self.assertFalse(service._evidence_allows_generation(weak_context))
        self.assertFalse(service._evidence_allows_generation(conflicting_context))
        self.assertIn(
            "conflicting",
            service._build_evidence_deflection_message(conflicting_context),
        )

    def test_ambiguous_but_sufficient_evidence_adds_caution_instruction(self) -> None:
        """
        Ensure usable ambiguous evidence is routed with grounding caution.
        """

        service = RetrievalService.__new__(RetrievalService)
        ambiguous_context = self._build_evidence_context(
            EvidenceQualityClassification(
                strength="strong",
                ambiguity="ambiguous",
                conflict="none",
                sufficient_for_answer=True,
                close_competitor_chunk_ids=["chunk_exception"],
            )
        )

        self.assertTrue(service._evidence_allows_generation(ambiguous_context))
        self.assertIn(
            "close legal competitors",
            service._build_grounding_instruction(ambiguous_context),
        )

    def test_grounding_validation_accepts_supported_aligned_answer(self) -> None:
        """
        Ensure aligned article citations and supported claims pass validation.
        """

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 5 do Regulamento de Propinas, "
                "o prazo e de 10 dias."
            ),
            context=self._build_legal_context(),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "strong_alignment")
        self.assertEqual(result.citation_status, "aligned")
        self.assertEqual(result.article_alignment, "aligned")
        self.assertEqual(result.document_alignment, "aligned")

    def test_grounding_validation_rejects_mismatched_citation(self) -> None:
        """
        Ensure answers citing unavailable legal anchors are rejected.
        """

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 7 do Regulamento de Propinas, "
                "o prazo e de 10 dias."
            ),
            context=self._build_legal_context(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "citation_mismatch")
        self.assertEqual(result.article_alignment, "mismatch")
        self.assertEqual(result.mismatched_citations, ["article=7"])
        self.assertIn("grounding.citation_mismatch", result.reasons)

    def test_grounding_validation_rejects_unsupported_legal_claim(self) -> None:
        """
        Ensure numeric legal claims absent from context are rejected.
        """

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 5 do Regulamento de Propinas, "
                "o prazo e de 20 dias."
            ),
            context=self._build_legal_context(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "unsupported_claim")
        self.assertEqual(result.unsupported_claims, ["20 dias"])
        self.assertIn("grounding.unsupported_numeric_claim", result.reasons)


if __name__ == "__main__":
    unittest.main()
