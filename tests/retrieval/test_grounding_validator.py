"""Regression tests for deterministic grounding validation."""

from __future__ import annotations

import unittest

from retrieval.grounding_validator import GroundingValidator
from retrieval.models import RetrievalContext, RetrievedChunkResult


class GroundingValidatorTests(unittest.TestCase):
    """Protect post-generation grounding and citation verification behavior."""

    def _build_context(self) -> RetrievalContext:
        """
        Build one selected legal context with explicit article anchors.

        Returns
        -------
        RetrievalContext
            Minimal grounded context used by validator tests.
        """

        chunk = RetrievedChunkResult(
            chunk_id="chunk_5",
            doc_id="reg_a",
            text="O artigo 5 estabelece o prazo de 10 dias para o pedido.",
            source_file="data/chunks/reg_a.json",
            chunk_metadata={
                "article_number": "5",
                "article_title": "Prazos",
                "section_title": "Artigo 5",
                "page_start": 3,
            },
            document_metadata={"document_title": "Regulation A"},
        )

        return RetrievalContext(
            chunks=[chunk],
            context_text=(
                "[Source 1 | doc_id=reg_a | article_number=5]\n"
                "O artigo 5 estabelece o prazo de 10 dias para o pedido."
            ),
        )

    def _build_portuguese_document_context(self) -> RetrievalContext:
        """
        Build one selected context with a longer Portuguese document title.

        Returns
        -------
        RetrievalContext
            Minimal context used to verify partial document-title matching.
        """

        chunk = RetrievedChunkResult(
            chunk_id="chunk_propinas_5",
            doc_id="reg_propinas",
            text="O artigo 5 estabelece o pagamento de propinas em 10 dias.",
            source_file="data/chunks/regulamento_propinas.json",
            chunk_metadata={
                "article_number": "5",
                "article_title": "Pagamento",
                "section_title": "Artigo 5",
                "page_start": 4,
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

    def _build_enumerated_payment_plan_context(self) -> RetrievalContext:
        """
        Build one context with an enumerated installment payment schedule.

        Returns
        -------
        RetrievalContext
            Minimal context for derived legal-claim validation.
        """

        chunk = RetrievedChunkResult(
            chunk_id="chunk_article_5_schedule",
            doc_id="reg_payment",
            text=(
                "O artigo 5 permite o pagamento em 8 prestacoes. A primeira "
                "prestacao corresponde a 30% da propina anual. Da 2 a 8 "
                "prestacao, cada prestacao corresponde a 10% da propina anual."
            ),
            source_file="data/chunks/reg_payment.json",
            chunk_metadata={
                "article_number": "5",
                "article_title": "Plano Geral de Pagamento",
                "page_start": 4,
            },
            document_metadata={"document_title": "Regulamento de Pagamentos"},
        )

        return RetrievalContext(
            chunks=[chunk],
            context_text=(
                "[Source 1 | doc_id=reg_payment | article_number=5 | "
                "document_title=Regulamento de Pagamentos]\n"
                "O artigo 5 permite o pagamento em 8 prestacoes. A primeira "
                "prestacao corresponde a 30% da propina anual. Da 2 a 8 "
                "prestacao, cada prestacao corresponde a 10% da propina anual."
            ),
        )

    def test_validate_flags_article_citation_mismatch(self) -> None:
        """Ensure an answer citing another article is rejected."""

        result = GroundingValidator().validate(
            answer_text="De acordo com o artigo 7, o prazo e de 10 dias.",
            context=self._build_context(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "citation_mismatch")
        self.assertEqual(result.article_alignment, "mismatch")
        self.assertEqual(result.mismatched_citations, ["article=7"])
        self.assertIn("grounding.citation_mismatch", result.reasons)
        self.assertEqual(
            [signal.code for signal in result.diagnostic_signals],
            ["answer_citation_mismatch"],
        )

    def test_validate_flags_document_citation_mismatch(self) -> None:
        """Ensure an answer citing another document is rejected."""

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 5 do Regulamento de Avaliacao, "
                "o prazo e de 10 dias."
            ),
            context=self._build_portuguese_document_context(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "citation_mismatch")
        self.assertEqual(result.document_alignment, "mismatch")
        self.assertEqual(
            result.mismatched_citations,
            ["document=Regulamento de Avaliacao"],
        )
        self.assertIn("grounding.citation_mismatch", result.reasons)

    def test_validate_flags_unsupported_numeric_legal_claim(self) -> None:
        """Ensure a numeric rule absent from context is rejected."""

        result = GroundingValidator().validate(
            answer_text="Nos termos do artigo 5, o prazo e de 20 dias.",
            context=self._build_context(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "unsupported_claim")
        self.assertEqual(result.article_alignment, "aligned")
        self.assertEqual(result.unsupported_claims, ["20 dias"])
        self.assertIn("grounding.unsupported_numeric_claim", result.reasons)

    def test_validate_explains_missing_required_article_anchor(self) -> None:
        """Ensure vague article references are explained as weak grounding."""

        result = GroundingValidator().validate(
            answer_text="O artigo aplicavel estabelece o prazo de 10 dias.",
            context=self._build_context(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "missing_required_anchor")
        self.assertEqual(result.article_alignment, "missing_required_anchor")
        self.assertEqual(result.missing_required_facts, ["article_number"])
        self.assertIn("grounding.missing_required_anchor", result.reasons)

    def test_validate_accepts_aligned_article_and_supported_claim(self) -> None:
        """Ensure aligned article citations and supported numeric claims pass."""

        result = GroundingValidator().validate(
            answer_text="Nos termos do artigo 5 da Regulation A, o prazo e de 10 dias.",
            context=self._build_context(),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "strong_alignment")
        self.assertEqual(result.citation_status, "aligned")
        self.assertEqual(result.article_alignment, "aligned")
        self.assertEqual(result.document_alignment, "aligned")

    def test_validate_accepts_partial_portuguese_document_title(self) -> None:
        """Ensure concise Portuguese document references align to longer titles."""

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 5 do Regulamento de Propinas, "
                "o prazo e de 10 dias."
            ),
            context=self._build_portuguese_document_context(),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "strong_alignment")
        self.assertEqual(result.document_alignment, "aligned")
        self.assertEqual(result.cited_documents, ["Regulamento de Propinas"])

    def test_validate_accepts_numeric_article_metadata_with_service_citation(
        self,
    ) -> None:
        """Ensure numeric article metadata aligns with generated service citations."""

        chunk = RetrievedChunkResult(
            chunk_id="chunk_propinas_8",
            doc_id="reg_propinas",
            text="O regime de pagamento esta definido para a propina anual.",
            source_file="data/chunks/regulamento_propinas.json",
            chunk_metadata={
                "article_number": 8,
                "article_title": "Pagamento",
                "page_start": 7,
                "page_end": 8,
            },
            document_metadata={
                "document_title": (
                    "Despacho P.PORTO-P- 043-2025_Regulamento de Propinas"
                )
            },
        )
        context = RetrievalContext(
            chunks=[chunk],
            context_text=(
                "[Source 1 | doc_id=reg_propinas | article_number=8 | pages=7-8]\n"
                "O regime de pagamento esta definido para a propina anual."
            ),
        )

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 8 do Regulamento de Propinas, "
                "o regime de pagamento aplica-se a propina anual."
            ),
            context=context,
            citations=[
                (
                    "Despacho P.PORTO-P- 043-2025_Regulamento de Propinas "
                    "| article=8 | pages=7-8"
                )
            ],
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "strong_alignment")
        self.assertEqual(result.article_alignment, "aligned")
        self.assertEqual(result.mismatched_citations, [])

    def test_validate_accepts_derived_enumerative_payment_summary(self) -> None:
        """Ensure compressed installment summaries pass when context enumerates them."""

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 5 do Regulamento de Pagamentos, "
                "o plano inclui 7 prestacoes de 10% apos a primeira prestacao."
            ),
            context=self._build_enumerated_payment_plan_context(),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "strong_alignment")
        self.assertEqual(result.article_alignment, "aligned")
        self.assertEqual(result.unsupported_claims, [])
        self.assertEqual(result.supported_derived_claims, ["7 prestacoes"])
        self.assertIn("grounding.aligned", result.reasons)
        self.assertIn(
            "supported_derived_claim",
            [signal.code for signal in result.diagnostic_signals],
        )

    def test_validate_accepts_article_anchor_provided_by_citation_payload(
        self,
    ) -> None:
        """Ensure service citations can provide the concrete article anchor."""

        result = GroundingValidator().validate(
            answer_text=(
                "O plano geral permite o pagamento em 8 prestacoes, com as "
                "restantes prestacoes de 10% apos a primeira."
            ),
            context=self._build_enumerated_payment_plan_context(),
            citations=["Regulamento de Pagamentos | article=5 | page=4"],
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "strong_alignment")
        self.assertEqual(result.article_alignment, "aligned")
        self.assertEqual(result.cited_article_numbers, ["5"])
        self.assertEqual(result.mismatched_citations, [])

    def test_validate_rejects_unsupported_derived_installment_count(self) -> None:
        """Ensure inflated derived installment summaries are rejected."""

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 5 do Regulamento de Pagamentos, "
                "o plano inclui 9 prestacoes de 10%."
            ),
            context=self._build_enumerated_payment_plan_context(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "unsupported_claim")
        self.assertEqual(result.article_alignment, "aligned")
        self.assertEqual(result.unsupported_claims, ["9 prestacoes"])
        self.assertIn("grounding.unsupported_numeric_claim", result.reasons)

    def test_validate_uses_metadata_anchors_when_context_text_has_only_body(
        self,
    ) -> None:
        """Ensure citations can align to metadata without metadata in chunk text."""

        chunk = RetrievedChunkResult(
            chunk_id="chunk_propinas_5_2",
            doc_id="reg_propinas",
            text=(
                "Para estudante internacional, em 8 prestacoes, com percentagens "
                "e datas-limite de pagamento."
            ),
            source_file="data/chunks/regulamento_propinas.json",
            chunk_metadata={
                "article_number": "5",
                "article_title": "PLANO GERAL DE PAGAMENTO DE PROPINAS",
                "page_start": 4,
                "page_end": 4,
            },
            document_metadata={"document_title": "Regulamento de Propinas"},
        )
        context = RetrievalContext(
            chunks=[chunk],
            context_text=(
                "Para estudante internacional, em 8 prestacoes, com percentagens "
                "e datas-limite de pagamento."
            ),
        )

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 5 do Regulamento de Propinas, "
                "o estudante internacional pode pagar em 8 prestacoes."
            ),
            context=context,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "strong_alignment")
        self.assertEqual(result.article_alignment, "aligned")
        self.assertEqual(result.document_alignment, "aligned")
        self.assertEqual(result.cited_article_numbers, ["5"])
        self.assertEqual(result.metadata["context_article_numbers"], ["5"])

    def test_validate_uses_serialized_context_header_document_metadata(
        self,
    ) -> None:
        """Ensure serialized context headers can supply document grounding anchors."""

        chunk = RetrievedChunkResult(
            chunk_id="chunk_without_scoped_metadata",
            doc_id="reg_propinas",
            text="O estudante internacional pode pagar em 8 prestacoes.",
            source_file="data/chunks/regulamento_propinas.json",
        )
        context = RetrievalContext(
            chunks=[chunk],
            context_text=(
                "[Source 1 | doc_id=reg_propinas | legal_anchor=Regulamento "
                "de Propinas > Article 5 - Plano Geral | "
                "document_title=Regulamento de Propinas | article_number=5]\n"
                "O estudante internacional pode pagar em 8 prestacoes."
            ),
        )

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 5 do Regulamento de Propinas, "
                "o estudante internacional pode pagar em 8 prestacoes."
            ),
            context=context,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "strong_alignment")
        self.assertEqual(result.article_alignment, "aligned")
        self.assertEqual(result.document_alignment, "aligned")

    def test_validate_rejects_wrong_article_when_correct_competitor_supports_claim(
        self,
    ) -> None:
        """Ensure same-document competing articles do not mask wrong citations."""

        article_5 = RetrievedChunkResult(
            chunk_id="chunk_article_5",
            doc_id="reg_propinas",
            text=(
                "Para estudante internacional, o plano geral de pagamento "
                "permite 8 prestacoes mensais."
            ),
            source_file="data/chunks/regulamento_propinas.json",
            chunk_metadata={
                "article_number": "5",
                "article_title": "Plano Geral de Pagamento de Propinas",
            },
            document_metadata={"document_title": "Regulamento de Propinas"},
        )
        article_24 = RetrievedChunkResult(
            chunk_id="chunk_article_24",
            doc_id="reg_propinas",
            text=(
                "O artigo sobre estudantes internacionais define regras "
                "especificas de regularizacao."
            ),
            source_file="data/chunks/regulamento_propinas.json",
            chunk_metadata={
                "article_number": "24",
                "article_title": "Estudantes Internacionais",
            },
            document_metadata={"document_title": "Regulamento de Propinas"},
        )
        context = RetrievalContext(
            chunks=[article_5, article_24],
            context_text=(
                "[Source 1 | doc_id=reg_propinas | article_number=5]\n"
                "Para estudante internacional, o plano geral de pagamento "
                "permite 8 prestacoes mensais.\n\n"
                "[Source 2 | doc_id=reg_propinas | article_number=24]\n"
                "O artigo sobre estudantes internacionais define regras "
                "especificas de regularizacao."
            ),
            metadata={
                "primary_anchor": "article=5",
                "primary_anchor_chunk_ids": ["chunk_article_5"],
            },
        )

        result = GroundingValidator().validate(
            answer_text=(
                "Nos termos do artigo 24 do Regulamento de Propinas, "
                "o estudante internacional pode pagar em 8 prestacoes mensais."
            ),
            context=context,
            citations=[
                "Regulamento de Propinas | article=5",
                "Regulamento de Propinas | article=24",
            ],
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "citation_mismatch")
        self.assertEqual(result.article_alignment, "mismatch")
        self.assertIn("article_claim=24:8 prestacoes", result.mismatched_citations)
        self.assertIn("primary_article=5", result.mismatched_citations)
        self.assertIn("grounding.citation_mismatch", result.reasons)
        self.assertEqual(result.metadata["primary_article_key"], "5")
        self.assertEqual(
            [signal.code for signal in result.diagnostic_signals],
            ["answer_citation_mismatch", "wrong_primary_anchor_selected"],
        )


if __name__ == "__main__":
    unittest.main()
