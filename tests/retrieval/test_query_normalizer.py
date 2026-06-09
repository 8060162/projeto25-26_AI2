"""Regression tests for deterministic semantic query normalization."""

from __future__ import annotations

import unittest

from Chunking.config.settings import PipelineSettings
from retrieval.query_normalizer import SemanticQueryNormalizer


class SemanticQueryNormalizerTests(unittest.TestCase):
    """Protect semantic query cleanup without leaking Task 5 integration."""

    def _build_settings(self, *, enabled: bool = True) -> PipelineSettings:
        """
        Build shared settings focused on query-normalization behavior.

        Parameters
        ----------
        enabled : bool
            Whether query normalization should be enabled.

        Returns
        -------
        PipelineSettings
            Deterministic settings object for the unit tests.
        """

        settings = PipelineSettings()
        settings.retrieval_query_normalization_enabled = enabled
        settings.retrieval_query_normalization_strip_formatting_instructions = True
        settings.retrieval_query_normalization_extract_formatting_directives = True
        return settings

    def _normalize_query_metadata(self, question_text: str) -> dict[str, object]:
        """
        Normalize one question and return its query metadata.

        Parameters
        ----------
        question_text : str
            Raw legal question to normalize.

        Returns
        -------
        dict[str, object]
            Query metadata emitted by the semantic normalizer.
        """

        normalizer = SemanticQueryNormalizer(settings=self._build_settings())
        return normalizer.normalize(question_text).query_metadata

    def _assert_intent_subset(
        self,
        metadata: dict[str, object],
        expected_intents: set[str],
    ) -> None:
        """
        Assert that normalized metadata includes the expected legal intents.

        Parameters
        ----------
        metadata : dict[str, object]
            Query metadata emitted by the semantic normalizer.

        expected_intents : set[str]
            Legal-intent signals that must be present.
        """

        legal_intents = set(metadata.get("legal_intent_signals", []))

        self.assertTrue(
            expected_intents <= legal_intents,
            f"Missing legal intents: {expected_intents - legal_intents}",
        )

    def test_normalize_separates_formatting_directives_from_legal_semantic_query(
        self,
    ) -> None:
        """
        Ensure legal semantics survive while formatting instructions are extracted.
        """

        normalizer = SemanticQueryNormalizer(settings=self._build_settings())

        normalized_question = normalizer.normalize(
            "Responde em PT-PT, indica o regulamento e o artigo aplicavel, "
            "com base nos regulamentos recuperados: qual e o prazo de matricula?"
        )

        self.assertEqual(
            normalized_question.normalized_query_text,
            "qual e o prazo de matricula?",
        )
        self.assertEqual(
            normalized_question.formatting_instructions,
            [
                "Responde em PT-PT",
                "indica o regulamento e o artigo aplicavel",
                "com base nos regulamentos recuperados",
            ],
        )
        self.assertEqual(
            normalized_question.query_metadata["requested_output_language"],
            "pt-pt",
        )
        self.assertTrue(
            normalized_question.query_metadata["citation_directive_detected"]
        )
        self.assertTrue(
            normalized_question.query_metadata["grounding_directive_detected"]
        )
        self.assertEqual(
            normalized_question.query_metadata["formatting_directive_count"],
            3,
        )

    def test_normalize_preserves_legal_question_when_no_formatting_directives_exist(
        self,
    ) -> None:
        """
        Ensure normalizer keeps semantically clean legal questions unchanged.
        """

        normalizer = SemanticQueryNormalizer(settings=self._build_settings())

        normalized_question = normalizer.normalize(
            "Qual e o prazo para anular a matricula em regime pos-laboral?"
        )

        self.assertEqual(
            normalized_question.normalized_query_text,
            "Qual e o prazo para anular a matricula em regime pos-laboral?",
        )
        self.assertEqual(normalized_question.formatting_instructions, [])
        self.assertEqual(
            normalized_question.query_metadata["formatting_directive_count"],
            0,
        )

    def test_normalize_falls_back_to_original_question_when_only_directives_exist(
        self,
    ) -> None:
        """
        Ensure directive-only prompts do not collapse into an empty embedding query.
        """

        normalizer = SemanticQueryNormalizer(settings=self._build_settings())

        normalized_question = normalizer.normalize(
            "Respond in English and cite the regulation and article."
        )

        self.assertEqual(
            normalized_question.normalized_query_text,
            "Respond in English and cite the regulation and article.",
        )
        self.assertEqual(
            normalized_question.formatting_instructions,
            [
                "Respond in English and cite the regulation and article",
            ],
        )
        self.assertEqual(
            normalized_question.query_metadata["requested_output_language"],
            "en",
        )

    def test_normalize_returns_original_question_when_feature_is_disabled(self) -> None:
        """
        Ensure Task 4 remains settings-driven and can be turned off cleanly.
        """

        normalizer = SemanticQueryNormalizer(settings=self._build_settings(enabled=False))

        normalized_question = normalizer.normalize(
            "Responde em PT-PT e indica o artigo aplicavel ao prazo."
        )

        self.assertEqual(
            normalized_question.normalized_query_text,
            "Responde em PT-PT e indica o artigo aplicavel ao prazo.",
        )
        self.assertEqual(normalized_question.formatting_instructions, [])
        self.assertEqual(normalized_question.query_metadata, {})

    def test_normalize_handles_broader_ptpt_directive_clauses(self) -> None:
        """
        Ensure clause-level PT-PT directives are removed beyond narrow templates.
        """

        normalizer = SemanticQueryNormalizer(settings=self._build_settings())

        normalized_question = normalizer.normalize(
            "Quero a resposta em portugues europeu, com referencia ao artigo e "
            "ao regulamento aplicavel, usando apenas o contexto fornecido: em "
            "que situacoes posso pedir reinscricao?"
        )

        self.assertEqual(
            normalized_question.normalized_query_text,
            "em que situacoes posso pedir reinscricao?",
        )
        self.assertEqual(
            normalized_question.formatting_instructions,
            [
                "Quero a resposta em portugues europeu",
                "com referencia ao artigo e ao regulamento aplicavel",
                "usando apenas o contexto fornecido",
            ],
        )
        self.assertEqual(
            normalized_question.query_metadata["requested_output_language"],
            "pt-pt",
        )
        self.assertTrue(
            normalized_question.query_metadata["citation_directive_detected"]
        )
        self.assertTrue(
            normalized_question.query_metadata["grounding_directive_detected"]
        )
        self.assertEqual(
            normalized_question.query_metadata["formatting_directive_count"],
            3,
        )

    def test_normalize_handles_natural_ptpt_question_with_language_and_list_request(
        self,
    ) -> None:
        """
        Ensure natural PT-PT phrasing still separates output-language directives.
        """

        normalizer = SemanticQueryNormalizer(settings=self._build_settings())

        normalized_question = normalizer.normalize(
            "Quero resposta em portugues europeu, em formato de lista, "
            "qual e o prazo para reinscricao?"
        )

        self.assertEqual(
            normalized_question.normalized_query_text,
            "qual e o prazo para reinscricao?",
        )
        self.assertEqual(
            normalized_question.formatting_instructions,
            [
                "Quero resposta em portugues europeu",
                "em formato de lista",
            ],
        )
        self.assertEqual(
            normalized_question.query_metadata["requested_output_language"],
            "pt-pt",
        )
        self.assertTrue(
            normalized_question.query_metadata["language_directive_detected"]
        )
        self.assertTrue(
            normalized_question.query_metadata["formatting_directive_detected"]
        )
        self.assertEqual(
            normalized_question.query_metadata["formatting_directive_count"],
            2,
        )

    def test_normalize_handles_natural_ptpt_question_with_grounding_and_citation(
        self,
    ) -> None:
        """
        Ensure natural Portuguese legal prompts keep only the semantic core.
        """

        normalizer = SemanticQueryNormalizer(settings=self._build_settings())

        normalized_question = normalizer.normalize(
            "Com base no contexto fornecido, indica o regulamento e o artigo: "
            "como funciona o pagamento em prestacoes?"
        )

        self.assertEqual(
            normalized_question.normalized_query_text,
            "como funciona o pagamento em prestacoes?",
        )
        self.assertEqual(
            normalized_question.formatting_instructions,
            [
                "Com base no contexto fornecido",
                "indica o regulamento e o artigo",
            ],
        )
        self.assertTrue(
            normalized_question.query_metadata["citation_directive_detected"]
        )
        self.assertTrue(
            normalized_question.query_metadata["grounding_directive_detected"]
        )
        self.assertEqual(
            normalized_question.query_metadata["formatting_directive_count"],
            2,
        )

    def test_normalize_extracts_structural_query_metadata_from_semantic_text(
        self,
    ) -> None:
        """
        Ensure legal article and document-title cues survive for context selection.
        """

        normalizer = SemanticQueryNormalizer(settings=self._build_settings())

        normalized_question = normalizer.normalize(
            "Resposta em PT-PT; em bullet points; com citacao do despacho aplicavel; "
            "qual e o prazo previsto no artigo 12 do regulamento de propinas?"
        )

        self.assertEqual(
            normalized_question.normalized_query_text,
            "qual e o prazo previsto no artigo 12 do regulamento de propinas?",
        )
        self.assertEqual(
            normalized_question.formatting_instructions,
            [
                "Resposta em PT-PT",
                "em bullet points",
                "com citacao do despacho aplicavel",
            ],
        )
        self.assertEqual(
            normalized_question.query_metadata["article_numbers"],
            ["12"],
        )
        self.assertEqual(
            normalized_question.query_metadata["article_number"],
            "12",
        )
        self.assertEqual(
            normalized_question.query_metadata["document_titles"],
            ["regulamento de propinas"],
        )
        self.assertEqual(
            normalized_question.query_metadata["document_title"],
            "regulamento de propinas",
        )

    def test_normalize_keeps_payment_plan_intent_when_document_is_omitted(
        self,
    ) -> None:
        """
        Ensure equivalent payment-plan questions keep compatible legal intent.
        """

        explicit_metadata = self._normalize_query_metadata(
            "No Regulamento de Propinas, como funciona o plano geral de "
            "pagamento para estudante internacional?"
        )
        implicit_metadata = self._normalize_query_metadata(
            "Qual e o plano de pagamento de propinas para estudantes "
            "internacionais?"
        )

        expected_intents = {"payment_plan", "international_student"}
        self._assert_intent_subset(explicit_metadata, expected_intents)
        self._assert_intent_subset(implicit_metadata, expected_intents)
        self.assertTrue(explicit_metadata["single_intent_question"])
        self.assertTrue(implicit_metadata["single_intent_question"])
        self.assertEqual(
            explicit_metadata["document_title"],
            "Regulamento de Propinas",
        )
        self.assertNotIn("document_title", implicit_metadata)

    def test_normalize_keeps_installment_variant_compatible_with_payment_plan(
        self,
    ) -> None:
        """
        Ensure installment wording still maps to payment-plan legal intent.
        """

        plan_metadata = self._normalize_query_metadata(
            "Qual e o plano de pagamento de propinas para estudantes "
            "internacionais?"
        )
        installment_metadata = self._normalize_query_metadata(
            "Como funciona o pagamento em prestacoes para estudantes "
            "internacionais?"
        )

        expected_common_intents = {"payment_plan", "international_student"}
        self._assert_intent_subset(plan_metadata, expected_common_intents)
        self._assert_intent_subset(installment_metadata, expected_common_intents)
        self.assertTrue(installment_metadata["installment_schedule"])
        self.assertTrue(plan_metadata["single_intent_question"])
        self.assertTrue(installment_metadata["single_intent_question"])

    def test_normalize_keeps_document_requirement_intent_across_ptpt_variants(
        self,
    ) -> None:
        """
        Ensure document-requirement formulations emit stable legal metadata.
        """

        necessary_metadata = self._normalize_query_metadata(
            "Que documentos sao necessarios para pedir reinscricao?"
        )
        required_metadata = self._normalize_query_metadata(
            "Quais sao os documentos exigidos para a reinscricao?"
        )

        expected_intents = {"document_requirement_question"}
        self._assert_intent_subset(necessary_metadata, expected_intents)
        self._assert_intent_subset(required_metadata, expected_intents)
        self.assertEqual(
            necessary_metadata["legal_intent_signals"],
            required_metadata["legal_intent_signals"],
        )
        self.assertTrue(necessary_metadata["single_intent_question"])
        self.assertTrue(required_metadata["single_intent_question"])

    def test_normalize_keeps_general_payment_plan_distinct_from_regularization_plan(
        self,
    ) -> None:
        """
        Ensure adjacent payment concepts keep their own legal-intent signals.
        """

        general_plan_metadata = self._normalize_query_metadata(
            "Como funciona o plano geral de pagamento de propinas?"
        )
        regularization_plan_metadata = self._normalize_query_metadata(
            "Como funciona o plano de regularizacao das propinas em divida?"
        )

        self._assert_intent_subset(
            general_plan_metadata,
            {"payment_plan", "general_payment_plan"},
        )
        self._assert_intent_subset(
            regularization_plan_metadata,
            {"payment_plan", "regularization_plan"},
        )
        self.assertNotIn(
            "regularization_plan",
            set(general_plan_metadata.get("legal_intent_signals", [])),
        )
        self.assertNotIn(
            "general_payment_plan",
            set(regularization_plan_metadata.get("legal_intent_signals", [])),
        )
        self.assertTrue(general_plan_metadata["single_intent_question"])
        self.assertTrue(regularization_plan_metadata["single_intent_question"])

    def test_normalize_extracts_operational_legal_signals_for_close_ptpt_questions(
        self,
    ) -> None:
        """
        Ensure close operational legal topics emit the intended stable signals.
        """

        cancellation_metadata = self._normalize_query_metadata(
            "Posso pedir a anulacao da matricula?"
        )
        attendance_metadata = self._normalize_query_metadata(
            "A assiduidade obrigatoria aplica-se?"
        )
        rejection_metadata = self._normalize_query_metadata(
            "Quando pode haver indeferimento liminar?"
        )
        exclusion_metadata = self._normalize_query_metadata(
            "Em que situacoes e aplicada a exclusao?"
        )

        self._assert_intent_subset(
            cancellation_metadata,
            {"matriculation_cancellation"},
        )
        self._assert_intent_subset(
            attendance_metadata,
            {"mandatory_attendance"},
        )
        self._assert_intent_subset(
            rejection_metadata,
            {"liminary_rejection"},
        )
        self._assert_intent_subset(
            exclusion_metadata,
            {"exclusion"},
        )
        self.assertTrue(cancellation_metadata["single_intent_question"])
        self.assertTrue(attendance_metadata["single_intent_question"])
        self.assertTrue(rejection_metadata["single_intent_question"])
        self.assertTrue(exclusion_metadata["single_intent_question"])

    def test_normalize_keeps_minimum_legal_intent_cues_for_short_vague_questions(
        self,
    ) -> None:
        """
        Ensure short broad formulations still expose minimum legal intent when possible.
        """

        deadline_metadata = self._normalize_query_metadata("Qual e o prazo?")
        exclusion_metadata = self._normalize_query_metadata("E a exclusao?")
        non_compliance_metadata = self._normalize_query_metadata(
            "Quais sao as consequencias do incumprimento?"
        )

        self._assert_intent_subset(deadline_metadata, {"deadline_question"})
        self._assert_intent_subset(exclusion_metadata, {"exclusion"})
        self._assert_intent_subset(
            non_compliance_metadata,
            {"non_compliance_consequence"},
        )
        self.assertTrue(deadline_metadata["single_intent_question"])
        self.assertTrue(exclusion_metadata["single_intent_question"])
        self.assertTrue(non_compliance_metadata["single_intent_question"])


if __name__ == "__main__":
    unittest.main()
