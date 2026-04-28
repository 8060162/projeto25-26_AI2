"""Regression tests for deterministic retrieval routing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from Chunking.config.settings import PipelineSettings
from retrieval.models import UserQuestionInput
from retrieval.retrieval_router import RetrievalRouter


class RetrievalRouterTests(unittest.TestCase):
    """Protect explicit routing between query metadata and retrieval behavior."""

    def _build_settings(self, *, enabled: bool = True) -> PipelineSettings:
        """
        Build deterministic settings for retrieval-router tests.

        Parameters
        ----------
        enabled : bool
            Whether explicit retrieval routing is enabled.

        Returns
        -------
        PipelineSettings
            Pipeline settings with routing controls enabled for tests.
        """

        settings = PipelineSettings()
        settings.retrieval_routing_enabled = enabled
        settings.retrieval_routing_article_scoping_enabled = True
        settings.retrieval_routing_document_scoping_enabled = True
        settings.retrieval_routing_comparative_retrieval_enabled = True
        settings.retrieval_routing_weak_evidence_retry_enabled = True
        settings.retrieval_routing_document_inference_enabled = True
        settings.retrieval_routing_document_inference_min_score = 3.0
        settings.retrieval_routing_document_inference_min_margin = 2.0
        settings.retrieval_candidate_pool_size = 5
        settings.retrieval_routing_scoped_candidate_pool_size = 6
        settings.retrieval_routing_broad_candidate_pool_size = 18
        settings.retrieval_second_pass_retry_candidate_pool_size = 24
        return settings

    def _write_chunk_file(
        self,
        root_path: Path,
        doc_id: str,
        document_title: str,
        article_number: str = "1",
        text: str = "Conteudo regulamentar.",
        article_title: str = "",
    ) -> None:
        """
        Write one minimal active chunk output for router metadata indexing.

        Parameters
        ----------
        root_path : Path
            Temporary embedding input root.
        doc_id : str
            Document identifier to write.
        document_title : str
            Document title stored in chunk metadata.
        article_number : str
            Structural article label stored in chunk metadata.
        text : str
            Chunk text used for legal-intent signal discovery.
        article_title : str
            Optional article title stored in chunk metadata.
        """

        chunk_path = root_path / doc_id / "article_smart" / "05_chunks.json"
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_payload = [
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_0001",
                "text": text,
                "source_node_label": article_number,
                "hierarchy_path": [article_number],
                "metadata": {
                    "document_title": document_title,
                    "source_file_name": f"{document_title}.pdf",
                    "article_number": article_number,
                    "article_title": article_title,
                },
            }
        ]
        chunk_path.write_text(json.dumps(chunk_payload), encoding="utf-8")

    def test_document_explicit_question_routes_to_scoped_retrieval(self) -> None:
        """
        Ensure explicit document metadata selects scoped retrieval behavior.
        """

        router = RetrievalRouter(settings=self._build_settings())

        decision = router.route(
            UserQuestionInput(
                question_text="Qual e o prazo no Regulamento de Propinas?",
                normalized_query_text="Qual e o prazo no Regulamento de Propinas?",
                query_metadata={
                    "document_title": "Regulamento de Propinas",
                },
            )
        )

        self.assertEqual(decision.route_name, "document_scoped")
        self.assertEqual(decision.retrieval_scope, "scoped")
        self.assertEqual(decision.retrieval_profile, "document_scoped")
        self.assertEqual(
            decision.target_document_titles,
            ["Regulamento de Propinas"],
        )
        self.assertEqual(decision.metadata["candidate_pool_size"], 6)
        self.assertIn("document_target_detected", decision.reasons)

    def test_document_explicit_question_resolves_known_doc_id(self) -> None:
        """
        Ensure local chunk metadata can turn document aliases into doc ids.
        """

        router = RetrievalRouter(settings=self._build_settings())

        decision = router.route(
            UserQuestionInput(
                question_text=(
                    "No Regulamento de Propinas, como funciona o pagamento?"
                ),
                normalized_query_text=(
                    "No Regulamento de Propinas, como funciona o pagamento?"
                ),
                query_metadata={
                    "document_title": "Regulamento de Propinas",
                },
            )
        )

        self.assertIn("Despacho_P_PORTO_P_043_2025", decision.target_doc_ids)

    def test_strong_document_signals_without_explicit_title_route_to_scoped_retrieval(
        self,
    ) -> None:
        """
        Ensure reusable document metadata can infer a scoped document target.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory)
            self._write_chunk_file(
                input_root,
                "doc_bolsas",
                "Regulamento de Bolsas de Estudo",
            )
            self._write_chunk_file(
                input_root,
                "doc_estagios",
                "Regulamento de Estagios Curriculares",
            )

            settings = self._build_settings()
            settings.embedding_input_root = input_root
            settings.chunking_strategy = "article_smart"
            router = RetrievalRouter(settings=settings)

            decision = router.route(
                UserQuestionInput(
                    question_text="Quais sao os criterios para bolsas de estudo?",
                    normalized_query_text=(
                        "Quais sao os criterios para bolsas de estudo?"
                    ),
                )
            )

        self.assertEqual(decision.route_name, "document_scoped")
        self.assertEqual(decision.retrieval_scope, "scoped")
        self.assertEqual(decision.target_doc_ids, ["doc_bolsas"])
        self.assertEqual(
            decision.target_document_titles,
            ["Regulamento de Bolsas de Estudo"],
        )
        self.assertIn("dynamic_document_target_inferred", decision.reasons)
        self.assertTrue(decision.metadata["document_inferred"])

    def test_legal_intent_without_explicit_title_keeps_broad_retry_candidate_scope(
        self,
    ) -> None:
        """
        Ensure inferred legal intent exposes a document retry target.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory)
            self._write_chunk_file(
                input_root,
                "doc_propinas",
                "Regulamento de Propinas",
                article_number="5",
                article_title="Plano geral de pagamento",
                text=(
                    "O plano geral de pagamento permite pagamento de propinas "
                    "em prestacoes para estudantes internacionais."
                ),
            )
            self._write_chunk_file(
                input_root,
                "doc_matriculas",
                "Regulamento de Matriculas e Inscricoes",
                article_number="7",
                article_title="Inscricao em unidades curriculares",
                text="As inscricoes seguem prazos e regras academicas gerais.",
            )

            settings = self._build_settings()
            settings.embedding_input_root = input_root
            settings.chunking_strategy = "article_smart"
            router = RetrievalRouter(settings=settings)

            decision = router.route(
                UserQuestionInput(
                    question_text=(
                        "Qual e o plano geral de pagamento para estudantes "
                        "internacionais?"
                    ),
                    normalized_query_text=(
                        "Qual e o plano geral de pagamento para estudantes "
                        "internacionais?"
                    ),
                    query_metadata={
                        "legal_intents": [
                            "payment_plan",
                            "general_payment_plan",
                            "international_student",
                        ],
                    },
                )
            )

        self.assertEqual(decision.route_name, "retry_candidate_document_scoped")
        self.assertEqual(decision.retrieval_scope, "retry_candidate_document_scoped")
        self.assertEqual(decision.retrieval_profile, "retry_candidate_document_scoped")
        self.assertEqual(decision.target_doc_ids, ["doc_propinas"])
        self.assertEqual(decision.target_document_titles, ["Regulamento de Propinas"])
        self.assertTrue(decision.allow_second_pass)
        self.assertEqual(decision.metadata["candidate_pool_size"], 24)
        self.assertEqual(
            decision.metadata["inferred_target_doc_ids"],
            ["doc_propinas"],
        )
        self.assertEqual(
            decision.metadata["inferred_target_document_titles"],
            ["Regulamento de Propinas"],
        )
        self.assertIn("payment_plan", decision.metadata["legal_intents"])
        self.assertIn("general_payment_plan", decision.metadata["legal_intents"])
        self.assertIn("international_student", decision.metadata["legal_intents"])
        self.assertIn("dynamic_document_target_inferred", decision.reasons)
        self.assertIn("retry_candidate_document_scope_selected", decision.reasons)

    def test_legal_intent_without_inference_uses_broad_expanded_pool(self) -> None:
        """
        Ensure legal intent broadens retrieval when no document target is inferred.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = self._build_settings()
            settings.embedding_input_root = Path(temporary_directory)
            settings.chunking_strategy = "article_smart"
            router = RetrievalRouter(settings=settings)

            decision = router.route(
                UserQuestionInput(
                    question_text="Quais sao os prazos para entregar documentos?",
                    normalized_query_text=(
                        "Quais sao os prazos para entregar documentos?"
                    ),
                    query_metadata={
                        "legal_intents": [
                            "deadline_question",
                            "document_requirement_question",
                        ],
                    },
                )
            )

        self.assertEqual(decision.route_name, "broad_expanded")
        self.assertEqual(decision.retrieval_scope, "broad_expanded")
        self.assertEqual(decision.retrieval_profile, "broad_expanded")
        self.assertEqual(decision.target_doc_ids, [])
        self.assertEqual(decision.metadata["candidate_pool_size"], 24)
        self.assertEqual(
            decision.metadata["legal_intents"],
            ["deadline_question", "document_requirement_question"],
        )
        self.assertTrue(decision.metadata["document_inference_attempted"])
        self.assertIn("legal_intent_broad_expansion_selected", decision.reasons)

    def test_explicit_document_name_overrides_dynamic_payment_plan_inference(
        self,
    ) -> None:
        """
        Ensure explicit document naming keeps scoped retrieval over retry inference.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory)
            self._write_chunk_file(
                input_root,
                "doc_propinas",
                "Regulamento de Propinas",
                article_number="5",
                article_title="Plano geral de pagamento",
                text=(
                    "O plano geral de pagamento regula prestacoes e "
                    "regularizacao de propinas."
                ),
            )

            settings = self._build_settings()
            settings.embedding_input_root = input_root
            settings.chunking_strategy = "article_smart"
            router = RetrievalRouter(settings=settings)

            decision = router.route(
                UserQuestionInput(
                    question_text=(
                        "No Regulamento de Propinas, qual e o plano geral "
                        "de pagamento?"
                    ),
                    normalized_query_text=(
                        "No Regulamento de Propinas, qual e o plano geral "
                        "de pagamento?"
                    ),
                    query_metadata={
                        "document_title": "Regulamento de Propinas",
                        "legal_intents": [
                            "payment_plan",
                            "general_payment_plan",
                        ],
                    },
                )
            )

        self.assertEqual(decision.route_name, "document_scoped")
        self.assertEqual(decision.retrieval_scope, "scoped")
        self.assertEqual(decision.target_doc_ids, ["doc_propinas"])
        self.assertEqual(decision.target_document_titles, ["Regulamento de Propinas"])
        self.assertEqual(decision.metadata["candidate_pool_size"], 6)
        self.assertFalse(decision.metadata["document_inferred"])
        self.assertEqual(
            decision.metadata["legal_intents"],
            ["payment_plan", "general_payment_plan"],
        )
        self.assertIn("document_target_detected", decision.reasons)
        self.assertNotIn("retry_candidate_document_scope_selected", decision.reasons)

    def test_dynamic_document_inference_covers_domain_specific_router_regressions(
        self,
    ) -> None:
        """
        Ensure explicit legal topics keep stable dynamic routing behavior.
        """

        regression_cases = [
            {
                "name": "payment_plan",
                "doc_id": "doc_propinas",
                "document_title": "Regulamento Financeiro Academico",
                "article_title": "Plano geral de pagamento",
                "text": (
                    "O plano geral de pagamento permite prestacoes e "
                    "regularizacao de propinas."
                ),
                "question_text": (
                    "Qual e o plano geral de pagamento em prestacoes?"
                ),
                "query_metadata": {
                    "legal_intents": [
                        "payment_plan",
                        "general_payment_plan",
                    ]
                },
                "expected_route_name": "retry_candidate_document_scoped",
                "expected_scope": "retry_candidate_document_scoped",
                "expected_candidate_pool_size": 24,
                "expected_reason": "retry_candidate_document_scope_selected",
                "expect_retry_metadata": True,
            },
            {
                "name": "matriculation_cancellation",
                "doc_id": "doc_matricula_cancelamento",
                "document_title": "Regulamento de Anulacao de Matricula",
                "article_title": "Prazos e efeitos",
                "text": (
                    "A anulacao de matricula segue prazos academicos e "
                    "efeitos administrativos."
                ),
                "question_text": "Como funciona a anulacao da matricula?",
                "query_metadata": {
                    "legal_intents": ["matriculation_cancellation"]
                },
                "expected_route_name": "document_scoped",
                "expected_scope": "scoped",
                "expected_candidate_pool_size": 6,
                "expected_reason": "dynamic_document_target_inferred",
                "expect_retry_metadata": False,
            },
            {
                "name": "mandatory_attendance",
                "doc_id": "doc_assiduidade",
                "document_title": "Regulamento de Assiduidade Obrigatoria",
                "article_title": "Assiduidade obrigatoria",
                "text": (
                    "A assiduidade obrigatoria aplica-se a unidades "
                    "curriculares com presenca controlada."
                ),
                "question_text": "Quando existe assiduidade obrigatoria?",
                "query_metadata": {
                    "legal_intents": ["mandatory_attendance"]
                },
                "expected_route_name": "document_scoped",
                "expected_scope": "scoped",
                "expected_candidate_pool_size": 6,
                "expected_reason": "dynamic_document_target_inferred",
                "expect_retry_metadata": False,
            },
            {
                "name": "liminary_rejection",
                "doc_id": "doc_indeferimento",
                "document_title": "Regulamento de Indeferimento Liminar",
                "article_title": "Fundamentos de rejeicao",
                "text": (
                    "O indeferimento liminar aplica-se a pedidos "
                    "manifestamente irregulares."
                ),
                "question_text": "Quando pode haver indeferimento liminar?",
                "query_metadata": {
                    "legal_intents": ["liminary_rejection"]
                },
                "expected_route_name": "document_scoped",
                "expected_scope": "scoped",
                "expected_candidate_pool_size": 6,
                "expected_reason": "dynamic_document_target_inferred",
                "expect_retry_metadata": False,
            },
            {
                "name": "exclusion",
                "doc_id": "doc_exclusao",
                "document_title": "Regulamento de Exclusao Academica",
                "article_title": "Causas de exclusao",
                "text": (
                    "A exclusao disciplinar depende da verificacao de "
                    "infracoes graves."
                ),
                "question_text": "Em que casos pode existir exclusao academica?",
                "query_metadata": {"legal_intents": ["exclusion"]},
                "expected_route_name": "document_scoped",
                "expected_scope": "scoped",
                "expected_candidate_pool_size": 6,
                "expected_reason": "dynamic_document_target_inferred",
                "expect_retry_metadata": False,
            },
        ]

        for regression_case in regression_cases:
            with self.subTest(regression_case["name"]):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    input_root = Path(temporary_directory)
                    self._write_chunk_file(
                        input_root,
                        regression_case["doc_id"],
                        regression_case["document_title"],
                        article_number="5",
                        article_title=regression_case["article_title"],
                        text=regression_case["text"],
                    )
                    self._write_chunk_file(
                        input_root,
                        f"{regression_case['doc_id']}_competitor",
                        "Regulamento Academico Geral",
                        article_number="7",
                        article_title="Disposicoes gerais",
                        text=(
                            "Normas academicas gerais sem foco no topico "
                            "principal da pergunta."
                        ),
                    )

                    settings = self._build_settings()
                    settings.embedding_input_root = input_root
                    settings.chunking_strategy = "article_smart"
                    router = RetrievalRouter(settings=settings)

                    decision = router.route(
                        UserQuestionInput(
                            question_text=regression_case["question_text"],
                            normalized_query_text=regression_case["question_text"],
                            query_metadata=regression_case["query_metadata"],
                        )
                    )

                self.assertEqual(
                    decision.route_name,
                    regression_case["expected_route_name"],
                )
                self.assertEqual(
                    decision.retrieval_scope,
                    regression_case["expected_scope"],
                )
                self.assertEqual(
                    decision.target_doc_ids,
                    [regression_case["doc_id"]],
                )
                self.assertEqual(
                    decision.target_document_titles,
                    [regression_case["document_title"]],
                )
                self.assertEqual(
                    decision.metadata["candidate_pool_size"],
                    regression_case["expected_candidate_pool_size"],
                )
                self.assertTrue(decision.metadata["routing_enabled"])
                self.assertIn(
                    regression_case["expected_reason"],
                    decision.reasons,
                )

                if regression_case["expect_retry_metadata"]:
                    self.assertTrue(decision.metadata["document_inferred"])
                    self.assertEqual(
                        decision.metadata["inferred_target_doc_ids"],
                        [regression_case["doc_id"]],
                    )
                    self.assertEqual(
                        decision.metadata["inferred_target_document_titles"],
                        [regression_case["document_title"]],
                    )
                else:
                    self.assertTrue(decision.metadata["document_inferred"])
                    self.assertNotIn(
                        "inferred_target_doc_ids",
                        decision.metadata,
                    )

    def test_ambiguous_document_signals_keep_broad_retrieval_with_larger_pool(
        self,
    ) -> None:
        """
        Ensure uncertain document inference preserves broad retrieval behavior.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory)
            self._write_chunk_file(
                input_root,
                "doc_avaliacao_geral",
                "Regulamento de Avaliacao Geral",
            )
            self._write_chunk_file(
                input_root,
                "doc_avaliacao_especial",
                "Regulamento de Avaliacao Especial",
            )

            settings = self._build_settings()
            settings.embedding_input_root = input_root
            settings.chunking_strategy = "article_smart"
            router = RetrievalRouter(settings=settings)

            decision = router.route(
                UserQuestionInput(
                    question_text="Como funciona a avaliacao?",
                    normalized_query_text="Como funciona a avaliacao?",
                )
            )

        self.assertEqual(decision.route_name, "standard_broad")
        self.assertEqual(decision.retrieval_scope, "broad")
        self.assertEqual(decision.target_doc_ids, [])
        self.assertEqual(decision.metadata["candidate_pool_size"], 18)
        self.assertTrue(decision.metadata["document_inference_attempted"])

    def test_comparative_question_stays_broad_and_multi_document(self) -> None:
        """
        Ensure comparative wording preserves broad multi-document retrieval.
        """

        router = RetrievalRouter(settings=self._build_settings())

        decision = router.route(
            UserQuestionInput(
                question_text=(
                    "Compara as regras entre o Regulamento de Propinas e "
                    "o Regulamento Academico."
                ),
                normalized_query_text=(
                    "Compara as regras entre o Regulamento de Propinas e "
                    "o Regulamento Academico."
                ),
                query_metadata={
                    "document_titles": [
                        "Regulamento de Propinas",
                        "Regulamento Academico",
                    ],
                },
            )
        )

        self.assertEqual(decision.route_name, "comparative_broad")
        self.assertEqual(decision.retrieval_scope, "broad")
        self.assertEqual(decision.retrieval_profile, "comparative")
        self.assertTrue(decision.comparative)
        self.assertEqual(decision.metadata["candidate_pool_size"], 18)
        self.assertIn("multi_document_retrieval_preserved", decision.reasons)

    def test_article_explicit_question_emits_article_bias_signals(self) -> None:
        """
        Ensure explicit article references bias retrieval without document scoping.
        """

        router = RetrievalRouter(settings=self._build_settings())

        decision = router.route(
            UserQuestionInput(
                question_text="O que diz o artigo 12 sobre prazos?",
                normalized_query_text="O que diz o artigo 12 sobre prazos?",
            )
        )

        self.assertEqual(decision.route_name, "article_biased")
        self.assertEqual(decision.retrieval_scope, "broad")
        self.assertEqual(decision.retrieval_profile, "article_biased")
        self.assertEqual(decision.target_article_numbers, ["12"])
        self.assertEqual(decision.metadata["candidate_pool_size"], 18)
        self.assertIn("article_bias_selected", decision.reasons)

    def test_disabled_routing_remains_explainable_and_standard(self) -> None:
        """
        Ensure disabled routing returns a broad standard decision with metadata.
        """

        router = RetrievalRouter(settings=self._build_settings(enabled=False))

        decision = router.route(
            UserQuestionInput(
                question_text="O que diz o artigo 8 do Regulamento de Propinas?",
                normalized_query_text="O que diz o artigo 8 do Regulamento de Propinas?",
                query_metadata={"document_title": "Regulamento de Propinas"},
            )
        )

        self.assertEqual(decision.route_name, "routing_disabled")
        self.assertEqual(decision.retrieval_scope, "broad")
        self.assertEqual(decision.retrieval_profile, "standard")
        self.assertFalse(decision.allow_second_pass)
        self.assertFalse(decision.metadata["routing_enabled"])
        self.assertEqual(decision.metadata["candidate_pool_size"], 5)
        self.assertIn("routing_disabled", decision.reasons)


if __name__ == "__main__":
    unittest.main()
