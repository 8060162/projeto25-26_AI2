"""Regression tests for retrieval context selection and packing."""

from __future__ import annotations

import unittest

from Chunking.config.settings import PipelineSettings
from retrieval.context_builder import RetrievalContextBuilder
from retrieval.models import RetrievalRouteDecision, RetrievedChunkResult


class RetrievalContextBuilderTests(unittest.TestCase):
    """Protect deterministic context selection and packing behavior."""

    def test_build_context_keeps_correct_chunk_after_deduplication_and_filtering(
        self,
    ) -> None:
        """Ensure the relevant chunk survives when early ranks collapse before packing."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_candidate_pool_size=4,
                retrieval_context_max_chunks=2,
                retrieval_context_max_characters=500,
                retrieval_score_filtering_enabled=True,
                retrieval_min_similarity_score=0.80,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_duplicate",
                    doc_id="doc_a",
                    text="Duplicate legal excerpt kept from the first rank.",
                    record_id="record_duplicate",
                    rank=1,
                    similarity_score=0.97,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_duplicate",
                    doc_id="doc_a",
                    text="Duplicate legal excerpt kept from the first rank.",
                    record_id="record_duplicate",
                    rank=2,
                    similarity_score=0.96,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_filtered",
                    doc_id="doc_b",
                    text="This chunk is retrieved but should not survive the similarity filter.",
                    record_id="record_filtered",
                    rank=3,
                    similarity_score=0.32,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_correct",
                    doc_id="doc_regulation",
                    text="Article 12 defines the exception that answers the question.",
                    record_id="record_correct",
                    rank=4,
                    similarity_score=0.91,
                    source_file="data/chunks/regulation.json",
                    chunk_metadata={
                        "article_number": "12",
                        "article_title": "Exceptional Cases",
                        "section_title": "Article 12",
                        "page_start": 9,
                    },
                    document_metadata={"document_title": "Student Regulation"},
                ),
            ]
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["chunk_duplicate", "chunk_correct"],
        )
        self.assertEqual(context.chunk_count, 2)
        self.assertEqual(context.metadata["candidate_chunk_count"], 2)
        self.assertEqual(context.metadata["duplicate_count"], 1)
        self.assertEqual(context.metadata["score_filtered_count"], 1)
        self.assertEqual(context.metadata["omitted_by_rank_limit_count"], 0)
        self.assertIn("chunk_correct", context.metadata["selected_chunk_ids"])

    def test_build_context_uses_broader_candidate_pool_before_final_chunk_limit(
        self,
    ) -> None:
        """Ensure lower-ranked valid chunks survive after early duplicates are removed."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_candidate_pool_size=4,
                retrieval_context_max_chunks=3,
                retrieval_context_max_characters=500,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_2",
                    doc_id="doc_a",
                    text="Second ranked chunk.",
                    record_id="record_2",
                    rank=3,
                    similarity_score=0.90,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_duplicate",
                    doc_id="doc_a",
                    text="Duplicate legal excerpt.",
                    record_id="record_duplicate",
                    rank=1,
                    similarity_score=0.95,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_duplicate",
                    doc_id="doc_a",
                    text="Duplicate legal excerpt.",
                    record_id="record_duplicate",
                    rank=2,
                    similarity_score=0.94,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_relevant_lower_rank",
                    doc_id="doc_b",
                    text="Article 9 contains the relevant exception.",
                    record_id="record_3",
                    rank=4,
                    similarity_score=0.91,
                    source_file="data/chunks/doc_b.json",
                    chunk_metadata={
                        "article_number": "9",
                        "article_title": "Exceptional Deadline",
                        "section_title": "Article 9",
                        "page_start": 7,
                    },
                    document_metadata={"document_title": "Regulation B"},
                ),
            ]
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["chunk_duplicate", "chunk_2", "chunk_relevant_lower_rank"],
        )
        self.assertEqual(context.chunk_count, 3)
        self.assertFalse(context.truncated)
        self.assertEqual(context.metadata["candidate_chunk_count"], 3)
        self.assertEqual(context.metadata["duplicate_count"], 1)
        self.assertEqual(context.metadata["omitted_by_rank_limit_count"], 0)
        self.assertEqual(context.metadata["omitted_by_max_chunks_count"], 0)
        self.assertIn("chunk_relevant_lower_rank", context.metadata["selected_chunk_ids"])

    def test_build_context_prioritizes_structural_match_over_raw_rank(self) -> None:
        """Ensure article-aligned chunks can outrank generic higher-ranked results."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_candidate_pool_size=4,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=500,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_generic",
                    doc_id="doc_generic",
                    text="General rule about deadlines without the requested legal anchor.",
                    record_id="record_generic",
                    rank=1,
                    similarity_score=0.98,
                    document_metadata={"document_title": "Academic Calendar"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_12",
                    doc_id="doc_regulation",
                    text="Article 12 defines the special payment exception.",
                    record_id="record_article_12",
                    rank=3,
                    similarity_score=0.89,
                    chunk_metadata={
                        "article_number": "12",
                        "article_title": "Special Payment Rules",
                        "section_title": "Article 12",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
            ],
            query_text="Qual e o conteudo do artigo 12 sobre pagamento especial?",
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_article_12"])
        self.assertEqual(context.metadata["selection_query_article_numbers"], ["12"])

    def test_build_context_prioritizes_structural_match_from_query_metadata(self) -> None:
        """Ensure metadata-derived structural cues can outrank weaker higher-ranked chunks."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_candidate_pool_size=4,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=500,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_generic_deadline",
                    doc_id="doc_calendar",
                    text="General calendar deadline without the requested legal basis.",
                    record_id="record_generic_deadline",
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={"section_title": "General Deadlines"},
                    document_metadata={"document_title": "Academic Calendar"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_7",
                    doc_id="doc_regulation",
                    text="Article 7 defines the special enrolment exception.",
                    record_id="record_article_7",
                    rank=3,
                    similarity_score=0.87,
                    chunk_metadata={
                        "article_number": "7",
                        "article_title": "Special Enrolment Exception",
                        "section_title": "Article 7",
                    },
                    document_metadata={"document_title": "Student Regulation"},
                ),
            ],
            query_text="qual e o prazo aplicavel?",
            query_metadata={
                "article_number": "7",
                "article_title": "Special Enrolment Exception",
                "document_title": "Student Regulation",
            },
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_article_7"])
        self.assertEqual(context.metadata["selection_query_article_numbers"], ["7"])
        self.assertEqual(context.metadata["selected_chunk_ids"], ["chunk_article_7"])

    def test_build_context_uses_route_targets_for_legal_competition(self) -> None:
        """Ensure route targets can preserve the legally specific chunk."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=3,
                retrieval_candidate_pool_size=3,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=700,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        route_decision = RetrievalRouteDecision(
            route_name="article_biased",
            retrieval_profile="article_biased",
            target_document_titles=["Tuition Regulation"],
            target_article_numbers=["9"],
            target_article_titles=["Special Payment Plans"],
            reasons=["article_bias_selected"],
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_standard_plan",
                    doc_id="doc_tuition",
                    text="Article 8 defines the standard payment plan for tuition fees.",
                    record_id="record_standard_plan",
                    rank=1,
                    similarity_score=0.97,
                    chunk_metadata={
                        "article_number": "8",
                        "article_title": "Standard Payment Plans",
                        "section_title": "Article 8",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_special_plan",
                    doc_id="doc_tuition",
                    text="Article 9 defines the special payment plan exception.",
                    record_id="record_special_plan",
                    rank=2,
                    similarity_score=0.91,
                    chunk_metadata={
                        "article_number": "9",
                        "article_title": "Special Payment Plans",
                        "section_title": "Article 9",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
            ],
            query_text="Qual e o plano de pagamento especial?",
            route_decision=route_decision,
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_special_plan"])
        self.assertIs(context.route_decision, route_decision)
        self.assertEqual(context.metadata["selection_query_article_numbers"], ["9"])
        self.assertEqual(
            context.metadata["selection_query_document_titles"],
            ["tuition regulation"],
        )
        self.assertIn("selection_reason=matched_article_number", context.context_text)
        self.assertIn("matched_document_title", context.context_text)
        self.assertIn("matched_article_title", context.context_text)

    def test_build_context_uses_route_candidate_pool_before_final_selection(
        self,
    ) -> None:
        """Ensure route-specific breadth is not discarded before context selection."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=5,
                retrieval_candidate_pool_size=2,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=900,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        route_decision = RetrievalRouteDecision(
            route_name="document_scoped",
            retrieval_profile="document_scoped",
            retrieval_scope="scoped",
            target_document_titles=["Regulamento de Propinas"],
            metadata={"candidate_pool_size": 5},
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_generic_payment",
                    doc_id="Despacho_P_PORTO_P_043_2025",
                    text="O pagamento da propina tem consequencias academicas.",
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={
                        "article_number": "8",
                        "article_title": "Consequencias do Incumprimento",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_late_payment",
                    doc_id="Despacho_P_PORTO_P_043_2025",
                    text="O pagamento fora de prazo implica juros de mora.",
                    rank=2,
                    similarity_score=0.97,
                    chunk_metadata={
                        "article_number": "7",
                        "article_title": "Pagamento Fora de Prazo",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_other",
                    doc_id="Despacho_P_PORTO_P_043_2025",
                    text="O pagamento e efetuado por meios eletronicos.",
                    rank=3,
                    similarity_score=0.96,
                    chunk_metadata={
                        "article_number": "10",
                        "article_title": "Tipos de Procedimentos",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="Despacho_P_PORTO_P_043_2025_chunk_0008",
                    doc_id="Despacho_P_PORTO_P_043_2025",
                    text=(
                        "Para estudante internacional, em 8 prestacoes, com "
                        "percentagens e datas-limite de pagamento."
                    ),
                    rank=5,
                    similarity_score=0.88,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "PLANO GERAL DE PAGAMENTO DE PROPINAS",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
            ],
            top_k=5,
            query_text=(
                "No Regulamento de Propinas, como funciona o plano geral de "
                "pagamento para estudante internacional?"
            ),
            route_decision=route_decision,
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["Despacho_P_PORTO_P_043_2025_chunk_0008"],
        )
        self.assertEqual(context.metadata["candidate_chunk_count"], 4)
        self.assertEqual(
            context.metadata["effective_candidate_pool_size"],
            5,
        )

    def test_build_context_prioritizes_primary_normative_intent_over_modifier_match(
        self,
    ) -> None:
        """Ensure primary payment-plan intent outranks nearby modifier-only chunks."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_candidate_pool_size=4,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=2200,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_article_24",
                    doc_id="doc_tuition",
                    text=(
                        "Os planos de regularizacao celebrados com estudantes "
                        "internacionais seguem limites proprios."
                    ),
                    record_id="record_article_24",
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={
                        "article_number": "24",
                        "article_title": "Estudantes Internacionais",
                        "section_title": "Article 24",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_6_a",
                    doc_id="doc_tuition",
                    text=(
                        "O plano especifico de pagamento depende de requerimento "
                        "fundamentado do estudante."
                    ),
                    record_id="record_article_6_a",
                    rank=2,
                    similarity_score=0.96,
                    chunk_metadata={
                        "article_number": "6",
                        "article_title": "Plano Especifico de Pagamento de Propinas",
                        "section_title": "Article 6",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_6_b",
                    doc_id="doc_tuition",
                    text=(
                        "As prestacoes do plano especifico nao podem ser "
                        "inferiores ao limite previsto."
                    ),
                    record_id="record_article_6_b",
                    rank=3,
                    similarity_score=0.95,
                    chunk_metadata={
                        "article_number": "6",
                        "article_title": "Plano Especifico de Pagamento de Propinas",
                        "section_title": "Article 6",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_5",
                    doc_id="doc_tuition",
                    text=(
                        "Para estudante internacional, o plano geral de pagamento "
                        "admite oito prestacoes."
                    ),
                    record_id="record_article_5",
                    rank=4,
                    similarity_score=0.90,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "Plano Geral de Pagamento de Propinas",
                        "section_title": "Article 5",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
            ],
            query_text=(
                "No Regulamento de Propinas, como funciona o plano de "
                "pagamento para estudante internacional?"
            ),
        )

        self.assertEqual(context.chunks[0].chunk_id, "chunk_article_5")
        self.assertEqual(context.chunks[0].context_metadata.article_number, "5")
        self.assertIn("selection_reason=matched_legal_terms", context.context_text)
        self.assertEqual(
            context.metadata["close_competitor_chunk_ids"],
            ["chunk_article_6_a", "chunk_article_6_b", "chunk_article_24"],
        )

    def test_build_context_uses_article_anchor_intent_before_body_only_overlap(
        self,
    ) -> None:
        """Ensure legal anchors beat competing chunks with generic body overlap."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_candidate_pool_size=4,
                retrieval_context_max_chunks=2,
                retrieval_context_max_characters=2200,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_article_8",
                    doc_id="doc_tuition",
                    text=(
                        "O pagamento da propina em atraso implica regularizacao "
                        "da situacao do estudante e consequencias academicas."
                    ),
                    record_id="record_article_8",
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={
                        "article_number": "8",
                        "article_title": "Consequencias do Incumprimento do Pagamento",
                        "section_title": "Article 8",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_6",
                    doc_id="doc_tuition",
                    text=(
                        "O plano especifico de pagamento depende de requerimento "
                        "fundamentado e inclui montantes vencidos."
                    ),
                    record_id="record_article_6",
                    rank=2,
                    similarity_score=0.96,
                    chunk_metadata={
                        "article_number": "6",
                        "article_title": "Plano Especifico de Pagamento",
                        "section_title": "Article 6",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_5_general",
                    doc_id="doc_tuition",
                    text=(
                        "A propina pode ser paga numa unica prestacao ou em "
                        "prestacoes mensais."
                    ),
                    record_id="record_article_5_general",
                    rank=3,
                    similarity_score=0.90,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "Plano Geral de Pagamento de Propinas",
                        "section_title": "Article 5",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_5_international",
                    doc_id="doc_tuition",
                    text=(
                        "Para estudante internacional, o pagamento da propina "
                        "pode ser efetuado em oito prestacoes mensais."
                    ),
                    record_id="record_article_5_international",
                    rank=4,
                    similarity_score=0.89,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "Plano Geral de Pagamento de Propinas",
                        "section_title": "Article 5",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
            ],
            query_text=(
                "Como funciona o plano geral de pagamento para estudante "
                "internacional?"
            ),
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["chunk_article_5_international", "chunk_article_5_general"],
        )
        self.assertEqual(context.evidence_quality.conflict, "none")
        self.assertTrue(context.evidence_quality.sufficient_for_answer)

    def test_build_context_grades_general_plan_competitors_without_blocking(
        self,
    ) -> None:
        """Ensure real legal competitors do not block the general payment plan."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_candidate_pool_size=4,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=2200,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_article_6_specific_plan",
                    doc_id="doc_tuition",
                    text=(
                        "O plano especifico de pagamento de propinas e autorizado "
                        "mediante requerimento fundamentado e define prestacoes "
                        "proprias."
                    ),
                    record_id="record_article_6_specific_plan",
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={
                        "article_number": "6",
                        "article_title": "Plano Especifico de Pagamento de Propinas",
                        "section_title": "Artigo 6",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_5_general_plan",
                    doc_id="doc_tuition",
                    text=(
                        "O plano geral de pagamento de propinas para estudante "
                        "internacional admite pagamento em oito prestacoes."
                    ),
                    record_id="record_article_5_general_plan",
                    rank=2,
                    similarity_score=0.97,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "Plano Geral de Pagamento de Propinas",
                        "section_title": "Artigo 5",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_24_regularization",
                    doc_id="doc_tuition",
                    text=(
                        "O estudante internacional com divida pode celebrar plano "
                        "de regularizacao nos termos aplicaveis."
                    ),
                    record_id="record_article_24_regularization",
                    rank=3,
                    similarity_score=0.96,
                    chunk_metadata={
                        "article_number": "24",
                        "article_title": "Estudantes Internacionais",
                        "section_title": "Artigo 24",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
            ],
            query_text=(
                "Como funciona o plano geral de pagamento de propinas para "
                "estudante internacional?"
            ),
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["chunk_article_5_general_plan"],
        )
        self.assertEqual(
            context.metadata["primary_anchor"],
            (
                "Regulamento de Propinas > Article 5 - "
                "Plano Geral de Pagamento de Propinas"
            ),
        )
        self.assertEqual(
            context.metadata["primary_anchor_chunk_ids"],
            ["chunk_article_5_general_plan"],
        )
        self.assertEqual(
            context.metadata["alternative_scope_competitor_chunk_ids"],
            ["chunk_article_6_specific_plan", "chunk_article_24_regularization"],
        )
        self.assertEqual(context.metadata["blocking_conflict_chunk_ids"], [])
        self.assertEqual(context.evidence_quality.conflict, "none")
        self.assertTrue(context.evidence_quality.sufficient_for_answer)

    def test_build_context_marks_specific_plan_competitors_as_blocking(
        self,
    ) -> None:
        """Ensure special-scope payment questions surface blocking competitors."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_candidate_pool_size=4,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=2200,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_article_6_specific_plan",
                    doc_id="doc_tuition",
                    text=(
                        "O plano especifico de pagamento de propinas e autorizado "
                        "mediante requerimento fundamentado e define prestacoes "
                        "proprias."
                    ),
                    record_id="record_article_6_specific_plan",
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={
                        "article_number": "6",
                        "article_title": "Plano Especifico de Pagamento de Propinas",
                        "section_title": "Artigo 6",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_5_general_plan",
                    doc_id="doc_tuition",
                    text=(
                        "O plano geral de pagamento de propinas para estudante "
                        "internacional admite pagamento em oito prestacoes."
                    ),
                    record_id="record_article_5_general_plan",
                    rank=2,
                    similarity_score=0.97,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "Plano Geral de Pagamento de Propinas",
                        "section_title": "Artigo 5",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_24_regularization",
                    doc_id="doc_tuition",
                    text=(
                        "O estudante internacional com divida pode celebrar plano "
                        "de regularizacao nos termos aplicaveis."
                    ),
                    record_id="record_article_24_regularization",
                    rank=3,
                    similarity_score=0.96,
                    chunk_metadata={
                        "article_number": "24",
                        "article_title": "Estudantes Internacionais",
                        "section_title": "Artigo 24",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
            ],
            query_text="Como funciona o plano especifico de pagamento de propinas?",
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["chunk_article_6_specific_plan"],
        )
        self.assertEqual(
            context.metadata["primary_anchor"],
            (
                "Regulamento de Propinas > Article 6 - "
                "Plano Especifico de Pagamento de Propinas"
            ),
        )
        self.assertEqual(
            context.metadata["blocking_conflict_chunk_ids"],
            ["chunk_article_5_general_plan", "chunk_article_24_regularization"],
        )
        self.assertEqual(
            context.metadata["conflicting_chunk_ids"],
            ["chunk_article_5_general_plan", "chunk_article_24_regularization"],
        )
        self.assertEqual(context.evidence_quality.conflict, "conflicting")
        self.assertFalse(context.evidence_quality.sufficient_for_answer)

    def test_build_context_keeps_annulment_of_enrolment_as_primary_anchor(
        self,
    ) -> None:
        """Ensure enrolment annulment does not falsely conflict with nearby regimes."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=3,
                retrieval_candidate_pool_size=3,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=2200,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_article_14_annulment",
                    doc_id="doc_enrolment",
                    text=(
                        "A anulacao da matricula implica a anulacao da inscricao "
                        "em todas as unidades curriculares e fixa a propina a "
                        "pagar conforme a data do pedido."
                    ),
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={
                        "article_number": "14",
                        "article_title": "Anulacao da matricula",
                        "section_title": "Artigo 14",
                    },
                    document_metadata={
                        "document_title": (
                            "Regulamento Geral de Matriculas e Inscricoes"
                        )
                    },
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_19_relocation",
                    doc_id="doc_reingresso",
                    text=(
                        "Os estudantes recolocados podem requerer a anulacao da "
                        "matricula no curso antecedente no prazo de sete dias "
                        "apos a nova matricula."
                    ),
                    rank=2,
                    similarity_score=0.97,
                    chunk_metadata={
                        "article_number": "19",
                        "article_title": (
                            "Estudantes colocados com matricula valida em outro "
                            "curso"
                        ),
                        "section_title": "Artigo 19",
                    },
                    document_metadata={
                        "document_title": (
                            "Regulamento de Reingresso e Mudanca de Par "
                            "Instituicao/Curso"
                        )
                    },
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_3_enrolment_fees",
                    doc_id="doc_enrolment",
                    text=(
                        "A falta de pagamento das taxas de matricula e seguro "
                        "escolar torna a matricula nao valida."
                    ),
                    rank=3,
                    similarity_score=0.96,
                    chunk_metadata={
                        "article_number": "3",
                        "article_title": "Matricula e inscricao",
                        "section_title": "Artigo 3",
                    },
                    document_metadata={
                        "document_title": (
                            "Regulamento Geral de Matriculas e Inscricoes"
                        )
                    },
                ),
            ],
            query_text="O que acontece e quanto pago se pedir a anulacao da matricula?",
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["chunk_article_14_annulment"],
        )
        self.assertEqual(
            context.metadata["primary_anchor"],
            (
                "Regulamento Geral de Matriculas e Inscricoes > Article 14 - "
                "Anulacao da matricula"
            ),
        )
        self.assertEqual(
            context.metadata["primary_anchor_chunk_ids"],
            ["chunk_article_14_annulment"],
        )
        self.assertEqual(
            context.metadata["alternative_scope_competitor_chunk_ids"],
            ["chunk_article_3_enrolment_fees", "chunk_article_19_relocation"],
        )
        self.assertEqual(context.metadata["blocking_conflict_chunk_ids"], [])
        self.assertEqual(context.evidence_quality.conflict, "none")
        self.assertTrue(context.evidence_quality.sufficient_for_answer)

    def test_build_context_keeps_attendance_rule_primary_without_exam_conflict(
        self,
    ) -> None:
        """Ensure attendance questions stay anchored on article 6 without false conflicts."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=3,
                retrieval_candidate_pool_size=3,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=2200,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_article_4_modalities",
                    doc_id="doc_evaluation",
                    text=(
                        "A FUC define modalidades de avaliacao, incluindo "
                        "avaliacao durante o periodo letivo com exame obrigatorio."
                    ),
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={
                        "article_number": "4",
                        "article_title": (
                            "Modalidades, criterios de avaliacao e ficha de "
                            "unidade curricular"
                        ),
                        "section_title": "Artigo 4",
                    },
                    document_metadata={
                        "document_title": (
                            "Regulamento de Avaliacao de Aproveitamento"
                        )
                    },
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_6_attendance",
                    doc_id="doc_evaluation",
                    text=(
                        "O ensino e presencial, nao sendo obrigatoria a "
                        "assiduidade as aulas, exceto exigencia contraria na FUC; "
                        "a falta injustificada superior a um terco implica perda "
                        "de assiduidade quando ela e obrigatoria."
                    ),
                    rank=2,
                    similarity_score=0.97,
                    chunk_metadata={
                        "article_number": "6",
                        "article_title": "Regime de assiduidade",
                        "section_title": "Artigo 6",
                    },
                    document_metadata={
                        "document_title": (
                            "Regulamento de Avaliacao de Aproveitamento"
                        )
                    },
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_9_exam_registration",
                    doc_id="doc_evaluation",
                    text=(
                        "A inscricao em exame e obrigatoria nas epocas de "
                        "recurso e especial e depende de pagamento da taxa "
                        "respetiva."
                    ),
                    rank=3,
                    similarity_score=0.96,
                    chunk_metadata={
                        "article_number": "9",
                        "article_title": "Inscricao nas provas de exame",
                        "section_title": "Artigo 9",
                    },
                    document_metadata={
                        "document_title": (
                            "Regulamento de Avaliacao de Aproveitamento"
                        )
                    },
                ),
            ],
            query_text="A assiduidade as aulas e obrigatoria?",
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["chunk_article_6_attendance"],
        )
        self.assertEqual(
            context.metadata["primary_anchor"],
            (
                "Regulamento de Avaliacao de Aproveitamento > Article 6 - "
                "Regime de assiduidade"
            ),
        )
        self.assertEqual(
            context.metadata["primary_anchor_chunk_ids"],
            ["chunk_article_6_attendance"],
        )
        self.assertEqual(
            context.metadata["alternative_scope_competitor_chunk_ids"],
            ["chunk_article_4_modalities", "chunk_article_9_exam_registration"],
        )
        self.assertEqual(context.metadata["blocking_conflict_chunk_ids"], [])
        self.assertEqual(context.evidence_quality.conflict, "none")
        self.assertTrue(context.evidence_quality.sufficient_for_answer)

    def test_build_context_classifies_same_article_general_plan_as_supportive(
        self,
    ) -> None:
        """Ensure same-article chunks support the selected primary anchor."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_candidate_pool_size=4,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=1600,
                retrieval_context_primary_anchor_max_count=1,
                retrieval_context_single_intent_max_supporting_chunks=1,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_article_5_schedule",
                    doc_id="doc_tuition",
                    text=(
                        "O plano geral de pagamento de propinas para estudante "
                        "internacional admite pagamento em oito prestacoes."
                    ),
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "Plano Geral de Pagamento de Propinas",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_5_deadlines",
                    doc_id="doc_tuition",
                    text=(
                        "As prestacoes do plano geral para estudante internacional "
                        "vencem em datas-limite mensais."
                    ),
                    rank=2,
                    similarity_score=0.96,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "Plano Geral de Pagamento de Propinas",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_6_specific_plan",
                    doc_id="doc_tuition",
                    text=(
                        "O plano especifico de pagamento depende de requerimento "
                        "fundamentado."
                    ),
                    rank=3,
                    similarity_score=0.95,
                    chunk_metadata={
                        "article_number": "6",
                        "article_title": "Plano Especifico de Pagamento de Propinas",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
            ],
            query_text=(
                "Como funciona o plano geral de pagamento de propinas para "
                "estudante internacional?"
            ),
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["chunk_article_5_schedule"],
        )
        self.assertEqual(
            context.metadata["supportive_competitor_chunk_ids"],
            ["chunk_article_5_deadlines"],
        )
        self.assertEqual(
            context.metadata["alternative_scope_competitor_chunk_ids"],
            ["chunk_article_6_specific_plan"],
        )
        self.assertEqual(context.metadata["blocking_conflict_chunk_ids"], [])
        self.assertEqual(context.evidence_quality.conflict, "none")

    def test_build_context_prioritizes_content_intent_over_subject_title_match(
        self,
    ) -> None:
        """Ensure subject-only article titles do not outrank applicable content."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=3,
                retrieval_candidate_pool_size=3,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=1600,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_article_24",
                    doc_id="doc_tuition",
                    text=(
                        "Os planos de regularizacao celebrados com estudantes "
                        "internacionais seguem limites proprios."
                    ),
                    record_id="record_article_24",
                    rank=1,
                    similarity_score=0.98,
                    chunk_metadata={
                        "article_number": "24",
                        "article_title": "Estudantes Internacionais",
                        "section_title": "Article 24",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_article_5",
                    doc_id="doc_tuition",
                    text=(
                        "Para estudante internacional, o pagamento da propina "
                        "pode ser efetuado em oito prestacoes mensais."
                    ),
                    record_id="record_article_5",
                    rank=3,
                    similarity_score=0.90,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "Plano Geral de Pagamento de Propinas",
                        "section_title": "Article 5",
                    },
                    document_metadata={"document_title": "Regulamento de Propinas"},
                ),
            ],
            query_text="Quais sao as prestacoes do estudante internacional?",
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_article_5"])
        self.assertEqual(context.chunks[0].context_metadata.article_number, "5")
        self.assertEqual(
            context.metadata["close_competitor_chunk_ids"],
            ["chunk_article_24"],
        )

    def test_build_context_exposes_close_legal_competitors_in_metadata(self) -> None:
        """Ensure omitted legal competitors remain visible for downstream routing."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=3,
                retrieval_candidate_pool_size=3,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=700,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_exception",
                    doc_id="doc_tuition",
                    text="Article 12 states the exceptional deadline for payment plans.",
                    record_id="record_exception",
                    rank=1,
                    similarity_score=0.95,
                    chunk_metadata={
                        "article_number": "12",
                        "article_title": "Exceptional Payment Deadline",
                        "section_title": "Article 12",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_standard",
                    doc_id="doc_tuition",
                    text="Article 10 states the standard deadline for payment plans.",
                    record_id="record_standard",
                    rank=2,
                    similarity_score=0.94,
                    chunk_metadata={
                        "article_number": "10",
                        "article_title": "Standard Payment Deadline",
                        "section_title": "Article 10",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
            ],
            query_text="Qual e o prazo excecional para plano de pagamento?",
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_exception"])
        self.assertEqual(
            context.metadata["close_competitor_chunk_ids"],
            ["chunk_standard"],
        )
        self.assertEqual(context.metadata["conflicting_chunk_ids"], ["chunk_standard"])
        self.assertIsNotNone(context.evidence_quality)
        self.assertEqual(context.evidence_quality.ambiguity, "ambiguous")
        self.assertEqual(context.evidence_quality.conflict, "conflicting")

    def test_build_context_focuses_single_intent_query_on_dominant_anchor(
        self,
    ) -> None:
        """Ensure generic same-document competitors are not packed as evidence."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=3,
                retrieval_candidate_pool_size=3,
                retrieval_context_max_chunks=2,
                retrieval_context_max_characters=1500,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_payment_general",
                    doc_id="doc_tuition",
                    text="Article 5 defines the general tuition payment plan.",
                    record_id="record_payment_general",
                    rank=1,
                    similarity_score=0.96,
                    chunk_metadata={
                        "article_number": "5",
                        "article_title": "General Tuition Payment Plan",
                        "section_title": "Article 5",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_payment_scope",
                    doc_id="doc_tuition",
                    text="Article 7 defines payment applicability for enrolment.",
                    record_id="record_payment_scope",
                    rank=2,
                    similarity_score=0.95,
                    chunk_metadata={
                        "article_number": "7",
                        "article_title": "Payment Applicability",
                        "section_title": "Article 7",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_payment_specific",
                    doc_id="doc_tuition",
                    text="Article 10 defines specific payment plan conditions.",
                    record_id="record_payment_specific",
                    rank=3,
                    similarity_score=0.94,
                    chunk_metadata={
                        "article_number": "10",
                        "article_title": "Specific Payment Plan",
                        "section_title": "Article 10",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
            ],
            query_text="How does the tuition payment plan work?",
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in context.chunks],
            ["chunk_payment_general"],
        )
        self.assertEqual(
            context.metadata["close_competitor_chunk_ids"],
            ["chunk_payment_scope", "chunk_payment_specific"],
        )
        self.assertEqual(context.metadata["conflicting_chunk_ids"], [])
        self.assertIsNotNone(context.evidence_quality)
        self.assertEqual(context.evidence_quality.ambiguity, "ambiguous")
        self.assertEqual(context.evidence_quality.conflict, "none")

    def test_build_context_marks_selected_close_anchors_as_insufficient(
        self,
    ) -> None:
        """Ensure unresolved selected legal anchors do not reach normal generation."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=2,
                retrieval_candidate_pool_size=2,
                retrieval_context_max_chunks=2,
                retrieval_context_max_characters=1500,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_payment_plan",
                    doc_id="doc_tuition",
                    text="Article 6 describes a student payment plan.",
                    record_id="record_payment_plan",
                    rank=1,
                    similarity_score=0.95,
                    chunk_metadata={
                        "article_number": "6",
                        "article_title": "Payment Plan",
                        "section_title": "Article 6",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_payment_scheme",
                    doc_id="doc_tuition",
                    text="Article 8 describes a student payment plan.",
                    record_id="record_payment_scheme",
                    rank=2,
                    similarity_score=0.94,
                    chunk_metadata={
                        "article_number": "8",
                        "article_title": "Payment Plan",
                        "section_title": "Article 8",
                    },
                    document_metadata={"document_title": "Tuition Regulation"},
                ),
            ],
            query_text="How does the student payment plan work?",
        )

        self.assertEqual(context.evidence_quality.ambiguity, "clear")
        self.assertEqual(context.evidence_quality.conflict, "conflicting")
        self.assertFalse(context.evidence_quality.sufficient_for_answer)
        self.assertEqual(
            context.evidence_quality.conflicting_chunk_ids,
            ["chunk_payment_scheme"],
        )

    def test_build_context_uses_distance_to_break_ties_without_similarity_scores(
        self,
    ) -> None:
        """Ensure smaller vector distance wins when rank ties and similarity is absent."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=2,
                retrieval_candidate_pool_size=2,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=300,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_farther",
                    doc_id="doc_a",
                    text="This chunk is farther from the query embedding.",
                    record_id="record_farther",
                    rank=1,
                    distance=0.21,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_closer",
                    doc_id="doc_b",
                    text="This chunk is closer to the query embedding.",
                    record_id="record_closer",
                    rank=1,
                    distance=0.08,
                ),
            ]
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_closer"])
        self.assertEqual(context.metadata["selected_chunk_ids"], ["chunk_closer"])
        self.assertEqual(context.metadata["omitted_by_max_chunks_count"], 1)

    def test_build_context_applies_similarity_filter_and_budget_truncation(self) -> None:
        """Ensure score filtering and character-budget truncation remain deterministic."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=3,
                retrieval_context_max_chunks=3,
                retrieval_context_max_characters=85,
                retrieval_score_filtering_enabled=True,
                retrieval_min_similarity_score=0.80,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_high",
                    doc_id="doc_a",
                    text="This grounded excerpt is long enough to exceed the compact context budget.",
                    record_id="record_high",
                    rank=1,
                    similarity_score=0.93,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_low",
                    doc_id="doc_b",
                    text="This excerpt should be filtered by similarity.",
                    record_id="record_low",
                    rank=2,
                    similarity_score=0.40,
                ),
            ]
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_high"])
        self.assertTrue(context.truncated)
        self.assertLessEqual(context.character_count, 85)
        self.assertEqual(context.metadata["score_filtered_count"], 1)
        self.assertEqual(context.metadata["omitted_by_budget_count"], 0)

    def test_build_context_exposes_article_metadata_in_serialized_context(self) -> None:
        """Ensure article-level structural metadata is explicit in the final context."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=3,
                retrieval_candidate_pool_size=3,
                retrieval_context_max_chunks=2,
                retrieval_context_max_characters=500,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
                retrieval_context_include_parent_structure=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_article",
                    doc_id="doc_reg_a",
                    text="The filing deadline is 10 working days.",
                    record_id="record_article",
                    rank=1,
                    similarity_score=0.96,
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
            ]
        )

        self.assertIn(
            "legal_anchor=Regulation A > Article 5 - Deadlines",
            context.context_text,
        )
        self.assertIn("document_title=Regulation A", context.context_text)
        self.assertIn("article_number=5", context.context_text)
        self.assertIn("article_title=Deadlines", context.context_text)
        self.assertIn("section_title=Article 5", context.context_text)
        self.assertIn("parent_structure=Chapter II > Applications", context.context_text)
        self.assertIn("pages=3", context.context_text)
        self.assertEqual(
            [metadata.article_number for metadata in context.selected_context_metadata],
            ["5"],
        )
        self.assertEqual(context.retrieval_quality.structural_metadata_chunk_count, 1)
        self.assertEqual(context.retrieval_quality.selected_chunk_ids, ["chunk_article"])

    def test_build_context_truncates_rich_article_serialization_within_budget(self) -> None:
        """Ensure richer structural serialization still truncates deterministically."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=2,
                retrieval_candidate_pool_size=2,
                retrieval_context_max_chunks=1,
                retrieval_context_max_characters=360,
                retrieval_context_include_article_number=True,
                retrieval_context_include_article_title=True,
                retrieval_context_include_parent_structure=True,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_rich_article",
                    doc_id="doc_reg_a",
                    text=(
                        "The deadline is suspended during the documented exceptional "
                        "period and resumes on the next working day after notification."
                    ),
                    record_id="record_rich_article",
                    rank=1,
                    similarity_score=0.96,
                    source_file="data/chunks/reg_a.json",
                    chunk_metadata={
                        "article_number": "14",
                        "article_title": "Exceptional Suspension of Deadlines",
                        "section_title": "Article 14",
                        "parent_structure": ["Chapter III", "Deadlines"],
                        "page_start": 12,
                    },
                    document_metadata={"document_title": "Regulation A"},
                )
            ]
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_rich_article"])
        self.assertTrue(context.truncated)
        self.assertLessEqual(context.character_count, 360)
        self.assertEqual(context.metadata["omitted_by_budget_count"], 0)
        self.assertIn(
            "legal_anchor=Regulation A > Article 14 - Exceptional Suspension of Deadlines",
            context.context_text,
        )
        self.assertIn("article_number=14", context.context_text)
        self.assertIn("article_title=Exceptional Suspension of Deadlines", context.context_text)
        self.assertIn("section_title=Article 14", context.context_text)
        self.assertIn("parent_structure=Chapter III > Deadlines", context.context_text)

    def test_build_context_keeps_missing_similarity_scores_when_filtering_enabled(self) -> None:
        """Ensure missing similarity scores do not discard otherwise valid chunks."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=2,
                retrieval_context_max_chunks=2,
                retrieval_context_max_characters=300,
                retrieval_score_filtering_enabled=True,
                retrieval_min_similarity_score=0.80,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_missing_score",
                    doc_id="doc_a",
                    text="Grounded excerpt without explicit similarity score.",
                    record_id="record_missing_score",
                    rank=1,
                )
            ]
        )

        self.assertEqual(context.chunk_count, 1)
        self.assertFalse(context.truncated)
        self.assertEqual(context.metadata["missing_similarity_score_count"], 1)
        self.assertIn("chunk_missing_score", context.metadata["selected_chunk_ids"])


if __name__ == "__main__":
    unittest.main()
