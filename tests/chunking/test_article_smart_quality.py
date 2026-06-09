"""Regression tests for known article_smart chunk-quality failures."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import (
    DocumentMetadata,
    ExtractedDocument,
    ExtractedPage,
    PageText,
    StructuralNode,
)
from Chunking.chunking.strategy_article_smart import ArticleSmartChunkingStrategy
from Chunking.cleaning.normalizer import TextNormalizer
from Chunking.config.settings import PipelineSettings
from Chunking.parsing.structure_parser import StructureParser


def build_document_metadata() -> DocumentMetadata:
    """Build stable document metadata for in-memory chunking tests."""
    return DocumentMetadata(
        doc_id="quality_regression",
        file_name="quality_regression.pdf",
        title="Quality Regression Fixture",
        source_path="quality_regression.pdf",
    )


def build_article_smart_chunks(
    raw_text: str,
    settings: PipelineSettings | None = None,
):
    """Run the minimal in-memory pipeline needed for article_smart tests."""
    effective_settings = settings or PipelineSettings()
    normalized = TextNormalizer().normalize(
        [PageText(page_number=1, text=raw_text)]
    )
    structure_root = StructureParser().parse(normalized)
    strategy = ArticleSmartChunkingStrategy(effective_settings)
    return strategy.build_chunks(build_document_metadata(), structure_root)


def build_article_smart_chunks_from_extracted_pages(
    pages: list[ExtractedPage],
    settings: PipelineSettings | None = None,
):
    """Run article_smart while preserving extracted-page quality signals."""
    effective_settings = settings or PipelineSettings()
    normalized = TextNormalizer().normalize(
        ExtractedDocument(
            source_path="quality_regression.pdf",
            page_count=len(pages),
            pages=pages,
        )
    )
    structure_root = StructureParser().parse(normalized)
    strategy = ArticleSmartChunkingStrategy(effective_settings)
    return strategy.build_chunks(build_document_metadata(), structure_root)


class ArticleSmartQualityRegressionTests(unittest.TestCase):
    """Protect known chunk-quality fixes from regression."""

    def test_editorial_residue_is_removed_from_article_smart_output(self) -> None:
        """Ensure Diario da Republica residue does not survive into chunks."""
        chunks = build_article_smart_chunks(
            (
                "DIÁRIO o DA REPÚBLICA | 07-06-2024\n"
                "Artigo 1 - Objeto\n"
                "O regulamento aplica-se a todos os cursos."
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].text,
            "O regulamento aplica-se a todos os cursos.",
        )
        self.assertNotIn("DIÁRIO", chunks[0].meta_text.upper())
        self.assertNotIn("REPÚBLICA", chunks[0].meta_text.upper())

    def test_broken_hyphenation_is_repaired_before_chunk_export(self) -> None:
        """Ensure broken same-line hyphenation is fixed in final chunk text."""
        chunks = build_article_smart_chunks(
            (
                "Artigo 3 - Créditos\n"
                "A ava- liação e os cré- ditos são apli- cáveis "
                "para inscrever -se."
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].text,
            "A avaliação e os créditos são aplicáveis para inscrever-se.",
        )
        self.assertNotIn("ava- liação", chunks[0].text)
        self.assertNotIn("cré- ditos", chunks[0].text)
        self.assertNotIn("apli- cáveis", chunks[0].text)
        self.assertNotIn("inscrever -se", chunks[0].text)

    def test_meta_text_does_not_keep_artificial_title_separators(self) -> None:
        """Ensure meta-text headers normalize titles polluted with pipe separators."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_5",
            title="Âmbito | Aplicação",
            text="O presente regulamento define o âmbito de aplicação.",
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "5",
                "article_title": "Âmbito | Aplicação",
                "document_part": "regulation_body",
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 1)
        self.assertIn("Artigo 5 - Âmbito Aplicação", chunks[0].meta_text)
        self.assertNotIn(" | ", chunks[0].meta_text)

    def test_long_enumerated_article_splits_into_multiple_chunks(self) -> None:
        """Ensure long enumerated article content no longer stays monolithic."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_2",
            title="Definições",
            text=(
                "Definições:\n"
                "1. Primeiro conceito com detalhe suficiente para crescer "
                "e justificar divisão.\n"
                "2. Segundo conceito com detalhe suficiente para crescer "
                "e justificar divisão.\n"
                "3. Terceiro conceito com detalhe suficiente para crescer "
                "e justificar divisão."
            ),
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "2",
                "article_title": "Definições",
                "document_part": "regulation_body",
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        settings = PipelineSettings(
            target_chunk_chars=110,
            min_chunk_chars=50,
            hard_max_chunk_chars=140,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(chunk.chunk_reason == "fallback_legal_signal_split" for chunk in chunks)
        )
        self.assertTrue(
            all(chunk.char_count <= settings.hard_max_chunk_chars for chunk in chunks)
        )
        self.assertTrue(chunks[0].text.startswith("Definições:"))
        self.assertTrue(any("2." in chunk.text for chunk in chunks[1:]))
        self.assertTrue(any("3." in chunk.text for chunk in chunks[1:]))

    def test_access_footnote_is_removed_from_final_chunk_text(self) -> None:
        """Ensure access footnotes do not survive final article_smart cleanup."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_10",
            title="Disposições finais",
            text=(
                "Disposições finais\n"
                "O presente regulamento entra em vigor no próximo ano letivo.\n"
                "(1) Acessível em https://domus.ipp.pt/."
            ),
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "10",
                "article_title": "Disposições finais",
                "document_part": "regulation_body",
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].text,
            "O presente regulamento entra em vigor no próximo ano letivo.",
        )
        self.assertNotIn("Acessível", chunks[0].text)
        self.assertNotIn("https://", chunks[0].text)

    def test_unnumbered_access_footnote_is_removed_from_final_chunk_text(self) -> None:
        """Ensure editorial access notes are removed even without footnote numbers."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_10A",
            title="Disposições finais",
            text=(
                "Disposições finais\n"
                "O presente regulamento entra em vigor no próximo ano letivo.\n"
                "Acessível em https://domus.ipp.pt/."
            ),
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "10-A",
                "article_title": "Disposições finais",
                "document_part": "regulation_body",
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].text,
            "O presente regulamento entra em vigor no próximo ano letivo.",
        )
        self.assertNotIn("Acessível", chunks[0].text)
        self.assertNotIn("https://", chunks[0].text)

    def test_residual_article_title_does_not_leak_into_chunk_text(self) -> None:
        """Ensure merged article titles are removed from visible chunk text."""
        chunks = build_article_smart_chunks(
            (
                "Artigo 2.o\n"
                "ÂMBITO O presente regulamento aplica-se a todos os ciclos de estudo."
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].text,
            "O presente regulamento aplica-se a todos os ciclos de estudo.",
        )
        self.assertNotIn("ÂMBITO O presente", chunks[0].text)

    def test_inline_heading_residue_is_removed_from_chunk_text(self) -> None:
        """Ensure glued uppercase heading residue is removed from final text."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_11",
            title="Projeto/Dissertação/Estágio",
            text=(
                "PROJETO/DISSERTAÇÃO/ESTÁGIO Os/as estudantes de mestrado "
                "devem formalizar a inscrição no prazo definido."
            ),
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "11",
                "article_title": "Projeto/Dissertação/Estágio",
                "document_part": "regulation_body",
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].text,
            "Os/as estudantes de mestrado devem formalizar a inscrição no prazo definido.",
        )
        self.assertNotIn("PROJETO/DISSERTAÇÃO/ESTÁGIO", chunks[0].text)

    def test_split_overlap_repeats_intro_context_for_numbered_parts(self) -> None:
        """Ensure later legal-split chunks keep the article intro context."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_12",
            title="Situações especiais",
            text=(
                "Nas situações previstas no presente artigo:\n"
                "1. O estudante deve submeter o pedido no prazo definido "
                "pela escola e juntar a documentação necessária.\n"
                "2. O estudante deve aguardar a validação administrativa "
                "antes de concluir a inscrição.\n"
                "3. O estudante deve regularizar os elementos em falta "
                "logo que seja notificado."
            ),
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "12",
                "article_title": "Situações especiais",
                "document_part": "regulation_body",
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        settings = PipelineSettings(
            target_chunk_chars=135,
            min_chunk_chars=60,
            hard_max_chunk_chars=180,
            overlap_chars=70,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[1].text.startswith("Nas situações previstas no presente artigo:"))
        self.assertIn("2.", chunks[1].text)
        self.assertTrue(chunks[-1].text.startswith("Nas situações previstas no presente artigo:"))
        self.assertIn("3.", chunks[-1].text)

    def test_partial_trailing_overlap_is_not_exported_as_orphan_prefix(self) -> None:
        """Ensure overlap does not prepend truncated continuation fragments."""
        settings = PipelineSettings(
            target_chunk_chars=260,
            min_chunk_chars=120,
            hard_max_chunk_chars=360,
            overlap_chars=80,
        )
        strategy = ArticleSmartChunkingStrategy(settings)
        parts = [
            (
                "O presente regulamento fixa as normas gerais relativas a matrículas e "
                "inscrições nos cursos ministrados pelas Escolas do Instituto Politécnico "
                "do Porto. O processo de matrícula/inscrição num determinado curso é da "
                "responsabilidade da Escola onde o curso é ministrado."
            ),
            (
                "O órgão legal e estatutariamente competente da Escola poderá fixar em "
                "regulamento específico normas adicionais em matérias complementares."
            ),
        ]

        overlapped_parts = strategy._apply_split_overlap(parts, "\n\n".join(parts))

        self.assertEqual(len(overlapped_parts), 2)
        self.assertFalse(overlapped_parts[1].startswith("determinado curso"))
        self.assertTrue(
            overlapped_parts[1].startswith(
                "O órgão legal e estatutariamente competente da Escola"
            )
        )

    def test_unsafe_overlap_is_skipped_when_it_only_repeats_item_tail(self) -> None:
        """Ensure split overlap does not create em-dash-only orphan continuations."""
        settings = PipelineSettings(
            target_chunk_chars=320,
            min_chunk_chars=140,
            hard_max_chunk_chars=420,
            overlap_chars=80,
        )
        strategy = ArticleSmartChunkingStrategy(settings)
        parts = [
            (
                "3 — Inscrição em exame — É o ato pelo qual o estudante formaliza a sua "
                "intenção de realizar um exame."
            ),
            (
                "4 — Regime de precedências — Regime que estabelece que a inscrição numa ou "
                "mais unidades curriculares depende de aprovação anterior."
            ),
        ]

        overlapped_parts = strategy._apply_split_overlap(parts, "\n\n".join(parts))

        self.assertEqual(len(overlapped_parts), 2)
        self.assertFalse(overlapped_parts[1].startswith("— É o ato"))
        self.assertTrue(overlapped_parts[1].startswith("4 — Regime de precedências"))

    def test_article_smart_output_keeps_full_non_truncated_article_body(self) -> None:
        """Ensure final chunks keep the complete article body in citation cases."""
        chunks = build_article_smart_chunks(
            (
                "ANEXO\n"
                "Artigo 17.o\n"
                "Inscrições em estágio profissional\n"
                "1 - O presente artigo regulamenta as medidas de apoio aos licenciados "
                "e mestres, previstas no\n"
                "artigo 46.o -B do Decreto -Lei n.o 74/2006, de 24 de março.\n"
                "2 - Aplica-se aos titulares do grau de licenciatura."
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertIn(
            "artigo 46.o -B do Decreto -Lei n.o 74/2006, de 24 de março.",
            chunks[0].text,
        )
        self.assertIn(
            "2 - Aplica-se aos titulares do grau de licenciatura.",
            chunks[0].text,
        )

    def test_default_settings_keep_article_smart_chunks_within_1024_chars(self) -> None:
        """Ensure the default chunk-size source of truth respects the 1024 ceiling."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_18",
            title="Medidas aplicáveis",
            text="\n".join(
                [
                    (
                        f"{number}. O presente regulamento estabelece a medida {number} "
                        "com detalhe suficiente para manter contexto juridico, "
                        "preservar autonomia semantica e exigir divisao controlada."
                    )
                    for number in range(1, 10)
                ]
            ),
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "18",
                "article_title": "Medidas aplicáveis",
                "document_part": "regulation_body",
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        settings = PipelineSettings()
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(settings.hard_max_chunk_chars, 1024)
        self.assertTrue(all(chunk.char_count <= 1024 for chunk in chunks))

    def test_preamble_tail_does_not_keep_signature_or_truncated_title_residue(self) -> None:
        """Ensure preamble chunks drop trailing signature and cut title fragments."""
        chunks = build_article_smart_chunks(
            (
                "Considerando que o projeto de regulamento foi objeto de consulta publica.\n"
                "Determino: a) A aprovacao do Regulamento de Propinas do P.PORTO, "
                "anexo ao presente despacho e que\n"
                "c) A revogacao do Despacho P.PORTO/P-042/2023.\n"
                "Paulo Pereira\n"
                "Regulamento de\n"
                "Artigo 1 - Objeto\n"
                "O regulamento aplica-se a todos os cursos."
            )
        )

        self.assertEqual(len(chunks), 2)
        self.assertNotIn("Paulo Pereira", chunks[0].text)
        self.assertFalse(
            any(line.strip() == "Regulamento de" for line in chunks[0].text.splitlines())
        )
        self.assertFalse(chunks[0].text.endswith("e que"))
        self.assertEqual(chunks[1].text, "O regulamento aplica-se a todos os cursos.")

    def test_front_matter_node_is_not_exported_as_visible_chunk(self) -> None:
        """Ensure front matter stays structural metadata and never becomes chunk text."""
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[
                StructuralNode(
                    node_type="FRONT_MATTER",
                    label="FRONT_MATTER",
                    text=(
                        "Escola Superior de Tecnologia e Gestao do Instituto Politecnico do Porto."
                    ),
                    page_start=1,
                    page_end=1,
                ),
                StructuralNode(
                    node_type="PREAMBLE",
                    label="PREAMBLE",
                    text="Considerando a necessidade de atualizar o regulamento.",
                    page_start=1,
                    page_end=1,
                ),
                StructuralNode(
                    node_type="ARTICLE",
                    label="ART_1",
                    title="Objeto",
                    text="O presente regulamento define o objeto.",
                    page_start=2,
                    page_end=2,
                    metadata={
                        "article_number": "1",
                        "article_title": "Objeto",
                        "document_part": "regulation_body",
                    },
                ),
            ],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [chunk.source_node_type for chunk in chunks],
            ["PREAMBLE", "ARTICLE"],
        )
        self.assertTrue(
            all(
                "Escola Superior de Tecnologia e Gestao"
                not in chunk.meta_text
                for chunk in chunks
            )
        )

    def test_preamble_split_does_not_start_following_chunk_with_dangling_fragment(self) -> None:
        """Ensure legal-signal preamble splits do not emit weak lowercase continuations."""
        preamble = StructuralNode(
            node_type="PREAMBLE",
            label="PREAMBLE",
            text="\n".join(
                [
                    "Considerando:",
                    (
                        "O regulamento estabelece o regime aplicavel aos pedidos e "
                        "procedimentos seguintes, mantendo contexto suficiente para "
                        "forcar divisao controlada nesta fase."
                    ),
                    (
                        "1. Reingresso em ciclos de estudo com regras, prazos e "
                        "condicoes detalhadas para justificar divisao semantica."
                    ),
                    (
                        "2. Mudanca de par instituicao/curso com regras, prazos e "
                        "condicoes detalhadas para justificar divisao semantica."
                    ),
                    (
                        "3. Concurso especial com regras, prazos e condicoes "
                        "detalhadas para justificar divisao semantica."
                    ),
                ]
            ),
            page_start=1,
            page_end=1,
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[preamble],
        )
        settings = PipelineSettings(
            target_chunk_chars=170,
            min_chunk_chars=80,
            hard_max_chunk_chars=220,
            overlap_chars=60,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= settings.hard_max_chunk_chars for chunk in chunks))
        self.assertFalse(any(chunk.text.startswith("em ciclos de estudo") for chunk in chunks))
        self.assertFalse(any(chunk.text.startswith("de par instituicao/curso") for chunk in chunks))
        self.assertFalse(any(chunk.text.startswith("especial com regras") for chunk in chunks))

    def test_locally_degraded_index_page_does_not_pollute_first_exported_chunk(self) -> None:
        """Ensure toxic TOC pages do not leak into the first article_smart chunks."""
        chunks = build_article_smart_chunks_from_extracted_pages(
            [
                ExtractedPage(
                    page_number=1,
                    text="Considerando a necessidade de fixar regras comuns.",
                    selected_mode="dict",
                    quality_score=120.0,
                    corruption_flags=[],
                ),
                ExtractedPage(
                    page_number=2,
                    text=(
                        "REGULAMENTO P.PORTO/P-002/2025\n"
                        "PROPINAS DO INSTITUTO POLITECNICO DO PORTO\n"
                        "1|14\n"
                        "CAPITULO I ........................................ 3\n"
                        "DISPOSICOES GERAIS ................................ 3\n"
                        "ARTIGO 1 ........................................ 3\n"
                        "OBJETO ........................................ 3\n"
                        "CAPITULO II ........................................ 7"
                    ),
                    selected_mode="text",
                    quality_score=-4.0,
                    corruption_flags=["low_alpha_ratio", "high_symbol_ratio"],
                ),
                ExtractedPage(
                    page_number=3,
                    text=(
                        "Artigo 1 - Objeto\n"
                        "O presente regulamento define o objeto."
                    ),
                    selected_mode="dict",
                    quality_score=118.0,
                    corruption_flags=[],
                ),
            ]
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "Considerando a necessidade de fixar regras comuns.")
        self.assertEqual(chunks[0].source_node_type, "PREAMBLE")
        self.assertNotIn("CAPITULO I", chunks[0].meta_text)
        self.assertNotIn("1|14", chunks[0].meta_text)
        self.assertEqual(chunks[1].text, "O presente regulamento define o objeto.")

    def test_default_settings_split_long_preamble_within_1024_chars(self) -> None:
        """Ensure oversized preamble text is split under the hard size ceiling."""
        preamble = StructuralNode(
            node_type="PREAMBLE",
            label="PREAMBLE",
            text="\n".join(
                [
                    (
                        f"Considerando {number}, que o presente regulamento estabelece "
                        "um conjunto de normas e procedimentos com detalhe suficiente "
                        "para exigir divisao controlada e preservar contexto juridico."
                    )
                    for number in range(1, 12)
                ]
            ),
            page_start=1,
            page_end=1,
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[preamble],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 1024 for chunk in chunks))
        self.assertTrue(all(chunk.chunk_reason != "preamble_group" for chunk in chunks))

    def test_weak_split_boundary_is_reattached_when_next_part_starts_with_connector(self) -> None:
        """Ensure weak split boundaries are repaired before export."""
        settings = PipelineSettings(
            target_chunk_chars=210,
            min_chunk_chars=90,
            hard_max_chunk_chars=260,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        repaired_parts = strategy._repair_weak_split_boundaries(
            [
                (
                    "O estudante pode requerer a melhoria da classificação mediante "
                    "pedido apresentado dentro do prazo e nos termos de"
                ),
                (
                    " acordo com o regulamento próprio da unidade orgânica e com "
                    "os critérios aplicáveis ao ciclo de estudos."
                ),
            ]
        )

        self.assertEqual(len(repaired_parts), 1)
        self.assertIn("nos termos de acordo com o regulamento próprio", repaired_parts[0])

    def test_split_oversized_text_remerges_weak_connector_boundary(self) -> None:
        """Ensure final oversized splitting does not export connector-led tails."""
        settings = PipelineSettings(
            target_chunk_chars=150,
            min_chunk_chars=80,
            hard_max_chunk_chars=280,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        split_parts, split_mode = strategy._split_oversized_text(
            (
                "O estudante pode requerer a melhoria da classificação mediante "
                "pedido apresentado dentro do prazo e nos termos de\n\n"
                "acordo com o regulamento próprio da unidade orgânica e com "
                "os critérios aplicáveis ao ciclo de estudos."
            )
        )

        self.assertEqual(split_mode, "paragraph_split")
        self.assertEqual(len(split_parts), 1)
        self.assertFalse(split_parts[0].endswith("nos termos de"))
        self.assertFalse(split_parts[0].startswith("acordo com"))
        self.assertIn("nos termos de acordo com o regulamento próprio", split_parts[0])

    def test_article_fallback_split_does_not_export_connector_led_tail_chunk(self) -> None:
        """Ensure fallback article splitting does not emit a connector-led tail."""
        settings = PipelineSettings(
            target_chunk_chars=130,
            min_chunk_chars=80,
            hard_max_chunk_chars=240,
        )

        chunks = build_article_smart_chunks(
            (
                "Artigo 7 - Requerimento\n"
                "O estudante pode requerer a melhoria da classificacao mediante "
                "pedido apresentado dentro do prazo e nos termos de\n\n"
                "acordo com o regulamento proprio da unidade organica e com "
                "os criterios aplicaveis ao ciclo de estudos."
            ),
            settings=settings,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_node_type, "ARTICLE_PART")
        self.assertIn(
            "nos termos de acordo com o regulamento proprio da unidade organica",
            chunks[0].text,
        )
        self.assertFalse(chunks[0].text.startswith("acordo com"))
        self.assertFalse(chunks[0].text.endswith("nos termos de"))

    def test_preamble_split_does_not_export_connector_led_tail_chunk(self) -> None:
        """Ensure preamble splitting does not emit a connector-led tail."""
        preamble = StructuralNode(
            node_type="PREAMBLE",
            label="PREAMBLE",
            text=(
                "Considerando que o presente regulamento fixa regras de acesso, "
                "procedimentos e prazos aplicaveis aos pedidos formulados pelos "
                "estudantes e nos termos de\n\n"
                "acordo com o regulamento proprio de cada unidade organica."
            ),
            page_start=1,
            page_end=1,
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[preamble],
        )
        settings = PipelineSettings(
            target_chunk_chars=130,
            min_chunk_chars=80,
            hard_max_chunk_chars=240,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_node_type, "PREAMBLE")
        self.assertIn(
            "nos termos de\nacordo com o regulamento proprio de cada unidade organica.",
            chunks[0].text,
        )
        self.assertFalse(chunks[0].text.startswith("acordo com"))
        self.assertFalse(chunks[0].text.endswith("nos termos de"))

    def test_structurally_incomplete_truncated_article_is_not_exported(self) -> None:
        """Ensure incomplete truncated articles do not become unsafe visible chunks."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_18",
            title="Dúvidas e omissões",
            text=(
                "As dúvidas e omissões resultantes da aplicação do presente regulamento "
                "serão resolvidas por aplicação do Regulamento de Exames do Instituto "
                "Politécnico do Porto (P PORTO) e na ausência de"
            ),
            page_start=9,
            page_end=9,
            metadata={
                "article_number": "18",
                "article_title": "Dúvidas e omissões",
                "document_part": "regulation_body",
                "is_structurally_incomplete": True,
                "truncation_signals": ["suspicious_truncated_ending"],
                "integrity_warnings": [],
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(chunks, [])

    def test_incomplete_section_groups_are_split_into_safe_single_section_exports(self) -> None:
        """Ensure incomplete multi-section groups are not exported unchanged."""
        first_section_text = (
            "1. Credito - ato pelo qual o estudante obtém reconhecimento formal "
            "do percurso anterior com elementos suficientes para autonomia e "
            "contexto juridico."
        )
        second_section_text = (
            "2. Inscricao - ato pelo qual o estudante formaliza o pedido de "
            "acesso ou continuidade no curso com elementos suficientes para "
            "autonomia e contexto juridico."
        )
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_18A",
            title="Definicoes",
            text=f"{first_section_text}\n{second_section_text}",
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "18-A",
                "article_title": "Definicoes",
                "document_part": "regulation_body",
                "is_structurally_incomplete": True,
            },
        )
        article.children = [
            StructuralNode(
                node_type="SECTION",
                label="1",
                text=first_section_text,
                page_start=1,
                page_end=1,
            ),
            StructuralNode(
                node_type="SECTION",
                label="2",
                text=second_section_text,
                page_start=1,
                page_end=1,
            ),
        ]
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        settings = PipelineSettings(
            target_chunk_chars=220,
            min_chunk_chars=80,
            hard_max_chunk_chars=320,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [chunk.source_node_type for chunk in chunks],
            ["SECTION_GROUP", "SECTION_GROUP"],
        )
        self.assertEqual(
            [chunk.source_node_label for chunk in chunks],
            ["1", "2"],
        )
        self.assertEqual(
            [chunk.metadata.get("section_labels") for chunk in chunks],
            [["1"], ["2"]],
        )
        self.assertTrue(all(chunk.chunk_reason == "grouped_sections" for chunk in chunks))
        self.assertTrue(all("," not in chunk.source_node_label for chunk in chunks))
        self.assertTrue(all("Inscricao -" not in chunk.text for chunk in chunks[:1]))

    def test_weak_section_lead_in_is_reattached_to_subordinate_continuation(self) -> None:
        """Ensure introductory numbered lead-ins do not survive as standalone groups."""
        lead_in_text = (
            "2. O pagamento da propina pode ser efetuado numa das seguintes modalidades:"
        )
        continuation_text = (
            "2.1 Em dez prestacoes mensais, sucessivas e de igual valor, "
            "nos termos fixados anualmente."
        )
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_18AA",
            title="Pagamento",
            text=f"{lead_in_text}\n{continuation_text}",
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "18-AA",
                "article_title": "Pagamento",
                "document_part": "regulation_body",
                "is_structurally_incomplete": True,
                "integrity_warnings": ["possible_interrupted_definition_capture"],
            },
        )
        article.children = [
            StructuralNode(
                node_type="SECTION",
                label="2",
                text=lead_in_text,
                page_start=1,
                page_end=1,
            ),
            StructuralNode(
                node_type="SECTION",
                label="2.1",
                text=continuation_text,
                page_start=1,
                page_end=1,
            ),
        ]
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        settings = PipelineSettings(
            target_chunk_chars=80,
            min_chunk_chars=40,
            hard_max_chunk_chars=220,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_node_type, "SECTION_GROUP")
        self.assertEqual(chunks[0].source_node_label, "2,2.1")
        self.assertEqual(chunks[0].metadata.get("section_labels"), ["2", "2.1"])
        self.assertIn("seguintes modalidades:", chunks[0].text)
        self.assertIn("Em dez prestacoes mensais", chunks[0].text)

    def test_weak_lettered_lead_in_is_reattached_to_following_item(self) -> None:
        """Ensure introductory lettered lead-ins do not survive as standalone groups."""
        lead_in_text = (
            "a) O procedimento pode seguir uma das seguintes modalidades:"
        )
        continuation_text = (
            "b) Modalidade ordinaria, aplicavel aos estudantes regularmente inscritos."
        )
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_18AB",
            title="Modalidades",
            text=f"{lead_in_text}\n{continuation_text}",
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "18-AB",
                "article_title": "Modalidades",
                "document_part": "regulation_body",
                "is_structurally_incomplete": True,
                "integrity_warnings": ["possible_broken_lettered_enumeration"],
            },
        )
        article.children = [
            StructuralNode(
                node_type="LETTERED_ITEM",
                label="a",
                text=lead_in_text,
                page_start=1,
                page_end=1,
            ),
            StructuralNode(
                node_type="LETTERED_ITEM",
                label="b",
                text=continuation_text,
                page_start=1,
                page_end=1,
            ),
        ]
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        settings = PipelineSettings(
            target_chunk_chars=70,
            min_chunk_chars=30,
            hard_max_chunk_chars=220,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_node_type, "LETTERED_GROUP")
        self.assertEqual(chunks[0].source_node_label, "a,b")
        self.assertEqual(chunks[0].metadata.get("lettered_labels"), ["a", "b"])
        self.assertIn("seguintes modalidades:", chunks[0].text)
        self.assertIn("Modalidade ordinaria", chunks[0].text)

    def test_orphaned_numbered_continuation_sections_are_not_exported_as_chunks(self) -> None:
        """Ensure subordinate continuation groups are rejected when parser warns."""
        first_section_text = (
            "2.1 O estudante apresenta o requerimento no prazo fixado pelo "
            "regulamento com contexto suficiente para forcar agrupamento inicial."
        )
        second_section_text = (
            "2.2 Junta os documentos exigidos pelo regulamento com contexto "
            "suficiente para forcar agrupamento inicial."
        )
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_18B",
            title="Continuacao",
            text=f"{first_section_text}\n{second_section_text}",
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "18-B",
                "article_title": "Continuacao",
                "document_part": "regulation_body",
                "is_structurally_incomplete": True,
                "integrity_warnings": ["possible_orphaned_numbered_continuation"],
            },
        )
        article.children = [
            StructuralNode(
                node_type="SECTION",
                label="2.1",
                text=first_section_text,
                page_start=1,
                page_end=1,
            ),
            StructuralNode(
                node_type="SECTION",
                label="2.2",
                text=second_section_text,
                page_start=1,
                page_end=1,
            ),
        ]
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        settings = PipelineSettings(
            target_chunk_chars=180,
            min_chunk_chars=60,
            hard_max_chunk_chars=260,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(chunks, [])

    def test_broken_lettered_continuation_groups_are_not_exported_as_chunks(self) -> None:
        """Ensure broken non-initial lettered groups do not survive final export."""
        first_item_text = (
            "b) O estudante pode requerer equivalencia em condicoes excecionais "
            "devidamente fundamentadas e comprovadas documentalmente."
        )
        second_item_text = (
            "c) O pedido depende de parecer favoravel do orgao competente e de "
            "verificacao administrativa previa."
        )
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_18C",
            title="Alineas",
            text=f"{first_item_text}\n{second_item_text}",
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "18-C",
                "article_title": "Alineas",
                "document_part": "regulation_body",
                "is_structurally_incomplete": True,
                "integrity_warnings": ["possible_broken_lettered_enumeration"],
            },
        )
        article.children = [
            StructuralNode(
                node_type="LETTERED_ITEM",
                label="b",
                text=first_item_text,
                page_start=1,
                page_end=1,
            ),
            StructuralNode(
                node_type="LETTERED_ITEM",
                label="c",
                text=second_item_text,
                page_start=1,
                page_end=1,
            ),
        ]
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        settings = PipelineSettings(
            target_chunk_chars=180,
            min_chunk_chars=60,
            hard_max_chunk_chars=280,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(chunks, [])

    def test_single_line_clause_like_preamble_stays_within_1024_chars(self) -> None:
        """Ensure clause-like preambles do not exceed the hard ceiling."""
        preamble_text = "\n".join(
            [
                "Considerando:",
                (
                    "A Portaria n.o 181-D/2015, de 19 de junho, regula os regimes "
                    "de reingresso e de mudanca com detalhe suficiente para manter "
                    "contexto juridico e ainda exigir divisao controlada nesta fase;"
                ),
                (
                    "O Decreto-Lei n.o 62/2018, de 6 de agosto, altera e republica "
                    "o estatuto aplicavel em termos extensos para que o primeiro "
                    "bloco de clausulas continue claramente acima do tamanho alvo;"
                ),
                (
                    "O Regulamento n.o 181/2017, de 5 de maio, fixa regras "
                    "institucionais complementares com densidade textual suficiente "
                    "para manter o preambulo longo e semanticamente autonomo;"
                ),
                (
                    "O Regulamento n.o 450/2020, de 12 de junho, acrescenta "
                    "criterios e procedimentos complementares num bloco final que "
                    "continua a ser relevante para retrieval e grounding;"
                ),
                (
                    "O Regulamento n.o 512/2021, de 30 de setembro, introduz "
                    "novos requisitos operacionais e garantias procedimentais com "
                    "densidade suficiente para manter este preambulo acima do "
                    "tamanho maximo aceitavel num unico chunk;"
                ),
                (
                    "A aplicacao articulada destes regimes exige ainda uma "
                    "explicacao acumulada sobre prazos, condicoes e excecoes, "
                    "preservando contexto semantico mas impondo divisao controlada;"
                ),
                "2. E revogado o Despacho IPP/P-042/2022, de 27 de julho.",
            ]
        )
        preamble = StructuralNode(
            node_type="PREAMBLE",
            label="PREAMBLE",
            text=preamble_text,
            page_start=1,
            page_end=1,
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[preamble],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 1024 for chunk in chunks))
        self.assertTrue(
            all(chunk.chunk_reason == "preamble_legal_signal_split" for chunk in chunks)
        )

    def test_preamble_split_repeats_intro_context_for_semantic_continuity(self) -> None:
        """Ensure split preamble chunks keep the leading context that frames the clauses."""
        preamble = StructuralNode(
            node_type="PREAMBLE",
            label="PREAMBLE",
            text="\n".join(
                [
                    "Considerando que o presente regulamento depende dos seguintes fundamentos:",
                    (
                        "1. A harmonizacao dos procedimentos institucionais exige "
                        "regras detalhadas, prazos objetivos e criterios uniformes "
                        "para todas as unidades organicas."
                    ),
                    (
                        "2. A protecao da igualdade de tratamento entre estudantes "
                        "exige medidas operacionais complementares e controlo "
                        "administrativo consistente."
                    ),
                    (
                        "3. A articulacao com regulamentos especiais impõe "
                        "clarificacoes adicionais, validacao documental e "
                        "publicitacao das decisoes finais."
                    ),
                ]
            ),
            page_start=1,
            page_end=1,
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[preamble],
        )
        settings = PipelineSettings(
            target_chunk_chars=180,
            min_chunk_chars=90,
            hard_max_chunk_chars=230,
            overlap_chars=70,
        )
        strategy = ArticleSmartChunkingStrategy(settings)

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            chunks[1].text.startswith(
                "que o presente regulamento depende dos seguintes fundamentos:"
            )
        )
        self.assertIn("2.", chunks[1].text)
        self.assertTrue(
            chunks[-1].text.startswith(
                "que o presente regulamento depende dos seguintes fundamentos:"
            )
        )
        self.assertIn("3.", chunks[-1].text)

    def test_formula_garbage_tail_is_removed_from_final_chunk_text(self) -> None:
        """Ensure OCR-like formula residue does not survive at the end of chunks."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_20",
            title="Creditacao",
            text=(
                "1. Quando aplicavel, as unidades curriculares creditadas conservam "
                "as classificacoes obtidas.\n\n"
                "2. Quando se trate de unidades curriculares realizadas em "
                "estabelecimentos de ensino superior estrangeiros, aplicar-se-a a "
                "seguinte formula de calculo:\n"
                "_\n\n10 1\n_\n_\nIPP\nCIESe\nCSESe lmp\nC\nCSESe lMp\n"
                "CSESe lmp\nSection\nparagraph\nsymbol\nnoise"
            ),
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "20",
                "article_title": "Creditacao",
                "document_part": "regulation_body",
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 1)
        self.assertIn(
            "2. Quando se trate de unidades curriculares realizadas em",
            chunks[0].text,
        )
        self.assertNotIn("CSESe lmp", chunks[0].text)
        self.assertNotIn("CSESe lMp", chunks[0].text)
        self.assertNotIn("Section", chunks[0].text)
        self.assertNotIn("symbol", chunks[0].text)

    def test_garbled_fragment_line_is_removed_from_final_chunk_text(self) -> None:
        """Ensure strongly garbled residue does not survive chunk cleanup."""
        article = StructuralNode(
            node_type="ARTICLE",
            label="ART_13",
            title="Objeto",
            text=(
                "O presente regulamento define o objeto do procedimento.\n"
                "[]]--==//__~~\n"
                "Aplica-se a todos os cursos conferentes de grau."
            ),
            page_start=1,
            page_end=1,
            metadata={
                "article_number": "13",
                "article_title": "Objeto",
                "document_part": "regulation_body",
            },
        )
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            children=[article],
        )
        strategy = ArticleSmartChunkingStrategy(PipelineSettings())

        chunks = strategy.build_chunks(build_document_metadata(), root)

        self.assertEqual(len(chunks), 1)
        self.assertNotIn("[]]--==//__~~", chunks[0].text)
        self.assertIn("Aplica-se a todos os cursos conferentes de grau.", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
