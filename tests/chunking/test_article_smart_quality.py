"""Regression tests for known article_smart chunk-quality failures."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import DocumentMetadata, PageText, StructuralNode
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
        self.assertNotIn("DIÁRIO", chunks[0].text_for_embedding.upper())
        self.assertNotIn("REPÚBLICA", chunks[0].text_for_embedding.upper())

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

    def test_embedding_text_does_not_keep_artificial_title_separators(self) -> None:
        """Ensure embedding headers normalize titles polluted with pipe separators."""
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
        self.assertIn("Artigo 5 - Âmbito Aplicação", chunks[0].text_for_embedding)
        self.assertNotIn(" | ", chunks[0].text_for_embedding)

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
