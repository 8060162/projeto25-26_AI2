"""Regression tests for parser fixes that affect article_smart chunk quality."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import (
    ExtractedDocument,
    ExtractedPage,
    PageText,
    StructuralNode,
)
from Chunking.cleaning.normalizer import TextNormalizer
from Chunking.parsing.structure_parser import StructureParser


def build_structure(raw_text: str) -> StructuralNode:
    """Run the minimal in-memory pipeline required for parser regression tests."""
    normalized = TextNormalizer().normalize(
        [PageText(page_number=1, text=raw_text)]
    )
    return StructureParser().parse(normalized)


def build_structure_from_lines(raw_text: str) -> StructuralNode:
    """Run parser-only tests against an already normalized line stream."""
    return StructureParser().parse([(1, raw_text)])


def build_structure_from_extracted_pages(
    pages: list[ExtractedPage],
) -> StructuralNode:
    """Run the pipeline with extracted-page quality signals preserved."""
    normalized = TextNormalizer().normalize(
        ExtractedDocument(
            source_path="quality_regression.pdf",
            page_count=len(pages),
            pages=pages,
        )
    )
    return StructureParser().parse(normalized)


def collect_articles(node: StructuralNode) -> list[StructuralNode]:
    """Collect ARTICLE nodes recursively for focused parser assertions."""
    articles: list[StructuralNode] = []

    if node.node_type == "ARTICLE":
        articles.append(node)

    for child in node.children:
        articles.extend(collect_articles(child))

    return articles


def collect_nodes_by_type(node: StructuralNode, node_type: str) -> list[StructuralNode]:
    """Collect nodes recursively for focused structural boundary assertions."""
    collected_nodes: list[StructuralNode] = []

    if node.node_type == node_type:
        collected_nodes.append(node)

    for child in node.children:
        collected_nodes.extend(collect_nodes_by_type(child, node_type))

    return collected_nodes


class StructureParserQualityRegressionTests(unittest.TestCase):
    """Protect parser behavior required by the chunk-quality implementation plan."""

    def test_institutional_title_line_stays_in_front_matter_before_preamble(self) -> None:
        """Ensure standalone institutional title lines do not pollute PREAMBLE."""
        root = build_structure_from_lines(
            (
                "Escola Superior de Tecnologia e Gestao do Instituto Politecnico do Porto.\n"
                "Considerando a necessidade de atualizar o regulamento.\n"
                "Artigo 1 - Objeto\n"
                "O presente regulamento define o objeto."
            )
        )

        front_matter_nodes = collect_nodes_by_type(root, "FRONT_MATTER")
        preamble_nodes = collect_nodes_by_type(root, "PREAMBLE")

        self.assertEqual(len(front_matter_nodes), 1)
        self.assertEqual(
            front_matter_nodes[0].text,
            "Escola Superior de Tecnologia e Gestao do Instituto Politecnico do Porto.",
        )
        self.assertEqual(len(preamble_nodes), 1)
        self.assertEqual(
            preamble_nodes[0].text,
            "Considerando a necessidade de atualizar o regulamento.",
        )

    def test_inline_header_suffix_is_split_between_title_and_body(self) -> None:
        """Ensure inline article header suffixes do not leak body text into titles."""
        root = build_structure(
            (
                "Artigo 5 - Definições 1. Para efeitos do presente regulamento, "
                "aplica-se o disposto no número seguinte."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Definições")
        self.assertEqual(
            articles[0].text,
            "1. Para efeitos do presente regulamento, aplica-se o disposto no número seguinte.",
        )

    def test_inline_title_and_body_line_is_separated_cleanly(self) -> None:
        """Ensure merged title/body lines are recovered without title leakage."""
        root = build_structure(
            (
                "Artigo 2.o\n"
                "ÂMBITO O presente regulamento aplica-se a todos os ciclos de estudo."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "ÂMBITO")
        self.assertEqual(
            articles[0].text,
            "O presente regulamento aplica-se a todos os ciclos de estudo.",
        )

    def test_title_case_title_line_is_consumed_before_numbered_body(self) -> None:
        """Ensure longer title-case article titles are not left inside article text."""
        root = build_structure_from_lines(
            (
                "Artigo 4\n"
                "Modalidades, critérios de avaliação e ficha de unidade curricular\n"
                "1 - A avaliação das competências e conhecimentos pode ser efetuada "
                "durante o período letivo."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertEqual(
            articles[0].title,
            "Modalidades, critérios de avaliação e ficha de unidade curricular",
        )
        self.assertEqual(
            articles[0].text,
            "1 - A avaliação das competências e conhecimentos pode ser efetuada durante o período letivo.",
        )

    def test_title_case_inline_title_and_body_are_separated_cleanly(self) -> None:
        """Ensure merged title-case title/body lines are recovered conservatively."""
        root = build_structure_from_lines(
            (
                "Artigo 6\n"
                "Prazos de matrícula e de inscrição As matrículas e inscrições "
                "realizam-se nos prazos fixados no calendário escolar."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Prazos de matrícula e de inscrição")
        self.assertEqual(
            articles[0].text,
            "As matrículas e inscrições realizam-se nos prazos fixados no calendário escolar.",
        )

    def test_definition_body_without_title_is_not_promoted_to_article_title(self) -> None:
        """Ensure definition-style first lines remain body text when no title exists."""
        root = build_structure_from_lines(
            (
                "Artigo 3\n"
                "Considera-se estudante a tempo integral, aquele/a que se encontre "
                "inscrito/a a, pelo menos, 51% do número máximo de créditos ECTS."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "")
        self.assertEqual(
            articles[0].text,
            "Considera-se estudante a tempo integral, aquele/a que se encontre inscrito/a a, pelo menos, 51% do número máximo de créditos ECTS.",
        )

    def test_citation_line_does_not_create_spurious_article_node(self) -> None:
        """Ensure legal citations do not truncate the current article body."""
        root = build_structure(
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

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].label, "ART_17")
        self.assertEqual(articles[0].title, "Inscrições em estágio profissional")
        self.assertIn(
            "artigo 46.o -B do Decreto -Lei n.o 74/2006, de 24 de março.",
            articles[0].text,
        )
        self.assertNotIn("ART_46", [article.label for article in articles])

    def test_preamble_tail_residue_is_trimmed_before_first_article(self) -> None:
        """Ensure sign-off names and title fragments do not remain in PREAMBLE."""
        root = build_structure_from_lines(
            (
                "Considerando que o projeto foi submetido a consulta publica.\n"
                "Determino: a aprovacao do regulamento anexo ao presente despacho e que\n"
                "Paulo Pereira\n"
                "Regulamento de\n"
                "Artigo 1 - Objeto\n"
                "O presente regulamento aplica-se a todos os cursos."
            )
        )

        preamble_nodes = collect_nodes_by_type(root, "PREAMBLE")
        articles = collect_articles(root)

        self.assertEqual(len(preamble_nodes), 1)
        self.assertEqual(
            preamble_nodes[0].text,
            "Considerando que o projeto foi submetido a consulta publica.\nDetermino: a aprovacao do regulamento anexo ao presente despacho",
        )
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Objeto")
        self.assertEqual(
            articles[0].text,
            "O presente regulamento aplica-se a todos os cursos.",
        )

    def test_chapter_body_starts_before_first_article_without_polluting_preamble(self) -> None:
        """Ensure preamble closes at the first normative chapter boundary."""
        root = build_structure_from_lines(
            (
                "Considerando a necessidade de fixar regras comuns.\n"
                "CAPITULO I\n"
                "Disposicoes gerais\n"
                "O presente capitulo estabelece principios gerais aplicaveis.\n"
                "Artigo 1 - Objeto\n"
                "O presente regulamento define o objeto."
            )
        )

        preamble_nodes = collect_nodes_by_type(root, "PREAMBLE")
        chapters = collect_nodes_by_type(root, "CHAPTER")
        articles = collect_articles(root)

        self.assertEqual(len(preamble_nodes), 1)
        self.assertEqual(
            preamble_nodes[0].text,
            "Considerando a necessidade de fixar regras comuns.",
        )
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0].title, "Disposicoes gerais")
        self.assertEqual(
            chapters[0].text,
            "O presente capitulo estabelece principios gerais aplicaveis.",
        )
        self.assertEqual(len(articles), 1)

    def test_locally_degraded_index_page_does_not_pollute_structure_start(self) -> None:
        """Ensure toxic TOC pages are quarantined before structural parsing."""
        root = build_structure_from_extracted_pages(
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

        front_matter_nodes = collect_nodes_by_type(root, "FRONT_MATTER")
        preamble_nodes = collect_nodes_by_type(root, "PREAMBLE")
        articles = collect_articles(root)

        self.assertEqual(len(front_matter_nodes), 0)
        self.assertEqual(len(preamble_nodes), 1)
        self.assertEqual(
            preamble_nodes[0].text,
            "Considerando a necessidade de fixar regras comuns.",
        )
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Objeto")

    def test_locally_flagged_page_with_real_body_text_is_not_quarantined(self) -> None:
        """Ensure defensive cleanup does not delete valid normative body text."""
        root = build_structure_from_extracted_pages(
            [
                ExtractedPage(
                    page_number=1,
                    text=(
                        "Artigo 2 - Pagamento\n"
                        "O pagamento da propina e efetuado em prestacoes mensais."
                    ),
                    selected_mode="text",
                    quality_score=8.0,
                    corruption_flags=["low_alpha_ratio"],
                )
            ]
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Pagamento")
        self.assertEqual(
            articles[0].text,
            "O pagamento da propina e efetuado em prestacoes mensais.",
        )


if __name__ == "__main__":
    unittest.main()
