"""Regression tests for parser fixes that affect article_smart chunk quality."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import PageText, StructuralNode
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


def collect_articles(node: StructuralNode) -> list[StructuralNode]:
    """Collect ARTICLE nodes recursively for focused parser assertions."""
    articles: list[StructuralNode] = []

    if node.node_type == "ARTICLE":
        articles.append(node)

    for child in node.children:
        articles.extend(collect_articles(child))

    return articles


class StructureParserQualityRegressionTests(unittest.TestCase):
    """Protect parser behavior required by the chunk-quality implementation plan."""

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


if __name__ == "__main__":
    unittest.main()
