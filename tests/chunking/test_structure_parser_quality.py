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
