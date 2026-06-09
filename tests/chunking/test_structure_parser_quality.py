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

    def test_preamble_leading_tail_fragment_is_trimmed_before_opening(self) -> None:
        """Ensure leaked title tails do not remain at the start of PREAMBLE."""
        root = build_structure_from_lines(
            (
                "de Tecnologia e Gestao do Instituto Politecnico do Porto.\n"
                "Considerando a necessidade de atualizar o regulamento.\n"
                "Artigo 1 - Objeto\n"
                "O presente regulamento define o objeto."
            )
        )

        preamble_nodes = collect_nodes_by_type(root, "PREAMBLE")

        self.assertEqual(len(preamble_nodes), 1)
        self.assertEqual(
            preamble_nodes[0].text,
            "Considerando a necessidade de atualizar o regulamento.",
        )

    def test_document_start_boundaries_remain_separate_from_clean_article(self) -> None:
        """Ensure document-start zones stay separated without false article flags."""
        root = build_structure_from_lines(
            (
                "POLITECNICO DO PORTO\n"
                "Considerando a necessidade de atualizar o regulamento.\n"
                "Artigo 1 - Objeto\n"
                "O presente regulamento define o objeto."
            )
        )

        self.assertEqual(
            [child.node_type for child in root.children],
            ["FRONT_MATTER", "PREAMBLE", "ARTICLE"],
        )
        self.assertEqual(root.children[0].text, "POLITECNICO DO PORTO")
        self.assertEqual(
            root.children[1].text,
            "Considerando a necessidade de atualizar o regulamento.",
        )
        self.assertEqual(root.children[2].title, "Objeto")
        self.assertFalse(root.children[2].metadata["is_structurally_incomplete"])
        self.assertEqual(root.children[2].metadata["truncation_signals"], [])
        self.assertEqual(root.children[2].metadata["integrity_warnings"], [])

    def test_broken_dispatch_item_block_is_trimmed_from_preamble_tail(self) -> None:
        """Ensure broken approval-list tails do not remain in PREAMBLE."""
        root = build_structure_from_lines(
            (
                "Considerando que o projeto foi submetido a consulta publica.\n"
                "Determino, no uso das competencias previstas nos Estatutos do\n"
                "a) A aprovacao do regulamento anexo ao presente despacho e que\n"
                "c) A revogacao do despacho anterior.\n"
                "Artigo 1 - Objeto\n"
                "O presente regulamento aplica-se a todos os cursos."
            )
        )

        preamble_nodes = collect_nodes_by_type(root, "PREAMBLE")

        self.assertEqual(len(preamble_nodes), 1)
        self.assertEqual(
            preamble_nodes[0].text,
            "Considerando que o projeto foi submetido a consulta publica.",
        )

    def test_header_title_can_continue_on_next_line_before_body(self) -> None:
        """Ensure split short titles are merged before body routing starts."""
        root = build_structure_from_lines(
            (
                "Artigo 19 - Entrada\n"
                "em vigor O presente regulamento entra em vigor no dia seguinte."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Entrada em vigor")
        self.assertEqual(
            articles[0].text,
            "O presente regulamento entra em vigor no dia seguinte.",
        )

    def test_inline_title_body_split_prefers_full_short_title(self) -> None:
        """Ensure inline splitting does not stop at a lowercase title tail."""
        root = build_structure_from_lines(
            (
                "Artigo 19.o\n"
                "Entrada em vigor O presente regulamento entra em vigor no dia seguinte."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Entrada em vigor")
        self.assertEqual(
            articles[0].text,
            "O presente regulamento entra em vigor no dia seguinte.",
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
        self.assertFalse(articles[0].metadata["is_structurally_incomplete"])
        self.assertEqual(articles[0].metadata["truncation_signals"], [])
        self.assertEqual(articles[0].metadata["integrity_warnings"], [])

    def test_truncated_article_before_next_header_is_marked_incomplete(self) -> None:
        """Ensure abrupt truncated article endings become explicit parser metadata."""
        root = build_structure_from_lines(
            (
                "Artigo 7 - Pagamento\n"
                "O estudante deve efetuar o pagamento da propina no prazo e\n"
                "Artigo 8 - Incumprimento\n"
                "O incumprimento determina a aplicacao das medidas previstas."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 2)
        self.assertTrue(articles[0].metadata["is_structurally_incomplete"])
        self.assertEqual(
            articles[0].metadata["truncation_signals"],
            ["suspicious_truncated_ending", "abrupt_transition_to_article"],
        )
        self.assertEqual(articles[0].metadata["integrity_warnings"], [])
        self.assertFalse(articles[1].metadata["is_structurally_incomplete"])

    def test_complete_article_is_not_marked_incomplete(self) -> None:
        """Ensure ordinary article closure does not trigger integrity noise."""
        root = build_structure_from_lines(
            (
                "Artigo 3 - Ambito\n"
                "O presente regulamento aplica-se a todos os cursos.\n"
                "Artigo 4 - Entrada em vigor\n"
                "O regulamento entra em vigor no dia seguinte a sua publicacao."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 2)
        self.assertFalse(articles[0].metadata["is_structurally_incomplete"])
        self.assertEqual(articles[0].metadata["truncation_signals"], [])
        self.assertEqual(articles[0].metadata["integrity_warnings"], [])
        self.assertFalse(articles[1].metadata["is_structurally_incomplete"])

    def test_missing_title_boundary_is_reported_as_integrity_warning(self) -> None:
        """Ensure strong unrecovered title/body patterns are exposed explicitly."""
        root = build_structure_from_lines(
            (
                "Artigo 9 - Entrada em vigor\n"
                "REGIME TRANSITORIO PARA ESTUDANTES DOS CURSOS DE MESTRADO E POS GRADUACAO EM FUNCIONAMENTO NO ANO LETIVO EM CURSO\n"
                "O presente regulamento entra em vigor no dia seguinte."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertTrue(articles[0].metadata["is_structurally_incomplete"])
        self.assertEqual(articles[0].metadata["truncation_signals"], [])
        self.assertEqual(
            articles[0].metadata["integrity_warnings"],
            ["possible_unrecovered_title_body_boundary"],
        )

    def test_orphaned_decimal_numbering_is_reported_as_integrity_warning(self) -> None:
        """Ensure sub-numbering without its parent unit is marked as suspicious."""
        root = build_structure_from_lines(
            (
                "Artigo 19 - Regime aplicavel\n"
                "Os estudantes abrangidos pelo regime transitório mantêm os direitos já constituídos.\n"
                "1.1. No ato de inscrição, devem confirmar os dados pessoais junto dos serviços."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertTrue(articles[0].metadata["is_structurally_incomplete"])
        self.assertEqual(articles[0].metadata["truncation_signals"], [])
        self.assertEqual(
            articles[0].metadata["integrity_warnings"],
            ["possible_orphaned_numbered_continuation"],
        )

    def test_interrupted_definition_capture_is_reported_as_integrity_warning(self) -> None:
        """Ensure colon-ended sections without subordinate content are marked."""
        root = build_structure_from_lines(
            (
                "Artigo 12 - Definicoes\n"
                "1. Para efeitos do presente regulamento, considera-se:\n"
                "2. O estudante internacional beneficia do regime previsto na lei."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertTrue(articles[0].metadata["is_structurally_incomplete"])
        self.assertEqual(articles[0].metadata["truncation_signals"], [])
        self.assertEqual(
            articles[0].metadata["integrity_warnings"],
            ["possible_interrupted_definition_capture"],
        )

    def test_complete_definition_capture_with_contiguous_alineas_is_not_flagged(self) -> None:
        """Ensure valid definition lists do not trigger continuity warnings."""
        root = build_structure_from_lines(
            (
                "Artigo 12 - Definicoes\n"
                "1. Para efeitos do presente regulamento, considera-se:\n"
                "a) Estudante internacional o estudante abrangido pelo regime legal aplicavel;\n"
                "b) Estudante em mobilidade o estudante inscrito ao abrigo de programa especifico.\n"
                "2. O disposto no numero anterior aplica-se sem prejuizo da legislacao em vigor."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertFalse(articles[0].metadata["is_structurally_incomplete"])
        self.assertEqual(articles[0].metadata["truncation_signals"], [])
        self.assertEqual(articles[0].metadata["integrity_warnings"], [])

    def test_decimal_subnumbering_after_colon_is_not_reported_as_interrupted(self) -> None:
        """Ensure decimal subsection continuity is not treated as missing content."""
        root = build_structure_from_lines(
            (
                "Artigo 5 - Plano geral de pagamento de propinas\n"
                "1. A propina e devida no momento da matricula.\n"
                "2. A propina pode ser paga em prestacoes:\n"
                "2.1 Para estudante nacional, em dez prestacoes mensais.\n"
                "2.2 Para estudante internacional, em oito prestacoes mensais.\n"
                "3. O pagamento deve respeitar os prazos fixados pela escola."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertFalse(articles[0].metadata["is_structurally_incomplete"])
        self.assertEqual(articles[0].metadata["truncation_signals"], [])
        self.assertEqual(articles[0].metadata["integrity_warnings"], [])

    def test_broken_lettered_enumeration_is_reported_as_integrity_warning(self) -> None:
        """Ensure missing alinea continuity is exposed in article metadata."""
        root = build_structure_from_lines(
            (
                "Artigo 14 - Exclusao\n"
                "1. O candidato e excluido quando:\n"
                "a) Nao apresente a documentacao exigida;\n"
                "c) Preste falsas declaracoes.\n"
                "2. A decisao final e notificada por via eletronica."
            )
        )

        articles = collect_articles(root)

        self.assertEqual(len(articles), 1)
        self.assertTrue(articles[0].metadata["is_structurally_incomplete"])
        self.assertEqual(articles[0].metadata["truncation_signals"], [])
        self.assertEqual(
            articles[0].metadata["integrity_warnings"],
            ["possible_broken_lettered_enumeration"],
        )


if __name__ == "__main__":
    unittest.main()
