"""Focused regression tests for normalizer residue and reconstruction cleanup."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import PageText
from Chunking.cleaning.normalizer import TextNormalizer


class TextNormalizerQualityRegressionTests(unittest.TestCase):
    """Protect conservative normalization against recurring residue leakage."""

    def test_early_document_cover_and_signoff_residue_is_trimmed(self) -> None:
        """Ensure early cover/title and sign-off residue does not survive normalization."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Escola Superior de Tecnologia e Gestao do Instituto Politecnico do Porto.\n"
                        "Considerando a necessidade de atualizar o regulamento.\n"
                        "Paulo Pereira\n"
                        "Regulamento de\n"
                    ),
                )
            ]
        )

        self.assertEqual(
            normalized.pages[0].text,
            "Considerando a necessidade de atualizar o regulamento.",
        )
        self.assertEqual(
            normalized.dropped_lines_report["institutional_cover_title_line"],
            1,
        )
        self.assertEqual(
            normalized.dropped_lines_report["early_document_tail_residue_line"],
            2,
        )

    def test_trailing_formula_garbage_block_is_removed(self) -> None:
        """Ensure trailing OCR-like formula garbage is removed before parsing."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Artigo 8 - Creditacao\n"
                        "1. Quando aplicavel, as unidades curriculares creditadas conservam as classificacoes obtidas.\n"
                        "2. Quando se trate de unidades curriculares realizadas em estabelecimentos de ensino superior "
                        "estrangeiros, aplicar-se-a a seguinte formula de calculo:\n"
                        "_\n10 1\n_\n_\nIPP\nCIESe\nCSESe lmp\nC\nCSESe lMp\n"
                        "CSESe lmp\nSection\nparagraph\nsymbol\nnoise"
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn(
            "2. Quando se trate de unidades curriculares realizadas em estabelecimentos de ensino superior estrangeiros",
            page_text,
        )
        self.assertNotIn("CSESe lmp", page_text)
        self.assertNotIn("CSESe lMp", page_text)
        self.assertNotIn("Section", page_text)
        self.assertNotIn("symbol", page_text)
        self.assertEqual(
            normalized.dropped_lines_report["trailing_garbled_block_line"],
            14,
        )

    def test_local_formula_garbage_block_is_removed_without_losing_context(self) -> None:
        """Ensure interior garbled formula residue is removed while useful prose survives."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Artigo 12 - Formula final\n"
                        "A classificacao final e calculada pela seguinte formula:\n"
                        "_\n10 1\nIPP\nCSESe lmp\nonde:\n"
                        "CF = classificacao final do estudante.\n"
                        "A classificacao e arredondada a unidade mais proxima."
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn("Artigo 12 - Formula final", page_text)
        self.assertIn(
            "A classificacao final e calculada pela seguinte formula:",
            page_text,
        )
        self.assertIn(
            "CF = classificacao final do estudante.",
            page_text,
        )
        self.assertIn(
            "A classificacao e arredondada a unidade mais proxima.",
            page_text,
        )
        self.assertNotIn("CSESe lmp", page_text)
        self.assertNotIn("IPP", page_text)
        self.assertEqual(
            normalized.dropped_lines_report["local_garbled_block_line"],
            5,
        )

    def test_normative_numbered_and_lettered_lines_are_not_removed_as_garbled(self) -> None:
        """Ensure conservative cleanup does not strip valid normative list items."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Artigo 5 - Creditos\n"
                        "1. Podem ser creditadas unidades curriculares realizadas noutro ciclo de estudos.\n"
                        "2. A creditação depende de deliberação do órgão competente.\n"
                        "a) A decisão deve identificar as unidades curriculares abrangidas.\n"
                        "b) A decisão deve indicar a classificação atribuída."
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn(
            "1. Podem ser creditadas unidades curriculares realizadas noutro ciclo de estudos.",
            page_text,
        )
        self.assertIn(
            "2. A creditação depende de deliberação do órgão competente.",
            page_text,
        )
        self.assertIn(
            "a) A decisão deve identificar as unidades curriculares abrangidas.",
            page_text,
        )
        self.assertIn(
            "b) A decisão deve indicar a classificação atribuída.",
            page_text,
        )
        self.assertNotIn("local_garbled_block_line", normalized.dropped_lines_report)
        self.assertNotIn(
            "trailing_garbled_block_line",
            normalized.dropped_lines_report,
        )

    def test_malformed_short_date_expression_is_repaired(self) -> None:
        """Ensure short date expressions recover missing spacing conservatively."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Considerando que o projeto de regulamento foi objeto de consulta publica, "
                        "nos termos do artigo 101.o do Codigo do Procedimento Administrativo, "
                        "atraves do Despacho P.PORTO/P-036/2025, de 13de junho."
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn("de 13 de junho.", page_text)
        self.assertNotIn("13de junho", page_text)

    def test_malformed_numbering_prefix_is_normalized(self) -> None:
        """Ensure fused numbering prefixes become valid legal list markers."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Artigo 1 - Ambito\n"
                        "1-— O presente regulamento e aplicavel.\n"
                        "1-—A pratica de fraude implica anulacao.\n"
                        "2 — A outras formacoes aplica-se o disposto em capitulo proprio."
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn("1 — O presente regulamento e aplicavel.", page_text)
        self.assertIn("1 — A pratica de fraude implica anulacao.", page_text)
        self.assertNotIn("1-—", page_text)

    def test_broken_hyphenation_is_repaired_conservatively_during_normalization(
        self,
    ) -> None:
        """Ensure local lexical breaks are repaired without altering valid wording."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "O regula-\n"
                        "mento aplica-se aos estudantes inscritos.\n"
                        "O pedido deve fazer -se por requerimento escrito."
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn(
            "O regulamento aplica-se aos estudantes inscritos.",
            page_text,
        )
        self.assertIn(
            "O pedido deve fazer-se por requerimento escrito.",
            page_text,
        )
        self.assertNotIn("regula-\nmento", page_text)
        self.assertNotIn("fazer -se", page_text)

    def test_adjacent_prose_continuation_is_merged_without_crossing_legal_item_boundary(
        self,
    ) -> None:
        """Ensure broken prose carry-over is merged while structural items stay separate."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Artigo 7 - Propinas\n"
                        "O pagamento pode ser efetuado em prestacoes\n"
                        "quando autorizado pelo orgao competente.\n"
                        "2. O incumprimento determina a aplicacao de juros."
                    ),
                )
            ]
        )

        normalized_lines = normalized.pages[0].text.splitlines()

        self.assertEqual(normalized_lines[0], "Artigo 7 - Propinas")
        self.assertEqual(
            normalized_lines[1],
            "O pagamento pode ser efetuado em prestacoes quando autorizado pelo orgao competente.",
        )
        self.assertEqual(
            normalized_lines[2],
            "2. O incumprimento determina a aplicacao de juros.",
        )

    def test_inline_garbled_residue_is_removed_without_losing_surrounding_prose(
        self,
    ) -> None:
        """Ensure inline garbled residue embedded in prose does not survive normalization."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Artigo 9 - Formula\n"
                        "A classificacao final corresponde a abc©Þ§£Ý¤¦¬ conforme deliberacao do conselho tecnico-cientifico.\n"
                        "1. O resultado e expresso na escala inteira."
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn(
            "A classificacao final corresponde a conforme deliberacao do conselho tecnico-cientifico.",
            page_text,
        )
        self.assertIn(
            "1. O resultado e expresso na escala inteira.",
            page_text,
        )
        self.assertNotIn("abc©Þ§£Ý¤¦¬", page_text)

    def test_leading_garbled_continuation_fragment_is_dropped(self) -> None:
        """Ensure recovered leading garbage does not survive as fake continuation prose."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "A mudanca de par instituicao/curso para os ciclos de estudos conducentes ao grau de licenciado em\n"
                        "Musica e em Teatro da Escola Superior de Musica e Artes do Espetaculo (ESMAE) esta ainda\n"
                        "(60$(realizadas no ano da candidatura, nos termos do regulamento aplicavel a essas provas."
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn(
            "Musica e em Teatro da Escola Superior de Musica e Artes do Espetaculo (ESMAE) esta ainda",
            page_text,
        )
        self.assertNotIn("realizadas no ano da candidatura", page_text)
        self.assertEqual(
            normalized.dropped_lines_report["displaced_continuation_fragment_line"],
            1,
        )

    def test_displaced_continuation_fragment_is_dropped_between_prose_and_legal_item(
        self,
    ) -> None:
        """Ensure displaced continuation residue does not bridge prose into a numbered item."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Artigo 4 - Requerimento\n"
                        "O requerimento deve indicar o numero de creditos ja obtidos\n"
                        "e o plano de estudos pretendido.\n"
                        "(60$(realizadas no ano anterior, nos termos do regulamento aplicavel.\n"
                        "2. O conselho tecnico-cientifico decide sobre o pedido."
                    ),
                )
            ]
        )

        normalized_lines = normalized.pages[0].text.splitlines()

        self.assertEqual(normalized_lines[0], "Artigo 4 - Requerimento")
        self.assertEqual(
            normalized_lines[1],
            "O requerimento deve indicar o numero de creditos ja obtidos e o plano de estudos pretendido.",
        )
        self.assertEqual(
            normalized_lines[2],
            "2. O conselho tecnico-cientifico decide sobre o pedido.",
        )
        self.assertNotIn("realizadas no ano anterior", normalized.pages[0].text)
        self.assertEqual(
            normalized.dropped_lines_report["displaced_continuation_fragment_line"],
            1,
        )

    def test_isolated_formula_residue_lines_are_removed_between_explanatory_lines(
        self,
    ) -> None:
        """Ensure short formula residue lines do not survive between explanatory prose."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Artigo 21 - Classificacao\n"
                        "Quando se trate de unidades curriculares realizadas em estabelecimentos de ensino superior "
                        "estrangeiros, aplicar-se-a a seguinte formula de calculo:\n"
                        "- Classificacao da unidade curricular no P.PORTO, arredondada as unidades.\n"
                        "CIESe\n"
                        "- Classificacao da unidade curricular na Instituicao de Ensino Superior Estrangeira.\n"
                        "_\n"
                        "CSESe lmp\n"
                        "- Classificacao minima para obtencao de aprovacao na escala de classificacao do Sistema "
                        "de Ensino Superior Estrangeiro.\n"
                        "_\n"
                        "CSESe lMp\n"
                        "- Classificacao maxima na escala de classificacao do Sistema de Ensino Superior Estrangeiro."
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn(
            "- Classificacao da unidade curricular no P.PORTO, arredondada as unidades.",
            page_text,
        )
        self.assertIn(
            "- Classificacao minima para obtencao de aprovacao na escala de classificacao do Sistema de Ensino Superior Estrangeiro.",
            page_text,
        )
        self.assertNotIn("CIESe", page_text)
        self.assertNotIn("CSESe lmp", page_text)
        self.assertNotIn("CSESe lMp", page_text)
        self.assertNotIn("\n_\n", page_text)
        self.assertEqual(
            normalized.dropped_lines_report["isolated_formula_residue_line"],
            5,
        )

    def test_local_garbled_block_is_removed_between_useful_prose_regions(self) -> None:
        """Ensure local garbled blocks are removed when bounded by useful prose on both sides."""
        normalized = TextNormalizer().normalize(
            [
                PageText(
                    page_number=1,
                    text=(
                        "Artigo 6 - Creditacao\n"
                        "A creditacao depende de deliberacao do orgao competente.\n"
                        "_\n"
                        "10 1\n"
                        "CSESe lmp\n"
                        "###\n"
                        "O pedido deve ser acompanhado dos comprovativos exigidos."
                    ),
                )
            ]
        )

        page_text = normalized.pages[0].text

        self.assertIn("Artigo 6 - Creditacao", page_text)
        self.assertIn(
            "A creditacao depende de deliberacao do orgao competente.",
            page_text,
        )
        self.assertIn(
            "O pedido deve ser acompanhado dos comprovativos exigidos.",
            page_text,
        )
        self.assertNotIn("CSESe lmp", page_text)
        self.assertNotIn("10 1", page_text)
        self.assertNotIn("###", page_text)
        self.assertEqual(
            normalized.dropped_lines_report["local_garbled_block_line"],
            4,
        )


if __name__ == "__main__":
    unittest.main()
