"""Focused regression tests for Task 3 normalizer residue cleanup."""

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


if __name__ == "__main__":
    unittest.main()
