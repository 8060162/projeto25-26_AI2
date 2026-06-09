"""Regression tests for canonical document identifier generation."""

from __future__ import annotations

import unittest
from pathlib import Path

from Chunking.pipeline import _validate_unique_document_ids
from Chunking.utils.text import build_canonical_document_id


class CanonicalDocumentIdTests(unittest.TestCase):
    """Protect concise and stable document IDs used by chunks and embeddings."""

    def test_builds_short_id_for_pporto_dispatch_reference(self) -> None:
        """Ensure long P.PORTO dispatch file names keep only the official ID."""

        document_id = build_canonical_document_id(
            "Despacho P.PORTO-P-029-2024_Regulamento Regimes Reingresso "
            "e de Mudança de Par Instituição_signed (2)"
        )

        self.assertEqual(document_id, "Despacho_P_PORTO_P_029_2024")

    def test_builds_short_id_for_dispatch_number_reference(self) -> None:
        """Ensure dispatch numbers are preserved without the full title suffix."""

        document_id = build_canonical_document_id(
            "Despacho nº 7088-2023 - Regulamento de Matrículas e Inscrições "
            "do Instituto Politécnico do Porto"
        )

        self.assertEqual(document_id, "Despacho_no_7088_2023")

    def test_builds_short_id_for_regulation_number_reference(self) -> None:
        """Ensure regulation numbers are preserved without the full title suffix."""

        document_id = build_canonical_document_id(
            "Regulamento nº 633-2024_Alteração ao Regulamento de Avaliação "
            "de Aproveitamento"
        )

        self.assertEqual(document_id, "Regulamento_no_633_2024")

    def test_falls_back_to_limited_slug_for_unknown_document_name(self) -> None:
        """Ensure unknown names still produce bounded filesystem-safe IDs."""

        document_id = build_canonical_document_id("Documento académico com acentuação")

        self.assertEqual(document_id, "Documento_academico_com_acentuacao")

    def test_duplicate_canonical_document_ids_are_rejected(self) -> None:
        """Ensure duplicate official IDs do not overwrite chunk output folders."""

        with self.assertRaisesRegex(ValueError, "same document id"):
            _validate_unique_document_ids(
                [
                    Path("Despacho P.PORTO-P-029-2024_Regulamento A.pdf"),
                    Path("Despacho P.PORTO-P-029-2024_Regulamento B.pdf"),
                ]
            )


if __name__ == "__main__":
    unittest.main()
