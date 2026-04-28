"""Regression tests for embedding text preparation behavior."""

from __future__ import annotations

import unittest

from embedding.models import EmbeddingInputRecord
from embedding.text_builder import build_embedding_text


class EmbeddingTextBuilderTests(unittest.TestCase):
    """Protect conservative cleanup and metadata-free embedding text behavior."""

    def test_build_embedding_text_applies_conservative_cleanup(self) -> None:
        """Ensure wrapping noise is cleaned without removing paragraph structure."""

        record = EmbeddingInputRecord(
            chunk_id="chunk_1",
            doc_id="doc_1",
            text="  First regula-\ntion line\ncontinues here.\n\nSecond paragraph.  ",
            metadata={},
        )

        self.assertEqual(
            build_embedding_text(record),
            "First regulation line continues here.\n\nSecond paragraph.",
        )

    def test_build_embedding_text_keeps_structural_metadata_out_of_text(
        self,
    ) -> None:
        """Ensure legal metadata does not pollute the provider text payload."""

        record = EmbeddingInputRecord(
            chunk_id="chunk_2",
            doc_id="doc_1",
            text="Body text about eight installments.",
            metadata={
                "document_title": "Regulation A",
                "article_number": "7",
                "article_title": "Deadlines",
            },
        )

        self.assertEqual(
            build_embedding_text(record),
            "Body text about eight installments.",
        )

    def test_build_embedding_text_keeps_split_payment_anchors_in_metadata_only(
        self,
    ) -> None:
        """Ensure the real payment-plan body stays free of structural anchors."""

        record = EmbeddingInputRecord(
            chunk_id="chunk_propinas_5_2",
            doc_id="reg_propinas",
            text=(
                "Para estudante internacional, em 8 prestacoes, com percentagens "
                "e datas-limite de pagamento."
            ),
            metadata={
                "document_title": "Regulamento de Propinas",
                "article_number": "5",
                "article_title": "PLANO GERAL DE PAGAMENTO DE PROPINAS",
            },
        )

        self.assertEqual(
            build_embedding_text(record),
            (
                "Para estudante internacional, em 8 prestacoes, com percentagens "
                "e datas-limite de pagamento."
            ),
        )

    def test_build_embedding_text_remains_clean_when_metadata_is_absent(self) -> None:
        """Ensure plain chunks stay prefix-free when no structural metadata exists."""

        record = EmbeddingInputRecord(
            chunk_id="chunk_4",
            doc_id="doc_1",
            text="  Plain\nchunk text.  ",
            metadata={},
        )

        self.assertEqual(build_embedding_text(record), "Plain chunk text.")


if __name__ == "__main__":
    unittest.main()
