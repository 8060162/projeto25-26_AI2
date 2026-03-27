"""Focused regression tests for chunk-quality validator acceptance gaps."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import Chunk
from Chunking.quality.chunk_quality_validator import ChunkQualityValidator


def build_chunk(**overrides: object) -> Chunk:
    """Build a minimal valid chunk for validator-focused tests."""
    payload = {
        "chunk_id": "validator_chunk",
        "doc_id": "validator_doc",
        "strategy": "article_smart",
        "text": "O presente regulamento aplica-se a todos os cursos.",
        "text_for_embedding": "O presente regulamento aplica-se a todos os cursos.",
        "page_start": 1,
        "page_end": 1,
        "source_node_type": "ARTICLE",
        "source_node_label": "ART_1",
        "hierarchy_path": ["DOCUMENT:DOCUMENT", "ARTICLE:ART_1"],
        "chunk_reason": "direct_article",
        "metadata": {
            "article_number": "1",
            "article_title": "Objeto",
            "document_part": "regulation_body",
        },
    }
    payload.update(overrides)
    return Chunk(**payload)


class ChunkQualityValidatorRegressionTests(unittest.TestCase):
    """Protect validator behavior against known chunk-quality acceptance gaps."""

    def test_validator_flags_standalone_url_line_as_footnote_leakage(self) -> None:
        """Ensure naked editorial URLs no longer pass as valid chunk text."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="url_leak",
            text=(
                "A nao publicacao dos resultados depende da regularizacao da situacao.\n"
                "https://portal.isep.ipp.pt/\n"
                "A emissao de certidoes fica igualmente condicionada."
            ),
        )

        report = validator.validate_chunk(chunk)

        self.assertFalse(report["is_valid"])
        self.assertIn("note_or_footnote_in_text", report["issue_codes"])

    def test_validator_flags_document_title_residue_on_chunk_edge(self) -> None:
        """Ensure standalone document-title residue blocks acceptance."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="title_residue",
            chunk_reason="preamble_group",
            source_node_type="PREAMBLE",
            source_node_label="PREAMBLE",
            hierarchy_path=["DOCUMENT:DOCUMENT", "PREAMBLE:PREAMBLE"],
            metadata={
                "document_part": "dispatch_or_intro",
                "source_span_type": "preamble",
            },
            text=(
                "Determino a aprovacao do regulamento anexo ao presente despacho.\n"
                "Regulamento P.PORTO/P -005/2023"
            ),
            text_for_embedding=(
                "Preamble\n\nDetermino a aprovacao do regulamento anexo ao presente despacho.\n"
                "Regulamento P.PORTO/P -005/2023"
            ),
        )

        report = validator.validate_chunk(chunk)

        self.assertFalse(report["is_valid"])
        self.assertIn("document_title_residue_in_text", report["issue_codes"])

    def test_validator_flags_dash_started_split_fragment_as_low_autonomy(self) -> None:
        """Ensure split chunks that open mid-definition no longer pass validation."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="dash_fragment",
            chunk_reason="fallback_legal_signal_split",
            text=(
                "- E o ato pelo qual o estudante formaliza a sua intencao de realizar um exame.\n\n"
                "15 - Regime de precedencias - Regime que estabelece condicoes adicionais."
            ),
            metadata={
                "article_number": "2",
                "article_title": "Definicoes",
                "document_part": "regulation_body",
                "part_index": 5,
                "part_count": 6,
            },
        )

        report = validator.validate_chunk(chunk)

        self.assertFalse(report["is_valid"])
        self.assertIn("low_semantic_autonomy_chunk", report["issue_codes"])

    def test_validator_does_not_treat_regular_body_text_as_document_title_residue(self) -> None:
        """Ensure ordinary body references to regulations stay acceptable."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="regular_body_reference",
            text=(
                "O regulamento especifico da escola aplica-se apenas quando exista norma complementar."
            ),
        )

        report = validator.validate_chunk(chunk)

        self.assertTrue(report["is_valid"])


if __name__ == "__main__":
    unittest.main()
