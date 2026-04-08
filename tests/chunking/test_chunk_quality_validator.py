"""Focused regression tests for chunk-quality validator acceptance gaps."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import Chunk
from Chunking.config.settings import PipelineSettings
from Chunking.quality.chunk_quality_validator import ChunkQualityValidator


def build_chunk(**overrides: object) -> Chunk:
    """Build a minimal valid chunk for validator-focused tests."""
    payload = {
        "chunk_id": "validator_chunk",
        "doc_id": "validator_doc",
        "strategy": "article_smart",
        "text": "O presente regulamento aplica-se a todos os cursos.",
        "meta_text": "O presente regulamento aplica-se a todos os cursos.",
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
            meta_text=(
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

    def test_validator_blocks_short_numbered_split_fragment(self) -> None:
        """Ensure short split fragments do not pass only because they fit the size limit."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="short_numbered_fragment",
            chunk_reason="fallback_legal_signal_split",
            text="15 - Regime de precedencias.",
            metadata={
                "article_number": "2",
                "article_title": "Definicoes",
                "document_part": "regulation_body",
                "part_index": 4,
                "part_count": 6,
            },
        )

        report = validator.validate_chunk(chunk)

        self.assertFalse(report["is_valid"])
        self.assertIn("problematic_undersized_chunk", report["issue_codes"])

    def test_validator_respects_configured_split_fragment_thresholds(self) -> None:
        """Ensure split-fragment rejection thresholds remain configurable."""
        chunk = build_chunk(
            chunk_id="configurable_split_fragment",
            chunk_reason="fallback_legal_signal_split",
            text="Regime geral muito aplicavel.",
            metadata={
                "article_number": "2",
                "article_title": "Definicoes",
                "document_part": "regulation_body",
                "part_index": 2,
                "part_count": 4,
            },
        )

        default_report = ChunkQualityValidator().validate_chunk(chunk)
        configured_report = ChunkQualityValidator(
            PipelineSettings(
                validator_problematic_split_chunk_max_chars=20,
                validator_low_autonomy_min_word_count=4,
            )
        ).validate_chunk(chunk)

        self.assertFalse(default_report["is_valid"])
        self.assertIn("problematic_undersized_chunk", default_report["issue_codes"])
        self.assertTrue(configured_report["is_valid"])

    def test_validator_blocks_structurally_incomplete_article_chunk(self) -> None:
        """Ensure parser-declared truncation metadata blocks unsafe article chunks."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="structural_incomplete",
            text=(
                "As situacoes omissas sao resolvidas pelo conselho tecnico-cientifico "
                "nos termos do regulamento aplicavel."
            ),
            metadata={
                "article_number": "18",
                "article_title": "Duvidas e omissoes",
                "document_part": "regulation_body",
                "is_structurally_incomplete": True,
                "truncation_signals": [
                    "suspicious_truncated_ending",
                    "abrupt_transition_to_article",
                ],
                "integrity_warnings": [],
            },
        )

        report = validator.validate_chunk(chunk)

        self.assertFalse(report["is_valid"])
        self.assertIn("structurally_incomplete_source_article", report["issue_codes"])

    def test_validator_blocks_parser_integrity_warning_without_truncation_signal(self) -> None:
        """Ensure parser integrity warnings still block unsafe article chunks."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="integrity_warning_only",
            text=(
                "Para efeitos do presente regulamento, considera-se:\n"
                "2. O estudante internacional beneficia do regime previsto na lei."
            ),
            metadata={
                "article_number": "12",
                "article_title": "Definicoes",
                "document_part": "regulation_body",
                "is_structurally_incomplete": True,
                "truncation_signals": [],
                "integrity_warnings": ["possible_interrupted_definition_capture"],
            },
        )

        report = validator.validate_chunk(chunk)

        self.assertFalse(report["is_valid"])
        self.assertIn("structurally_incomplete_source_article", report["issue_codes"])

    def test_validator_blocks_broken_enumeration_warning_without_truncation_signal(self) -> None:
        """Ensure continuity warnings for malformed enumeration are treated as blocking."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="broken_enumeration_warning",
            text=(
                "1. O candidato e excluido quando:\n"
                "a) Nao apresente a documentacao exigida;\n"
                "c) Preste falsas declaracoes."
            ),
            metadata={
                "article_number": "14",
                "article_title": "Exclusao",
                "document_part": "regulation_body",
                "is_structurally_incomplete": True,
                "truncation_signals": [],
                "integrity_warnings": ["possible_broken_lettered_enumeration"],
            },
        )

        report = validator.validate_chunk(chunk)

        self.assertFalse(report["is_valid"])
        self.assertIn("structurally_incomplete_source_article", report["issue_codes"])

    def test_validator_blocks_chunk_with_dominant_symbolic_garbage(self) -> None:
        """Ensure chunks dominated by formula-like residue are rejected."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="dominant_garbage",
            text=(
                "++==//__--**\n"
                "00112233445566778899\n"
                "A decisao final depende de ato formal."
            ),
        )

        report = validator.validate_chunk(chunk)

        self.assertFalse(report["is_valid"])
        self.assertIn("dominant_non_semantic_content", report["issue_codes"])

    def test_validator_blocks_grouped_chunk_with_broken_definition_capture(self) -> None:
        """Ensure damaged grouped definitions no longer pass validation."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="broken_grouped_definition",
            source_node_type="LETTERED_GROUP",
            source_node_label="d,e,f,g",
            chunk_reason="grouped_lettered_items",
            text=(
                "«Escala de classificacao portuguesa» aquela a que se refere o artigo 15.o do Decreto-Lei\n\n"
                "«Regime geral de acesso» o regime de acesso e ingresso regulado pelo Decreto-Lei\n"
                "junho;\n\n"
                "«Matricula» e o ato pelo qual o estudante concretiza o ingresso num curso.\n\n"
                "«Inscricao» e o ato pelo qual o estudante, tendo matricula valida num curso, adquire "
                "o direito de frequentar as unidades curriculares em que se inscreve."
            ),
            metadata={
                "article_number": "3",
                "article_title": "Conceitos",
                "document_part": "regulation_body",
                "lettered_labels": ["d", "e", "f", "g"],
                "source_span_type": "article_lettered_group",
            },
        )

        report = validator.validate_chunk(chunk)

        self.assertFalse(report["is_valid"])
        self.assertIn("broken_grouped_legal_unit", report["issue_codes"])

    def test_validator_accepts_grouped_chunk_with_complete_definitions(self) -> None:
        """Ensure coherent grouped definitions remain acceptable."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="complete_grouped_definition",
            source_node_type="LETTERED_GROUP",
            source_node_label="a,b,c",
            chunk_reason="grouped_lettered_items",
            text=(
                "«Reingresso» e o ato pelo qual um estudante retoma os estudos na mesma instituicao;\n\n"
                "«Mudanca de par instituicao/curso» e o ato pelo qual um estudante se matricula em par "
                "instituicao/curso diferente;\n\n"
                "«Creditos» sao os creditos segundo o ECTS, cuja atribuicao e regulada pelo diploma aplicavel."
            ),
            metadata={
                "article_number": "3",
                "article_title": "Conceitos",
                "document_part": "regulation_body",
                "lettered_labels": ["a", "b", "c"],
                "source_span_type": "article_lettered_group",
            },
        )

        report = validator.validate_chunk(chunk)

        self.assertTrue(report["is_valid"])

    def test_validator_accepts_short_complete_legal_chunk(self) -> None:
        """Ensure short but complete legal text remains acceptable."""
        validator = ChunkQualityValidator()
        chunk = build_chunk(
            chunk_id="short_complete_legal",
            text="Aplica-se a todos os cursos.",
            meta_text="Aplica-se a todos os cursos.",
            metadata={
                "article_number": "4",
                "article_title": "Ambito",
                "document_part": "regulation_body",
                "is_structurally_incomplete": False,
                "truncation_signals": [],
                "integrity_warnings": [],
            },
        )

        report = validator.validate_chunk(chunk)

        self.assertTrue(report["is_valid"])

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
