"""Tests for pipeline chunk-quality acceptance and exported summary behavior."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.cleaning.normalizer import NormalizedDocument
from Chunking.config.settings import PipelineSettings
from Chunking.pipeline import _build_chunk_quality_summary


def build_document_metadata() -> DocumentMetadata:
    """Build stable document metadata for pipeline summary tests."""
    return DocumentMetadata(
        doc_id="pipeline_quality_summary",
        file_name="pipeline_quality_summary.pdf",
        title="Pipeline Quality Summary Fixture",
        source_path="pipeline_quality_summary.pdf",
    )


def build_normalized_document() -> NormalizedDocument:
    """Build the smallest normalized document accepted by the summary helper."""
    return NormalizedDocument(
        pages=[],
        full_text="",
        dropped_lines_report={},
    )


def build_structure_root() -> StructuralNode:
    """Build a minimal structure root for chunk-quality summary tests."""
    return StructuralNode(
        node_type="DOCUMENT",
        label="DOCUMENT",
    )


class PipelineChunkQualitySummaryTests(unittest.TestCase):
    """Validate pipeline acceptance and summary export against validator output."""

    def test_summary_accepts_clean_chunk_sequence(self) -> None:
        """Ensure clean chunks remain acceptable for the next phase."""
        settings = PipelineSettings()
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="doc_clean",
                strategy="article_smart",
                text="O presente regulamento aplica-se a todos os cursos.",
                text_for_embedding="O presente regulamento aplica-se a todos os cursos.",
                page_start=1,
                page_end=1,
                source_node_type="ARTICLE",
                source_node_label="ART_1",
                hierarchy_path=["DOCUMENT:DOCUMENT", "ARTICLE:ART_1"],
                chunk_reason="direct_article",
                metadata={
                    "article_number": "1",
                    "article_title": "Objeto",
                    "document_part": "regulation_body",
                },
            )
        ]

        summary = _build_chunk_quality_summary(
            document_metadata=build_document_metadata(),
            extraction_quality={},
            normalized=build_normalized_document(),
            structure_root=build_structure_root(),
            chunks=chunks,
            strategy_name="article_smart",
            extraction_mode_used="native",
            settings=settings,
        )

        self.assertTrue(summary["acceptable_for_next_phase"])
        self.assertFalse(summary["has_blocking_failures"])
        self.assertEqual(summary["blocking_failure_count"], 0)
        self.assertEqual(summary["blocking_failure_types"], [])
        self.assertEqual(summary["failed_chunk_ids"], [])
        self.assertEqual(summary["acceptance_basis"], "validator_blocking_failures")
        self.assertIn(
            "acceptable for downstream embedding consumption",
            summary["acceptance_reason"],
        )
        self.assertEqual(summary["next_phase_decision"]["decision"], "accept")
        self.assertEqual(summary["validator_summary"]["blocking_failure_types"], [])
        self.assertFalse(summary["validator_summary"]["has_blocking_failures"])

    def test_summary_rejects_and_explains_blocking_failures(self) -> None:
        """Ensure rejected summaries expose blocking failure categories clearly."""
        settings = PipelineSettings()
        chunks = [
            Chunk(
                chunk_id="c_bad",
                doc_id="doc_bad",
                strategy="article_smart",
                text="Disposições finais\n(1) Acessível em https://domus.ipp.pt/",
                text_for_embedding=(
                    "Disposições finais\n(1) Acessível em https://domus.ipp.pt/"
                ),
                page_start=1,
                page_end=1,
                source_node_type="ARTICLE",
                source_node_label="ART_10",
                hierarchy_path=["DOCUMENT:DOCUMENT", "ARTICLE:ART_10"],
                chunk_reason="direct_article",
                metadata={
                    "article_number": "10",
                    "article_title": "Disposições finais",
                    "document_part": "regulation_body",
                },
            )
        ]

        summary = _build_chunk_quality_summary(
            document_metadata=build_document_metadata(),
            extraction_quality={},
            normalized=build_normalized_document(),
            structure_root=build_structure_root(),
            chunks=chunks,
            strategy_name="article_smart",
            extraction_mode_used="native",
            settings=settings,
        )

        self.assertFalse(summary["acceptable_for_next_phase"])
        self.assertTrue(summary["has_blocking_failures"])
        self.assertEqual(summary["next_phase_decision"]["decision"], "reject")
        self.assertGreater(summary["blocking_failure_count"], 0)
        self.assertIn(
            "note_or_footnote_in_text",
            summary["blocking_failure_types"],
        )
        self.assertEqual(summary["failed_chunk_ids"], ["c_bad"])
        self.assertEqual(summary["acceptance_basis"], "validator_blocking_failures")
        self.assertEqual(
            summary["acceptance_reason"],
            summary["next_phase_decision"]["reason"],
        )
        self.assertIn(
            "note_or_footnote_in_text",
            summary["validator_summary"]["blocking_failure_types"],
        )
        self.assertTrue(summary["validator_summary"]["has_blocking_failures"])
        self.assertIn(
            "note_or_footnote_in_text",
            summary["next_phase_decision"]["reason"],
        )
        self.assertEqual(
            summary["next_phase_decision"]["failed_chunk_ids"],
            ["c_bad"],
        )
        self.assertEqual(summary["invalid_chunk_count"], 1)
        self.assertEqual(summary["next_phase_decision"]["invalid_chunk_count"], 1)

    def test_summary_distinguishes_invalid_chunk_count_from_issue_count(self) -> None:
        """Ensure summary keeps chunk rejection counts aligned with validator reality."""
        settings = PipelineSettings()
        chunks = [
            Chunk(
                chunk_id="c_multi_issue",
                doc_id="doc_multi_issue",
                strategy="article_smart",
                text=(
                    "Disposicoes finais do regulamento aplicavel.\n"
                    "https://domus.ipp.pt/"
                ),
                text_for_embedding=(
                    "Preamble\n\nDisposicoes finais do regulamento aplicavel.\n"
                    "https://domus.ipp.pt/"
                ),
                page_start=1,
                page_end=1,
                source_node_type="PREAMBLE",
                source_node_label="PREAMBLE",
                hierarchy_path=["DOCUMENT:DOCUMENT", "PREAMBLE:PREAMBLE"],
                chunk_reason="preamble_group",
                metadata={
                    "document_part": "dispatch_or_intro",
                    "source_span_type": "preamble",
                },
            )
        ]

        summary = _build_chunk_quality_summary(
            document_metadata=build_document_metadata(),
            extraction_quality={},
            normalized=build_normalized_document(),
            structure_root=build_structure_root(),
            chunks=chunks,
            strategy_name="article_smart",
            extraction_mode_used="native",
            settings=settings,
        )

        self.assertFalse(summary["acceptable_for_next_phase"])
        self.assertTrue(summary["has_blocking_failures"])
        self.assertEqual(summary["valid_chunk_count"], 0)
        self.assertEqual(summary["invalid_chunk_count"], 1)
        self.assertEqual(summary["validator_summary"]["invalid_chunk_count"], 1)
        self.assertEqual(summary["next_phase_decision"]["invalid_chunk_count"], 1)
        self.assertGreater(summary["blocking_failure_count"], summary["invalid_chunk_count"])
        self.assertEqual(summary["failed_chunk_ids"], ["c_multi_issue"])
        self.assertIn("1 invalid chunk(s)", summary["next_phase_decision"]["reason"])


if __name__ == "__main__":
    unittest.main()
