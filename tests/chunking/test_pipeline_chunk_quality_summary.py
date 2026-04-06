"""Tests for pipeline chunk-quality acceptance and exported summary behavior."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.cleaning.normalizer import NormalizedDocument, NormalizedPage
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


def build_hybrid_normalized_document() -> NormalizedDocument:
    """Build a minimal normalized document with mixed page-level extraction modes."""
    return NormalizedDocument(
        pages=[
            NormalizedPage(
                page_number=1,
                text="Artigo 1.o Objeto\nTexto nativo legivel do regulamento.",
                selected_mode="dict",
                upstream_quality_score=78.0,
            ),
            NormalizedPage(
                page_number=2,
                text="Artigo 2.o Ambito\nTexto OCR legivel do regulamento.",
                selected_mode="ocr",
                upstream_quality_score=65.0,
            ),
        ],
        full_text=(
            "Artigo 1.o Objeto\nTexto nativo legivel do regulamento.\n\n"
            "Artigo 2.o Ambito\nTexto OCR legivel do regulamento."
        ),
        dropped_lines_report={},
    )


def build_structure_root() -> StructuralNode:
    """Build a minimal structure root for chunk-quality summary tests."""
    return StructuralNode(
        node_type="DOCUMENT",
        label="DOCUMENT",
    )


def build_structure_root_with_integrity_warning() -> StructuralNode:
    """Build a structure root containing one incomplete article warning."""
    article_node = StructuralNode(
        node_type="ARTICLE",
        label="ART_4",
        title="Ambito",
        page_start=2,
        page_end=2,
        metadata={
            "is_structurally_incomplete": True,
            "truncation_signals": ["suspicious_truncated_ending"],
            "integrity_warnings": ["body_capture_incomplete"],
        },
    )
    return StructuralNode(
        node_type="DOCUMENT",
        label="DOCUMENT",
        children=[article_node],
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
        self.assertEqual(summary["diagnostic_summary"]["final_decision"]["decision"], "accept")
        self.assertIn(
            "Validator reported no blocking chunk-quality failures",
            summary["diagnostic_summary"]["final_decision"]["decision_drivers"][1],
        )

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
        self.assertEqual(summary["diagnostic_summary"]["validation"]["invalid_chunk_count"], 1)
        self.assertEqual(
            summary["diagnostic_summary"]["validation"]["blocking_failures"][0]["messages"],
            ["Visible chunk text still contains note or footnote residue."],
        )

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

    def test_summary_exports_audit_ready_diagnostics_for_hybrid_rejection(self) -> None:
        """Ensure rejection summaries remain understandable from diagnostics alone."""
        settings = PipelineSettings()
        chunks = [
            Chunk(
                chunk_id="c_split_semantic",
                doc_id="doc_hybrid_reject",
                strategy="article_smart",
                text="- sem contexto autonomo",
                text_for_embedding="- sem contexto autonomo",
                page_start=2,
                page_end=2,
                source_node_type="ARTICLE",
                source_node_label="ART_4",
                hierarchy_path=["DOCUMENT:DOCUMENT", "ARTICLE:ART_4"],
                chunk_reason="article_split_tail",
                metadata={
                    "article_number": "4",
                    "article_title": "Ambito",
                    "document_part": "regulation_body",
                    "part_count": 2,
                },
            )
        ]
        extraction_decision = {
            "comparison_performed": True,
            "document_composition": "hybrid",
            "page_count": 2,
            "compared_page_count": 2,
            "native_page_count": 1,
            "ocr_page_count": 1,
            "ocr_comparison_trigger": {
                "document_likely_corrupted": False,
                "has_local_pages_requiring_ocr_comparison": True,
                "pages_requiring_ocr_comparison": [2],
                "local_page_comparison_supported": True,
                "should_run_ocr_comparison": True,
            },
            "page_decisions": [
                {
                    "page_number": 1,
                    "selected_source": "native",
                    "selected_mode": "dict",
                    "decision": "keep_native",
                    "score_gap": 12.0,
                    "reason_codes": ["native_stronger"],
                },
                {
                    "page_number": 2,
                    "selected_source": "ocr",
                    "selected_mode": "ocr",
                    "decision": "switch_to_ocr",
                    "score_gap": 38.0,
                    "reason_codes": ["ocr_replaces_garbled_native"],
                },
            ],
        }

        summary = _build_chunk_quality_summary(
            document_metadata=build_document_metadata(),
            extraction_quality={},
            normalized=build_hybrid_normalized_document(),
            structure_root=build_structure_root_with_integrity_warning(),
            chunks=chunks,
            strategy_name="article_smart",
            extraction_mode_used="hybrid",
            extraction_decision=extraction_decision,
            settings=settings,
        )

        self.assertFalse(summary["acceptable_for_next_phase"])
        self.assertEqual(summary["final_decision"], "reject")
        self.assertTrue(summary["has_structural_integrity_warnings"])
        self.assertEqual(
            summary["diagnostic_summary"]["extraction"]["pages_switched_to_ocr"],
            [2],
        )
        self.assertEqual(
            summary["diagnostic_summary"]["extraction"]["ocr_comparison_trigger"],
            extraction_decision["ocr_comparison_trigger"],
        )
        self.assertEqual(
            summary["diagnostic_summary"]["structure"][
                "structurally_incomplete_article_count"
            ],
            1,
        )
        self.assertEqual(
            summary["diagnostic_summary"]["validation"]["blocking_failures"][0]["code"],
            "low_semantic_autonomy_chunk",
        )
        self.assertEqual(
            summary["diagnostic_summary"]["validation"]["blocking_failures"][0][
                "messages"
            ],
            ["Split chunk still behaves like a semantically orphaned fragment."],
        )
        self.assertIn(
            "Hybrid extraction kept native text where it remained stronger",
            summary["diagnostic_summary"]["final_decision"]["decision_drivers"][0],
        )
        self.assertIn(
            "Parser integrity warnings remained present",
            summary["diagnostic_summary"]["final_decision"]["decision_drivers"][1],
        )
        self.assertIn(
            "low_semantic_autonomy_chunk",
            summary["diagnostic_summary"]["final_decision"]["decision_drivers"][2],
        )

    def test_summary_preserves_hybrid_page_selection_in_exported_artifacts(self) -> None:
        """Ensure hybrid extraction summaries expose the final page-by-page composition."""
        settings = PipelineSettings()
        chunks = [
            Chunk(
                chunk_id="c_hybrid",
                doc_id="doc_hybrid",
                strategy="article_smart",
                text=(
                    "Artigo 1.o Objeto\n"
                    "O presente regulamento define as regras aplicaveis aos cursos."
                ),
                text_for_embedding=(
                    "Artigo 1.o Objeto\n"
                    "O presente regulamento define as regras aplicaveis aos cursos."
                ),
                page_start=1,
                page_end=2,
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
        extraction_decision = {
            "comparison_performed": True,
            "document_composition": "hybrid",
            "page_count": 2,
            "compared_page_count": 2,
            "native_page_count": 1,
            "ocr_page_count": 1,
            "page_decisions": [
                {
                    "page_number": 1,
                    "selected_source": "native",
                    "selected_mode": "dict",
                    "reason_codes": ["native_stronger"],
                },
                {
                    "page_number": 2,
                    "selected_source": "ocr",
                    "selected_mode": "ocr",
                    "reason_codes": ["ocr_replaces_garbled_native"],
                },
            ],
        }

        summary = _build_chunk_quality_summary(
            document_metadata=build_document_metadata(),
            extraction_quality={},
            normalized=build_hybrid_normalized_document(),
            structure_root=build_structure_root(),
            chunks=chunks,
            strategy_name="article_smart",
            extraction_mode_used="hybrid",
            extraction_decision=extraction_decision,
            settings=settings,
        )

        self.assertEqual(summary["extraction_mode_used"], "hybrid")
        self.assertEqual(summary["extraction_summary"]["document_mode"], "hybrid")
        self.assertEqual(summary["extraction_summary"]["document_composition"], "hybrid")
        self.assertTrue(summary["extraction_summary"]["comparison_performed"])
        self.assertEqual(summary["extraction_summary"]["native_page_count"], 1)
        self.assertEqual(summary["extraction_summary"]["ocr_page_count"], 1)
        self.assertTrue(summary["extraction_summary"]["selected_page_modes_present"])
        self.assertTrue(summary["extraction_summary"]["hybrid_pages_selected"])
        self.assertEqual(
            summary["extraction_summary"]["page_modes_selected"],
            [
                {
                    "page_number": 1,
                    "selected_source": "native",
                    "selected_mode": "dict",
                },
                {
                    "page_number": 2,
                    "selected_source": "ocr",
                    "selected_mode": "ocr",
                },
            ],
        )
        self.assertEqual(
            summary["extraction_summary"]["page_selection_reason_codes"],
            {
                "1": ["native_stronger"],
                "2": ["ocr_replaces_garbled_native"],
            },
        )
        self.assertEqual(
            summary["extraction_summary"]["ocr_comparison_trigger"],
            {
                "document_likely_corrupted": False,
                "has_local_pages_requiring_ocr_comparison": False,
                "pages_requiring_ocr_comparison": [],
                "local_page_comparison_supported": False,
                "should_run_ocr_comparison": False,
            },
        )
        self.assertEqual(summary["extraction_decision"], extraction_decision)
        self.assertEqual(summary["normalized_page_count"], 2)
        self.assertEqual(summary["non_empty_normalized_pages"], 2)

    def test_summary_distinguishes_native_pages_kept_after_comparison(self) -> None:
        """Ensure diagnostics do not report untested native pages as compared winners."""
        settings = PipelineSettings()
        chunks = [
            Chunk(
                chunk_id="c_hybrid_scope",
                doc_id="doc_hybrid_scope",
                strategy="article_smart",
                text="Artigo 1.o Objeto\nTexto regulamentar estavel.",
                text_for_embedding="Artigo 1.o Objeto\nTexto regulamentar estavel.",
                page_start=1,
                page_end=3,
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
        extraction_decision = {
            "comparison_performed": True,
            "document_composition": "hybrid",
            "page_count": 3,
            "compared_page_count": 2,
            "native_page_count": 2,
            "ocr_page_count": 1,
            "ocr_comparison_trigger": {
                "document_likely_corrupted": False,
                "has_local_pages_requiring_ocr_comparison": True,
                "pages_requiring_ocr_comparison": [2, 3],
                "local_page_comparison_supported": True,
                "should_run_ocr_comparison": True,
            },
            "page_decisions": [
                {
                    "page_number": 1,
                    "compared_with_ocr": False,
                    "selected_source": "native",
                    "selected_mode": "dict",
                    "decision": "keep_native_without_ocr_comparison",
                    "reason_codes": ["page_not_flagged_for_ocr_comparison"],
                },
                {
                    "page_number": 2,
                    "compared_with_ocr": True,
                    "selected_source": "native",
                    "selected_mode": "dict",
                    "decision": "keep_native",
                    "reason_codes": ["native_not_worse_by_quality_score"],
                },
                {
                    "page_number": 3,
                    "compared_with_ocr": True,
                    "selected_source": "ocr",
                    "selected_mode": "ocr",
                    "decision": "switch_to_ocr",
                    "reason_codes": ["ocr_replaces_garbled_native"],
                },
            ],
        }

        summary = _build_chunk_quality_summary(
            document_metadata=build_document_metadata(),
            extraction_quality={},
            normalized=build_hybrid_normalized_document(),
            structure_root=build_structure_root(),
            chunks=chunks,
            strategy_name="article_smart",
            extraction_mode_used="hybrid",
            extraction_decision=extraction_decision,
            settings=settings,
        )

        self.assertEqual(
            summary["diagnostic_summary"]["extraction"]["pages_kept_native_after_comparison"],
            [2],
        )
        self.assertEqual(
            summary["diagnostic_summary"]["extraction"]["pages_kept_native_without_comparison"],
            [1],
        )
        self.assertEqual(
            summary["diagnostic_summary"]["extraction"]["pages_switched_to_ocr"],
            [3],
        )
        self.assertEqual(
            summary["diagnostic_summary"]["extraction"]["page_decision_trace"],
            [
                {
                    "page_number": 1,
                    "compared_with_ocr": False,
                    "selected_source": "native",
                    "selected_mode": "dict",
                    "decision": "keep_native_without_ocr_comparison",
                    "score_gap": None,
                    "reason_codes": ["page_not_flagged_for_ocr_comparison"],
                },
                {
                    "page_number": 2,
                    "compared_with_ocr": True,
                    "selected_source": "native",
                    "selected_mode": "dict",
                    "decision": "keep_native",
                    "score_gap": None,
                    "reason_codes": ["native_not_worse_by_quality_score"],
                },
                {
                    "page_number": 3,
                    "compared_with_ocr": True,
                    "selected_source": "ocr",
                    "selected_mode": "ocr",
                    "decision": "switch_to_ocr",
                    "score_gap": None,
                    "reason_codes": ["ocr_replaces_garbled_native"],
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
