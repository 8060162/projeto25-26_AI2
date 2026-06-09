"""Tests for hybrid page-by-page extraction assembly in the pipeline."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import ExtractedDocument, ExtractedPage
from Chunking.config.settings import PipelineSettings
from Chunking.extraction.extraction_quality import ExtractionQualityAnalyzer
from Chunking.pipeline import (
    _build_extraction_diagnostic_summary,
    _build_extraction_summary,
    _build_ocr_comparison_trigger,
    _build_hybrid_extraction_result,
    _build_native_only_extraction_decision,
    _build_ocr_only_extraction_decision,
)


def build_page(
    *,
    page_number: int,
    text: str,
    selected_mode: str,
    quality_score: float,
    corruption_flags: list[str] | None = None,
) -> ExtractedPage:
    """Build an extracted page fixture for hybrid extraction tests."""
    return ExtractedPage(
        page_number=page_number,
        text=text,
        selected_mode=selected_mode,
        quality_score=quality_score,
        blocks=[],
        corruption_flags=corruption_flags or [],
    )


class PipelineHybridExtractionTests(unittest.TestCase):
    """Validate page-level hybrid extraction assembly in the pipeline."""

    def test_build_ocr_comparison_trigger_runs_for_local_candidates_when_hybrid_is_enabled(self) -> None:
        """Ensure local page defects can trigger OCR comparison without global corruption."""
        analyzer = ExtractionQualityAnalyzer()
        native_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=3,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="dict",
                    quality_score=80.0,
                    text="Artigo 1.o Objeto\nTexto legivel do regulamento.",
                ),
                build_page(
                    page_number=2,
                    selected_mode="dict",
                    quality_score=6.0,
                    text=(
                        "CAPITULO II\n"
                        "0 estudante deve apresentar o requerimento no prazo de 10 dias uteis /\n"
                        "* ^ _"
                    ),
                ),
                build_page(
                    page_number=3,
                    selected_mode="dict",
                    quality_score=74.0,
                    text="Artigo 3.o Procedimento\nTexto legivel do regulamento.",
                ),
            ],
        )

        comparison_trigger = _build_ocr_comparison_trigger(
            native_extraction_quality=analyzer.analyze_document(native_document),
            enable_hybrid_page_selection=True,
        )

        self.assertFalse(comparison_trigger["document_likely_corrupted"])
        self.assertTrue(comparison_trigger["has_local_pages_requiring_ocr_comparison"])
        self.assertEqual(comparison_trigger["pages_requiring_ocr_comparison"], [2])
        self.assertTrue(comparison_trigger["local_page_comparison_supported"])
        self.assertTrue(comparison_trigger["should_run_ocr_comparison"])

    def test_build_ocr_comparison_trigger_does_not_force_ocr_only_for_local_candidates(self) -> None:
        """Ensure local defects alone do not trigger document-wide OCR when hybrid is disabled."""
        analyzer = ExtractionQualityAnalyzer()
        native_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=3,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="dict",
                    quality_score=80.0,
                    text="Artigo 1.o Objeto\nTexto legivel do regulamento.",
                ),
                build_page(
                    page_number=2,
                    selected_mode="dict",
                    quality_score=6.0,
                    text=(
                        "CAPITULO II\n"
                        "0 estudante deve apresentar o requerimento no prazo de 10 dias uteis /\n"
                        "* ^ _"
                    ),
                ),
                build_page(
                    page_number=3,
                    selected_mode="dict",
                    quality_score=74.0,
                    text="Artigo 3.o Procedimento\nTexto legivel do regulamento.",
                ),
            ],
        )

        comparison_trigger = _build_ocr_comparison_trigger(
            native_extraction_quality=analyzer.analyze_document(native_document),
            enable_hybrid_page_selection=False,
        )

        self.assertFalse(comparison_trigger["document_likely_corrupted"])
        self.assertTrue(comparison_trigger["has_local_pages_requiring_ocr_comparison"])
        self.assertFalse(comparison_trigger["local_page_comparison_supported"])
        self.assertFalse(comparison_trigger["should_run_ocr_comparison"])

    def test_build_hybrid_extraction_result_keeps_native_and_replaces_only_weak_pages(self) -> None:
        """Ensure mixed-quality documents are assembled from the best page source."""
        analyzer = ExtractionQualityAnalyzer()
        native_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=3,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="dict",
                    quality_score=78.0,
                    text=(
                        "Artigo 1.o Objeto\n"
                        "O presente regulamento estabelece as regras aplicaveis ao curso."
                    ),
                ),
                build_page(
                    page_number=2,
                    selected_mode="dict",
                    quality_score=-15.0,
                    corruption_flags=[
                        "replacement_like_characters",
                        "high_suspicious_symbol_density",
                    ],
                    text="*+,-*-.-/-1/2*-34+*4/-5/-1/6-/",
                ),
                build_page(
                    page_number=3,
                    selected_mode="dict",
                    quality_score=80.0,
                    text=(
                        "Artigo 3.o Vigencia\n"
                        "O presente regulamento entra em vigor no dia seguinte."
                    ),
                ),
            ],
        )
        ocr_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=3,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="ocr",
                    quality_score=52.0,
                    text=(
                        "Artigo 1.o Objcto\n"
                        "O presente regulamento estabelece as regras aplicaveis ao curso"
                    ),
                ),
                build_page(
                    page_number=2,
                    selected_mode="ocr",
                    quality_score=65.0,
                    text=(
                        "Artigo 2.o Ambito\n"
                        "As normas aplicam-se aos estudantes matriculados no instituto."
                    ),
                ),
                build_page(
                    page_number=3,
                    selected_mode="ocr",
                    quality_score=40.0,
                    text=(
                        "Artigo 3.o Vigencia\n"
                        "O presentc regulamento entra em vigor no dia seguinte."
                    ),
                ),
            ],
        )

        hybrid_document, extraction_mode_used, extraction_decision = (
            _build_hybrid_extraction_result(
                source_path="sample.pdf",
                native_extracted_document=native_document,
                native_extraction_quality=analyzer.analyze_document(native_document),
                ocr_extracted_document=ocr_document,
                ocr_extraction_quality=analyzer.analyze_document(ocr_document),
                extraction_quality_analyzer=analyzer,
                ocr_comparison_trigger=_build_ocr_comparison_trigger(
                    native_extraction_quality=analyzer.analyze_document(native_document),
                    enable_hybrid_page_selection=True,
                ),
            )
        )

        self.assertEqual(extraction_mode_used, "hybrid")
        self.assertEqual(
            [page.selected_mode for page in hybrid_document.pages],
            ["dict", "ocr", "dict"],
        )
        self.assertEqual(extraction_decision["document_composition"], "hybrid")
        self.assertEqual(extraction_decision["native_page_count"], 2)
        self.assertEqual(extraction_decision["ocr_page_count"], 1)
        self.assertEqual(extraction_decision["compared_page_count"], 1)
        self.assertTrue(
            extraction_decision["ocr_comparison_trigger"]["should_run_ocr_comparison"]
        )
        self.assertEqual(
            extraction_decision["page_decisions"][0]["selected_source"],
            "native",
        )
        self.assertFalse(
            extraction_decision["page_decisions"][0]["compared_with_ocr"]
        )
        self.assertEqual(
            extraction_decision["page_decisions"][0]["selected_mode"],
            "dict",
        )
        self.assertEqual(
            extraction_decision["page_decisions"][0]["decision"],
            "keep_native_without_ocr_comparison",
        )
        self.assertEqual(
            extraction_decision["page_decisions"][0]["reason_codes"],
            ["page_not_flagged_for_ocr_comparison"],
        )
        self.assertEqual(
            extraction_decision["page_decisions"][1]["selected_source"],
            "ocr",
        )
        self.assertTrue(extraction_decision["page_decisions"][1]["compared_with_ocr"])
        self.assertEqual(
            extraction_decision["page_decisions"][1]["selected_mode"],
            "ocr",
        )
        self.assertIn(
            "native_has_more_suspicious_symbol_noise",
            extraction_decision["page_decisions"][1]["reason_codes"],
        )
        self.assertIn(
            "ocr_looks_more_like_prose",
            extraction_decision["page_decisions"][1]["reason_codes"],
        )
        self.assertEqual(
            extraction_decision["page_decisions"][2]["selected_source"],
            "native",
        )
        self.assertFalse(
            extraction_decision["page_decisions"][2]["compared_with_ocr"]
        )
        self.assertEqual(
            extraction_decision["page_decisions"][2]["reason_codes"],
            ["page_not_flagged_for_ocr_comparison"],
        )

    def test_build_extraction_summary_preserves_selected_modes_for_hybrid_document(self) -> None:
        """Ensure exported extraction summaries retain per-page mode choices."""
        analyzer = ExtractionQualityAnalyzer()
        native_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=3,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="dict",
                    quality_score=78.0,
                    text=(
                        "Artigo 1.o Objeto\n"
                        "O presente regulamento estabelece as regras aplicaveis ao curso."
                    ),
                ),
                build_page(
                    page_number=2,
                    selected_mode="dict",
                    quality_score=-15.0,
                    corruption_flags=[
                        "replacement_like_characters",
                        "high_suspicious_symbol_density",
                    ],
                    text="*+,-*-.-/-1/2*-34+*4/-5/-1/6-/",
                ),
                build_page(
                    page_number=3,
                    selected_mode="dict",
                    quality_score=80.0,
                    text=(
                        "Artigo 3.o Vigencia\n"
                        "O presente regulamento entra em vigor no dia seguinte."
                    ),
                ),
            ],
        )
        ocr_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=3,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="ocr",
                    quality_score=52.0,
                    text=(
                        "Artigo 1.o Objcto\n"
                        "O presente regulamento estabelece as regras aplicaveis ao curso"
                    ),
                ),
                build_page(
                    page_number=2,
                    selected_mode="ocr",
                    quality_score=65.0,
                    text=(
                        "Artigo 2.o Ambito\n"
                        "As normas aplicam-se aos estudantes matriculados no instituto."
                    ),
                ),
                build_page(
                    page_number=3,
                    selected_mode="ocr",
                    quality_score=40.0,
                    text=(
                        "Artigo 3.o Vigencia\n"
                        "O presentc regulamento entra em vigor no dia seguinte."
                    ),
                ),
            ],
        )
        trigger = _build_ocr_comparison_trigger(
            native_extraction_quality=analyzer.analyze_document(native_document),
            enable_hybrid_page_selection=True,
        )
        _, extraction_mode_used, extraction_decision = _build_hybrid_extraction_result(
            source_path="sample.pdf",
            native_extracted_document=native_document,
            native_extraction_quality=analyzer.analyze_document(native_document),
            ocr_extracted_document=ocr_document,
            ocr_extraction_quality=analyzer.analyze_document(ocr_document),
            extraction_quality_analyzer=analyzer,
            ocr_comparison_trigger=trigger,
        )

        summary = _build_extraction_summary(
            extraction_mode_used=extraction_mode_used,
            extraction_decision=extraction_decision,
        )

        self.assertEqual(summary["document_mode"], "hybrid")
        self.assertEqual(summary["document_composition"], "hybrid")
        self.assertTrue(summary["comparison_performed"])
        self.assertTrue(summary["selected_page_modes_present"])
        self.assertTrue(summary["hybrid_pages_selected"])
        self.assertEqual(
            summary["page_modes_selected"],
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
                {
                    "page_number": 3,
                    "selected_source": "native",
                    "selected_mode": "dict",
                },
            ],
        )
        self.assertEqual(
            summary["page_selection_reason_codes"]["1"],
            ["page_not_flagged_for_ocr_comparison"],
        )
        self.assertEqual(
            summary["page_selection_reason_codes"]["3"],
            ["page_not_flagged_for_ocr_comparison"],
        )
        self.assertIn(
            "native_has_more_suspicious_symbol_noise",
            summary["page_selection_reason_codes"]["2"],
        )
        self.assertIn(
            "ocr_looks_more_like_prose",
            summary["page_selection_reason_codes"]["2"],
        )
        self.assertEqual(summary["ocr_comparison_trigger"], trigger)

    def test_build_extraction_diagnostic_summary_preserves_page_level_reasons(self) -> None:
        """Ensure exported diagnostics keep page-level comparison reasons."""
        analyzer = ExtractionQualityAnalyzer()
        native_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=3,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="dict",
                    quality_score=78.0,
                    text=(
                        "Artigo 1.o Objeto\n"
                        "O presente regulamento estabelece as regras aplicaveis ao curso."
                    ),
                ),
                build_page(
                    page_number=2,
                    selected_mode="dict",
                    quality_score=22.0,
                    text=(
                        "Artigo 2.o Definicoes\n"
                        "1 - Para efeitos do presente regulamento, entende-se por\n"
                        "2 - O pedido deve ser apresentado pelos estudantes no portal academico."
                    ),
                ),
                build_page(
                    page_number=3,
                    selected_mode="dict",
                    quality_score=80.0,
                    text=(
                        "Artigo 3.o Vigencia\n"
                        "O presente regulamento entra em vigor no dia seguinte."
                    ),
                ),
            ],
        )
        ocr_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=3,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="ocr",
                    quality_score=52.0,
                    text=(
                        "Artigo 1.o Objcto\n"
                        "O presente regulamento estabelece as regras aplicaveis ao curso"
                    ),
                ),
                build_page(
                    page_number=2,
                    selected_mode="ocr",
                    quality_score=80.0,
                    text=(
                        "Artigo 2.o Definicoes\n"
                        "1 - Para efeitos do presente regulamento, entende-se por estudante "
                        "o titular de inscricao valida no instituto.\n"
                        "2 - O pedido deve ser apresentado pelos estudantes no portal academico."
                    ),
                ),
                build_page(
                    page_number=3,
                    selected_mode="ocr",
                    quality_score=40.0,
                    text=(
                        "Artigo 3.o Vigencia\n"
                        "O presentc regulamento entra em vigor no dia seguinte."
                    ),
                ),
            ],
        )
        trigger = _build_ocr_comparison_trigger(
            native_extraction_quality=analyzer.analyze_document(native_document),
            enable_hybrid_page_selection=True,
        )
        _, extraction_mode_used, extraction_decision = _build_hybrid_extraction_result(
            source_path="sample.pdf",
            native_extracted_document=native_document,
            native_extraction_quality=analyzer.analyze_document(native_document),
            ocr_extracted_document=ocr_document,
            ocr_extraction_quality=analyzer.analyze_document(ocr_document),
            extraction_quality_analyzer=analyzer,
            ocr_comparison_trigger=trigger,
        )

        diagnostic_summary = _build_extraction_diagnostic_summary(
            extraction_mode_used=extraction_mode_used,
            extraction_decision=extraction_decision,
        )

        self.assertEqual(diagnostic_summary["pages_switched_to_ocr"], [2])
        self.assertEqual(diagnostic_summary["pages_kept_native_after_comparison"], [])
        self.assertEqual(diagnostic_summary["pages_kept_native_without_comparison"], [1, 3])
        self.assertEqual(
            diagnostic_summary["page_decision_trace"][0],
            {
                "page_number": 1,
                "compared_with_ocr": False,
                "selected_source": "native",
                "selected_mode": "dict",
                "decision": "keep_native_without_ocr_comparison",
                "score_gap": None,
                "reason_codes": ["page_not_flagged_for_ocr_comparison"],
            },
        )
        self.assertEqual(
            diagnostic_summary["page_decision_trace"][2],
            {
                "page_number": 3,
                "compared_with_ocr": False,
                "selected_source": "native",
                "selected_mode": "dict",
                "decision": "keep_native_without_ocr_comparison",
                "score_gap": None,
                "reason_codes": ["page_not_flagged_for_ocr_comparison"],
            },
        )
        compared_page_trace = diagnostic_summary["page_decision_trace"][1]
        self.assertEqual(compared_page_trace["page_number"], 2)
        self.assertTrue(compared_page_trace["compared_with_ocr"])
        self.assertEqual(compared_page_trace["selected_source"], "ocr")
        self.assertEqual(compared_page_trace["selected_mode"], "ocr")
        self.assertEqual(compared_page_trace["decision"], "use_ocr")
        self.assertGreater(float(compared_page_trace["score_gap"]), 0.0)
        self.assertIn(
            "ocr_better_preserves_legal_continuity",
            compared_page_trace["reason_codes"],
        )
        self.assertEqual(diagnostic_summary["ocr_comparison_trigger"], trigger)

    def test_build_native_only_extraction_decision_marks_all_pages_as_native(self) -> None:
        """Ensure non-compared documents still export a stable decision payload."""
        analyzer = ExtractionQualityAnalyzer()
        native_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=1,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="dict",
                    quality_score=78.0,
                    text="Artigo 1.o Objeto\nTexto legivel do regulamento.",
                )
            ],
        )

        extraction_decision = _build_native_only_extraction_decision(
            native_extracted_document=native_document,
            native_extraction_quality=analyzer.analyze_document(native_document),
        )

        self.assertFalse(extraction_decision["comparison_performed"])
        self.assertEqual(extraction_decision["document_composition"], "all_native")
        self.assertEqual(extraction_decision["native_page_count"], 1)
        self.assertEqual(extraction_decision["ocr_page_count"], 0)
        self.assertEqual(extraction_decision["ocr_comparison_trigger"], {})
        self.assertEqual(extraction_decision["page_decisions"], [])

    def test_build_ocr_only_extraction_decision_marks_all_pages_as_ocr(self) -> None:
        """Ensure disabled hybrid selection still exports a stable OCR decision."""
        settings = PipelineSettings(enable_hybrid_page_selection=False)
        analyzer = ExtractionQualityAnalyzer(settings)
        ocr_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=1,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="ocr",
                    quality_score=65.0,
                    text="Artigo 1.o Objeto\nTexto OCR legivel do regulamento.",
                )
            ],
        )
        native_document = ExtractedDocument(
            source_path="sample.pdf",
            page_count=1,
            pages=[
                build_page(
                    page_number=1,
                    selected_mode="dict",
                    quality_score=-12.0,
                    corruption_flags=["replacement_like_characters"],
                    text="*+,-*-.-/-1/2*-34+*4/-5/-1/6-/",
                )
            ],
        )

        extraction_decision = _build_ocr_only_extraction_decision(
            ocr_extracted_document=ocr_document,
            native_extraction_quality=analyzer.analyze_document(native_document),
            ocr_extraction_quality=analyzer.analyze_document(ocr_document),
        )

        self.assertFalse(extraction_decision["comparison_performed"])
        self.assertEqual(extraction_decision["document_composition"], "all_ocr")
        self.assertEqual(extraction_decision["native_page_count"], 0)
        self.assertEqual(extraction_decision["ocr_page_count"], 1)
        self.assertEqual(extraction_decision["ocr_comparison_trigger"], {})
        self.assertEqual(extraction_decision["page_decisions"], [])


if __name__ == "__main__":
    unittest.main()
