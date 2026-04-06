"""Tests for page-level native-versus-OCR extraction comparison."""

from __future__ import annotations

import unittest

from Chunking.chunking.models import ExtractedPage
from Chunking.config.settings import PipelineSettings
from Chunking.extraction.extraction_quality import ExtractionQualityAnalyzer


def build_page(
    *,
    page_number: int,
    text: str,
    selected_mode: str,
    quality_score: float,
    corruption_flags: list[str] | None = None,
) -> ExtractedPage:
    """Build an extracted page fixture for analyzer comparison tests."""
    return ExtractedPage(
        page_number=page_number,
        text=text,
        selected_mode=selected_mode,
        quality_score=quality_score,
        blocks=[],
        corruption_flags=corruption_flags or [],
    )


class ExtractionQualityComparisonTests(unittest.TestCase):
    """Validate explicit page-level comparison between native and OCR text."""

    def test_analyze_document_flags_local_page_for_ocr_comparison_without_global_corruption(self) -> None:
        """Ensure one bad page is exposed even when the full document stays native."""
        analyzer = ExtractionQualityAnalyzer()
        document_pages = [
            build_page(
                page_number=1,
                selected_mode="dict",
                quality_score=76.0,
                text=(
                    "Artigo 1.o Objeto\n"
                    "O presente regulamento estabelece as regras aplicaveis ao curso."
                ),
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
                quality_score=71.0,
                text=(
                    "Artigo 3.o Procedimento\n"
                    "A decisao e comunicada aos estudantes pelos servicos academicos."
                ),
            ),
        ]

        analysis = analyzer.analyze_document(document_pages)
        flagged_page_reports = analyzer.identify_pages_requiring_ocr_comparison(document_pages)

        self.assertFalse(analysis["document_likely_corrupted"])
        self.assertEqual(analysis["pages_requiring_ocr_comparison"], [2])
        self.assertTrue(analysis["has_local_pages_requiring_ocr_comparison"])
        self.assertEqual(analysis["ocr_comparison_candidate_count"], 1)
        self.assertEqual(len(flagged_page_reports), 1)
        self.assertEqual(flagged_page_reports[0]["page_number"], 2)
        self.assertTrue(flagged_page_reports[0]["should_trigger_ocr_comparison"])
        self.assertIn(
            "strong_suspicious_symbol_density",
            flagged_page_reports[0]["comparison_trigger_reasons"],
        )

    def test_compare_page_versions_keeps_native_when_ocr_is_not_clearly_better(self) -> None:
        """Ensure good native prose is not replaced by weaker OCR output."""
        analyzer = ExtractionQualityAnalyzer()
        native_page = build_page(
            page_number=1,
            selected_mode="dict",
            quality_score=72.0,
            text=(
                "Artigo 1.o Objeto\n"
                "O presente regulamento define as regras de inscricao, avaliacao "
                "e funcionamento dos cursos do instituto."
            ),
        )
        ocr_page = build_page(
            page_number=1,
            selected_mode="ocr",
            quality_score=48.0,
            text=(
                "Artigo 1.o Objcto\n"
                "O presente regulamento define as regras dc inscricao e "
                "avaliacao dos cursos do instituto"
            ),
        )

        comparison = analyzer.compare_page_versions(native_page, ocr_page)

        self.assertEqual(comparison["preferred_source"], "native")
        self.assertEqual(comparison["decision"], "keep_native")
        self.assertIn(
            "native_not_worse_by_quality_score",
            comparison["reason_codes"],
        )

    def test_compare_page_versions_prefers_ocr_for_garbled_native_page(self) -> None:
        """Ensure OCR wins when native extraction is strongly degraded."""
        analyzer = ExtractionQualityAnalyzer()
        native_page = build_page(
            page_number=2,
            selected_mode="dict",
            quality_score=-10.0,
            corruption_flags=["replacement_like_characters", "high_suspicious_symbol_density"],
            text="*+,-*-.-/-1/2*-34+*4/-5/-1/6-/",
        )
        ocr_page = build_page(
            page_number=2,
            selected_mode="ocr",
            quality_score=64.0,
            text=(
                "Artigo 2.o Ambito\n"
                "As normas do presente regulamento aplicam-se a todos os estudantes "
                "matriculados no instituto."
            ),
        )

        comparison = analyzer.compare_page_versions(native_page, ocr_page)

        self.assertEqual(comparison["preferred_source"], "ocr")
        self.assertEqual(comparison["decision"], "use_ocr")
        self.assertIn(
            "native_has_more_suspicious_symbol_noise",
            comparison["reason_codes"],
        )
        self.assertIn(
            "ocr_looks_more_like_prose",
            comparison["reason_codes"],
        )

    def test_compare_page_versions_prefers_less_harmful_candidate_when_both_are_weak(self) -> None:
        """Ensure the comparison still selects the less harmful page version."""
        analyzer = ExtractionQualityAnalyzer()
        native_page = build_page(
            page_number=3,
            selected_mode="dict",
            quality_score=5.0,
            corruption_flags=["replacement_like_characters"],
            text=(
                "Artigo 3.o\n"
                "O estudante deve apresentar o reque�imento no prazo de 10 dias uteis;"
            ),
        )
        ocr_page = build_page(
            page_number=3,
            selected_mode="ocr",
            quality_score=18.0,
            text=(
                "Artigo 3.o\n"
                "O estudante deve apresentar o requerimento no prazo de 10 dias uteis"
            ),
        )

        comparison = analyzer.compare_page_versions(native_page, ocr_page)

        self.assertEqual(comparison["preferred_source"], "ocr")
        self.assertEqual(comparison["decision"], "use_ocr")
        self.assertIn(
            "native_has_more_replacement_like_characters",
            comparison["reason_codes"],
        )

    def test_analyze_document_flags_readable_structural_page_with_interrupted_legal_continuity(self) -> None:
        """Ensure readable but incomplete legal enumeration triggers OCR comparison."""
        analyzer = ExtractionQualityAnalyzer()

        analysis = analyzer.analyze_document(
            [
                build_page(
                    page_number=4,
                    selected_mode="dict",
                    quality_score=42.0,
                    text=(
                        "Artigo 7.o Definicoes\n"
                        "1 - Para efeitos do presente regulamento, entende-se por\n"
                        "2 - O pedido deve ser apresentado pelos estudantes no portal academico."
                    ),
                )
            ]
        )

        page_report = analysis["page_reports"][0]

        self.assertTrue(page_report["should_trigger_ocr_comparison"])
        self.assertEqual(page_report["semantic_risk_level"], "acceptable_borderline")
        self.assertIn(
            "interrupted_legal_enumeration",
            page_report["semantic_fragility_signals"],
        )
        self.assertIn(
            "broken_legal_enumeration_continuity",
            page_report["comparison_trigger_reasons"],
        )

    def test_compare_page_versions_prefers_ocr_when_it_restores_legal_continuity(self) -> None:
        """Ensure OCR can win on semantically risky pages that still look readable."""
        analyzer = ExtractionQualityAnalyzer()
        native_page = build_page(
            page_number=5,
            selected_mode="dict",
            quality_score=25.0,
            text=(
                "Artigo 7.o Definicoes\n"
                "1 - Para efeitos do presente regulamento, entende-se por\n"
                "2 - O pedido deve ser apresentado pelos estudantes no portal academico."
            ),
        )
        ocr_page = build_page(
            page_number=5,
            selected_mode="ocr",
            quality_score=80.0,
            text=(
                "Artigo 7.o Definicoes\n"
                "1 - Para efeitos do presente regulamento, entende-se por estudante "
                "o titular de inscricao valida no instituto.\n"
                "2 - O pedido deve ser apresentado pelos estudantes no portal academico."
            ),
        )

        comparison = analyzer.compare_page_versions(native_page, ocr_page)

        self.assertEqual(comparison["preferred_source"], "ocr")
        self.assertEqual(comparison["decision"], "use_ocr")
        self.assertEqual(comparison["native_assessment"], "acceptable_borderline")
        self.assertEqual(comparison["ocr_assessment"], "acceptable")
        self.assertIn(
            "ocr_better_preserves_legal_continuity",
            comparison["reason_codes"],
        )

    def test_compare_page_versions_respects_configured_hybrid_thresholds(self) -> None:
        """Ensure page replacement thresholds are controlled by runtime settings."""
        settings = PipelineSettings(
            hybrid_ocr_page_min_score_gap=20.0,
            hybrid_ocr_page_min_badness_gap=1.0,
            hybrid_ocr_strong_signal_min_score_gap=30.0,
            hybrid_ocr_less_harmful_min_score_gap=30.0,
        )
        analyzer = ExtractionQualityAnalyzer(settings)
        native_page = build_page(
            page_number=4,
            selected_mode="dict",
            quality_score=50.0,
            text=(
                "Artigo 4.o Taxas\n"
                "O estudante paga a taxa anual nos prazos definidos pelo calendario."
            ),
        )
        ocr_page = build_page(
            page_number=4,
            selected_mode="ocr",
            quality_score=64.0,
            text=(
                "Artigo 4.o Taxas\n"
                "O estudante paga a taxa anual nos prazos definidos pelo calendario academico."
            ),
        )

        comparison = analyzer.compare_page_versions(native_page, ocr_page)

        self.assertEqual(comparison["preferred_source"], "native")
        self.assertEqual(comparison["decision"], "keep_native_not_clearly_worse")
        self.assertEqual(comparison["reason_codes"], ["ocr_not_clearly_better"])

    def test_analyze_document_marks_strong_symbolic_degradation_as_ocr_comparison_candidate(self) -> None:
        """Ensure strong symbolic degradation becomes an explicit comparison trigger."""
        analyzer = ExtractionQualityAnalyzer()
        analysis = analyzer.analyze_document(
            [
                build_page(
                    page_number=5,
                    selected_mode="dict",
                    quality_score=-12.0,
                    corruption_flags=[
                        "replacement_like_characters",
                        "high_suspicious_symbol_density",
                    ],
                    text="*+,-*-.-/-1/2*-34+*4/-5/-1/6-/",
                )
            ]
        )

        page_report = analysis["page_reports"][0]

        self.assertTrue(page_report["should_trigger_ocr_comparison"])
        self.assertTrue(page_report["is_locally_unreliable"])
        self.assertIn(
            "strong_suspicious_symbol_density",
            page_report["comparison_trigger_reasons"],
        )
        self.assertIn(
            "poor_prose_likeness",
            page_report["comparison_trigger_reasons"],
        )

    def test_analyze_document_respects_configured_local_ocr_trigger_thresholds(self) -> None:
        """Ensure page-level OCR trigger thresholds are controlled by runtime settings."""
        settings = PipelineSettings(
            local_ocr_trigger_page_quality_score=5.0,
            local_ocr_trigger_suspicious_symbol_ratio=0.05,
            local_ocr_trigger_min_lexical_completeness=0.20,
            local_ocr_trigger_min_line_readability=0.20,
            local_ocr_trigger_min_prose_likeness=0.20,
        )
        analyzer = ExtractionQualityAnalyzer(settings)

        analysis = analyzer.analyze_document(
            [
                build_page(
                    page_number=5,
                    selected_mode="dict",
                    quality_score=6.0,
                    text=(
                        "CAPITULO II\n"
                        "0 estudante deve apresentar o requerimento no prazo de 10 dias uteis /\n"
                        "* ^ _"
                    ),
                )
            ]
        )

        page_report = analysis["page_reports"][0]

        self.assertFalse(page_report["should_trigger_ocr_comparison"])
        self.assertEqual(page_report["comparison_trigger_reasons"], [])

    def test_analyze_document_respects_configured_local_unreliability_thresholds(self) -> None:
        """Ensure local unreliability thresholds are configurable from settings."""
        settings = PipelineSettings(
            local_unreliable_page_min_quality_score=-5.0,
            local_unreliable_page_hard_floor_score=-20.0,
        )
        analyzer = ExtractionQualityAnalyzer(settings)

        analysis = analyzer.analyze_document(
            [
                build_page(
                    page_number=6,
                    selected_mode="dict",
                    quality_score=6.0,
                    text=(
                        "CAPITULO II\n"
                        "0 estudante deve apresentar o requerimento no prazo de 10 dias uteis /\n"
                        "* ^ _"
                    ),
                )
            ]
        )

        page_report = analysis["page_reports"][0]

        self.assertFalse(page_report["is_locally_unreliable"])
        self.assertIn(
            "strong_suspicious_symbol_density",
            page_report["comparison_trigger_reasons"],
        )

    def test_analyze_document_respects_configured_document_replacement_ratio_threshold(self) -> None:
        """Ensure document-level OCR fallback remains configurable for replacement-heavy inputs."""
        pages = [
            build_page(
                page_number=1,
                selected_mode="dict",
                quality_score=80.0,
                text="Artigo 1.o Objeto\nTexto legivel do regulamento.",
            ),
            build_page(
                page_number=2,
                selected_mode="dict",
                quality_score=78.0,
                text="Artigo 2.o Ambito\nTexto legivel do regulamento.",
            ),
            build_page(
                page_number=3,
                selected_mode="dict",
                quality_score=40.0,
                corruption_flags=["replacement_like_characters"],
                text="Artigo 3.o Definicoes\nO estudante beneficia do regime � aplicavel.",
            ),
        ]

        default_analysis = ExtractionQualityAnalyzer().analyze_document(pages)
        configured_analysis = ExtractionQualityAnalyzer(
            PipelineSettings(
                document_ocr_replacement_ratio_threshold=0.50,
                document_ocr_replacement_mix_suspicious_ratio_threshold=0.50,
                document_ocr_replacement_mix_replacement_ratio_threshold=0.50,
            )
        ).analyze_document(pages)

        self.assertTrue(default_analysis["document_likely_corrupted"])
        self.assertFalse(configured_analysis["document_likely_corrupted"])


if __name__ == "__main__":
    unittest.main()
