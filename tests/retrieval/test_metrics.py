"""Regression tests for retrieval safety and latency metrics collection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from Chunking.config.settings import PipelineSettings
from retrieval.metrics import RetrievalMetricsCollector
from retrieval.models import (
    ContextChunkMetadata,
    EvidenceQualityClassification,
    GuardrailDecision,
    RetrievalContext,
    RetrievalQualitySignals,
    RetrievedChunkResult,
)


class RetrievalMetricsCollectorTests(unittest.TestCase):
    """Protect the lightweight retrieval metrics collector contract."""

    def _build_retrieval_context(
        self,
        *,
        total_input_chunks: int | None = None,
        candidate_chunk_count: int,
        selected_chunk_count: int,
        truncated: bool = False,
        structural_metadata_chunk_count: int = 0,
        selected_chunk_ids: list[str] | None = None,
        effective_top_k: int | None = None,
    ) -> RetrievalContext:
        """Build one deterministic retrieval context carrying explicit quality signals."""

        resolved_total_input_chunks = (
            candidate_chunk_count if total_input_chunks is None else total_input_chunks
        )
        resolved_effective_top_k = (
            resolved_total_input_chunks
            if effective_top_k is None
            else effective_top_k
        )

        return RetrievalContext(
            context_text="Grounded context.",
            chunk_count=selected_chunk_count,
            character_count=len("Grounded context."),
            truncated=truncated,
            retrieval_quality=RetrievalQualitySignals(
                total_input_chunks=resolved_total_input_chunks,
                candidate_chunk_count=candidate_chunk_count,
                selected_chunk_count=selected_chunk_count,
                structural_metadata_chunk_count=structural_metadata_chunk_count,
                truncated=truncated,
                selected_chunk_ids=list(selected_chunk_ids or []),
                metadata={"effective_top_k": resolved_effective_top_k},
            ),
        )

    def _build_metadata_only_retrieval_context(self) -> RetrievalContext:
        """Build one context that resolves retrieval-quality signals from metadata only."""

        selected_chunk = RetrievedChunkResult(
            chunk_id="chunk_expected",
            doc_id="doc_1",
            text="Selected legal chunk.",
            chunk_metadata={
                "article_number": "12",
                "article_title": "Propinas",
            },
            context_metadata=ContextChunkMetadata(),
        )

        return RetrievalContext(
            chunks=[selected_chunk],
            context_text="Selected legal chunk.",
            truncated=True,
            metadata={
                "total_input_chunks": 4,
                "candidate_chunk_count": 2,
                "selected_chunk_ids": ["chunk_expected"],
                "effective_top_k": "invalid",
            },
        )

    def test_build_snapshot_tracks_request_outcomes_and_stage_latency(self) -> None:
        """Ensure request counters and stage timings are exposed in snapshots."""
        collector = RetrievalMetricsCollector(PipelineSettings())

        collector.record_request_started()
        collector.record_request_outcome(successful=True)
        collector.record_request_started()
        collector.record_request_outcome(blocked=True, deflected=True)
        collector.record_stage_latency("pre_guardrails", 12.5)
        collector.record_stage_latency("retrieval", 30.0)

        snapshot = collector.build_snapshot()
        report = collector.build_metric_report()

        self.assertEqual(snapshot.total_requests, 2)
        self.assertEqual(snapshot.successful_requests, 1)
        self.assertEqual(snapshot.blocked_requests, 1)
        self.assertEqual(snapshot.deflected_requests, 1)
        self.assertEqual(snapshot.stage_latency_ms["pre_guardrails"], 12.5)
        self.assertEqual(snapshot.stage_latency_ms["retrieval"], 30.0)
        self.assertEqual(snapshot.total_latency_ms, 42.5)
        self.assertEqual(report["deflection_rate"], 0.5)
        self.assertEqual(report["total_latency_ms"], 42.5)

    def test_record_guardrail_decision_tracks_false_positives_and_jailbreak_resistance(self) -> None:
        """Ensure labeled guardrail outcomes feed robustness counters correctly."""
        collector = RetrievalMetricsCollector(PipelineSettings())
        collector.record_request_started()
        collector.record_request_started()

        collector.record_guardrail_decision(
            GuardrailDecision(
                stage="pre_request",
                allowed=False,
                category="offensive_language",
                action="block",
            ),
            expected_safe=True,
            expected_jailbreak=False,
        )
        collector.record_guardrail_decision(
            GuardrailDecision(
                stage="pre_request",
                allowed=False,
                category="dangerous_command",
                action="block",
            ),
            expected_jailbreak=True,
        )
        collector.record_guardrail_decision(
            GuardrailDecision(
                stage="pre_request",
                allowed=True,
                category="",
                action="allow",
            ),
            expected_jailbreak=True,
        )

        snapshot = collector.build_snapshot()
        report = collector.build_metric_report()

        self.assertEqual(snapshot.false_positive_count, 1)
        self.assertEqual(snapshot.jailbreak_attempt_count, 2)
        self.assertEqual(snapshot.blocked_jailbreak_attempt_count, 1)
        self.assertEqual(report["false_positive_rate"], 0.5)
        self.assertEqual(report["jailbreak_resistance"], 0.5)

    def test_measure_stage_records_positive_latency_automatically(self) -> None:
        """Ensure the collector can time one stage via the context-manager helper."""
        collector = RetrievalMetricsCollector(PipelineSettings())

        with collector.measure_stage("context_builder"):
            for _ in range(1000):
                pass

        snapshot = collector.build_snapshot()

        self.assertIn("context_builder", snapshot.stage_latency_ms)
        self.assertGreaterEqual(snapshot.stage_latency_ms["context_builder"], 0.0)

    def test_disabled_metric_families_do_not_accumulate_optional_data(self) -> None:
        """Ensure disabled metric toggles prevent optional counters and timings."""
        collector = RetrievalMetricsCollector(
            PipelineSettings(
                metrics_track_false_positive_rate=False,
                metrics_track_jailbreak_resistance=False,
                metrics_track_stage_latency=False,
            )
        )

        collector.record_request_started()
        collector.record_guardrail_decision(
            GuardrailDecision(
                stage="pre_request",
                allowed=False,
                category="dangerous_command",
                action="block",
            ),
            expected_safe=True,
            expected_jailbreak=True,
        )
        collector.record_stage_latency("retrieval", 99.0)

        snapshot = collector.build_snapshot()
        report = collector.build_metric_report()

        self.assertEqual(snapshot.false_positive_count, 0)
        self.assertEqual(snapshot.jailbreak_attempt_count, 0)
        self.assertEqual(snapshot.blocked_jailbreak_attempt_count, 0)
        self.assertEqual(snapshot.stage_latency_ms, {})
        self.assertEqual(snapshot.total_latency_ms, 0.0)
        self.assertEqual(report["false_positive_rate"], 0.0)
        self.assertEqual(report["jailbreak_resistance"], 0.0)

    def test_record_retrieval_context_tracks_candidate_selection_and_truncation_signals(
        self,
    ) -> None:
        """Ensure retrieval-quality counters expose candidate, selection, and truncation signals."""
        collector = RetrievalMetricsCollector(
            PipelineSettings(
                metrics_enabled=True,
                metrics_retrieval_quality_enabled=True,
                metrics_track_candidate_pool_size=True,
                metrics_track_selected_context_size=True,
                metrics_track_context_truncation=True,
            )
        )

        collector.record_retrieval_context(
            self._build_retrieval_context(
                total_input_chunks=5,
                candidate_chunk_count=5,
                selected_chunk_count=2,
                truncated=True,
                selected_chunk_ids=["chunk_a", "chunk_b"],
                effective_top_k=6,
            )
        )
        collector.record_retrieval_context(
            self._build_retrieval_context(
                total_input_chunks=3,
                candidate_chunk_count=3,
                selected_chunk_count=1,
                truncated=False,
                selected_chunk_ids=["chunk_c"],
                effective_top_k=4,
            )
        )

        snapshot = collector.build_snapshot()
        report = collector.build_metric_report()

        self.assertEqual(snapshot.retrieval_quality_sample_count, 2)
        self.assertEqual(snapshot.requested_retrieval_count, 10)
        self.assertEqual(snapshot.returned_retrieval_count, 8)
        self.assertEqual(snapshot.retrieved_candidate_count, 8)
        self.assertEqual(snapshot.selected_context_count, 3)
        self.assertEqual(snapshot.truncated_context_count, 1)
        self.assertEqual(report["average_requested_retrieval_breadth"], 5.0)
        self.assertEqual(report["average_returned_retrieval_breadth"], 4.0)
        self.assertEqual(report["retrieval_return_rate"], 0.8)
        self.assertEqual(report["average_candidate_pool_size"], 4.0)
        self.assertEqual(report["candidate_pool_retention_rate"], 1.0)
        self.assertEqual(report["average_selected_context_size"], 1.5)
        self.assertEqual(report["context_selection_rate"], 0.375)
        self.assertEqual(report["context_truncation_rate"], 0.5)

    def test_record_retrieval_context_tracks_structural_presence_and_labeled_recovery(
        self,
    ) -> None:
        """Ensure structural richness and labeled chunk recovery remain regression-protected."""
        collector = RetrievalMetricsCollector(
            PipelineSettings(
                metrics_enabled=True,
                metrics_retrieval_quality_enabled=True,
                metrics_track_structural_richness=True,
            )
        )

        collector.record_retrieval_context(
            self._build_retrieval_context(
                total_input_chunks=4,
                candidate_chunk_count=4,
                selected_chunk_count=2,
                structural_metadata_chunk_count=2,
                selected_chunk_ids=["chunk_expected", "chunk_support"],
                effective_top_k=5,
            ),
            expected_chunk_ids=["chunk_expected"],
        )
        collector.record_retrieval_context(
            self._build_retrieval_context(
                total_input_chunks=4,
                candidate_chunk_count=4,
                selected_chunk_count=1,
                structural_metadata_chunk_count=0,
                selected_chunk_ids=["chunk_other"],
                effective_top_k=5,
            ),
            expected_chunk_ids=["chunk_missing"],
        )

        snapshot = collector.build_snapshot()
        report = collector.build_metric_report()

        self.assertEqual(snapshot.contexts_with_structural_metadata_count, 1)
        self.assertEqual(snapshot.structural_metadata_chunk_count, 2)
        self.assertEqual(snapshot.labeled_recovery_sample_count, 2)
        self.assertEqual(snapshot.recovered_labeled_chunk_count, 1)
        self.assertEqual(report["structural_context_presence_rate"], 0.5)
        self.assertEqual(report["average_structural_metadata_chunks"], 1.0)
        self.assertEqual(report["labeled_chunk_recovery_rate"], 0.5)

    def test_record_retrieval_context_uses_metadata_fallbacks_for_requested_breadth_and_recovery(
        self,
    ) -> None:
        """Ensure metadata-derived quality signals remain deterministic when explicit payloads are absent."""
        collector = RetrievalMetricsCollector(
            PipelineSettings(
                metrics_enabled=True,
                metrics_retrieval_quality_enabled=True,
                metrics_track_candidate_pool_size=True,
                metrics_track_selected_context_size=True,
                metrics_track_context_truncation=True,
                metrics_track_structural_richness=True,
            )
        )

        collector.record_retrieval_context(
            self._build_metadata_only_retrieval_context(),
            expected_chunk_ids=["", "chunk_expected", 7],  # type: ignore[list-item]
        )

        snapshot = collector.build_snapshot()
        report = collector.build_metric_report()

        self.assertEqual(snapshot.retrieval_quality_sample_count, 1)
        self.assertEqual(snapshot.requested_retrieval_count, 4)
        self.assertEqual(snapshot.returned_retrieval_count, 4)
        self.assertEqual(snapshot.retrieved_candidate_count, 2)
        self.assertEqual(snapshot.selected_context_count, 1)
        self.assertEqual(snapshot.truncated_context_count, 1)
        self.assertEqual(snapshot.contexts_with_structural_metadata_count, 1)
        self.assertEqual(snapshot.structural_metadata_chunk_count, 1)
        self.assertEqual(snapshot.labeled_recovery_sample_count, 1)
        self.assertEqual(snapshot.recovered_labeled_chunk_count, 1)
        self.assertEqual(report["average_requested_retrieval_breadth"], 4.0)
        self.assertEqual(report["average_returned_retrieval_breadth"], 4.0)
        self.assertEqual(report["retrieval_return_rate"], 1.0)
        self.assertEqual(report["average_candidate_pool_size"], 2.0)
        self.assertEqual(report["candidate_pool_retention_rate"], 0.5)
        self.assertEqual(report["average_selected_context_size"], 1.0)
        self.assertEqual(report["context_selection_rate"], 0.5)
        self.assertEqual(report["context_truncation_rate"], 1.0)
        self.assertEqual(report["structural_context_presence_rate"], 1.0)
        self.assertEqual(report["average_structural_metadata_chunks"], 1.0)
        self.assertEqual(report["labeled_chunk_recovery_rate"], 1.0)

    def test_record_retrieval_context_tracks_primary_anchor_quality(self) -> None:
        """Ensure primary-anchor quality metrics expose stable document and article anchors."""
        collector = RetrievalMetricsCollector(
            PipelineSettings(
                metrics_enabled=True,
                metrics_retrieval_quality_enabled=True,
            )
        )
        primary_chunk = RetrievedChunkResult(
            chunk_id="chunk_primary",
            doc_id="doc_propinas",
            text="Primary legal evidence.",
            chunk_metadata={
                "article_number": "4",
                "article_title": "Plano geral de pagamento",
            },
            context_metadata=ContextChunkMetadata(),
        )

        collector.record_retrieval_context(
            RetrievalContext(
                chunks=[primary_chunk],
                context_text="Primary legal evidence.",
                metadata={
                    "total_input_chunks": 3,
                    "candidate_chunk_count": 2,
                    "selected_chunk_ids": ["chunk_primary"],
                    "primary_anchor": "doc_propinas:4",
                    "primary_anchor_chunk_ids": ["chunk_primary"],
                },
            )
        )

        report = collector.build_metric_report()

        self.assertEqual(report["primary_anchor_context_count"], 1)
        self.assertEqual(report["primary_anchor_stable_count"], 1)
        self.assertEqual(report["last_primary_anchor_doc_id"], "doc_propinas")
        self.assertEqual(report["last_primary_anchor_article_number"], "4")
        self.assertEqual(report["primary_anchor_presence_rate"], 1.0)
        self.assertEqual(report["primary_anchor_stability_rate"], 1.0)

    def test_record_retrieval_pass_metadata_tracks_retry_and_evidence_classes(
        self,
    ) -> None:
        """Ensure first-pass and second-pass retry metrics are exposed deterministically."""
        collector = RetrievalMetricsCollector(
            PipelineSettings(
                metrics_enabled=True,
                metrics_retrieval_quality_enabled=True,
            )
        )

        collector.record_retrieval_pass_metadata(
            {
                "selected_pass": "second_pass",
                "second_pass_triggered": True,
                "first_pass": {
                    "evidence_strength": "weak",
                    "evidence_conflict": "conflicting",
                    "sufficient_for_answer": False,
                },
                "selected": {
                    "pass_name": "second_pass",
                    "evidence_strength": "strong",
                    "evidence_conflict": "non_conflicting",
                    "sufficient_for_answer": True,
                },
            }
        )

        report = collector.build_metric_report()

        self.assertEqual(report["retry_triggered_count"], 1)
        self.assertEqual(report["retry_success_count"], 1)
        self.assertEqual(report["retry_success_rate"], 1.0)
        self.assertEqual(report["first_pass_evidence_classification_count"], 1)
        self.assertEqual(report["second_pass_evidence_classification_count"], 1)
        self.assertEqual(
            report["last_first_pass_evidence_classification"],
            "weak:conflicting:insufficient",
        )
        self.assertEqual(
            report["last_second_pass_evidence_classification"],
            "strong:non_conflicting:sufficient",
        )

    def test_record_retrieval_context_tracks_diagnostic_outcomes_and_cautious_answers(
        self,
    ) -> None:
        """Ensure selected-context diagnostics remain visible in runtime metrics."""
        collector = RetrievalMetricsCollector(
            PipelineSettings(
                metrics_enabled=True,
                metrics_retrieval_quality_enabled=True,
            )
        )

        collector.record_retrieval_context(
            RetrievalContext(
                context_text="Most likely governing anchor with close competitors.",
                evidence_quality=EvidenceQualityClassification(
                    strength="strong",
                    ambiguity="ambiguous",
                    conflict="none",
                    sufficient_for_answer=True,
                ),
            )
        )
        collector.record_retrieval_context(
            RetrievalContext(
                context_text="",
                evidence_quality=EvidenceQualityClassification(
                    strength="weak",
                    ambiguity="clear",
                    conflict="conflicting",
                    sufficient_for_answer=False,
                ),
            )
        )

        report = collector.build_metric_report()

        self.assertEqual(report["cautious_answer_outcome_count"], 1)
        self.assertEqual(report["diagnostic_cautious_answer_count"], 1)
        self.assertEqual(report["diagnostic_retrieval_failure_count"], 1)
        self.assertEqual(report["diagnostic_evidence_insufficiency_count"], 0)
        self.assertEqual(report["diagnostic_grounded_answer_count"], 0)
        self.assertEqual(
            report["last_diagnostic_outcome_category"],
            "retrieval_failure",
        )

    def test_record_request_outcome_tracks_deflection_reason_categories(self) -> None:
        """Ensure deflection categories remain explicit and ignore unknown labels."""
        collector = RetrievalMetricsCollector(PipelineSettings())

        collector.record_request_started()
        collector.record_request_outcome(
            deflected=True,
            deflection_reason_category="evidence_routing",
        )
        collector.record_request_started()
        collector.record_request_outcome(
            deflected=True,
            deflection_reason_category="grounding_validation",
        )
        collector.record_request_started()
        collector.record_request_outcome(
            deflected=True,
            deflection_reason_category="unknown_reason",
        )

        report = collector.build_metric_report()

        self.assertEqual(report["deflected_requests"], 3)
        self.assertEqual(report["deflection_evidence_routing_count"], 1)
        self.assertEqual(report["deflection_grounding_validation_count"], 1)
        self.assertEqual(report["deflection_pre_guardrails_count"], 0)
        self.assertEqual(report["deflection_post_guardrails_count"], 0)

    def test_disabled_retrieval_quality_metrics_do_not_accumulate_context_signals(
        self,
    ) -> None:
        """Ensure retrieval-quality counters stay disabled even when contexts are recorded."""
        with patch(
            "Chunking.config.settings._load_appsettings",
            return_value={"metrics": {"enabled": True}},
        ):
            collector = RetrievalMetricsCollector(
                PipelineSettings(
                    metrics_enabled=True,
                    metrics_retrieval_quality_enabled=False,
                    metrics_track_candidate_pool_size=True,
                    metrics_track_selected_context_size=True,
                    metrics_track_context_truncation=True,
                    metrics_track_structural_richness=True,
                )
            )

        collector.record_retrieval_context(
            self._build_retrieval_context(
                total_input_chunks=6,
                candidate_chunk_count=6,
                selected_chunk_count=2,
                truncated=True,
                structural_metadata_chunk_count=2,
                selected_chunk_ids=["chunk_expected"],
                effective_top_k=8,
            ),
            expected_chunk_ids=["chunk_expected"],
        )

        snapshot = collector.build_snapshot()
        report = collector.build_metric_report()

        self.assertEqual(snapshot.retrieval_quality_sample_count, 0)
        self.assertEqual(snapshot.requested_retrieval_count, 0)
        self.assertEqual(snapshot.returned_retrieval_count, 0)
        self.assertEqual(snapshot.retrieved_candidate_count, 0)
        self.assertEqual(snapshot.selected_context_count, 0)
        self.assertEqual(snapshot.truncated_context_count, 0)
        self.assertEqual(snapshot.contexts_with_structural_metadata_count, 0)
        self.assertEqual(snapshot.structural_metadata_chunk_count, 0)
        self.assertEqual(snapshot.labeled_recovery_sample_count, 0)
        self.assertEqual(snapshot.recovered_labeled_chunk_count, 0)
        self.assertEqual(report["average_requested_retrieval_breadth"], 0.0)
        self.assertEqual(report["average_returned_retrieval_breadth"], 0.0)
        self.assertEqual(report["retrieval_return_rate"], 0.0)
        self.assertEqual(report["average_candidate_pool_size"], 0.0)
        self.assertEqual(report["candidate_pool_retention_rate"], 0.0)
        self.assertEqual(report["average_selected_context_size"], 0.0)
        self.assertEqual(report["context_selection_rate"], 0.0)
        self.assertEqual(report["context_truncation_rate"], 0.0)
        self.assertEqual(report["structural_context_presence_rate"], 0.0)
        self.assertEqual(report["average_structural_metadata_chunks"], 0.0)
        self.assertEqual(report["labeled_chunk_recovery_rate"], 0.0)

    def test_build_metric_report_returns_zero_rates_when_denominators_are_missing(self) -> None:
        """Ensure metric ratios stay defined when no eligible samples exist yet."""
        collector = RetrievalMetricsCollector(PipelineSettings())

        report = collector.build_metric_report()

        self.assertEqual(report["deflection_rate"], 0.0)
        self.assertEqual(report["false_positive_rate"], 0.0)
        self.assertEqual(report["jailbreak_resistance"], 0.0)
        self.assertEqual(report["total_latency_ms"], 0.0)

    def test_reset_clears_counters_and_latency_aggregates(self) -> None:
        """Ensure reset clears previously accumulated metric calculations."""
        collector = RetrievalMetricsCollector(PipelineSettings())

        collector.record_request_started()
        collector.record_request_outcome(blocked=True, deflected=True)
        collector.record_guardrail_decision(
            GuardrailDecision(
                stage="pre_request",
                allowed=False,
                category="dangerous_command",
                action="block",
            ),
            expected_safe=True,
            expected_jailbreak=True,
        )
        collector.record_stage_latency("pre_guardrails", 8.0)

        collector.reset()

        snapshot = collector.build_snapshot()
        report = collector.build_metric_report()

        self.assertEqual(snapshot.total_requests, 0)
        self.assertEqual(snapshot.blocked_requests, 0)
        self.assertEqual(snapshot.deflected_requests, 0)
        self.assertEqual(snapshot.false_positive_count, 0)
        self.assertEqual(snapshot.jailbreak_attempt_count, 0)
        self.assertEqual(snapshot.blocked_jailbreak_attempt_count, 0)
        self.assertEqual(snapshot.stage_latency_ms, {})
        self.assertEqual(snapshot.total_latency_ms, 0.0)
        self.assertEqual(report["deflection_rate"], 0.0)
        self.assertEqual(report["false_positive_rate"], 0.0)
        self.assertEqual(report["jailbreak_resistance"], 0.0)
        self.assertEqual(report["total_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
