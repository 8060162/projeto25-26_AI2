"""Regression tests for retrieval safety and latency metrics collection."""

from __future__ import annotations

import unittest

from Chunking.config.settings import PipelineSettings
from retrieval.metrics import RetrievalMetricsCollector
from retrieval.models import GuardrailDecision


class RetrievalMetricsCollectorTests(unittest.TestCase):
    """Protect the lightweight retrieval metrics collector contract."""

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
