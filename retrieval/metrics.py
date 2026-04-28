from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, Generator, List, Optional

from Chunking.config.settings import PipelineSettings
from retrieval.models import (
    GuardrailDecision,
    MetricsSnapshot,
    RetrievalContext,
    RetrievalQualitySignals,
)


def _normalize_stage_name(stage_name: str) -> str:
    """
    Normalize one stage name into a stable non-empty identifier.

    Parameters
    ----------
    stage_name : str
        Raw stage name supplied by the caller.

    Returns
    -------
    str
        Stripped stage name.
    """

    normalized_stage_name = stage_name.strip()
    if not normalized_stage_name:
        raise ValueError("Stage name cannot be empty.")
    return normalized_stage_name


def _normalize_metric_label(value: Any, *, fallback_value: str = "unknown") -> str:
    """
    Normalize one metrics label into a stable non-empty identifier.

    Parameters
    ----------
    value : Any
        Candidate metrics label.

    fallback_value : str
        Label used when the candidate value is empty or invalid.

    Returns
    -------
    str
        Normalized label value.
    """

    if isinstance(value, str):
        normalized_value = value.strip()
        if normalized_value:
            return normalized_value

    return fallback_value


class RetrievalMetricsCollector:
    """
    Collect lightweight runtime metrics for retrieval safety, latency, and context quality.

    Design goals
    ------------
    - keep metrics collection local to the retrieval flow
    - expose deterministic counters and derived rates for tests and logs
    - respect shared runtime toggles from `PipelineSettings`
    """

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """
        Initialize the collector with the shared runtime settings.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared project settings. Default settings are loaded when omitted.
        """

        self.settings = settings or PipelineSettings()
        self._total_requests = 0
        self._successful_requests = 0
        self._blocked_requests = 0
        self._deflected_requests = 0
        self._false_positive_count = 0
        self._jailbreak_attempt_count = 0
        self._blocked_jailbreak_attempt_count = 0
        self._retrieval_quality_sample_count = 0
        self._requested_retrieval_count = 0
        self._returned_retrieval_count = 0
        self._retrieved_candidate_count = 0
        self._selected_context_count = 0
        self._truncated_context_count = 0
        self._contexts_with_structural_metadata_count = 0
        self._structural_metadata_chunk_count = 0
        self._labeled_recovery_sample_count = 0
        self._recovered_labeled_chunk_count = 0
        self._retry_triggered_count = 0
        self._retry_success_count = 0
        self._first_pass_evidence_classification_counts: Dict[str, int] = {}
        self._second_pass_evidence_classification_counts: Dict[str, int] = {}
        self._last_first_pass_evidence_classification = ""
        self._last_second_pass_evidence_classification = ""
        self._primary_anchor_context_count = 0
        self._primary_anchor_stable_count = 0
        self._last_primary_anchor_doc_id = ""
        self._last_primary_anchor_article_number = ""
        self._cautious_answer_outcome_count = 0
        self._diagnostic_outcome_counts: Dict[str, int] = {}
        self._last_diagnostic_outcome_category = ""
        self._deflection_reason_counts: Dict[str, int] = {}
        self._stage_latency_ms: Dict[str, float] = {}

    def record_request_started(self) -> None:
        """
        Record one retrieval request entering the flow.
        """

        if not self.settings.metrics_enabled:
            return

        self._total_requests += 1

    def record_request_outcome(
        self,
        *,
        successful: bool = False,
        blocked: bool = False,
        deflected: bool = False,
        deflection_reason_category: str = "",
    ) -> None:
        """
        Record the final outcome of one retrieval request.

        Parameters
        ----------
        successful : bool
            Whether the request completed with a grounded successful answer.

        blocked : bool
            Whether the request was blocked by deterministic guardrails.

        deflected : bool
            Whether the request was deflected instead of answered directly.

        deflection_reason_category : str
            Optional stable deflection category used for observability.
        """

        if not self.settings.metrics_enabled:
            return

        if successful:
            self._successful_requests += 1
        if blocked:
            self._blocked_requests += 1
        if deflected:
            self._deflected_requests += 1
            normalized_category = self._normalize_deflection_reason_category(
                deflection_reason_category
            )
            if normalized_category:
                self._deflection_reason_counts[normalized_category] = (
                    self._deflection_reason_counts.get(normalized_category, 0) + 1
                )

    def record_guardrail_decision(
        self,
        decision: GuardrailDecision,
        *,
        expected_safe: Optional[bool] = None,
        expected_jailbreak: Optional[bool] = None,
    ) -> None:
        """
        Record one guardrail decision for robustness and safety metrics.

        Parameters
        ----------
        decision : GuardrailDecision
            Guardrail decision emitted by pre-request or post-response checks.

        expected_safe : Optional[bool]
            Optional labeled expectation for whether the evaluated content
            should have been allowed. When explicitly `True` and the decision is
            not allowed, the event counts as a false positive.

        expected_jailbreak : Optional[bool]
            Optional labeled expectation for whether the evaluated content is a
            jailbreak attempt. When explicitly `True`, the attempt contributes
            to jailbreak-resistance counters.
        """

        if not self.settings.metrics_enabled:
            return

        if (
            self.settings.metrics_track_false_positive_rate
            and expected_safe is True
            and not decision.allowed
        ):
            self._false_positive_count += 1

        if self.settings.metrics_track_jailbreak_resistance and expected_jailbreak is True:
            self._jailbreak_attempt_count += 1
            if not decision.allowed:
                self._blocked_jailbreak_attempt_count += 1

    def record_stage_latency(self, stage_name: str, elapsed_ms: float) -> None:
        """
        Record one measured stage latency in milliseconds.

        Parameters
        ----------
        stage_name : str
            Stable retrieval stage identifier.

        elapsed_ms : float
            Elapsed time in milliseconds for the stage execution.
        """

        if not self.settings.metrics_enabled or not self.settings.metrics_track_stage_latency:
            return

        normalized_stage_name = _normalize_stage_name(stage_name)
        self._stage_latency_ms[normalized_stage_name] = max(0.0, float(elapsed_ms))

    def record_retrieval_context(
        self,
        retrieval_context: RetrievalContext,
        *,
        expected_chunk_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Record retrieval/context quality signals from one built context.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Final context assembled by the context builder.

        expected_chunk_ids : Optional[List[str]]
            Optional labeled chunk identifiers expected to survive into the
            selected context for deterministic recovery checks.
        """

        if (
            not self.settings.metrics_enabled
            or not self.settings.metrics_retrieval_quality_enabled
            or not isinstance(retrieval_context, RetrievalContext)
        ):
            return

        retrieval_quality = retrieval_context.retrieval_quality
        self._retrieval_quality_sample_count += 1
        self._requested_retrieval_count += self._read_non_negative_quality_metadata(
            retrieval_quality,
            key="effective_top_k",
            fallback_value=retrieval_quality.total_input_chunks,
        )
        self._returned_retrieval_count += retrieval_quality.total_input_chunks

        if self.settings.metrics_track_candidate_pool_size:
            self._retrieved_candidate_count += retrieval_quality.candidate_chunk_count

        if self.settings.metrics_track_selected_context_size:
            self._selected_context_count += retrieval_quality.selected_chunk_count

        if self.settings.metrics_track_context_truncation and retrieval_quality.truncated:
            self._truncated_context_count += 1

        if self.settings.metrics_track_structural_richness:
            self._structural_metadata_chunk_count += (
                retrieval_quality.structural_metadata_chunk_count
            )
            if retrieval_quality.structural_metadata_chunk_count > 0:
                self._contexts_with_structural_metadata_count += 1

        self._record_primary_anchor_quality(retrieval_context)
        self._record_context_outcome_signals(retrieval_context)

        normalized_expected_chunk_ids = self._normalize_expected_chunk_ids(
            expected_chunk_ids
        )
        if not normalized_expected_chunk_ids:
            return

        self._labeled_recovery_sample_count += 1
        selected_chunk_ids = set(retrieval_quality.selected_chunk_ids)
        if any(
            expected_chunk_id in selected_chunk_ids
            for expected_chunk_id in normalized_expected_chunk_ids
        ):
            self._recovered_labeled_chunk_count += 1

    def record_retrieval_pass_metadata(
        self,
        retrieval_pass_metadata: Optional[Dict[str, Any]],
    ) -> None:
        """
        Record first-pass and second-pass retrieval metadata.

        Parameters
        ----------
        retrieval_pass_metadata : Optional[Dict[str, Any]]
            Serializable service metadata describing first-pass and selected-pass
            evidence state.
        """

        if (
            not self.settings.metrics_enabled
            or not self.settings.metrics_retrieval_quality_enabled
            or not isinstance(retrieval_pass_metadata, dict)
        ):
            return

        first_pass_metadata = self._read_mapping(
            retrieval_pass_metadata.get("first_pass")
        )
        selected_pass_metadata = self._read_mapping(
            retrieval_pass_metadata.get("selected")
        )
        selected_pass_name = _normalize_metric_label(
            retrieval_pass_metadata.get("selected_pass"),
            fallback_value="",
        )
        second_pass_triggered = bool(
            retrieval_pass_metadata.get("second_pass_triggered")
        )

        if first_pass_metadata:
            self._last_first_pass_evidence_classification = (
                self._resolve_evidence_classification(first_pass_metadata)
            )
            self._increment_label_count(
                self._first_pass_evidence_classification_counts,
                self._last_first_pass_evidence_classification,
            )

        if second_pass_triggered:
            self._retry_triggered_count += 1

            second_pass_metadata = selected_pass_metadata
            explicit_second_pass_metadata = self._read_mapping(
                retrieval_pass_metadata.get("second_pass")
            )
            if explicit_second_pass_metadata:
                second_pass_metadata = explicit_second_pass_metadata

            if second_pass_metadata:
                self._last_second_pass_evidence_classification = (
                    self._resolve_evidence_classification(second_pass_metadata)
                )
                self._increment_label_count(
                    self._second_pass_evidence_classification_counts,
                    self._last_second_pass_evidence_classification,
                )

            if selected_pass_name == "second_pass" and self._pass_allows_generation(
                selected_pass_metadata
            ):
                self._retry_success_count += 1

    @contextmanager
    def measure_stage(self, stage_name: str) -> Generator[None, None, None]:
        """
        Measure one retrieval stage and store its latency automatically.

        Parameters
        ----------
        stage_name : str
            Stable retrieval stage identifier.
        """

        normalized_stage_name = _normalize_stage_name(stage_name)
        started_at = perf_counter()

        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            self.record_stage_latency(normalized_stage_name, elapsed_ms)

    def build_snapshot(self) -> MetricsSnapshot:
        """
        Build one immutable metrics snapshot from the current counters.

        Returns
        -------
        MetricsSnapshot
            Snapshot describing the current retrieval metrics state.
        """

        total_latency_ms = 0.0
        if self.settings.metrics_track_stage_latency:
            total_latency_ms = sum(self._stage_latency_ms.values())

        return MetricsSnapshot(
            total_requests=self._total_requests,
            successful_requests=self._successful_requests,
            blocked_requests=self._blocked_requests,
            deflected_requests=self._deflected_requests,
            false_positive_count=self._false_positive_count,
            jailbreak_attempt_count=self._jailbreak_attempt_count,
            blocked_jailbreak_attempt_count=self._blocked_jailbreak_attempt_count,
            retrieval_quality_sample_count=self._retrieval_quality_sample_count,
            requested_retrieval_count=self._requested_retrieval_count,
            returned_retrieval_count=self._returned_retrieval_count,
            retrieved_candidate_count=self._retrieved_candidate_count,
            selected_context_count=self._selected_context_count,
            truncated_context_count=self._truncated_context_count,
            contexts_with_structural_metadata_count=(
                self._contexts_with_structural_metadata_count
            ),
            structural_metadata_chunk_count=self._structural_metadata_chunk_count,
            labeled_recovery_sample_count=self._labeled_recovery_sample_count,
            recovered_labeled_chunk_count=self._recovered_labeled_chunk_count,
            stage_latency_ms=dict(self._stage_latency_ms),
            total_latency_ms=total_latency_ms,
        )

    def build_metric_report(self) -> Dict[str, float | int | str]:
        """
        Build one flat metric report for logs and assertions.

        Returns
        -------
        Dict[str, float | int | str]
            Counter and rate summary derived from the current snapshot.
        """

        snapshot = self.build_snapshot()

        return {
            "total_requests": snapshot.total_requests,
            "successful_requests": snapshot.successful_requests,
            "blocked_requests": snapshot.blocked_requests,
            "deflected_requests": snapshot.deflected_requests,
            "false_positive_count": snapshot.false_positive_count,
            "jailbreak_attempt_count": snapshot.jailbreak_attempt_count,
            "blocked_jailbreak_attempt_count": snapshot.blocked_jailbreak_attempt_count,
            "retrieval_quality_sample_count": snapshot.retrieval_quality_sample_count,
            "requested_retrieval_count": snapshot.requested_retrieval_count,
            "returned_retrieval_count": snapshot.returned_retrieval_count,
            "retrieved_candidate_count": snapshot.retrieved_candidate_count,
            "selected_context_count": snapshot.selected_context_count,
            "truncated_context_count": snapshot.truncated_context_count,
            "contexts_with_structural_metadata_count": (
                snapshot.contexts_with_structural_metadata_count
            ),
            "structural_metadata_chunk_count": (
                snapshot.structural_metadata_chunk_count
            ),
            "labeled_recovery_sample_count": snapshot.labeled_recovery_sample_count,
            "recovered_labeled_chunk_count": snapshot.recovered_labeled_chunk_count,
            "retry_triggered_count": self._retry_triggered_count,
            "retry_success_count": self._retry_success_count,
            "first_pass_evidence_classification_count": sum(
                self._first_pass_evidence_classification_counts.values()
            ),
            "second_pass_evidence_classification_count": sum(
                self._second_pass_evidence_classification_counts.values()
            ),
            "last_first_pass_evidence_classification": (
                self._last_first_pass_evidence_classification
            ),
            "last_second_pass_evidence_classification": (
                self._last_second_pass_evidence_classification
            ),
            "primary_anchor_context_count": self._primary_anchor_context_count,
            "primary_anchor_stable_count": self._primary_anchor_stable_count,
            "last_primary_anchor_doc_id": self._last_primary_anchor_doc_id,
            "last_primary_anchor_article_number": (
                self._last_primary_anchor_article_number
            ),
            "cautious_answer_outcome_count": self._cautious_answer_outcome_count,
            "last_diagnostic_outcome_category": (
                self._last_diagnostic_outcome_category
            ),
            "diagnostic_retrieval_failure_count": (
                self._diagnostic_outcome_counts.get("retrieval_failure", 0)
            ),
            "diagnostic_evidence_insufficiency_count": (
                self._diagnostic_outcome_counts.get("evidence_insufficiency", 0)
            ),
            "diagnostic_cautious_answer_count": (
                self._diagnostic_outcome_counts.get("cautious_answer", 0)
            ),
            "diagnostic_grounded_answer_count": (
                self._diagnostic_outcome_counts.get("grounded_answer", 0)
            ),
            "deflection_evidence_routing_count": self._deflection_reason_counts.get(
                "evidence_routing",
                0,
            ),
            "deflection_grounding_validation_count": (
                self._deflection_reason_counts.get("grounding_validation", 0)
            ),
            "deflection_pre_guardrails_count": self._deflection_reason_counts.get(
                "pre_guardrails",
                0,
            ),
            "deflection_post_guardrails_count": self._deflection_reason_counts.get(
                "post_guardrails",
                0,
            ),
            "deflection_rate": self._safe_divide(
                numerator=snapshot.deflected_requests,
                denominator=snapshot.total_requests,
                enabled=self.settings.metrics_track_deflection_rate,
            ),
            "false_positive_rate": self._safe_divide(
                numerator=snapshot.false_positive_count,
                denominator=snapshot.total_requests,
                enabled=self.settings.metrics_track_false_positive_rate,
            ),
            "jailbreak_resistance": self._safe_divide(
                numerator=snapshot.blocked_jailbreak_attempt_count,
                denominator=snapshot.jailbreak_attempt_count,
                enabled=self.settings.metrics_track_jailbreak_resistance,
            ),
            "average_requested_retrieval_breadth": self._safe_divide(
                numerator=snapshot.requested_retrieval_count,
                denominator=snapshot.retrieval_quality_sample_count,
                enabled=self.settings.metrics_retrieval_quality_enabled,
            ),
            "average_returned_retrieval_breadth": self._safe_divide(
                numerator=snapshot.returned_retrieval_count,
                denominator=snapshot.retrieval_quality_sample_count,
                enabled=self.settings.metrics_retrieval_quality_enabled,
            ),
            "retrieval_return_rate": self._safe_divide(
                numerator=snapshot.returned_retrieval_count,
                denominator=snapshot.requested_retrieval_count,
                enabled=self.settings.metrics_retrieval_quality_enabled,
            ),
            "average_candidate_pool_size": self._safe_divide(
                numerator=snapshot.retrieved_candidate_count,
                denominator=snapshot.retrieval_quality_sample_count,
                enabled=(
                    self.settings.metrics_retrieval_quality_enabled
                    and self.settings.metrics_track_candidate_pool_size
                ),
            ),
            "candidate_pool_retention_rate": self._safe_divide(
                numerator=snapshot.retrieved_candidate_count,
                denominator=snapshot.returned_retrieval_count,
                enabled=(
                    self.settings.metrics_retrieval_quality_enabled
                    and self.settings.metrics_track_candidate_pool_size
                ),
            ),
            "average_selected_context_size": self._safe_divide(
                numerator=snapshot.selected_context_count,
                denominator=snapshot.retrieval_quality_sample_count,
                enabled=(
                    self.settings.metrics_retrieval_quality_enabled
                    and self.settings.metrics_track_selected_context_size
                ),
            ),
            "context_selection_rate": self._safe_divide(
                numerator=snapshot.selected_context_count,
                denominator=snapshot.retrieved_candidate_count,
                enabled=(
                    self.settings.metrics_retrieval_quality_enabled
                    and self.settings.metrics_track_candidate_pool_size
                    and self.settings.metrics_track_selected_context_size
                ),
            ),
            "context_truncation_rate": self._safe_divide(
                numerator=snapshot.truncated_context_count,
                denominator=snapshot.retrieval_quality_sample_count,
                enabled=(
                    self.settings.metrics_retrieval_quality_enabled
                    and self.settings.metrics_track_context_truncation
                ),
            ),
            "structural_context_presence_rate": self._safe_divide(
                numerator=snapshot.contexts_with_structural_metadata_count,
                denominator=snapshot.retrieval_quality_sample_count,
                enabled=(
                    self.settings.metrics_retrieval_quality_enabled
                    and self.settings.metrics_track_structural_richness
                ),
            ),
            "average_structural_metadata_chunks": self._safe_divide(
                numerator=snapshot.structural_metadata_chunk_count,
                denominator=snapshot.retrieval_quality_sample_count,
                enabled=(
                    self.settings.metrics_retrieval_quality_enabled
                    and self.settings.metrics_track_structural_richness
                ),
            ),
            "labeled_chunk_recovery_rate": self._safe_divide(
                numerator=snapshot.recovered_labeled_chunk_count,
                denominator=snapshot.labeled_recovery_sample_count,
                enabled=self.settings.metrics_retrieval_quality_enabled,
            ),
            "retry_success_rate": self._safe_divide(
                numerator=self._retry_success_count,
                denominator=self._retry_triggered_count,
                enabled=self.settings.metrics_retrieval_quality_enabled,
            ),
            "primary_anchor_presence_rate": self._safe_divide(
                numerator=self._primary_anchor_context_count,
                denominator=snapshot.retrieval_quality_sample_count,
                enabled=self.settings.metrics_retrieval_quality_enabled,
            ),
            "primary_anchor_stability_rate": self._safe_divide(
                numerator=self._primary_anchor_stable_count,
                denominator=self._primary_anchor_context_count,
                enabled=self.settings.metrics_retrieval_quality_enabled,
            ),
            "total_latency_ms": snapshot.total_latency_ms,
        }

    def reset(self) -> None:
        """
        Reset all collected counters and latencies.
        """

        self._total_requests = 0
        self._successful_requests = 0
        self._blocked_requests = 0
        self._deflected_requests = 0
        self._false_positive_count = 0
        self._jailbreak_attempt_count = 0
        self._blocked_jailbreak_attempt_count = 0
        self._retrieval_quality_sample_count = 0
        self._requested_retrieval_count = 0
        self._returned_retrieval_count = 0
        self._retrieved_candidate_count = 0
        self._selected_context_count = 0
        self._truncated_context_count = 0
        self._contexts_with_structural_metadata_count = 0
        self._structural_metadata_chunk_count = 0
        self._labeled_recovery_sample_count = 0
        self._recovered_labeled_chunk_count = 0
        self._retry_triggered_count = 0
        self._retry_success_count = 0
        self._first_pass_evidence_classification_counts.clear()
        self._second_pass_evidence_classification_counts.clear()
        self._last_first_pass_evidence_classification = ""
        self._last_second_pass_evidence_classification = ""
        self._primary_anchor_context_count = 0
        self._primary_anchor_stable_count = 0
        self._last_primary_anchor_doc_id = ""
        self._last_primary_anchor_article_number = ""
        self._cautious_answer_outcome_count = 0
        self._diagnostic_outcome_counts.clear()
        self._last_diagnostic_outcome_category = ""
        self._deflection_reason_counts.clear()
        self._stage_latency_ms.clear()

    def _record_primary_anchor_quality(
        self,
        retrieval_context: RetrievalContext,
    ) -> None:
        """
        Record primary-anchor observability signals from one context.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Retrieval context carrying optional primary-anchor metadata.
        """

        primary_anchor_chunk_ids = set(
            self._normalize_expected_chunk_ids(
                retrieval_context.metadata.get("primary_anchor_chunk_ids")
            )
        )
        raw_primary_anchor = retrieval_context.metadata.get("primary_anchor")
        primary_anchor = _normalize_metric_label(raw_primary_anchor, fallback_value="")

        if not primary_anchor and not primary_anchor_chunk_ids:
            return

        self._primary_anchor_context_count += 1

        primary_chunks = [
            chunk
            for chunk in retrieval_context.chunks
            if not primary_anchor_chunk_ids
            or chunk.chunk_id in primary_anchor_chunk_ids
        ]
        doc_ids = {
            chunk.context_metadata.doc_id
            for chunk in primary_chunks
            if chunk.context_metadata.doc_id
        }
        article_numbers = {
            chunk.context_metadata.article_number
            for chunk in primary_chunks
            if chunk.context_metadata.article_number
        }

        if doc_ids:
            self._last_primary_anchor_doc_id = sorted(doc_ids)[0]
        if article_numbers:
            self._last_primary_anchor_article_number = sorted(article_numbers)[0]

        if self._primary_anchor_is_stable(doc_ids, article_numbers):
            self._primary_anchor_stable_count += 1

    def _record_context_outcome_signals(
        self,
        retrieval_context: RetrievalContext,
    ) -> None:
        """
        Record diagnostic and cautious-answer signals from one selected context.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Retrieval context carrying the selected-pass evidence state.
        """

        diagnostic_outcome_category = self._resolve_context_diagnostic_outcome(
            retrieval_context
        )
        if not diagnostic_outcome_category:
            return

        self._last_diagnostic_outcome_category = diagnostic_outcome_category
        self._increment_label_count(
            self._diagnostic_outcome_counts,
            diagnostic_outcome_category,
        )

        if diagnostic_outcome_category == "cautious_answer":
            self._cautious_answer_outcome_count += 1

    def _resolve_context_diagnostic_outcome(
        self,
        retrieval_context: RetrievalContext,
    ) -> str:
        """
        Resolve one stable diagnostic outcome from the selected context state.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Retrieval context carrying evidence classification and text.

        Returns
        -------
        str
            Stable diagnostic outcome category, otherwise an empty string.
        """

        evidence_quality = retrieval_context.evidence_quality
        if evidence_quality is None:
            if retrieval_context.context_text.strip():
                return ""
            return "evidence_insufficiency"

        explicit_category = _normalize_metric_label(
            evidence_quality.diagnostic_category,
            fallback_value="",
        )
        if explicit_category in {
            "retrieval_failure",
            "evidence_insufficiency",
            "cautious_answer",
            "grounded_answer",
        }:
            return explicit_category

        if evidence_quality.conflict == "conflicting":
            return "retrieval_failure"
        if not evidence_quality.sufficient_for_answer:
            return "evidence_insufficiency"
        if evidence_quality.ambiguity == "ambiguous":
            return "cautious_answer"
        return "grounded_answer"

    def _primary_anchor_is_stable(
        self,
        doc_ids: set[str],
        article_numbers: set[str],
    ) -> bool:
        """
        Determine whether one primary anchor resolves to a single legal anchor.

        Parameters
        ----------
        doc_ids : set[str]
            Distinct document identifiers found in primary-anchor chunks.

        article_numbers : set[str]
            Distinct article numbers found in primary-anchor chunks.

        Returns
        -------
        bool
            `True` when the anchor resolves to one document and at most one
            article number.
        """

        return len(doc_ids) == 1 and len(article_numbers) <= 1

    def _normalize_deflection_reason_category(
        self,
        category: str,
    ) -> str:
        """
        Normalize one deflection category into the supported metrics taxonomy.

        Parameters
        ----------
        category : str
            Candidate deflection category.

        Returns
        -------
        str
            Supported category, otherwise an empty string.
        """

        normalized_category = _normalize_metric_label(category, fallback_value="")
        if normalized_category in {
            "evidence_routing",
            "grounding_validation",
            "pre_guardrails",
            "post_guardrails",
        }:
            return normalized_category

        return ""

    def _read_mapping(
        self,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Read one mapping value into a detached dictionary.

        Parameters
        ----------
        value : Any
            Candidate mapping payload.

        Returns
        -------
        Dict[str, Any]
            Detached dictionary when valid, otherwise an empty dictionary.
        """

        if not isinstance(value, dict):
            return {}

        return dict(value)

    def _resolve_evidence_classification(
        self,
        pass_metadata: Dict[str, Any],
    ) -> str:
        """
        Resolve one stable evidence classification label from pass metadata.

        Parameters
        ----------
        pass_metadata : Dict[str, Any]
            Metadata summary for one retrieval pass.

        Returns
        -------
        str
            Composite evidence label.
        """

        evidence_strength = _normalize_metric_label(
            pass_metadata.get("evidence_strength")
        )
        evidence_conflict = _normalize_metric_label(
            pass_metadata.get("evidence_conflict")
        )
        sufficient_for_answer = bool(pass_metadata.get("sufficient_for_answer"))
        sufficiency_label = "sufficient" if sufficient_for_answer else "insufficient"

        return f"{evidence_strength}:{evidence_conflict}:{sufficiency_label}"

    def _pass_allows_generation(
        self,
        pass_metadata: Dict[str, Any],
    ) -> bool:
        """
        Determine whether one summarized retrieval pass allowed generation.

        Parameters
        ----------
        pass_metadata : Dict[str, Any]
            Metadata summary for one retrieval pass.

        Returns
        -------
        bool
            `True` when evidence is sufficient and not conflicting.
        """

        if not pass_metadata:
            return False

        return bool(pass_metadata.get("sufficient_for_answer")) and (
            _normalize_metric_label(pass_metadata.get("evidence_conflict"))
            != "conflicting"
        )

    def _increment_label_count(
        self,
        label_counts: Dict[str, int],
        label: str,
    ) -> None:
        """
        Increment one label counter in place.

        Parameters
        ----------
        label_counts : Dict[str, int]
            Mutable label counter mapping.

        label : str
            Label to increment.
        """

        normalized_label = _normalize_metric_label(label)
        label_counts[normalized_label] = label_counts.get(normalized_label, 0) + 1

    def _normalize_expected_chunk_ids(
        self,
        expected_chunk_ids: Optional[List[str]],
    ) -> List[str]:
        """
        Normalize optional labeled chunk identifiers into stable string values.

        Parameters
        ----------
        expected_chunk_ids : Optional[List[str]]
            Optional labeled chunk identifiers.

        Returns
        -------
        List[str]
            Ordered non-empty string identifiers.
        """

        if not isinstance(expected_chunk_ids, list):
            return []

        normalized_chunk_ids: List[str] = []

        for raw_chunk_id in expected_chunk_ids:
            if not isinstance(raw_chunk_id, str):
                continue

            normalized_chunk_id = raw_chunk_id.strip()
            if normalized_chunk_id:
                normalized_chunk_ids.append(normalized_chunk_id)

        return normalized_chunk_ids

    def _read_non_negative_quality_metadata(
        self,
        retrieval_quality: RetrievalQualitySignals,
        *,
        key: str,
        fallback_value: int = 0,
    ) -> int:
        """
        Read one non-negative integer signal from retrieval-quality metadata.

        Parameters
        ----------
        retrieval_quality : RetrievalQualitySignals
            Retrieval-quality payload carrying optional metadata values.

        key : str
            Metadata key containing the desired numeric signal.

        fallback_value : int
            Fallback value used when the metadata entry is absent or invalid.

        Returns
        -------
        int
            Non-negative integer signal.
        """

        if not isinstance(key, str) or not key.strip():
            return max(0, int(fallback_value))

        raw_value = retrieval_quality.metadata.get(key, fallback_value)
        try:
            normalized_value = int(raw_value)
        except (TypeError, ValueError):
            normalized_value = int(fallback_value)

        return max(0, normalized_value)

    def _safe_divide(
        self,
        *,
        numerator: int,
        denominator: int,
        enabled: bool,
    ) -> float:
        """
        Compute one normalized metric ratio safely.

        Parameters
        ----------
        numerator : int
            Numerator used by the metric ratio.

        denominator : int
            Denominator used by the metric ratio.

        enabled : bool
            Whether the metric family is enabled in settings.

        Returns
        -------
        float
            Ratio in the inclusive range `[0.0, 1.0]`, or `0.0` when disabled
            or undefined.
        """

        if not enabled or denominator <= 0:
            return 0.0

        return float(numerator) / float(denominator)
