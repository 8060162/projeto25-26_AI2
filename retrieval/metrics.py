from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Generator, Optional

from Chunking.config.settings import PipelineSettings
from retrieval.models import GuardrailDecision, MetricsSnapshot


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


class RetrievalMetricsCollector:
    """
    Collect lightweight runtime metrics for retrieval safety and latency.

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
        """

        if not self.settings.metrics_enabled:
            return

        if successful:
            self._successful_requests += 1
        if blocked:
            self._blocked_requests += 1
        if deflected:
            self._deflected_requests += 1

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
            stage_latency_ms=dict(self._stage_latency_ms),
            total_latency_ms=total_latency_ms,
        )

    def build_metric_report(self) -> Dict[str, float | int]:
        """
        Build one flat metric report for logs and assertions.

        Returns
        -------
        Dict[str, float | int]
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
        self._stage_latency_ms.clear()

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
