from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from Chunking.config.settings import PipelineSettings
from retrieval.evaluation.models import (
    BenchmarkQuestionCase,
    BenchmarkRunSummary,
    RetrievalEvaluationResult,
)
from retrieval.models import RetrievalContext, RetrievedChunkResult


class RetrievalBenchmarkEvaluator:
    """
    Evaluate offline retrieval rankings against versioned benchmark labels.
    """

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """
        Create a retrieval evaluator using shared benchmark defaults.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared project settings. Default settings are loaded when omitted.
        """

        resolved_settings = settings or PipelineSettings()
        self.default_top_k = max(1, int(resolved_settings.retrieval_top_k))

    def evaluate_case(
        self,
        benchmark_case: BenchmarkQuestionCase,
        retrieved_chunks: Iterable[RetrievedChunkResult],
        *,
        selected_context: Optional[RetrievalContext] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalEvaluationResult:
        """
        Evaluate one benchmark case against one deterministic retrieval ranking.

        Parameters
        ----------
        benchmark_case : BenchmarkQuestionCase
            Expected retrieval labels for one benchmark question.

        retrieved_chunks : Iterable[RetrievedChunkResult]
            Ordered retrieval candidates returned by the retrieval backend.

        selected_context : Optional[RetrievalContext]
            Optional final context selected from the retrieved candidates.

        top_k : Optional[int]
            Optional evaluation cutoff. The configured retrieval top-k is used
            when omitted.

        Returns
        -------
        RetrievalEvaluationResult
            Per-case benchmark metrics and diagnostic reasons.
        """

        cutoff = self._resolve_top_k(top_k)
        ranked_chunks = self._normalize_chunks(retrieved_chunks)
        top_chunks = ranked_chunks[:cutoff]
        selected_chunks = self._resolve_selected_chunks(selected_context)

        expected_chunk_ids = set(benchmark_case.expected_chunk_ids)
        expected_article_numbers = set(benchmark_case.expected_article_numbers)
        expected_doc_id = benchmark_case.expected_doc_id

        retrieved_chunk_ids = [chunk.chunk_id for chunk in top_chunks]
        selected_chunk_ids = [chunk.chunk_id for chunk in selected_chunks]
        retrieved_doc_ids = [chunk.doc_id for chunk in top_chunks if chunk.doc_id]
        retrieved_article_numbers = self._collect_article_numbers(top_chunks)

        expected_doc_recovered = self._has_expected_doc(
            chunks=top_chunks,
            expected_doc_id=expected_doc_id,
        )
        expected_article_recovered = self._has_expected_article(
            chunks=top_chunks,
            expected_article_numbers=expected_article_numbers,
        )
        expected_chunk_recovered = bool(
            expected_chunk_ids and expected_chunk_ids.intersection(retrieved_chunk_ids)
        )
        selected_context_hit = self._has_selected_context_hit(
            selected_chunks=selected_chunks,
            expected_chunk_ids=expected_chunk_ids,
            expected_doc_id=expected_doc_id,
            expected_article_numbers=expected_article_numbers,
        )
        reciprocal_rank = self._calculate_reciprocal_rank(
            ranked_chunks=ranked_chunks,
            expected_chunk_ids=expected_chunk_ids,
        )
        conflict_present = self._detect_conflict_presence(
            benchmark_case=benchmark_case,
            ranked_chunks=ranked_chunks,
            top_chunks=top_chunks,
            selected_context=selected_context,
        )
        reasons = self._build_reasons(
            benchmark_case=benchmark_case,
            expected_doc_recovered=expected_doc_recovered,
            expected_article_recovered=expected_article_recovered,
            expected_chunk_recovered=expected_chunk_recovered,
            selected_context_hit=selected_context_hit,
            conflict_present=conflict_present,
        )

        return RetrievalEvaluationResult(
            case_id=benchmark_case.case_id,
            expected_doc_id=expected_doc_id,
            expected_article_numbers=benchmark_case.expected_article_numbers,
            expected_chunk_ids=benchmark_case.expected_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            selected_chunk_ids=selected_chunk_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_article_numbers=retrieved_article_numbers,
            expected_doc_recovered=expected_doc_recovered,
            expected_article_recovered=expected_article_recovered,
            expected_chunk_recovered=expected_chunk_recovered,
            selected_context_hit=selected_context_hit,
            reciprocal_rank=reciprocal_rank,
            conflict_present=conflict_present,
            metrics={
                "recall_at_k": 1.0 if expected_chunk_recovered else 0.0,
                "mrr": reciprocal_rank or 0.0,
                "expected_document_recovered": (
                    1.0 if expected_doc_recovered else 0.0
                ),
                "expected_article_recovered": (
                    1.0 if expected_article_recovered else 0.0
                ),
                "expected_chunk_recovered": (
                    1.0 if expected_chunk_recovered else 0.0
                ),
                "selected_context_hit": 1.0 if selected_context_hit else 0.0,
                "conflict_presence": 1.0 if conflict_present else 0.0,
            },
            reasons=reasons,
            metadata={
                "top_k": cutoff,
                "retrieved_count": len(ranked_chunks),
                "selected_context_count": len(selected_chunks),
                "first_expected_chunk_rank": self._find_first_expected_rank(
                    ranked_chunks=ranked_chunks,
                    expected_chunk_ids=expected_chunk_ids,
                ),
            },
        )

    def evaluate_cases(
        self,
        benchmark_cases: Sequence[BenchmarkQuestionCase],
        retrieved_chunks_by_case_id: Mapping[str, Iterable[RetrievedChunkResult]],
        *,
        selected_context_by_case_id: Optional[Mapping[str, RetrievalContext]] = None,
        top_k: Optional[int] = None,
    ) -> BenchmarkRunSummary:
        """
        Evaluate multiple benchmark retrieval cases and aggregate their metrics.

        Parameters
        ----------
        benchmark_cases : Sequence[BenchmarkQuestionCase]
            Ordered benchmark cases to evaluate.

        retrieved_chunks_by_case_id : Mapping[str, Iterable[RetrievedChunkResult]]
            Retrieval rankings keyed by benchmark case identifier.

        selected_context_by_case_id : Optional[Mapping[str, RetrievalContext]]
            Optional selected contexts keyed by benchmark case identifier.

        top_k : Optional[int]
            Optional shared evaluation cutoff.

        Returns
        -------
        BenchmarkRunSummary
            Aggregate retrieval benchmark summary.
        """

        selected_contexts = selected_context_by_case_id or {}
        results = [
            self.evaluate_case(
                benchmark_case=benchmark_case,
                retrieved_chunks=retrieved_chunks_by_case_id.get(
                    benchmark_case.case_id,
                    [],
                ),
                selected_context=selected_contexts.get(benchmark_case.case_id),
                top_k=top_k,
            )
            for benchmark_case in benchmark_cases
        ]

        return BenchmarkRunSummary(
            mode="retrieval",
            question_case_count=len(benchmark_cases),
            retrieval_results=results,
            metrics=summarize_retrieval_results(results),
        )

    def _resolve_top_k(self, top_k: Optional[int]) -> int:
        """
        Resolve the positive evaluation cutoff.

        Parameters
        ----------
        top_k : Optional[int]
            Optional caller-supplied evaluation cutoff.

        Returns
        -------
        int
            Positive retrieval cutoff.
        """

        resolved_top_k = self.default_top_k if top_k is None else int(top_k)
        if resolved_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        return resolved_top_k

    def _normalize_chunks(
        self,
        retrieved_chunks: Iterable[RetrievedChunkResult],
    ) -> List[RetrievedChunkResult]:
        """
        Keep only typed retrieval chunks while preserving ranking order.

        Parameters
        ----------
        retrieved_chunks : Iterable[RetrievedChunkResult]
            Candidate ranking supplied by the caller.

        Returns
        -------
        List[RetrievedChunkResult]
            Ordered typed retrieval chunks.
        """

        return [
            chunk
            for chunk in retrieved_chunks
            if isinstance(chunk, RetrievedChunkResult)
        ]

    def _resolve_selected_chunks(
        self,
        selected_context: Optional[RetrievalContext],
    ) -> List[RetrievedChunkResult]:
        """
        Resolve selected chunks from an optional retrieval context.

        Parameters
        ----------
        selected_context : Optional[RetrievalContext]
            Final selected context produced by the context builder.

        Returns
        -------
        List[RetrievedChunkResult]
            Ordered selected chunks.
        """

        if not isinstance(selected_context, RetrievalContext):
            return []
        return self._normalize_chunks(selected_context.chunks)

    def _collect_article_numbers(
        self,
        chunks: Sequence[RetrievedChunkResult],
    ) -> List[str]:
        """
        Collect article numbers from ranked chunks without duplicates.

        Parameters
        ----------
        chunks : Sequence[RetrievedChunkResult]
            Ranked chunks whose legal article anchors are evaluated.

        Returns
        -------
        List[str]
            Ordered distinct article numbers.
        """

        article_numbers: List[str] = []
        seen_article_numbers: Set[str] = set()

        for chunk in chunks:
            article_number = self._resolve_article_number(chunk)
            if article_number and article_number not in seen_article_numbers:
                seen_article_numbers.add(article_number)
                article_numbers.append(article_number)

        return article_numbers

    def _has_expected_doc(
        self,
        *,
        chunks: Sequence[RetrievedChunkResult],
        expected_doc_id: Optional[str],
    ) -> bool:
        """
        Check whether the expected document appears in the evaluated ranking.

        Parameters
        ----------
        chunks : Sequence[RetrievedChunkResult]
            Ranked chunks evaluated within the cutoff.

        expected_doc_id : Optional[str]
            Expected source document identifier.

        Returns
        -------
        bool
            True when the expected document label is recovered.
        """

        return bool(
            expected_doc_id
            and any(chunk.doc_id == expected_doc_id for chunk in chunks)
        )

    def _has_expected_article(
        self,
        *,
        chunks: Sequence[RetrievedChunkResult],
        expected_article_numbers: Set[str],
    ) -> bool:
        """
        Check whether one expected article appears in the evaluated ranking.

        Parameters
        ----------
        chunks : Sequence[RetrievedChunkResult]
            Ranked chunks evaluated within the cutoff.

        expected_article_numbers : Set[str]
            Expected article numbers for the benchmark case.

        Returns
        -------
        bool
            True when any expected article number is recovered.
        """

        return bool(
            expected_article_numbers
            and any(
                self._resolve_article_number(chunk) in expected_article_numbers
                for chunk in chunks
            )
        )

    def _has_selected_context_hit(
        self,
        *,
        selected_chunks: Sequence[RetrievedChunkResult],
        expected_chunk_ids: Set[str],
        expected_doc_id: Optional[str],
        expected_article_numbers: Set[str],
    ) -> bool:
        """
        Check whether the expected source survives into final selected context.

        Parameters
        ----------
        selected_chunks : Sequence[RetrievedChunkResult]
            Chunks selected into the final grounded context.

        expected_chunk_ids : Set[str]
            Expected benchmark chunk identifiers.

        expected_doc_id : Optional[str]
            Expected source document identifier.

        expected_article_numbers : Set[str]
            Expected article numbers for the benchmark case.

        Returns
        -------
        bool
            True when selected context contains the expected source.
        """

        if not selected_chunks:
            return False

        if expected_chunk_ids:
            return any(chunk.chunk_id in expected_chunk_ids for chunk in selected_chunks)

        return any(
            self._chunk_matches_expected_source(
                chunk=chunk,
                expected_doc_id=expected_doc_id,
                expected_article_numbers=expected_article_numbers,
            )
            for chunk in selected_chunks
        )

    def _calculate_reciprocal_rank(
        self,
        *,
        ranked_chunks: Sequence[RetrievedChunkResult],
        expected_chunk_ids: Set[str],
    ) -> Optional[float]:
        """
        Calculate reciprocal rank for the first expected chunk match.

        Parameters
        ----------
        ranked_chunks : Sequence[RetrievedChunkResult]
            Full ordered retrieval ranking.

        expected_chunk_ids : Set[str]
            Expected benchmark chunk identifiers.

        Returns
        -------
        Optional[float]
            Reciprocal rank, or None when no expected chunk label exists.
        """

        first_rank = self._find_first_expected_rank(
            ranked_chunks=ranked_chunks,
            expected_chunk_ids=expected_chunk_ids,
        )
        if first_rank is None:
            return None
        return 1.0 / first_rank

    def _find_first_expected_rank(
        self,
        *,
        ranked_chunks: Sequence[RetrievedChunkResult],
        expected_chunk_ids: Set[str],
    ) -> Optional[int]:
        """
        Find the one-based rank of the first expected chunk in the ranking.

        Parameters
        ----------
        ranked_chunks : Sequence[RetrievedChunkResult]
            Full ordered retrieval ranking.

        expected_chunk_ids : Set[str]
            Expected benchmark chunk identifiers.

        Returns
        -------
        Optional[int]
            One-based rank, or None when no expected chunk is found.
        """

        if not expected_chunk_ids:
            return None

        for index, chunk in enumerate(ranked_chunks, start=1):
            if chunk.chunk_id in expected_chunk_ids:
                return index

        return None

    def _detect_conflict_presence(
        self,
        *,
        benchmark_case: BenchmarkQuestionCase,
        ranked_chunks: Sequence[RetrievedChunkResult],
        top_chunks: Sequence[RetrievedChunkResult],
        selected_context: Optional[RetrievalContext],
    ) -> bool:
        """
        Detect whether conflicting evidence appears before or within top hits.

        Parameters
        ----------
        benchmark_case : BenchmarkQuestionCase
            Benchmark case carrying expected legal labels.

        ranked_chunks : Sequence[RetrievedChunkResult]
            Full ordered retrieval ranking.

        top_chunks : Sequence[RetrievedChunkResult]
            Ranking slice evaluated within the cutoff.

        selected_context : Optional[RetrievalContext]
            Optional final selected context with evidence-quality metadata.

        Returns
        -------
        bool
            True when a distinct legal candidate competes with the expected one.
        """

        if self._context_reports_conflict(selected_context):
            return True

        first_expected_rank = self._find_first_expected_rank(
            ranked_chunks=ranked_chunks,
            expected_chunk_ids=set(benchmark_case.expected_chunk_ids),
        )
        conflict_candidates = (
            ranked_chunks[: first_expected_rank - 1]
            if first_expected_rank
            else top_chunks
        )

        return any(
            self._is_conflicting_candidate(
                chunk=chunk,
                benchmark_case=benchmark_case,
            )
            for chunk in conflict_candidates
        )

    def _context_reports_conflict(
        self,
        selected_context: Optional[RetrievalContext],
    ) -> bool:
        """
        Read explicit conflict signals already emitted by context selection.

        Parameters
        ----------
        selected_context : Optional[RetrievalContext]
            Optional selected context with evidence-quality metadata.

        Returns
        -------
        bool
            True when selected context metadata reports conflicting chunks.
        """

        if not isinstance(selected_context, RetrievalContext):
            return False

        evidence_quality = selected_context.evidence_quality
        if evidence_quality and evidence_quality.conflicting_chunk_ids:
            return True

        conflicting_chunk_ids = selected_context.metadata.get("conflicting_chunk_ids")
        return bool(conflicting_chunk_ids)

    def _is_conflicting_candidate(
        self,
        *,
        chunk: RetrievedChunkResult,
        benchmark_case: BenchmarkQuestionCase,
    ) -> bool:
        """
        Classify one retrieved candidate as conflicting with expected labels.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Candidate chunk being evaluated.

        benchmark_case : BenchmarkQuestionCase
            Benchmark case carrying expected legal labels.

        Returns
        -------
        bool
            True when the candidate has a distinct legal anchor.
        """

        if chunk.chunk_id in benchmark_case.expected_chunk_ids:
            return False

        expected_doc_id = benchmark_case.expected_doc_id
        expected_article_numbers = set(benchmark_case.expected_article_numbers)
        chunk_article_number = self._resolve_article_number(chunk)

        if expected_doc_id and chunk.doc_id and chunk.doc_id != expected_doc_id:
            return True

        if (
            expected_article_numbers
            and chunk_article_number
            and chunk_article_number not in expected_article_numbers
        ):
            return True

        return False

    def _chunk_matches_expected_source(
        self,
        *,
        chunk: RetrievedChunkResult,
        expected_doc_id: Optional[str],
        expected_article_numbers: Set[str],
    ) -> bool:
        """
        Check whether one chunk matches expected document and article labels.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Candidate chunk being evaluated.

        expected_doc_id : Optional[str]
            Expected document identifier.

        expected_article_numbers : Set[str]
            Expected article numbers.

        Returns
        -------
        bool
            True when available labels align with the expected source.
        """

        doc_matches = not expected_doc_id or chunk.doc_id == expected_doc_id
        article_number = self._resolve_article_number(chunk)
        article_matches = (
            not expected_article_numbers
            or article_number in expected_article_numbers
        )
        return doc_matches and article_matches

    def _resolve_article_number(self, chunk: RetrievedChunkResult) -> str:
        """
        Resolve the article number from normalized retrieval metadata scopes.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Chunk whose article label is inspected.

        Returns
        -------
        str
            Article number when available, otherwise an empty string.
        """

        candidate_values = (
            chunk.context_metadata.article_number,
            chunk.chunk_metadata.get("article_number"),
            chunk.metadata.get("article_number"),
        )

        for value in candidate_values:
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def _build_reasons(
        self,
        *,
        benchmark_case: BenchmarkQuestionCase,
        expected_doc_recovered: bool,
        expected_article_recovered: bool,
        expected_chunk_recovered: bool,
        selected_context_hit: bool,
        conflict_present: bool,
    ) -> List[str]:
        """
        Build deterministic diagnostic reasons for one retrieval result.

        Parameters
        ----------
        benchmark_case : BenchmarkQuestionCase
            Benchmark case carrying expected labels.

        expected_doc_recovered : bool
            Whether the expected document was recovered.

        expected_article_recovered : bool
            Whether the expected article was recovered.

        expected_chunk_recovered : bool
            Whether one expected chunk was recovered.

        selected_context_hit : bool
            Whether expected evidence survived into selected context.

        conflict_present : bool
            Whether conflicting legal evidence was detected.

        Returns
        -------
        List[str]
            Ordered reason codes.
        """

        reasons: List[str] = []

        if benchmark_case.expected_doc_id and not expected_doc_recovered:
            reasons.append("expected_document_not_recovered")
        if benchmark_case.expected_article_numbers and not expected_article_recovered:
            reasons.append("expected_article_not_recovered")
        if benchmark_case.expected_chunk_ids and not expected_chunk_recovered:
            reasons.append("expected_chunk_not_recovered")
        if benchmark_case.expected_chunk_ids and not selected_context_hit:
            reasons.append("expected_source_not_selected")
        if conflict_present:
            reasons.append("conflicting_candidate_present")

        if not reasons:
            reasons.append("retrieval_labels_satisfied")

        return reasons


def evaluate_retrieval_case(
    benchmark_case: BenchmarkQuestionCase,
    retrieved_chunks: Iterable[RetrievedChunkResult],
    *,
    selected_context: Optional[RetrievalContext] = None,
    top_k: Optional[int] = None,
    settings: Optional[PipelineSettings] = None,
) -> RetrievalEvaluationResult:
    """
    Evaluate one retrieval benchmark case with the shared evaluator.

    Parameters
    ----------
    benchmark_case : BenchmarkQuestionCase
        Expected retrieval labels for one benchmark question.

    retrieved_chunks : Iterable[RetrievedChunkResult]
        Ordered retrieval candidates returned by the retrieval backend.

    selected_context : Optional[RetrievalContext]
        Optional final selected context.

    top_k : Optional[int]
        Optional evaluation cutoff.

    settings : Optional[PipelineSettings]
        Shared project settings. Default settings are loaded when omitted.

    Returns
    -------
    RetrievalEvaluationResult
        Per-case benchmark metrics and diagnostic reasons.
    """

    return RetrievalBenchmarkEvaluator(settings=settings).evaluate_case(
        benchmark_case=benchmark_case,
        retrieved_chunks=retrieved_chunks,
        selected_context=selected_context,
        top_k=top_k,
    )


def summarize_retrieval_results(
    results: Sequence[RetrievalEvaluationResult],
) -> Dict[str, float]:
    """
    Aggregate retrieval benchmark results into stable numeric metrics.

    Parameters
    ----------
    results : Sequence[RetrievalEvaluationResult]
        Per-case retrieval benchmark results.

    Returns
    -------
    Dict[str, float]
        Aggregate benchmark metrics.
    """

    if not results:
        return {
            "case_count": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "expected_document_recovery_rate": 0.0,
            "expected_article_recovery_rate": 0.0,
            "expected_chunk_recovery_rate": 0.0,
            "selected_context_hit_rate": 0.0,
            "conflict_presence_rate": 0.0,
        }

    return {
        "case_count": float(len(results)),
        "recall_at_k": _mean_metric(results, "recall_at_k"),
        "mrr": _mean_metric(results, "mrr"),
        "expected_document_recovery_rate": _mean_labeled_bool(
            results,
            label_attribute="expected_doc_id",
            value_attribute="expected_doc_recovered",
        ),
        "expected_article_recovery_rate": _mean_labeled_bool(
            results,
            label_attribute="expected_article_numbers",
            value_attribute="expected_article_recovered",
        ),
        "expected_chunk_recovery_rate": _mean_labeled_bool(
            results,
            label_attribute="expected_chunk_ids",
            value_attribute="expected_chunk_recovered",
        ),
        "selected_context_hit_rate": _mean_labeled_bool(
            results,
            label_attribute="expected_chunk_ids",
            value_attribute="selected_context_hit",
        ),
        "conflict_presence_rate": sum(
            1.0 for result in results if result.conflict_present
        )
        / len(results),
    }


def _mean_metric(
    results: Sequence[RetrievalEvaluationResult],
    metric_name: str,
) -> float:
    """
    Average one numeric metric across all results.

    Parameters
    ----------
    results : Sequence[RetrievalEvaluationResult]
        Per-case retrieval benchmark results.

    metric_name : str
        Metric name to average from each result mapping.

    Returns
    -------
    float
        Mean metric value.
    """

    return sum(result.metrics.get(metric_name, 0.0) for result in results) / len(
        results
    )


def _mean_labeled_bool(
    results: Sequence[RetrievalEvaluationResult],
    *,
    label_attribute: str,
    value_attribute: str,
) -> float:
    """
    Average one boolean result only across cases that contain that label.

    Parameters
    ----------
    results : Sequence[RetrievalEvaluationResult]
        Per-case retrieval benchmark results.

    label_attribute : str
        Result attribute carrying the expected benchmark label.

    value_attribute : str
        Result boolean attribute to average.

    Returns
    -------
    float
        Labeled recovery rate, or zero when no cases carry that label.
    """

    labeled_results = [
        result for result in results if bool(getattr(result, label_attribute))
    ]
    if not labeled_results:
        return 0.0

    return sum(
        1.0 for result in labeled_results if bool(getattr(result, value_attribute))
    ) / len(labeled_results)
