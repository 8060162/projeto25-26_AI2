from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Ensure project-root imports work when this file is executed directly.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Chunking.config.settings import PipelineSettings
from embedding.chunk_input_loader import load_embedding_input_records
from embedding.models import EmbeddingInputRecord
from embedding.provider_factory import EmbeddingProvider, create_embedding_provider
from embedding.text_builder import build_embedding_text
from retrieval.context_builder import RetrievalContextBuilder
from retrieval.evaluation.benchmark_loader import BenchmarkLoader
from retrieval.evaluation.models import BenchmarkQuestionCase, BenchmarkRunSummary
from retrieval.evaluation.retrieval_evaluator import RetrievalBenchmarkEvaluator
from retrieval.models import RetrievalContext, RetrievedChunkResult, UserQuestionInput
from retrieval.query_normalizer import SemanticQueryNormalizer


QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
SUMMARY_FILE_NAME = "embedding_comparison_summary.json"


@dataclass(slots=True)
class EmbeddingModelBenchmarkResult:
    """
    Store retrieval benchmark output for one embedding model.
    """

    model: str
    provider: str
    question_case_count: int
    metrics: Dict[str, float] = field(default_factory=dict)
    retrieval_summary: BenchmarkRunSummary = field(default_factory=BenchmarkRunSummary)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the model benchmark result into a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model benchmark payload.
        """

        payload = asdict(self)
        payload["retrieval_summary"] = self.retrieval_summary.to_dict()
        return payload


@dataclass(slots=True)
class EmbeddingBenchmarkComparisonResult:
    """
    Store the full embedding model comparison benchmark output.
    """

    run_id: str
    baseline_model: str
    compared_models: List[str]
    top_k: int
    model_results: List[EmbeddingModelBenchmarkResult] = field(default_factory=list)
    metric_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    output_directory: str = ""
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the comparison result into a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable comparison payload.
        """

        return {
            "run_id": self.run_id,
            "baseline_model": self.baseline_model,
            "compared_models": list(self.compared_models),
            "top_k": self.top_k,
            "model_results": [result.to_dict() for result in self.model_results],
            "metric_deltas": dict(self.metric_deltas),
            "output_directory": self.output_directory,
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


def run_embedding_benchmark_comparison(
    *,
    settings: Optional[PipelineSettings] = None,
    questions_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    run_id: str = "latest",
    model_names: Optional[Sequence[str]] = None,
    providers_by_model: Optional[Mapping[str, EmbeddingProvider]] = None,
    input_records: Optional[Sequence[EmbeddingInputRecord]] = None,
    top_k: Optional[int] = None,
    write_artifacts: bool = True,
) -> EmbeddingBenchmarkComparisonResult:
    """
    Compare retrieval benchmark metrics across configured embedding models.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared project settings. Default settings are loaded when omitted.

    questions_path : Optional[Path]
        Optional factual legal QA benchmark dataset path override.

    output_root : Optional[Path]
        Optional output root override for comparison artifacts.

    run_id : str
        Stable run directory name under the comparison output root.

    model_names : Optional[Sequence[str]]
        Optional explicit model list. The active model is used as the baseline.

    providers_by_model : Optional[Mapping[str, EmbeddingProvider]]
        Optional deterministic provider overrides keyed by model name.

    input_records : Optional[Sequence[EmbeddingInputRecord]]
        Optional chunk inputs used by tests or controlled offline runs.

    top_k : Optional[int]
        Optional retrieval evaluation cutoff.

    write_artifacts : bool
        Whether to write the comparison JSON artifact.

    Returns
    -------
    EmbeddingBenchmarkComparisonResult
        Per-model retrieval metrics and candidate deltas from the baseline.
    """

    resolved_settings = settings or PipelineSettings()
    compared_models = _resolve_compared_models(
        settings=resolved_settings,
        model_names=model_names,
    )
    resolved_top_k = _resolve_top_k(top_k, resolved_settings.retrieval_top_k)
    benchmark_cases = BenchmarkLoader(
        settings=resolved_settings,
        questions_path=questions_path,
    ).load_question_cases()
    prepared_records = _resolve_embedding_input_records(
        settings=resolved_settings,
        input_records=input_records,
    )

    model_results = [
        _evaluate_model(
            model_name=model_name,
            settings=resolved_settings,
            benchmark_cases=benchmark_cases,
            input_records=prepared_records,
            providers_by_model=providers_by_model,
            top_k=resolved_top_k,
        )
        for model_name in compared_models
    ]
    comparison_result = EmbeddingBenchmarkComparisonResult(
        run_id=_normalize_run_id(run_id),
        baseline_model=compared_models[0],
        compared_models=compared_models,
        top_k=resolved_top_k,
        model_results=model_results,
        metric_deltas=_build_metric_deltas(model_results),
        metadata={
            "questions_path": str(
                questions_path or resolved_settings.benchmark_questions_path
            ),
            "input_record_count": len(prepared_records),
        },
    )

    if write_artifacts:
        output_directory = _resolve_output_directory(
            output_root=output_root
            or resolved_settings.embedding_comparison_output_root,
            run_id=comparison_result.run_id,
        )
        comparison_result.output_directory = str(output_directory)
        _write_comparison_summary(
            output_path=output_directory / SUMMARY_FILE_NAME,
            comparison_result=comparison_result,
        )

    return comparison_result


def _evaluate_model(
    *,
    model_name: str,
    settings: PipelineSettings,
    benchmark_cases: Sequence[BenchmarkQuestionCase],
    input_records: Sequence[EmbeddingInputRecord],
    providers_by_model: Optional[Mapping[str, EmbeddingProvider]],
    top_k: int,
) -> EmbeddingModelBenchmarkResult:
    """
    Evaluate one embedding model with the shared retrieval benchmark evaluator.
    """

    model_settings = replace(settings, embedding_model=model_name)
    provider = _resolve_provider(
        model_name=model_name,
        settings=model_settings,
        providers_by_model=providers_by_model,
    )
    query_normalizer = SemanticQueryNormalizer(model_settings)
    context_builder = RetrievalContextBuilder(model_settings)
    chunk_vectors = provider.embed_texts([record.text for record in input_records])

    _validate_vector_count(
        vectors=chunk_vectors,
        expected_count=len(input_records),
        label=f"chunk embeddings for model '{model_name}'",
    )

    retrieved_chunks_by_case_id: Dict[str, List[RetrievedChunkResult]] = {}
    selected_context_by_case_id: Dict[str, RetrievalContext] = {}

    for benchmark_case in benchmark_cases:
        normalized_question = query_normalizer.normalize(
            UserQuestionInput(
                question_text=benchmark_case.question,
                metadata={"benchmark_case_id": benchmark_case.case_id},
            )
        )
        query_vectors = provider.embed_texts([normalized_question.normalized_query_text])
        _validate_vector_count(
            vectors=query_vectors,
            expected_count=1,
            label=f"query embedding for case '{benchmark_case.case_id}'",
        )
        retrieved_chunks = _rank_records_by_similarity(
            input_records=input_records,
            chunk_vectors=chunk_vectors,
            query_vector=query_vectors[0],
            top_k=top_k,
        )
        selected_context = context_builder.build_context(
            retrieved_chunks,
            top_k=top_k,
            query_text=normalized_question.normalized_query_text,
            query_metadata=normalized_question.query_metadata,
        )
        retrieved_chunks_by_case_id[benchmark_case.case_id] = retrieved_chunks
        selected_context_by_case_id[benchmark_case.case_id] = selected_context

    retrieval_summary = RetrievalBenchmarkEvaluator(
        settings=model_settings,
    ).evaluate_cases(
        benchmark_cases=benchmark_cases,
        retrieved_chunks_by_case_id=retrieved_chunks_by_case_id,
        selected_context_by_case_id=selected_context_by_case_id,
        top_k=top_k,
    )
    retrieval_summary.run_id = _sanitize_model_name(model_name)
    retrieval_summary.metadata.update(
        {
            "embedding_model": model_name,
            "embedding_provider": model_settings.embedding_provider,
        }
    )

    return EmbeddingModelBenchmarkResult(
        model=model_name,
        provider=model_settings.embedding_provider,
        question_case_count=len(benchmark_cases),
        metrics=dict(retrieval_summary.metrics),
        retrieval_summary=retrieval_summary,
        metadata={"top_k": top_k},
    )


def _resolve_provider(
    *,
    model_name: str,
    settings: PipelineSettings,
    providers_by_model: Optional[Mapping[str, EmbeddingProvider]],
) -> EmbeddingProvider:
    """
    Resolve a provider override or build the configured provider for one model.
    """

    if providers_by_model and model_name in providers_by_model:
        return providers_by_model[model_name]
    return create_embedding_provider(settings)


def _resolve_embedding_input_records(
    *,
    settings: PipelineSettings,
    input_records: Optional[Sequence[EmbeddingInputRecord]],
) -> List[EmbeddingInputRecord]:
    """
    Load and prepare embedding input records for offline model comparison.
    """

    source_records = (
        list(input_records)
        if input_records is not None
        else list(load_embedding_input_records(settings))
    )
    prepared_records: List[EmbeddingInputRecord] = []

    for source_record in source_records:
        embedding_text = build_embedding_text(source_record)
        if not embedding_text:
            continue
        prepared_records.append(
            EmbeddingInputRecord(
                chunk_id=source_record.chunk_id,
                doc_id=source_record.doc_id,
                text=embedding_text,
                metadata=dict(source_record.metadata),
                source_file=source_record.source_file,
                hierarchy_path=list(source_record.hierarchy_path),
                page_start=source_record.page_start,
                page_end=source_record.page_end,
                record_id=source_record.record_id,
                chunk_metadata=dict(source_record.chunk_metadata),
                document_metadata=dict(source_record.document_metadata),
            )
        )

    if not prepared_records:
        raise ValueError("No embedding records with usable text were prepared.")

    return prepared_records


def _rank_records_by_similarity(
    *,
    input_records: Sequence[EmbeddingInputRecord],
    chunk_vectors: Sequence[Sequence[float]],
    query_vector: Sequence[float],
    top_k: int,
) -> List[RetrievedChunkResult]:
    """
    Rank prepared chunk records by cosine similarity to one query vector.
    """

    scored_records = [
        (
            _cosine_similarity(query_vector, chunk_vector),
            record_index,
            input_record,
        )
        for record_index, (input_record, chunk_vector) in enumerate(
            zip(input_records, chunk_vectors),
        )
    ]
    scored_records.sort(key=lambda item: (-item[0], item[1]))

    retrieved_chunks: List[RetrievedChunkResult] = []
    for rank, (similarity_score, _record_index, input_record) in enumerate(
        scored_records[:top_k],
        start=1,
    ):
        retrieved_chunks.append(
            RetrievedChunkResult(
                chunk_id=input_record.chunk_id,
                doc_id=input_record.doc_id,
                text=input_record.text,
                record_id=input_record.record_id,
                rank=rank,
                distance=1.0 - similarity_score,
                similarity_score=similarity_score,
                source_file=input_record.source_file,
                metadata=dict(input_record.metadata),
                chunk_metadata=dict(input_record.chunk_metadata),
                document_metadata=dict(input_record.document_metadata),
            )
        )

    return retrieved_chunks


def _cosine_similarity(
    left_vector: Sequence[float],
    right_vector: Sequence[float],
) -> float:
    """
    Calculate cosine similarity for two numeric vectors.
    """

    if len(left_vector) != len(right_vector):
        raise ValueError("Embedding vectors must have matching dimensions.")

    left_norm = math.sqrt(sum(float(value) * float(value) for value in left_vector))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right_vector))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(
        float(left_value) * float(right_value)
        for left_value, right_value in zip(left_vector, right_vector)
    )
    return dot_product / (left_norm * right_norm)


def _build_metric_deltas(
    model_results: Sequence[EmbeddingModelBenchmarkResult],
) -> Dict[str, Dict[str, float]]:
    """
    Compare candidate model metrics against the first model as the baseline.
    """

    if not model_results:
        return {}

    baseline_metrics = model_results[0].metrics
    metric_deltas: Dict[str, Dict[str, float]] = {}

    for model_result in model_results[1:]:
        compared_metric_names = sorted(
            set(baseline_metrics).union(model_result.metrics)
        )
        metric_deltas[model_result.model] = {
            metric_name: model_result.metrics.get(metric_name, 0.0)
            - baseline_metrics.get(metric_name, 0.0)
            for metric_name in compared_metric_names
        }

    return metric_deltas


def _resolve_compared_models(
    *,
    settings: PipelineSettings,
    model_names: Optional[Sequence[str]],
) -> List[str]:
    """
    Resolve the ordered model list, keeping the active model as baseline.
    """

    configured_models = list(
        model_names or settings.embedding_comparison_candidate_models
    )
    candidate_models = [
        settings.embedding_model,
        *configured_models,
        QWEN_EMBEDDING_MODEL,
    ]
    compared_models: List[str] = []
    seen_models = set()

    for model_name in candidate_models:
        normalized_model = model_name.strip() if isinstance(model_name, str) else ""
        if not normalized_model or normalized_model in seen_models:
            continue
        seen_models.add(normalized_model)
        compared_models.append(normalized_model)

    if len(compared_models) < 2:
        raise ValueError("At least two embedding models are required for comparison.")

    return compared_models


def _resolve_top_k(top_k: Optional[int], default_top_k: int) -> int:
    """
    Resolve a positive retrieval benchmark cutoff.
    """

    resolved_top_k = int(default_top_k if top_k is None else top_k)
    if resolved_top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    return resolved_top_k


def _validate_vector_count(
    *,
    vectors: Sequence[Sequence[float]],
    expected_count: int,
    label: str,
) -> None:
    """
    Validate that the provider returned one vector per requested input.
    """

    if len(vectors) != expected_count:
        raise ValueError(
            f"Embedding provider returned {len(vectors)} vectors for {label}; "
            f"expected {expected_count}."
        )


def _resolve_output_directory(*, output_root: Path, run_id: str) -> Path:
    """
    Resolve and create the embedding comparison output directory.
    """

    output_directory = Path(output_root) / _normalize_run_id(run_id)
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def _write_comparison_summary(
    *,
    output_path: Path,
    comparison_result: EmbeddingBenchmarkComparisonResult,
) -> None:
    """
    Write the embedding comparison summary as stable JSON.
    """

    output_path.write_text(
        json.dumps(
            comparison_result.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _normalize_run_id(run_id: str) -> str:
    """
    Normalize and validate one output run identifier.
    """

    normalized_run_id = run_id.strip() if isinstance(run_id, str) else ""
    if not normalized_run_id:
        raise ValueError("run_id cannot be empty.")
    return normalized_run_id


def _sanitize_model_name(model_name: str) -> str:
    """
    Convert one model name into a filesystem-safe identifier.
    """

    normalized_model = model_name.strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "_", normalized_model).strip("_")


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for the embedding comparison runner.
    """

    parser = argparse.ArgumentParser(
        description="Compare embedding models with retrieval benchmark metrics."
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        help="Optional benchmark questions JSONL path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional embedding comparison output root.",
    )
    parser.add_argument(
        "--run-id",
        default="latest",
        help="Stable output directory name under the output root.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Embedding model to compare. Can be passed more than once.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Optional retrieval evaluation cutoff.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the comparison without writing output artifacts.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the embedding benchmark comparison CLI.
    """

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        result = run_embedding_benchmark_comparison(
            settings=PipelineSettings(),
            questions_path=arguments.questions_path,
            output_root=arguments.output_root,
            run_id=arguments.run_id,
            model_names=arguments.models,
            top_k=arguments.top_k,
            write_artifacts=not arguments.no_write,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Baseline model: {result.baseline_model}")
    print(f"[INFO] Compared models: {', '.join(result.compared_models)}")
    if result.output_directory:
        print(f"[INFO] Output directory: {result.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
