from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Ensure project-root imports work when this file is executed directly.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Chunking.config.settings import PipelineSettings
from retrieval.evaluation.answer_evaluator import AnswerBenchmarkEvaluator
from retrieval.evaluation.benchmark_loader import BenchmarkLoader
from retrieval.evaluation.guardrails_evaluator import GuardrailBenchmarkEvaluator
from retrieval.evaluation.models import (
    BenchmarkQuestionCase,
    BenchmarkRunSummary,
)
from retrieval.evaluation.retrieval_evaluator import RetrievalBenchmarkEvaluator
from retrieval.models import (
    GroundingVerificationResult,
    RetrievalContext,
    RetrievedChunkResult,
    UserQuestionInput,
)
from retrieval.service import RetrievalService


_VALID_MODES = ("retrieval", "answer", "guardrails", "full")


def run_benchmark_main(
    *,
    mode: str = "full",
    settings: Optional[PipelineSettings] = None,
    questions_path: Optional[Path] = None,
    guardrails_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    run_id: str = "latest",
    retrieval_observations_path: Optional[Path] = None,
    answer_observations_path: Optional[Path] = None,
    top_k: Optional[int] = None,
) -> BenchmarkRunSummary:
    """
    Execute the benchmark runner and write JSON evaluation artifacts.

    Parameters
    ----------
    mode : str
        Benchmark mode to execute. Supported values are `retrieval`, `answer`,
        `guardrails`, and `full`.

    settings : Optional[PipelineSettings]
        Shared project settings. Default settings are loaded when omitted.

    questions_path : Optional[Path]
        Optional factual legal QA benchmark path override.

    guardrails_path : Optional[Path]
        Optional guardrail benchmark path override.

    output_root : Optional[Path]
        Optional output root override for benchmark artifacts.

    run_id : str
        Stable run directory name under the output root.

    retrieval_observations_path : Optional[Path]
        Optional deterministic retrieval observations JSON file.

    answer_observations_path : Optional[Path]
        Optional deterministic answer observations JSON file.

    top_k : Optional[int]
        Optional retrieval evaluation cutoff.

    Returns
    -------
    BenchmarkRunSummary
        Combined summary for the executed benchmark mode.
    """

    resolved_mode = _normalize_mode(mode)
    resolved_settings = settings or PipelineSettings()
    loader = BenchmarkLoader(
        settings=resolved_settings,
        questions_path=questions_path,
        guardrails_path=guardrails_path,
    )
    output_directory = _resolve_output_directory(
        output_root=output_root or resolved_settings.benchmark_output_root,
        run_id=run_id,
    )

    summaries: List[BenchmarkRunSummary] = []
    service_results: Dict[str, Any] = {}

    if resolved_mode in {"retrieval", "answer", "full"}:
        question_cases = loader.load_question_cases()
        service_results = _run_runtime_cases_when_needed(
            question_cases=question_cases,
            settings=resolved_settings,
            retrieval_observations_path=retrieval_observations_path,
            answer_observations_path=answer_observations_path,
            mode=resolved_mode,
        )
    else:
        question_cases = []

    if resolved_mode in {"retrieval", "full"}:
        retrieval_summary = _run_retrieval_benchmark(
            question_cases=question_cases,
            settings=resolved_settings,
            service_results=service_results,
            observations_path=retrieval_observations_path,
            top_k=top_k,
        )
        summaries.append(retrieval_summary)
        _write_summary(output_directory / "retrieval_summary.json", retrieval_summary)

    if resolved_mode in {"answer", "full"}:
        answer_summary = _run_answer_benchmark(
            question_cases=question_cases,
            service_results=service_results,
            observations_path=answer_observations_path,
        )
        summaries.append(answer_summary)
        _write_summary(output_directory / "answer_summary.json", answer_summary)

    if resolved_mode in {"guardrails", "full"}:
        guardrail_summary = _run_guardrail_benchmark(
            guardrail_cases=loader.load_guardrail_cases(),
            settings=resolved_settings,
        )
        summaries.append(guardrail_summary)
        _write_summary(output_directory / "guardrails_summary.json", guardrail_summary)

    combined_summary = _combine_summaries(
        mode=resolved_mode,
        run_id=run_id,
        summaries=summaries,
        output_directory=output_directory,
    )
    _write_summary(output_directory / "benchmark_summary.json", combined_summary)
    return combined_summary


def _run_retrieval_benchmark(
    *,
    question_cases: Sequence[BenchmarkQuestionCase],
    settings: PipelineSettings,
    service_results: Mapping[str, Any],
    observations_path: Optional[Path],
    top_k: Optional[int],
) -> BenchmarkRunSummary:
    """
    Run retrieval evaluation from observations or runtime service results.
    """

    if observations_path is not None:
        observations = _load_json_mapping(observations_path)
        retrieved_chunks_by_case_id = {
            case_id: _build_chunks(payload.get("retrieved_chunks", []))
            for case_id, payload in observations.items()
            if isinstance(payload, dict)
        }
        selected_context_by_case_id = {
            case_id: RetrievalContext(
                chunks=_build_chunks(payload.get("selected_chunks", []))
            )
            for case_id, payload in observations.items()
            if isinstance(payload, dict) and payload.get("selected_chunks") is not None
        }
    else:
        retrieved_chunks_by_case_id = {
            case_id: result.retrieval_context.chunks
            for case_id, result in service_results.items()
            if result.retrieval_context is not None
        }
        selected_context_by_case_id = {
            case_id: result.retrieval_context
            for case_id, result in service_results.items()
            if result.retrieval_context is not None
        }

    return RetrievalBenchmarkEvaluator(settings=settings).evaluate_cases(
        benchmark_cases=question_cases,
        retrieved_chunks_by_case_id=retrieved_chunks_by_case_id,
        selected_context_by_case_id=selected_context_by_case_id,
        top_k=top_k,
    )


def _run_answer_benchmark(
    *,
    question_cases: Sequence[BenchmarkQuestionCase],
    service_results: Mapping[str, Any],
    observations_path: Optional[Path],
) -> BenchmarkRunSummary:
    """
    Run answer and grounding evaluation from observations or runtime results.
    """

    if observations_path is not None:
        observations = _load_json_mapping(observations_path)
        answers_by_case_id = {
            case_id: _normalize_string(payload.get("answer_text"))
            for case_id, payload in observations.items()
            if isinstance(payload, dict)
        }
        observed_behavior_by_case_id = {
            case_id: _normalize_string(payload.get("observed_behavior"))
            for case_id, payload in observations.items()
            if isinstance(payload, dict)
        }
        citations_by_case_id = {
            case_id: _normalize_string_list(payload.get("citations"))
            for case_id, payload in observations.items()
            if isinstance(payload, dict)
        }
        grounding_by_case_id = {
            case_id: _build_grounding(payload.get("grounding_verification"))
            for case_id, payload in observations.items()
            if isinstance(payload, dict)
            and isinstance(payload.get("grounding_verification"), dict)
        }
        observed_route_by_case_id = {
            case_id: _normalize_string(payload.get("observed_route"))
            for case_id, payload in observations.items()
            if isinstance(payload, dict)
        }
    else:
        answers_by_case_id = {
            case_id: result.answer_text for case_id, result in service_results.items()
        }
        observed_behavior_by_case_id = {
            case_id: result.status for case_id, result in service_results.items()
        }
        citations_by_case_id = {
            case_id: result.citations for case_id, result in service_results.items()
        }
        grounding_by_case_id = {
            case_id: result.route_metadata.grounding_verification
            for case_id, result in service_results.items()
            if result.route_metadata is not None
            and result.route_metadata.grounding_verification is not None
        }
        observed_route_by_case_id = {
            case_id: _resolve_route_name(result)
            for case_id, result in service_results.items()
        }

    return AnswerBenchmarkEvaluator().evaluate_cases(
        benchmark_cases=question_cases,
        answers_by_case_id=answers_by_case_id,
        observed_behavior_by_case_id=observed_behavior_by_case_id,
        citations_by_case_id=citations_by_case_id,
        grounding_by_case_id=grounding_by_case_id,
        observed_route_by_case_id=observed_route_by_case_id,
    )


def _run_guardrail_benchmark(
    *,
    guardrail_cases: Sequence[Any],
    settings: PipelineSettings,
) -> BenchmarkRunSummary:
    """
    Run deterministic guardrail evaluation against the guardrail dataset.
    """

    return GuardrailBenchmarkEvaluator(settings=settings).evaluate_cases(
        guardrail_cases
    )


def _run_runtime_cases_when_needed(
    *,
    question_cases: Sequence[BenchmarkQuestionCase],
    settings: PipelineSettings,
    retrieval_observations_path: Optional[Path],
    answer_observations_path: Optional[Path],
    mode: str,
) -> Dict[str, Any]:
    """
    Run the retrieval service only for benchmark modes without observations.
    """

    retrieval_observed = retrieval_observations_path is not None
    answer_observed = answer_observations_path is not None

    if mode == "retrieval" and retrieval_observed:
        return {}
    if mode == "answer" and answer_observed:
        return {}
    if mode == "full" and retrieval_observed and answer_observed:
        return {}

    retrieval_service = RetrievalService(settings=settings)
    return {
        benchmark_case.case_id: retrieval_service.answer_question(
            UserQuestionInput(
                question_text=benchmark_case.question,
                metadata={
                    "benchmark_case_id": benchmark_case.case_id,
                    "expected_chunk_ids": benchmark_case.expected_chunk_ids,
                },
            )
        )
        for benchmark_case in question_cases
    }


def _combine_summaries(
    *,
    mode: str,
    run_id: str,
    summaries: Sequence[BenchmarkRunSummary],
    output_directory: Path,
) -> BenchmarkRunSummary:
    """
    Combine per-mode summaries into one benchmark-run artifact.
    """

    combined_metrics: Dict[str, float] = {}
    combined_errors: List[str] = []
    combined_summary = BenchmarkRunSummary(
        run_id=run_id,
        mode=mode,
        metadata={"output_directory": str(output_directory)},
    )

    for summary in summaries:
        metric_prefix = summary.mode or "benchmark"
        for metric_name, metric_value in summary.metrics.items():
            combined_metrics[f"{metric_prefix}.{metric_name}"] = metric_value
        combined_errors.extend(summary.errors)
        combined_summary.question_case_count = max(
            combined_summary.question_case_count,
            summary.question_case_count,
        )
        combined_summary.guardrail_case_count = max(
            combined_summary.guardrail_case_count,
            summary.guardrail_case_count,
        )
        combined_summary.retrieval_results.extend(summary.retrieval_results)
        combined_summary.answer_results.extend(summary.answer_results)
        combined_summary.guardrail_results.extend(summary.guardrail_results)

    combined_summary.metrics = combined_metrics
    combined_summary.errors = combined_errors
    return combined_summary


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for the benchmark runner.
    """

    parser = argparse.ArgumentParser(
        description="Run retrieval, answer, and guardrail benchmark evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=_VALID_MODES,
        default="full",
        help="Benchmark mode to execute.",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        help="Optional benchmark questions JSONL path.",
    )
    parser.add_argument(
        "--guardrails-path",
        type=Path,
        help="Optional benchmark guardrails JSONL path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional benchmark output root.",
    )
    parser.add_argument(
        "--run-id",
        default="latest",
        help="Stable output directory name under the output root.",
    )
    parser.add_argument(
        "--retrieval-observations",
        type=Path,
        help="Optional deterministic retrieval observations JSON file.",
    )
    parser.add_argument(
        "--answer-observations",
        type=Path,
        help="Optional deterministic answer observations JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Optional retrieval evaluation cutoff.",
    )
    return parser


def _build_chunks(raw_chunks: Any) -> List[RetrievedChunkResult]:
    """
    Convert raw chunk observation mappings into retrieval result contracts.
    """

    if not isinstance(raw_chunks, list):
        return []

    chunks: List[RetrievedChunkResult] = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        chunks.append(
            RetrievedChunkResult(
                chunk_id=_normalize_string(raw_chunk.get("chunk_id")),
                doc_id=_normalize_string(raw_chunk.get("doc_id")),
                text=_normalize_string(raw_chunk.get("text")),
                record_id=_normalize_string(raw_chunk.get("record_id")),
                rank=_normalize_optional_int(raw_chunk.get("rank")),
                distance=_normalize_optional_float(raw_chunk.get("distance")),
                similarity_score=_normalize_optional_float(
                    raw_chunk.get("similarity_score")
                ),
                source_file=_normalize_string(raw_chunk.get("source_file")),
                metadata=_normalize_mapping(raw_chunk.get("metadata")),
                chunk_metadata=_normalize_mapping(raw_chunk.get("chunk_metadata")),
                document_metadata=_normalize_mapping(
                    raw_chunk.get("document_metadata")
                ),
            )
        )
    return chunks


def _build_grounding(raw_grounding: Any) -> GroundingVerificationResult:
    """
    Convert one raw grounding observation into a grounding result contract.
    """

    payload = _normalize_mapping(raw_grounding)
    return GroundingVerificationResult(
        status=_normalize_string(payload.get("status")),
        accepted=bool(payload.get("accepted", True)),
        citation_status=_normalize_string(payload.get("citation_status")),
        document_alignment=_normalize_string(payload.get("document_alignment")),
        article_alignment=_normalize_string(payload.get("article_alignment")),
        cited_documents=_normalize_string_list(payload.get("cited_documents")),
        cited_article_numbers=_normalize_string_list(
            payload.get("cited_article_numbers")
        ),
        mismatched_citations=_normalize_string_list(
            payload.get("mismatched_citations")
        ),
        unsupported_claims=_normalize_string_list(payload.get("unsupported_claims")),
        missing_required_facts=_normalize_string_list(
            payload.get("missing_required_facts")
        ),
        reasons=_normalize_string_list(payload.get("reasons")),
        metadata=_normalize_mapping(payload.get("metadata")),
    )


def _resolve_route_name(result: Any) -> str:
    """
    Resolve the observed route name from a final runtime result.
    """

    if result.route_metadata is None or result.route_metadata.route_decision is None:
        return ""
    return result.route_metadata.route_decision.route_name


def _resolve_output_directory(*, output_root: Path, run_id: str) -> Path:
    """
    Resolve and create the benchmark output directory.
    """

    normalized_run_id = run_id.strip() if isinstance(run_id, str) else ""
    if not normalized_run_id:
        raise ValueError("run_id cannot be empty.")

    output_directory = Path(output_root) / normalized_run_id
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def _write_summary(output_path: Path, summary: BenchmarkRunSummary) -> None:
    """
    Write one benchmark summary as stable JSON.
    """

    output_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _load_json_mapping(input_path: Path) -> Dict[str, Any]:
    """
    Load one JSON object file used for deterministic observations.
    """

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file '{input_path}': {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read JSON file '{input_path}': {exc}") from exc

    if isinstance(payload, list):
        return {
            _normalize_string(item.get("case_id")): item
            for item in payload
            if isinstance(item, dict) and _normalize_string(item.get("case_id"))
        }
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"JSON file '{input_path}' must contain an object or a list.")


def _normalize_mode(mode: str) -> str:
    """
    Normalize and validate the requested benchmark mode.
    """

    normalized_mode = mode.strip().lower() if isinstance(mode, str) else ""
    if normalized_mode not in _VALID_MODES:
        raise ValueError(
            "Benchmark mode must be one of: " + ", ".join(sorted(_VALID_MODES))
        )
    return normalized_mode


def _normalize_mapping(value: Any) -> Dict[str, Any]:
    """
    Normalize optional mapping input into a detached dictionary.
    """

    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_string(value: Any) -> str:
    """
    Normalize optional string input into a stripped string.
    """

    if isinstance(value, str):
        return value.strip()
    return ""


def _normalize_string_list(value: Any) -> List[str]:
    """
    Normalize optional list input into a clean string list.
    """

    if not isinstance(value, list):
        return []
    return [_normalize_string(item) for item in value if _normalize_string(item)]


def _normalize_optional_int(value: Any) -> Optional[int]:
    """
    Normalize optional integer input without accepting boolean values.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_float(value: Any) -> Optional[float]:
    """
    Normalize optional float input without accepting boolean values.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the benchmark CLI and print the output artifact location.
    """

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        summary = run_benchmark_main(
            mode=arguments.mode,
            settings=PipelineSettings(),
            questions_path=arguments.questions_path,
            guardrails_path=arguments.guardrails_path,
            output_root=arguments.output_root,
            run_id=arguments.run_id,
            retrieval_observations_path=arguments.retrieval_observations,
            answer_observations_path=arguments.answer_observations,
            top_k=arguments.top_k,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    output_directory = summary.metadata.get("output_directory", "")
    print(f"[INFO] Benchmark mode: {summary.mode}")
    print(f"[INFO] Output directory: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
