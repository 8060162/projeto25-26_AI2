from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from embedding.models import EmbeddingVectorRecord
from retrieval.evaluation.models import BenchmarkQuestionCase


@dataclass(slots=True)
class BenchmarkOverlayExportResult:
    """
    Describe the benchmark overlay dataset generated for embedding inspection.

    Attributes
    ----------
    output_path : Path
        JSONL dataset created for visual inspection.

    chunk_point_count : int
        Number of exported chunk embedding points.

    benchmark_question_point_count : int
        Number of exported benchmark-question embedding points.

    embedding_dimension : int
        Shared vector dimension used by all exported points.
    """

    output_path: Path
    chunk_point_count: int
    benchmark_question_point_count: int
    embedding_dimension: int

    @property
    def record_count(self) -> int:
        """
        Return the total number of exported overlay points.

        Returns
        -------
        int
            Sum of chunk and benchmark-question points.
        """

        return self.chunk_point_count + self.benchmark_question_point_count


def export_benchmark_overlay_dataset(
    chunk_embedding_records: Sequence[EmbeddingVectorRecord],
    benchmark_question_cases: Sequence[BenchmarkQuestionCase],
    benchmark_question_embeddings: Mapping[str, Sequence[float]],
    output_path: Path,
) -> BenchmarkOverlayExportResult:
    """
    Export chunk and benchmark-question embeddings into one JSONL overlay dataset.

    Parameters
    ----------
    chunk_embedding_records : Sequence[EmbeddingVectorRecord]
        Existing embedded chunk records to include in the overlay.

    benchmark_question_cases : Sequence[BenchmarkQuestionCase]
        Benchmark question cases whose embeddings must be included.

    benchmark_question_embeddings : Mapping[str, Sequence[float]]
        Precomputed question vectors keyed by benchmark case id.

    output_path : Path
        Destination JSONL file.

    Returns
    -------
    BenchmarkOverlayExportResult
        Summary of the generated overlay dataset.
    """

    normalized_question_vectors = _normalize_question_vectors(
        benchmark_question_cases=benchmark_question_cases,
        benchmark_question_embeddings=benchmark_question_embeddings,
    )
    embedding_dimension = _validate_embedding_dimensions(
        chunk_embedding_records=chunk_embedding_records,
        benchmark_question_embeddings=normalized_question_vectors,
    )

    chunk_rows = [
        _build_chunk_overlay_row(chunk_record)
        for chunk_record in chunk_embedding_records
    ]
    question_rows = [
        _build_benchmark_question_overlay_row(
            benchmark_case=benchmark_case,
            embedding=normalized_question_vectors[benchmark_case.case_id],
        )
        for benchmark_case in benchmark_question_cases
    ]

    _write_jsonl(rows=[*chunk_rows, *question_rows], output_path=output_path)

    return BenchmarkOverlayExportResult(
        output_path=output_path,
        chunk_point_count=len(chunk_rows),
        benchmark_question_point_count=len(question_rows),
        embedding_dimension=embedding_dimension,
    )


def _normalize_question_vectors(
    benchmark_question_cases: Sequence[BenchmarkQuestionCase],
    benchmark_question_embeddings: Mapping[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """
    Normalize and validate question vectors for the requested benchmark cases.

    Parameters
    ----------
    benchmark_question_cases : Sequence[BenchmarkQuestionCase]
        Benchmark cases that must be represented in the overlay.

    benchmark_question_embeddings : Mapping[str, Sequence[float]]
        Raw vector mapping keyed by benchmark case id.

    Returns
    -------
    Dict[str, List[float]]
        Detached question vectors keyed by case id.
    """

    normalized_vectors: Dict[str, List[float]] = {}

    for benchmark_case in benchmark_question_cases:
        case_id = benchmark_case.case_id
        if not case_id:
            raise ValueError("Benchmark question case id cannot be empty.")

        if case_id not in benchmark_question_embeddings:
            raise ValueError(
                "Missing benchmark-question embedding for case id "
                f"'{case_id}'."
            )

        normalized_vectors[case_id] = _normalize_vector(
            benchmark_question_embeddings[case_id],
            label=f"benchmark-question embedding for case id '{case_id}'",
        )

    return normalized_vectors


def _validate_embedding_dimensions(
    chunk_embedding_records: Sequence[EmbeddingVectorRecord],
    benchmark_question_embeddings: Mapping[str, Sequence[float]],
) -> int:
    """
    Ensure all exported vectors use one shared embedding dimension.

    Parameters
    ----------
    chunk_embedding_records : Sequence[EmbeddingVectorRecord]
        Chunk vectors to validate.

    benchmark_question_embeddings : Mapping[str, Sequence[float]]
        Question vectors to validate.

    Returns
    -------
    int
        Shared vector dimension, or zero when no points are exported.
    """

    expected_dimension: Optional[int] = None

    for chunk_record in chunk_embedding_records:
        vector = _normalize_vector(
            chunk_record.vector,
            label=f"chunk embedding for chunk id '{chunk_record.chunk_id}'",
        )
        expected_dimension = _resolve_expected_dimension(
            expected_dimension=expected_dimension,
            vector=vector,
            label=f"chunk id '{chunk_record.chunk_id}'",
        )

    for case_id, vector in benchmark_question_embeddings.items():
        expected_dimension = _resolve_expected_dimension(
            expected_dimension=expected_dimension,
            vector=_normalize_vector(
                vector,
                label=f"benchmark-question embedding for case id '{case_id}'",
            ),
            label=f"benchmark case id '{case_id}'",
        )

    return expected_dimension or 0


def _resolve_expected_dimension(
    expected_dimension: Optional[int],
    vector: Sequence[float],
    label: str,
) -> int:
    """
    Resolve or compare the expected embedding dimension for one vector.

    Parameters
    ----------
    expected_dimension : Optional[int]
        Dimension established by earlier vectors.

    vector : Sequence[float]
        Current vector to compare.

    label : str
        Human-readable vector label for validation errors.

    Returns
    -------
    int
        Established shared embedding dimension.
    """

    vector_dimension = len(vector)
    if expected_dimension is None:
        return vector_dimension

    if vector_dimension != expected_dimension:
        raise ValueError(
            "Inconsistent embedding dimension for "
            f"{label}: expected {expected_dimension}, got {vector_dimension}."
        )

    return expected_dimension


def _build_chunk_overlay_row(record: EmbeddingVectorRecord) -> Dict[str, Any]:
    """
    Convert one chunk embedding record into a benchmark overlay row.

    Parameters
    ----------
    record : EmbeddingVectorRecord
        Chunk embedding record to export.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable overlay row.
    """

    metadata = _as_mapping(record.metadata)
    document_title = _read_document_title(record=record, metadata=metadata)
    article_number = _read_string(metadata.get("article_number"))

    return {
        "point_type": "chunk",
        "point_id": record.record_id or record.chunk_id,
        "chunk_id": record.chunk_id,
        "benchmark_case_id": "",
        "expected_doc_id": "",
        "expected_article_numbers": [],
        "expected_chunk_ids": [],
        "doc_id": record.doc_id,
        "document_title": document_title,
        "article_number": article_number,
        "article_title": _read_string(metadata.get("article_title")),
        "case_type": "",
        "color_group": _build_color_group("chunk", record.doc_id),
        "text": record.text,
        "embedding": _normalize_vector(
            record.vector,
            label=f"chunk embedding for chunk id '{record.chunk_id}'",
        ),
        "provider": record.provider,
        "model": record.model,
        "source_file": record.source_file,
        "hierarchy_path": _read_string_list(metadata.get("hierarchy_path")),
        "hierarchy_path_text": " > ".join(
            _read_string_list(metadata.get("hierarchy_path"))
        ),
        "page_start": _read_optional_int(metadata.get("page_start")),
        "page_end": _read_optional_int(metadata.get("page_end")),
    }


def _build_benchmark_question_overlay_row(
    benchmark_case: BenchmarkQuestionCase,
    embedding: Sequence[float],
) -> Dict[str, Any]:
    """
    Convert one benchmark question and vector into an overlay row.

    Parameters
    ----------
    benchmark_case : BenchmarkQuestionCase
        Benchmark question case represented by the vector.

    embedding : Sequence[float]
        Precomputed question embedding.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable overlay row.
    """

    expected_doc_id = benchmark_case.expected_doc_id or ""
    expected_document_title = _resolve_expected_document_title(benchmark_case)

    return {
        "point_type": "benchmark_question",
        "point_id": benchmark_case.case_id,
        "chunk_id": "",
        "benchmark_case_id": benchmark_case.case_id,
        "expected_doc_id": expected_doc_id,
        "expected_article_numbers": list(benchmark_case.expected_article_numbers),
        "expected_chunk_ids": list(benchmark_case.expected_chunk_ids),
        "doc_id": "",
        "document_title": expected_document_title,
        "article_number": "",
        "article_title": "",
        "case_type": benchmark_case.case_type,
        "color_group": _build_color_group(
            "benchmark_question",
            expected_doc_id or "unscoped",
        ),
        "text": benchmark_case.question,
        "embedding": _normalize_vector(
            embedding,
            label=(
                "benchmark-question embedding for case id "
                f"'{benchmark_case.case_id}'"
            ),
        ),
        "provider": "",
        "model": "",
        "source_file": "",
        "hierarchy_path": [],
        "hierarchy_path_text": "",
        "page_start": None,
        "page_end": None,
    }


def _resolve_expected_document_title(
    benchmark_case: BenchmarkQuestionCase,
) -> str:
    """
    Resolve the expected document title for one benchmark question.

    Parameters
    ----------
    benchmark_case : BenchmarkQuestionCase
        Benchmark case to inspect.

    Returns
    -------
    str
        First expected route document title when available.
    """

    if benchmark_case.expected_route.target_document_titles:
        return benchmark_case.expected_route.target_document_titles[0]
    return ""


def _read_document_title(
    record: EmbeddingVectorRecord,
    metadata: Mapping[str, Any],
) -> str:
    """
    Read a stable document title from record metadata when available.

    Parameters
    ----------
    record : EmbeddingVectorRecord
        Chunk embedding record used as the fallback source.

    metadata : Mapping[str, Any]
        Metadata payload attached to the embedding record.

    Returns
    -------
    str
        Human-readable document title or the document id fallback.
    """

    document_title = _read_string(metadata.get("document_title"))
    if document_title:
        return document_title
    return record.doc_id


def _build_color_group(point_type: str, group_id: str) -> str:
    """
    Build a stable visual grouping label for overlay consumers.

    Parameters
    ----------
    point_type : str
        Overlay point type.

    group_id : str
        Document or expected-document grouping key.

    Returns
    -------
    str
        Group label combining point type and document context.
    """

    normalized_group_id = group_id.strip() if isinstance(group_id, str) else ""
    if not normalized_group_id:
        normalized_group_id = "unknown"

    return f"{point_type}:{normalized_group_id}"


def _normalize_vector(vector: Sequence[float], label: str) -> List[float]:
    """
    Normalize one raw embedding vector into a detached float list.

    Parameters
    ----------
    vector : Sequence[float]
        Raw embedding vector to normalize.

    label : str
        Human-readable vector label for validation errors.

    Returns
    -------
    List[float]
        Detached vector containing numeric values.
    """

    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise ValueError(f"{label} must be a numeric sequence.")

    normalized_vector: List[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must contain only numeric values.")
        normalized_vector.append(float(value))

    if not normalized_vector:
        raise ValueError(f"{label} cannot be empty.")

    return normalized_vector


def _write_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """
    Persist overlay rows as line-delimited JSON.

    Parameters
    ----------
    rows : Iterable[Dict[str, Any]]
        Dataset rows to persist.

    output_path : Path
        Destination dataset path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False))
            output_file.write("\n")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """
    Safely interpret arbitrary metadata as a mapping.

    Parameters
    ----------
    value : Any
        Candidate metadata payload.

    Returns
    -------
    Mapping[str, Any]
        Mapping view of the metadata, or an empty mapping when invalid.
    """

    if isinstance(value, dict):
        return value
    return {}


def _read_string(value: Any) -> str:
    """
    Normalize a raw value into a stripped string when possible.

    Parameters
    ----------
    value : Any
        Raw value to inspect.

    Returns
    -------
    str
        Normalized string, or an empty string when unavailable.
    """

    if isinstance(value, str):
        return value.strip()
    return ""


def _read_string_list(value: Any) -> List[str]:
    """
    Normalize a raw value into a list of strings.

    Parameters
    ----------
    value : Any
        Raw value to inspect.

    Returns
    -------
    List[str]
        Ordered list of non-empty strings.
    """

    if not isinstance(value, list):
        return []

    string_values: List[str] = []
    for item in value:
        normalized_value = _read_string(item)
        if normalized_value:
            string_values.append(normalized_value)

    return string_values


def _read_optional_int(value: Any) -> Optional[int]:
    """
    Normalize a raw value into an integer when available.

    Parameters
    ----------
    value : Any
        Raw value to inspect.

    Returns
    -------
    Optional[int]
        Integer value, or `None` when unavailable.
    """

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
