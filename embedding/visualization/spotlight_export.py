from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from embedding.models import EmbeddingRunManifest, EmbeddingVectorRecord


@dataclass(slots=True)
class SpotlightExportResult:
    """
    Describe the dataset artifact generated for Spotlight inspection.

    Attributes
    ----------
    output_path : Path
        JSONL dataset created for Spotlight inspection.

    record_count : int
        Number of exported embedding records.
    """

    output_path: Path
    record_count: int


def export_spotlight_dataset(
    embedding_records: Sequence[EmbeddingVectorRecord],
    manifest: EmbeddingRunManifest,
    run_directory: Path,
    output_filename: str = "spotlight_dataset.jsonl",
) -> SpotlightExportResult:
    """
    Export embedding results into a Spotlight-friendly JSONL dataset.

    Parameters
    ----------
    embedding_records : Sequence[EmbeddingVectorRecord]
        Persistable embedding records from one embedding run.

    manifest : EmbeddingRunManifest
        Manifest describing the current embedding run.

    run_directory : Path
        Output directory that stores artifacts for the current run.

    output_filename : str
        Dataset file name created inside the run directory.

    Returns
    -------
    SpotlightExportResult
        Summary of the generated Spotlight dataset artifact.
    """

    normalized_output_filename = output_filename.strip()
    if not normalized_output_filename:
        raise ValueError("Spotlight output filename cannot be empty.")

    run_directory.mkdir(parents=True, exist_ok=True)
    output_path = run_directory / normalized_output_filename

    spotlight_rows = [
        _build_spotlight_row(record=record, manifest=manifest)
        for record in embedding_records
    ]
    _write_jsonl(rows=spotlight_rows, output_path=output_path)

    return SpotlightExportResult(
        output_path=output_path,
        record_count=len(spotlight_rows),
    )


def _build_spotlight_row(
    record: EmbeddingVectorRecord,
    manifest: EmbeddingRunManifest,
) -> Dict[str, Any]:
    """
    Convert one embedding record into one Spotlight dataset row.

    Parameters
    ----------
    record : EmbeddingVectorRecord
        Persisted embedding record to convert.

    manifest : EmbeddingRunManifest
        Manifest that provides run-level export metadata.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable row tailored for Spotlight inspection.
    """

    metadata = _as_mapping(record.metadata)
    hierarchy_path = _read_string_list(metadata.get("hierarchy_path"))
    page_start = _read_optional_int(metadata.get("page_start"))
    page_end = _read_optional_int(metadata.get("page_end"))
    strategy_name = _read_string(metadata.get("strategy")) or _read_manifest_strategy(
        manifest
    )

    spotlight_row: Dict[str, Any] = {
        "chunk_id": record.chunk_id,
        "doc_id": record.doc_id,
        "text": record.text,
        "document_title": _read_document_title(metadata, record.doc_id),
        "page_start": page_start,
        "page_end": page_end,
        "strategy": strategy_name,
        "embedding": list(record.vector),
        "provider": record.provider,
        "model": record.model,
        "source_file": record.source_file,
        "hierarchy_path": hierarchy_path,
        "hierarchy_path_text": " > ".join(hierarchy_path),
        "run_id": manifest.run_id,
        "generated_at_utc": manifest.generated_at_utc,
    }

    spotlight_row.update(_flatten_metadata(metadata))
    return spotlight_row


def _read_document_title(metadata: Mapping[str, Any], default_value: str) -> str:
    """
    Read a stable document title from record metadata when available.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        Metadata payload attached to one embedding record.

    default_value : str
        Fallback value used when no document title is present.

    Returns
    -------
    str
        Human-readable document title used by the Spotlight dataset.
    """

    document_title = _read_string(metadata.get("document_title"))
    if document_title:
        return document_title
    return default_value


def _read_manifest_strategy(manifest: EmbeddingRunManifest) -> str:
    """
    Read the strategy associated with one embedding run manifest.

    Parameters
    ----------
    manifest : EmbeddingRunManifest
        Manifest to inspect.

    Returns
    -------
    str
        Normalized strategy name when available.
    """

    if not isinstance(manifest.metadata, dict):
        return ""
    return _read_string(manifest.metadata.get("strategy"))


def _flatten_metadata(
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Flatten scalar metadata fields for easier Spotlight filtering.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        Original embedding metadata payload.

    Returns
    -------
    Dict[str, Any]
        Scalar-only flattened metadata columns prefixed with `metadata_`.
    """

    flattened_metadata: Dict[str, Any] = {}

    for key, value in metadata.items():
        normalized_key = _normalize_metadata_key(key)
        if not normalized_key:
            continue

        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened_metadata[f"metadata_{normalized_key}"] = value
            continue

        if isinstance(value, list) and all(
            isinstance(item, str) for item in value
        ):
            flattened_metadata[f"metadata_{normalized_key}"] = list(value)
            continue

    return flattened_metadata


def _normalize_metadata_key(key: Any) -> str:
    """
    Normalize one metadata key so it can be used as a dataset column name.

    Parameters
    ----------
    key : Any
        Raw metadata key to normalize.

    Returns
    -------
    str
        Lowercase normalized metadata key.
    """

    if not isinstance(key, str):
        return ""

    normalized_key = key.strip().lower()
    if not normalized_key:
        return ""

    return normalized_key.replace(" ", "_")


def _write_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """
    Persist Spotlight rows as line-delimited JSON.

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
