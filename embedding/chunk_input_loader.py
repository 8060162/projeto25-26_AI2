from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from Chunking.config.settings import PipelineSettings
from embedding.models import EmbeddingInputRecord
from embedding.text_builder import build_embedding_text


def load_embedding_input_records(
    settings: Optional[PipelineSettings] = None,
) -> List[EmbeddingInputRecord]:
    """
    Load embedding input records from chunk outputs of the active strategy.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared runtime settings. When omitted, default settings are loaded.

    Returns
    -------
    List[EmbeddingInputRecord]
        Ordered list of embedding-ready input records.
    """

    resolved_settings = settings or PipelineSettings()
    strategy_name = _resolve_active_embedding_strategy(resolved_settings)
    chunk_file_paths = _discover_chunk_file_paths(
        input_root=resolved_settings.embedding_input_root,
        strategy_name=strategy_name,
    )

    embedding_input_records: List[EmbeddingInputRecord] = []

    for chunk_file_path in chunk_file_paths:
        chunk_payload = _load_chunk_payload(chunk_file_path)

        for chunk_item in chunk_payload:
            embedding_input_record = _build_embedding_input_record(
                chunk_item=chunk_item,
                chunk_file_path=chunk_file_path,
                strategy_name=strategy_name,
                text_field_name=resolved_settings.embedding_input_text_field,
            )

            if embedding_input_record is not None:
                embedding_input_records.append(embedding_input_record)

    return embedding_input_records


def _resolve_active_embedding_strategy(settings: PipelineSettings) -> str:
    """
    Resolve the single chunking strategy allowed for embedding input loading.

    Parameters
    ----------
    settings : PipelineSettings
        Shared runtime settings.

    Returns
    -------
    str
        Active strategy name.
    """

    strategy_name = settings.chunking_strategy.strip().lower()

    if not strategy_name:
        raise ValueError("Chunking strategy cannot be empty for embedding input loading.")

    if strategy_name in {"all", "none"}:
        raise ValueError(
            "Embedding input loading requires a single active chunking strategy, "
            f"but received '{settings.chunking_strategy}'."
        )

    return strategy_name


def _discover_chunk_file_paths(input_root: Path, strategy_name: str) -> List[Path]:
    """
    Discover chunk JSON files that belong to the active strategy.

    Parameters
    ----------
    input_root : Path
        Root folder containing document chunk outputs.

    strategy_name : str
        Active chunking strategy name.

    Returns
    -------
    List[Path]
        Sorted list of chunk JSON files for the active strategy.
    """

    if not input_root.exists():
        raise FileNotFoundError(
            f"Embedding input root not found: '{input_root}'."
        )

    chunk_file_paths = sorted(
        path
        for path in input_root.rglob("05_chunks.json")
        if path.parent.name == strategy_name
    )

    if not chunk_file_paths:
        raise FileNotFoundError(
            "No chunk files were found for the active embedding strategy "
            f"'{strategy_name}' under '{input_root}'."
        )

    return chunk_file_paths


def _load_chunk_payload(chunk_file_path: Path) -> List[Dict[str, Any]]:
    """
    Read one chunk JSON file and validate its top-level payload shape.

    Parameters
    ----------
    chunk_file_path : Path
        Path to a `05_chunks.json` file.

    Returns
    -------
    List[Dict[str, Any]]
        Raw chunk dictionaries from the file.
    """

    try:
        with chunk_file_path.open("r", encoding="utf-8") as chunk_file:
            payload = json.load(chunk_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Failed to read chunk file '{chunk_file_path}': {exc}"
        ) from exc

    if not isinstance(payload, list):
        raise ValueError(
            f"Chunk file '{chunk_file_path}' must contain a top-level JSON list."
        )

    chunk_items: List[Dict[str, Any]] = []

    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(
                f"Chunk entry {index} in '{chunk_file_path}' must be a JSON object."
            )
        chunk_items.append(item)

    return chunk_items


def _build_embedding_input_record(
    chunk_item: Dict[str, Any],
    chunk_file_path: Path,
    strategy_name: str,
    text_field_name: str,
) -> Optional[EmbeddingInputRecord]:
    """
    Convert one exported chunk dictionary into an embedding input record.

    Parameters
    ----------
    chunk_item : Dict[str, Any]
        Raw chunk dictionary loaded from JSON.

    chunk_file_path : Path
        Source chunk file path.

    strategy_name : str
        Active chunking strategy name.

    text_field_name : str
        Preferred chunk field used as embedding text input.

    Returns
    -------
    Optional[EmbeddingInputRecord]
        Converted record, or `None` when the chunk has no usable text.
    """

    exported_strategy_name = _read_string_value(
        payload=chunk_item,
        key="strategy",
        default_value=strategy_name,
    )
    if exported_strategy_name and exported_strategy_name != strategy_name:
        return None

    chunk_id = _read_required_string_value(chunk_item, "chunk_id", chunk_file_path)
    doc_id = _read_required_string_value(chunk_item, "doc_id", chunk_file_path)
    raw_text_value = _resolve_embedding_text(chunk_item, text_field_name)

    if not raw_text_value:
        return None

    provisional_record = EmbeddingInputRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=raw_text_value,
        source_file=str(chunk_file_path),
        hierarchy_path=_read_string_list_value(chunk_item, "hierarchy_path"),
        page_start=_read_optional_int_value(chunk_item, "page_start"),
        page_end=_read_optional_int_value(chunk_item, "page_end"),
    )
    text_value = build_embedding_text(provisional_record)

    if not text_value:
        return None

    metadata = _build_embedding_metadata(
        chunk_item=chunk_item,
        chunk_file_path=chunk_file_path,
        strategy_name=strategy_name,
        text_field_name=text_field_name,
    )

    return EmbeddingInputRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text_value,
        metadata=metadata,
        source_file=str(chunk_file_path),
        hierarchy_path=_read_string_list_value(chunk_item, "hierarchy_path"),
        page_start=_read_optional_int_value(chunk_item, "page_start"),
        page_end=_read_optional_int_value(chunk_item, "page_end"),
    )


def _resolve_embedding_text(chunk_item: Dict[str, Any], text_field_name: str) -> str:
    """
    Resolve the text sent to the embedding stage from the exported chunk.

    Parameters
    ----------
    chunk_item : Dict[str, Any]
        Raw chunk dictionary loaded from JSON.

    text_field_name : str
        Preferred chunk field used as embedding text input.

    Returns
    -------
    str
        Clean text selected for embedding.
    """

    preferred_text = _read_string_value(
        payload=chunk_item,
        key=text_field_name,
        default_value="",
    ).strip()
    if preferred_text:
        return preferred_text

    return _read_string_value(
        payload=chunk_item,
        key="text",
        default_value="",
    ).strip()


def _build_embedding_metadata(
    chunk_item: Dict[str, Any],
    chunk_file_path: Path,
    strategy_name: str,
    text_field_name: str,
) -> Dict[str, Any]:
    """
    Build metadata preserved alongside the embedding input record.

    Parameters
    ----------
    chunk_item : Dict[str, Any]
        Raw chunk dictionary loaded from JSON.

    chunk_file_path : Path
        Source chunk file path.

    strategy_name : str
        Active chunking strategy name.

    text_field_name : str
        Preferred chunk field used as embedding text input.

    Returns
    -------
    Dict[str, Any]
        Metadata payload enriched with loader traceability.
    """

    raw_metadata = chunk_item.get("metadata", {})
    preserved_metadata = raw_metadata.copy() if isinstance(raw_metadata, dict) else {}

    preserved_metadata["strategy"] = strategy_name
    preserved_metadata["chunk_file_path"] = str(chunk_file_path)
    preserved_metadata["embedding_text_field"] = text_field_name
    preserved_metadata["source_node_type"] = _read_string_value(
        payload=chunk_item,
        key="source_node_type",
        default_value="",
    )
    preserved_metadata["source_node_label"] = _read_string_value(
        payload=chunk_item,
        key="source_node_label",
        default_value="",
    )
    preserved_metadata["chunk_reason"] = _read_string_value(
        payload=chunk_item,
        key="chunk_reason",
        default_value="",
    )
    preserved_metadata["prev_chunk_id"] = chunk_item.get("prev_chunk_id")
    preserved_metadata["next_chunk_id"] = chunk_item.get("next_chunk_id")
    preserved_metadata["char_count"] = _read_optional_int_value(chunk_item, "char_count")

    return preserved_metadata


def _read_required_string_value(
    payload: Dict[str, Any],
    key: str,
    chunk_file_path: Path,
) -> str:
    """
    Read a required non-empty string value from a chunk payload.

    Parameters
    ----------
    payload : Dict[str, Any]
        Raw chunk dictionary loaded from JSON.

    key : str
        Field name to read.

    chunk_file_path : Path
        Source chunk file path used for error context.

    Returns
    -------
    str
        Non-empty string value.
    """

    value = _read_string_value(payload=payload, key=key, default_value="").strip()
    if value:
        return value

    raise ValueError(
        f"Chunk file '{chunk_file_path}' contains a chunk without a valid '{key}' value."
    )


def _read_string_value(payload: Dict[str, Any], key: str, default_value: str) -> str:
    """
    Read a string value from a payload with a controlled fallback.

    Parameters
    ----------
    payload : Dict[str, Any]
        Source dictionary.

    key : str
        Field name to read.

    default_value : str
        Fallback value when the field is missing or invalid.
    """

    value = payload.get(key)
    if isinstance(value, str):
        return value
    return default_value


def _read_string_list_value(payload: Dict[str, Any], key: str) -> List[str]:
    """
    Read a list of strings from a payload with conservative filtering.

    Parameters
    ----------
    payload : Dict[str, Any]
        Source dictionary.

    key : str
        Field name to read.

    Returns
    -------
    List[str]
        String values preserved in original order.
    """

    raw_value = payload.get(key)
    if not isinstance(raw_value, list):
        return []

    return [item for item in raw_value if isinstance(item, str)]


def _read_optional_int_value(payload: Dict[str, Any], key: str) -> Optional[int]:
    """
    Read an optional integer value from a payload.

    Parameters
    ----------
    payload : Dict[str, Any]
        Source dictionary.

    key : str
        Field name to read.

    Returns
    -------
    Optional[int]
        Integer value when present and valid, otherwise `None`.
    """

    value = payload.get(key)
    if isinstance(value, int):
        return value
    return None
