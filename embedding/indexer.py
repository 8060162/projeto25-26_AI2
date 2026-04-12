from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Dict, List, Optional, Sequence

from Chunking.config.settings import PipelineSettings
from embedding.chunk_input_loader import load_embedding_input_records
from embedding.models import (
    EmbeddingInputRecord,
    EmbeddingRunManifest,
    EmbeddingVectorRecord,
)
from embedding.provider_factory import EmbeddingProvider, create_embedding_provider
from embedding.storage import ChromaEmbeddingStorage, EmbeddingStorageResult
from embedding.text_builder import build_embedding_text
from embedding.visualization.spotlight_export import export_spotlight_dataset


@dataclass(slots=True)
class EmbeddingIndexingResult:
    """
    Describe the outcome of one embedding indexing execution.

    Attributes
    ----------
    run_id : str
        Stable identifier assigned to the embedding run.

    input_record_count : int
        Number of records loaded from chunk outputs before final filtering.

    embedded_record_count : int
        Number of records successfully embedded and persisted.

    records_path : str
        Output path of the stored embedding records artifact.

    manifest_path : str
        Output path of the stored run manifest artifact.

    spotlight_export_path : Optional[str]
        Output path of the generated Spotlight dataset when visualization is enabled.
    """

    run_id: str
    input_record_count: int
    embedded_record_count: int
    records_path: str
    manifest_path: str
    spotlight_export_path: Optional[str] = None


def run_embedding_indexer(
    settings: Optional[PipelineSettings] = None,
) -> EmbeddingIndexingResult:
    """
    Execute the full embedding orchestration flow for the active strategy.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared runtime settings. When omitted, default settings are loaded.

    Returns
    -------
    EmbeddingIndexingResult
        Summary of the persisted embedding run artifacts.
    """

    resolved_settings = settings or PipelineSettings()
    loaded_input_records = load_embedding_input_records(resolved_settings)
    prepared_input_records = _prepare_input_records_for_embedding(loaded_input_records)

    if not prepared_input_records:
        raise ValueError("No embedding records with usable text were prepared.")

    provider = create_embedding_provider(resolved_settings)
    embedding_vector_records = _generate_embedding_vector_records(
        input_records=prepared_input_records,
        provider=provider,
        settings=resolved_settings,
    )

    run_manifest = _build_run_manifest(
        input_records=prepared_input_records,
        vector_records=embedding_vector_records,
        settings=resolved_settings,
    )

    storage = ChromaEmbeddingStorage(resolved_settings)
    storage_result = storage.save_run(
        embedding_records=embedding_vector_records,
        manifest=run_manifest,
    )
    spotlight_export_path = _export_spotlight_dataset_if_enabled(
        settings=resolved_settings,
        embedding_records=embedding_vector_records,
        manifest=run_manifest,
        storage_result=storage_result,
    )

    return _build_indexing_result(
        input_record_count=len(loaded_input_records),
        embedded_record_count=len(embedding_vector_records),
        manifest=run_manifest,
        storage_result=storage_result,
        spotlight_export_path=spotlight_export_path,
    )


def _prepare_input_records_for_embedding(
    input_records: Sequence[EmbeddingInputRecord],
) -> List[EmbeddingInputRecord]:
    """
    Rebuild and validate the final embedding text for each loaded input record.

    Parameters
    ----------
    input_records : Sequence[EmbeddingInputRecord]
        Records loaded from chunk outputs.

    Returns
    -------
    List[EmbeddingInputRecord]
        Records that still contain usable text after final normalization.
    """

    prepared_input_records: List[EmbeddingInputRecord] = []

    for input_record in input_records:
        embedding_text = build_embedding_text(input_record)
        if not embedding_text:
            continue

        prepared_input_records.append(
            EmbeddingInputRecord(
                chunk_id=input_record.chunk_id,
                doc_id=input_record.doc_id,
                text=embedding_text,
                metadata=dict(input_record.metadata),
                source_file=input_record.source_file,
                hierarchy_path=list(input_record.hierarchy_path),
                page_start=input_record.page_start,
                page_end=input_record.page_end,
            )
        )

    return prepared_input_records


def _generate_embedding_vector_records(
    input_records: Sequence[EmbeddingInputRecord],
    provider: EmbeddingProvider,
    settings: PipelineSettings,
) -> List[EmbeddingVectorRecord]:
    """
    Generate ordered embedding records in deterministic batches.

    Parameters
    ----------
    input_records : Sequence[EmbeddingInputRecord]
        Prepared embedding inputs.

    provider : EmbeddingProvider
        Configured provider used to generate vectors.

    settings : PipelineSettings
        Shared runtime settings.

    Returns
    -------
    List[EmbeddingVectorRecord]
        Persistable embedding records aligned with the input order.
    """

    embedding_vector_records: List[EmbeddingVectorRecord] = []

    for record_batch in _yield_record_batches(
        input_records=input_records,
        batch_size=settings.embedding_batch_size,
    ):
        batch_vectors = provider.embed_texts([record.text for record in record_batch])

        if len(batch_vectors) != len(record_batch):
            raise ValueError(
                "Embedding provider returned a vector count that does not match "
                "the requested batch size."
            )

        for input_record, vector in zip(record_batch, batch_vectors):
            strategy_name = _resolve_input_record_strategy_name(input_record)
            embedding_vector_records.append(
                EmbeddingVectorRecord(
                    chunk_id=input_record.chunk_id,
                    doc_id=input_record.doc_id,
                    vector=vector,
                    metadata=_build_vector_metadata(input_record),
                    model=settings.embedding_model,
                    provider=settings.embedding_provider,
                    source_file=input_record.source_file,
                    text=input_record.text,
                    record_id=_build_record_id(
                        strategy_name=strategy_name,
                        input_record=input_record,
                    ),
                    storage_record_id=_build_storage_record_id(
                        strategy_name=strategy_name,
                        input_record=input_record,
                    ),
                    storage_backend="chromadb",
                    storage_collection=settings.chromadb_collection_name,
                )
            )

    return embedding_vector_records


def _yield_record_batches(
    input_records: Sequence[EmbeddingInputRecord],
    batch_size: int,
) -> List[List[EmbeddingInputRecord]]:
    """
    Split ordered input records into deterministic batches.

    Parameters
    ----------
    input_records : Sequence[EmbeddingInputRecord]
        Prepared embedding inputs.

    batch_size : int
        Maximum number of records per batch.

    Returns
    -------
    List[List[EmbeddingInputRecord]]
        Ordered batches ready for provider calls.
    """

    if batch_size <= 0:
        raise ValueError("Embedding batch size must be greater than zero.")

    return [
        list(input_records[start_index : start_index + batch_size])
        for start_index in range(0, len(input_records), batch_size)
    ]


def _build_vector_metadata(input_record: EmbeddingInputRecord) -> Dict[str, object]:
    """
    Build the metadata payload stored with one embedding vector record.

    Parameters
    ----------
    input_record : EmbeddingInputRecord
        Prepared input record.

    Returns
    -------
    Dict[str, object]
        Metadata payload preserved for filtering, audit, and inspection.
    """

    vector_metadata = dict(input_record.metadata)
    vector_metadata["hierarchy_path"] = list(input_record.hierarchy_path)
    vector_metadata["page_start"] = input_record.page_start
    vector_metadata["page_end"] = input_record.page_end
    return vector_metadata


def _resolve_input_record_strategy_name(input_record: EmbeddingInputRecord) -> str:
    """
    Resolve the normalized strategy name stored on one prepared input record.

    Parameters
    ----------
    input_record : EmbeddingInputRecord
        Prepared input record being converted into a vector record.

    Returns
    -------
    str
        Non-empty normalized strategy name.
    """

    strategy_name = str(input_record.metadata.get("strategy", "")).strip().lower()
    if not strategy_name:
        raise ValueError(
            "Prepared embedding input record does not define a valid strategy."
        )

    return strategy_name


def _build_record_id(
    strategy_name: str,
    input_record: EmbeddingInputRecord,
) -> str:
    """
    Build a stable logical record identifier for one embedding output record.

    Parameters
    ----------
    strategy_name : str
        Active strategy associated with the prepared input record.

    input_record : EmbeddingInputRecord
        Prepared input record being converted into a vector record.

    Returns
    -------
    str
        Stable logical identifier preserved on the embedding record.
    """

    return "::".join(
        (
            strategy_name,
            input_record.doc_id.strip(),
            input_record.record_id.strip() or input_record.chunk_id.strip(),
        )
    )


def _build_storage_record_id(
    strategy_name: str,
    input_record: EmbeddingInputRecord,
) -> str:
    """
    Build a stable ChromaDB upsert identifier for one embedding output record.

    Parameters
    ----------
    strategy_name : str
        Active strategy associated with the prepared input record.

    input_record : EmbeddingInputRecord
        Prepared input record being converted into a vector record.

    Returns
    -------
    str
        Deterministic identifier suitable for repeated ChromaDB upserts.
    """

    logical_record_id = _build_record_id(
        strategy_name=strategy_name,
        input_record=input_record,
    )
    return f"emb_{sha256(logical_record_id.encode('utf-8')).hexdigest()[:32]}"


def _build_run_manifest(
    input_records: Sequence[EmbeddingInputRecord],
    vector_records: Sequence[EmbeddingVectorRecord],
    settings: PipelineSettings,
) -> EmbeddingRunManifest:
    """
    Build the manifest describing one completed embedding run.

    Parameters
    ----------
    input_records : Sequence[EmbeddingInputRecord]
        Prepared embedding inputs that were sent for vector generation.

    vector_records : Sequence[EmbeddingVectorRecord]
        Generated embedding records.

    settings : PipelineSettings
        Shared runtime settings.

    Returns
    -------
    EmbeddingRunManifest
        Manifest ready to be persisted with the embedding outputs.
    """

    strategy_name = _resolve_strategy_name(input_records)
    run_id = _build_run_id(strategy_name)
    source_files = sorted({record.source_file for record in input_records if record.source_file})
    vector_dimension = len(vector_records[0].vector) if vector_records else 0

    return EmbeddingRunManifest(
        run_id=run_id,
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        input_root=str(settings.embedding_input_root),
        output_root=str(settings.embedding_output_root),
        input_text_field=settings.embedding_input_text_field,
        batch_size=settings.embedding_batch_size,
        record_count=len(vector_records),
        source_files=source_files,
        metadata={
            "strategy": strategy_name,
            "input_record_count": len(input_records),
            "embedded_record_count": len(vector_records),
            "vector_dimension": vector_dimension,
        },
    )


def _resolve_strategy_name(input_records: Sequence[EmbeddingInputRecord]) -> str:
    """
    Resolve the single active strategy represented by the prepared inputs.

    Parameters
    ----------
    input_records : Sequence[EmbeddingInputRecord]
        Prepared embedding inputs.

    Returns
    -------
    str
        Normalized strategy name used for storage and audit metadata.
    """

    candidate_strategies = {
        str(record.metadata.get("strategy", "")).strip().lower()
        for record in input_records
        if str(record.metadata.get("strategy", "")).strip()
    }

    if not candidate_strategies:
        raise ValueError("Embedding input records do not define an active strategy.")

    if len(candidate_strategies) > 1:
        raise ValueError(
            "Embedding indexer received input records from multiple strategies: "
            f"{sorted(candidate_strategies)}."
        )

    return next(iter(candidate_strategies))


def _build_run_id(strategy_name: str) -> str:
    """
    Build a traceable run identifier for one embedding execution.

    Parameters
    ----------
    strategy_name : str
        Active strategy name for the current run.

    Returns
    -------
    str
        Stable string identifier combining strategy and UTC timestamp.
    """

    timestamp_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{strategy_name}_{timestamp_label}"


def _build_indexing_result(
    input_record_count: int,
    embedded_record_count: int,
    manifest: EmbeddingRunManifest,
    storage_result: EmbeddingStorageResult,
    spotlight_export_path: Optional[str],
) -> EmbeddingIndexingResult:
    """
    Convert persisted storage artifacts into the public indexer result model.

    Parameters
    ----------
    input_record_count : int
        Number of records loaded from chunk outputs before final filtering.

    embedded_record_count : int
        Number of records successfully embedded and persisted.

    manifest : EmbeddingRunManifest
        Manifest describing the completed embedding run.

    storage_result : EmbeddingStorageResult
        Paths created by local embedding storage.

    spotlight_export_path : Optional[str]
        Spotlight dataset path when visualization export was generated.

    Returns
    -------
    EmbeddingIndexingResult
        Result payload returned to callers of the indexer.
    """

    return EmbeddingIndexingResult(
        run_id=manifest.run_id,
        input_record_count=input_record_count,
        embedded_record_count=embedded_record_count,
        records_path=str(storage_result.records_path),
        manifest_path=str(storage_result.manifest_path),
        spotlight_export_path=spotlight_export_path,
    )


def _export_spotlight_dataset_if_enabled(
    settings: PipelineSettings,
    embedding_records: Sequence[EmbeddingVectorRecord],
    manifest: EmbeddingRunManifest,
    storage_result: EmbeddingStorageResult,
) -> Optional[str]:
    """
    Export a Spotlight dataset when visualization settings enable it.

    Parameters
    ----------
    settings : PipelineSettings
        Shared runtime settings.

    embedding_records : Sequence[EmbeddingVectorRecord]
        Generated embedding records from the current run.

    manifest : EmbeddingRunManifest
        Manifest describing the current embedding execution.

    storage_result : EmbeddingStorageResult
        Paths created by local embedding storage.

    Returns
    -------
    Optional[str]
        Dataset path when the export is generated, otherwise `None`.
    """

    if not settings.embedding_visualization_enabled:
        return None

    if not settings.embedding_visualization_spotlight_enabled:
        return None

    spotlight_result = export_spotlight_dataset(
        embedding_records=embedding_records,
        manifest=manifest,
        run_directory=storage_result.run_directory,
    )
    return str(spotlight_result.output_path)
