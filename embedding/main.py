from __future__ import annotations

import sys
from typing import Optional

from Chunking.config.settings import PipelineSettings
from embedding.indexer import EmbeddingIndexingResult, run_embedding_indexer


def run_embedding_main(
    settings: Optional[PipelineSettings] = None,
) -> EmbeddingIndexingResult:
    """
    Execute the embedding phase through the standalone module entrypoint.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared runtime settings. When omitted, default settings are loaded.

    Returns
    -------
    EmbeddingIndexingResult
        Summary of the completed embedding indexing run.
    """

    resolved_settings = settings or PipelineSettings()
    _validate_embedding_is_enabled(resolved_settings)
    return run_embedding_indexer(resolved_settings)


def _validate_embedding_is_enabled(settings: PipelineSettings) -> None:
    """
    Ensure the embedding phase is enabled in central pipeline settings.

    Parameters
    ----------
    settings : PipelineSettings
        Shared runtime settings.
    """

    if settings.embedding_enabled:
        return

    raise ValueError(
        "Embedding execution is disabled in 'config/appsettings.json'. "
        "Set 'embedding.enabled' to true before running the embedding entrypoint."
    )


def _print_result_summary(
    result: EmbeddingIndexingResult,
    settings: PipelineSettings,
) -> None:
    """
    Print a concise summary of the completed embedding execution.

    Parameters
    ----------
    result : EmbeddingIndexingResult
        Final result returned by the embedding indexer.

    settings : PipelineSettings
        Shared runtime settings used for the completed run.
    """

    print(f"[INFO] Embedding run completed: {result.run_id}")
    print(f"[INFO] Input records loaded: {result.input_record_count}")
    print(f"[INFO] Embedding records stored: {result.embedded_record_count}")
    print(f"[INFO] Provider used: {settings.embedding_provider}")
    print(f"[INFO] Model used: {settings.embedding_model}")
    print(f"[INFO] ChromaDB collection: {settings.chromadb_collection_name}")
    print(f"[INFO] ChromaDB target: {_build_chromadb_target_summary(settings)}")
    print(f"[INFO] ChromaDB audit path: {result.records_path}")
    print(f"[INFO] Run manifest path: {result.manifest_path}")
    if result.spotlight_export_path:
        print(f"[INFO] Spotlight export path: {result.spotlight_export_path}")


def _build_chromadb_target_summary(settings: PipelineSettings) -> str:
    """
    Build a human-readable ChromaDB target description for the final summary.

    Parameters
    ----------
    settings : PipelineSettings
        Shared runtime settings used for the completed run.

    Returns
    -------
    str
        Target summary describing the configured ChromaDB mode and location.
    """

    chromadb_mode = settings.chromadb_mode.strip().lower()
    if chromadb_mode == "persistent":
        return (
            "persistent "
            f"(path={settings.chromadb_persist_directory})"
        )

    tenant_label = settings.chromadb_cloud_tenant.strip() or "<unset>"
    database_label = settings.chromadb_cloud_database.strip() or "<unset>"
    return (
        "cloud "
        f"(tenant={tenant_label}; database={database_label})"
    )


def main() -> int:
    """
    Run the standalone embedding entrypoint and report the final result.

    Returns
    -------
    int
        Process exit code for shell execution.
    """

    resolved_settings = PipelineSettings()

    try:
        result = run_embedding_main(resolved_settings)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    _print_result_summary(result, resolved_settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
