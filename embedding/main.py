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


def _print_result_summary(result: EmbeddingIndexingResult) -> None:
    """
    Print a concise summary of the completed embedding execution.

    Parameters
    ----------
    result : EmbeddingIndexingResult
        Final result returned by the embedding indexer.
    """

    print(f"[INFO] Embedding run completed: {result.run_id}")
    print(f"[INFO] Input records loaded: {result.input_record_count}")
    print(f"[INFO] Embedding records stored: {result.embedded_record_count}")
    print(f"[INFO] Embedding records path: {result.records_path}")
    print(f"[INFO] Run manifest path: {result.manifest_path}")
    if result.spotlight_export_path:
        print(f"[INFO] Spotlight export path: {result.spotlight_export_path}")


def main() -> int:
    """
    Run the standalone embedding entrypoint and report the final result.

    Returns
    -------
    int
        Process exit code for shell execution.
    """

    try:
        result = run_embedding_main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    _print_result_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
