from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from Chunking.config.settings import PipelineSettings
from embedding.models import EmbeddingRunManifest, EmbeddingVectorRecord


@dataclass(slots=True)
class EmbeddingStorageResult:
    """
    Describe the filesystem artifacts written for one embedding run.

    Attributes
    ----------
    run_directory : Path
        Root directory created for the embedding run.

    records_path : Path
        JSON file containing the persisted embedding records.

    manifest_path : Path
        JSON file containing the persisted run manifest.
    """

    run_directory: Path
    records_path: Path
    manifest_path: Path


class LocalEmbeddingStorage:
    """
    Persist embedding outputs to deterministic local JSON files.

    Design goals
    ------------
    - keep embedding outputs auditable and easy to inspect
    - group all artifacts of one run under a dedicated directory
    - preserve stable metadata needed for later reload and verification
    """

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """
        Initialize the local storage helper from pipeline settings.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared runtime settings. When omitted, default settings are loaded.
        """

        resolved_settings = settings or PipelineSettings()
        self.output_root = resolved_settings.embedding_output_root

    def save_run(
        self,
        embedding_records: Sequence[EmbeddingVectorRecord],
        manifest: EmbeddingRunManifest,
    ) -> EmbeddingStorageResult:
        """
        Persist both embedding records and the corresponding run manifest.

        Parameters
        ----------
        embedding_records : Sequence[EmbeddingVectorRecord]
            Generated embedding records to store.

        manifest : EmbeddingRunManifest
            Manifest describing the embedding execution.

        Returns
        -------
        EmbeddingStorageResult
            Paths of the created storage artifacts.
        """

        run_directory = self._resolve_run_directory(
            run_id=manifest.run_id,
            embedding_records=embedding_records,
            manifest=manifest,
        )
        records_path = run_directory / "embedding_records.json"
        manifest_path = run_directory / "run_manifest.json"

        records_payload = [
            self._vector_record_to_dict(record)
            for record in embedding_records
        ]
        manifest_payload = self._manifest_to_dict(
            manifest=manifest,
            run_directory=run_directory,
            records_path=records_path,
        )

        self._write_json(payload=records_payload, output_path=records_path)
        self._write_json(payload=manifest_payload, output_path=manifest_path)

        return EmbeddingStorageResult(
            run_directory=run_directory,
            records_path=records_path,
            manifest_path=manifest_path,
        )

    def save_embedding_records(
        self,
        run_id: str,
        embedding_records: Sequence[EmbeddingVectorRecord],
    ) -> Path:
        """
        Persist embedding records without writing the manifest.

        Parameters
        ----------
        run_id : str
            Stable run identifier used to build the output path.

        embedding_records : Sequence[EmbeddingVectorRecord]
            Generated embedding records to store.

        Returns
        -------
        Path
            Path to the created embedding records JSON file.
        """

        run_directory = self._resolve_run_directory(
            run_id=run_id,
            embedding_records=embedding_records,
        )
        records_path = run_directory / "embedding_records.json"
        records_payload = [
            self._vector_record_to_dict(record)
            for record in embedding_records
        ]
        self._write_json(payload=records_payload, output_path=records_path)
        return records_path

    def save_run_manifest(self, manifest: EmbeddingRunManifest) -> Path:
        """
        Persist the run manifest without rewriting embedding records.

        Parameters
        ----------
        manifest : EmbeddingRunManifest
            Manifest describing the embedding execution.

        Returns
        -------
        Path
            Path to the created manifest JSON file.
        """

        run_directory = self._resolve_run_directory(
            run_id=manifest.run_id,
            manifest=manifest,
        )
        manifest_path = run_directory / "run_manifest.json"
        manifest_payload = self._manifest_to_dict(
            manifest=manifest,
            run_directory=run_directory,
            records_path=run_directory / "embedding_records.json",
        )
        self._write_json(payload=manifest_payload, output_path=manifest_path)
        return manifest_path

    def _resolve_run_directory(
        self,
        run_id: str,
        embedding_records: Sequence[EmbeddingVectorRecord] = (),
        manifest: Optional[EmbeddingRunManifest] = None,
    ) -> Path:
        """
        Build the directory used to store one embedding run.

        Parameters
        ----------
        run_id : str
            Stable run identifier used to build the output path.

        embedding_records : Sequence[EmbeddingVectorRecord]
            Records used to infer the active strategy when available.

        manifest : Optional[EmbeddingRunManifest]
            Manifest used to infer the active strategy when available.

        Returns
        -------
        Path
            Run-specific directory under the configured embedding output root.
        """

        normalized_run_id = run_id.strip()
        if not normalized_run_id:
            raise ValueError("Embedding run id cannot be empty.")

        strategy_name = self._resolve_strategy_name(
            embedding_records=embedding_records,
            manifest=manifest,
        )

        run_directory = self.output_root / strategy_name / normalized_run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory

    def _resolve_strategy_name(
        self,
        embedding_records: Sequence[EmbeddingVectorRecord],
        manifest: Optional[EmbeddingRunManifest],
    ) -> str:
        """
        Resolve the single strategy associated with the persisted run.

        Parameters
        ----------
        embedding_records : Sequence[EmbeddingVectorRecord]
            Records used to infer strategy metadata.

        manifest : Optional[EmbeddingRunManifest]
            Manifest used as fallback for strategy inference.

        Returns
        -------
        str
            Normalized strategy name used in the output directory structure.
        """

        candidate_strategies = {
            strategy_name
            for strategy_name in [
                self._read_strategy_from_record(record)
                for record in embedding_records
            ]
            if strategy_name
        }

        manifest_strategy = self._read_strategy_from_manifest(manifest)
        if manifest_strategy:
            candidate_strategies.add(manifest_strategy)

        if not candidate_strategies:
            return "unknown_strategy"

        if len(candidate_strategies) > 1:
            raise ValueError(
                "Embedding storage received records from multiple strategies: "
                f"{sorted(candidate_strategies)}."
            )

        return next(iter(candidate_strategies))

    def _read_strategy_from_record(self, record: EmbeddingVectorRecord) -> str:
        """
        Read the stored strategy value from one embedding record metadata payload.

        Parameters
        ----------
        record : EmbeddingVectorRecord
            Record to inspect.

        Returns
        -------
        str
            Normalized strategy name, or an empty string when absent.
        """

        strategy_value = record.metadata.get("strategy", "")
        if isinstance(strategy_value, str):
            return strategy_value.strip().lower()
        return ""

    def _read_strategy_from_manifest(
        self,
        manifest: Optional[EmbeddingRunManifest],
    ) -> str:
        """
        Read the stored strategy value from manifest metadata when present.

        Parameters
        ----------
        manifest : Optional[EmbeddingRunManifest]
            Manifest to inspect.

        Returns
        -------
        str
            Normalized strategy name, or an empty string when absent.
        """

        if manifest is None:
            return ""

        strategy_value = manifest.metadata.get("strategy", "")
        if isinstance(strategy_value, str):
            return strategy_value.strip().lower()
        return ""

    def _vector_record_to_dict(
        self,
        record: EmbeddingVectorRecord,
    ) -> Dict[str, Any]:
        """
        Convert one embedding record into the persisted JSON structure.

        Parameters
        ----------
        record : EmbeddingVectorRecord
            Embedding record to serialize.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable record payload.
        """

        return {
            "chunk_id": record.chunk_id,
            "doc_id": record.doc_id,
            "strategy": self._read_strategy_from_record(record),
            "model": record.model,
            "provider": record.provider,
            "text": record.text,
            "source_file": record.source_file,
            "metadata": record.metadata,
            "vector": list(record.vector),
        }

    def _manifest_to_dict(
        self,
        manifest: EmbeddingRunManifest,
        run_directory: Path,
        records_path: Path,
    ) -> Dict[str, Any]:
        """
        Convert one run manifest into the persisted JSON structure.

        Parameters
        ----------
        manifest : EmbeddingRunManifest
            Manifest to serialize.

        run_directory : Path
            Run directory created for this manifest.

        records_path : Path
            Expected location of the embedding records JSON file.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable manifest payload.
        """

        source_path = manifest.input_root
        if len(manifest.source_files) == 1:
            source_path = manifest.source_files[0]

        return {
            "run_id": manifest.run_id,
            "provider": manifest.provider,
            "model": manifest.model,
            "strategy": self._read_strategy_from_manifest(manifest),
            "timestamp": manifest.generated_at_utc,
            "generated_at_utc": manifest.generated_at_utc,
            "input_root": manifest.input_root,
            "output_root": manifest.output_root,
            "input_text_field": manifest.input_text_field,
            "batch_size": manifest.batch_size,
            "record_count": manifest.record_count,
            "source_files": list(manifest.source_files),
            "source_path": source_path,
            "output_path": str(run_directory),
            "records_path": str(records_path),
            "metadata": manifest.metadata,
        }

    def _write_json(self, payload: Any, output_path: Path) -> None:
        """
        Write one JSON payload to disk using stable readable formatting.

        Parameters
        ----------
        payload : Any
            JSON-serializable payload.

        output_path : Path
            Destination file path.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
