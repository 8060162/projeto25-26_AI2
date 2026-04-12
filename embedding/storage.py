from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from Chunking.config.settings import PipelineSettings
from embedding.models import EmbeddingRunManifest, EmbeddingVectorRecord


@dataclass(slots=True)
class EmbeddingStorageResult:
    """
    Describe the local audit artifacts written for one embedding run.

    Attributes
    ----------
    run_directory : Path
        Root directory created for the embedding run auxiliary artifacts.

    records_path : Path
        Local audit file describing the ChromaDB persistence result.

    manifest_path : Path
        JSON file containing the persisted run manifest.
    """

    run_directory: Path
    records_path: Path
    manifest_path: Path


class ChromaEmbeddingStorage:
    """
    Persist embedding outputs into ChromaDB and keep small local audit artifacts.

    Design goals
    ------------
    - use ChromaDB as the primary source of truth for stored embeddings
    - keep run manifests and storage summaries easy to inspect locally
    - preserve deterministic replacement behavior for one active strategy
    """

    _CHROMADB_METADATA_FIELD_LIMIT = 32
    _DIRECT_METADATA_KEYS = (
        "chunk_file_path",
        "source_node_type",
        "source_node_label",
        "chunk_reason",
        "char_count",
    )
    _DIRECT_CHUNK_METADATA_KEYS = (
        "hierarchy_path",
        "page_start",
        "page_end",
        "chunk_index",
        "chunk_kind",
        "section_title",
        "article_number",
        "article_title",
    )
    _DIRECT_DOCUMENT_METADATA_KEYS = (
        "document_title",
        "document_type",
        "source_document_path",
        "language",
        "jurisdiction",
    )

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """
        Initialize the storage helper from shared runtime settings.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared runtime settings. When omitted, default settings are loaded.
        """

        resolved_settings = settings or PipelineSettings()
        self.output_root = resolved_settings.embedding_output_root
        self.chromadb_mode = resolved_settings.chromadb_mode
        self.chromadb_persist_directory = resolved_settings.chromadb_persist_directory
        self.chromadb_collection_name = resolved_settings.chromadb_collection_name
        self.chromadb_cloud_tenant = resolved_settings.chromadb_cloud_tenant
        self.chromadb_cloud_database = resolved_settings.chromadb_cloud_database
        self.chromadb_cloud_host = resolved_settings.chromadb_cloud_host
        self.chromadb_cloud_port = resolved_settings.chromadb_cloud_port
        self.chromadb_cloud_api_key_env_var = (
            resolved_settings.chromadb_cloud_api_key_env_var
        )

    def save_run(
        self,
        embedding_records: Sequence[EmbeddingVectorRecord],
        manifest: EmbeddingRunManifest,
    ) -> EmbeddingStorageResult:
        """
        Persist embeddings to ChromaDB and write the run-level local artifacts.

        Parameters
        ----------
        embedding_records : Sequence[EmbeddingVectorRecord]
            Generated embedding records to store.

        manifest : EmbeddingRunManifest
            Manifest describing the embedding execution.

        Returns
        -------
        EmbeddingStorageResult
            Paths of the created local audit artifacts.
        """

        run_directory = self._resolve_run_directory(
            run_id=manifest.run_id,
            embedding_records=embedding_records,
            manifest=manifest,
            replace_existing_strategy_output=True,
        )
        records_path = run_directory / "chromadb_storage.json"
        manifest_path = run_directory / "run_manifest.json"

        strategy_name = self._resolve_strategy_name(
            embedding_records=embedding_records,
            manifest=manifest,
        )
        collection = self._get_collection()
        deleted_record_count = self._delete_existing_strategy_records(
            collection=collection,
            strategy_name=strategy_name,
        )
        upserted_record_count = self._upsert_embedding_records(
            collection=collection,
            embedding_records=embedding_records,
            strategy_name=strategy_name,
        )

        storage_summary_payload = self._build_storage_summary_payload(
            embedding_records=embedding_records,
            manifest=manifest,
            strategy_name=strategy_name,
            deleted_record_count=deleted_record_count,
            upserted_record_count=upserted_record_count,
            run_directory=run_directory,
        )
        manifest_payload = self._manifest_to_dict(
            manifest=manifest,
            run_directory=run_directory,
            strategy_name=strategy_name,
            upserted_record_count=upserted_record_count,
        )

        self._write_json(payload=storage_summary_payload, output_path=records_path)
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
            Path to the created ChromaDB audit file.
        """

        run_directory = self._resolve_run_directory(
            run_id=run_id,
            embedding_records=embedding_records,
            replace_existing_strategy_output=True,
        )
        records_path = run_directory / "chromadb_storage.json"
        strategy_name = self._resolve_strategy_name(
            embedding_records=embedding_records,
            manifest=None,
        )
        collection = self._get_collection()
        deleted_record_count = self._delete_existing_strategy_records(
            collection=collection,
            strategy_name=strategy_name,
        )
        upserted_record_count = self._upsert_embedding_records(
            collection=collection,
            embedding_records=embedding_records,
            strategy_name=strategy_name,
        )

        self._write_json(
            payload={
                "run_id": run_id.strip(),
                "storage_backend": "chromadb",
                "storage_scope": self._normalize_chromadb_mode(),
                "storage_collection": self.chromadb_collection_name,
                "storage_location": self._build_storage_location(),
                "strategy": strategy_name,
                "deleted_record_count": deleted_record_count,
                "upserted_record_count": upserted_record_count,
                "record_ids": [
                    self._resolve_storage_record_id(record)
                    for record in embedding_records
                ],
            },
            output_path=records_path,
        )
        return records_path

    def save_run_manifest(self, manifest: EmbeddingRunManifest) -> Path:
        """
        Persist the run manifest without rewriting ChromaDB records.

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
        strategy_name = self._resolve_strategy_name(
            embedding_records=(),
            manifest=manifest,
        )
        self._write_json(
            payload=self._manifest_to_dict(
                manifest=manifest,
                run_directory=run_directory,
                strategy_name=strategy_name,
                upserted_record_count=manifest.record_count,
            ),
            output_path=manifest_path,
        )
        return manifest_path

    def _get_collection(self) -> Any:
        """
        Build the ChromaDB collection used by the embedding storage layer.

        Returns
        -------
        Any
            ChromaDB collection object ready for delete and upsert operations.
        """

        client = self._build_chromadb_client()
        return client.get_or_create_collection(name=self.chromadb_collection_name)

    def _build_chromadb_client(self) -> Any:
        """
        Build the configured ChromaDB client.

        Returns
        -------
        Any
            ChromaDB client instance for the configured storage mode.
        """

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Package 'chromadb' is required to persist embeddings."
            ) from exc

        chromadb_mode = self._normalize_chromadb_mode()

        if chromadb_mode == "cloud":
            api_key_env_var = self.chromadb_cloud_api_key_env_var.strip()
            api_key = os.environ.get(api_key_env_var, "").strip()
            if not api_key:
                raise ValueError(
                    "Environment variable "
                    f"'{self.chromadb_cloud_api_key_env_var}' is required for "
                    "ChromaDB cloud storage."
                )
            if not self.chromadb_cloud_tenant.strip():
                raise ValueError("ChromaDB cloud tenant cannot be empty.")
            if not self.chromadb_cloud_database.strip():
                raise ValueError("ChromaDB cloud database cannot be empty.")
            if not self.chromadb_cloud_host.strip():
                raise ValueError("ChromaDB cloud host cannot be empty.")
            if self.chromadb_cloud_port <= 0:
                raise ValueError("ChromaDB cloud port must be greater than zero.")

            try:
                return chromadb.CloudClient(
                    api_key=api_key,
                    tenant=self.chromadb_cloud_tenant,
                    database=self.chromadb_cloud_database,
                    cloud_host=self.chromadb_cloud_host,
                    cloud_port=self.chromadb_cloud_port,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize the ChromaDB cloud client: "
                    f"{exc}"
                ) from exc

        if chromadb_mode == "persistent":
            self.chromadb_persist_directory.mkdir(parents=True, exist_ok=True)
            try:
                return chromadb.PersistentClient(
                    path=str(self.chromadb_persist_directory)
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize the persistent ChromaDB client: "
                    f"{exc}"
                ) from exc

        raise ValueError(
            "Unsupported ChromaDB mode configured in settings: "
            f"'{self.chromadb_mode}'. Supported modes: cloud, persistent."
        )

    def _normalize_chromadb_mode(self) -> str:
        """
        Normalize the configured ChromaDB mode for internal branching.

        Returns
        -------
        str
            Lowercase storage mode name.
        """

        normalized_mode = self.chromadb_mode.strip().lower()
        if not normalized_mode:
            raise ValueError("ChromaDB mode cannot be empty.")
        return normalized_mode

    def _delete_existing_strategy_records(
        self,
        collection: Any,
        strategy_name: str,
    ) -> int:
        """
        Delete any previously stored ChromaDB records for one strategy.

        Parameters
        ----------
        collection : Any
            ChromaDB collection used by the storage layer.

        strategy_name : str
            Normalized strategy name to replace.

        Returns
        -------
        int
            Number of deleted records when the collection reports them.
        """

        existing_records = collection.get(
            where={"strategy": strategy_name},
            include=[],
        )
        existing_ids = existing_records.get("ids") or []
        if not existing_ids:
            return 0

        collection.delete(ids=list(existing_ids))
        return len(existing_ids)

    def _upsert_embedding_records(
        self,
        collection: Any,
        embedding_records: Sequence[EmbeddingVectorRecord],
        strategy_name: str,
    ) -> int:
        """
        Upsert embedding records into the active ChromaDB collection.

        Parameters
        ----------
        collection : Any
            ChromaDB collection used by the storage layer.

        embedding_records : Sequence[EmbeddingVectorRecord]
            Generated embedding records to store.

        strategy_name : str
            Normalized strategy name attached to the stored metadata.

        Returns
        -------
        int
            Number of upserted embedding records.
        """

        if not embedding_records:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        for record in embedding_records:
            ids.append(self._resolve_storage_record_id(record))
            documents.append(record.text)
            embeddings.append(list(record.vector))
            metadatas.append(
                self._build_chromadb_metadata(
                    record=record,
                    strategy_name=strategy_name,
                )
            )

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    def _resolve_storage_record_id(self, record: EmbeddingVectorRecord) -> str:
        """
        Resolve the stable storage identifier for one embedding record.

        Parameters
        ----------
        record : EmbeddingVectorRecord
            Record being persisted.

        Returns
        -------
        str
            Non-empty record identifier used by ChromaDB.
        """

        storage_record_id = (
            record.storage_record_id.strip() or record.record_id.strip()
        )
        if storage_record_id:
            return storage_record_id
        return record.chunk_id.strip()

    def _build_chromadb_metadata(
        self,
        record: EmbeddingVectorRecord,
        strategy_name: str,
    ) -> Dict[str, Any]:
        """
        Build one flat ChromaDB metadata payload from an embedding record.

        Parameters
        ----------
        record : EmbeddingVectorRecord
            Record being persisted.

        strategy_name : str
            Normalized strategy name attached to the stored metadata.

        Returns
        -------
        Dict[str, Any]
            Flat metadata payload compatible with ChromaDB.
        """

        metadata: Dict[str, Any] = {
            "record_id": record.record_id,
            "storage_record_id": self._resolve_storage_record_id(record),
            "chunk_id": record.chunk_id,
            "doc_id": record.doc_id,
            "strategy": strategy_name,
            "provider": record.provider,
            "model": record.model,
        }

        if record.source_file:
            metadata["source_file"] = record.source_file

        metadata.update(
            self._build_scoped_chromadb_metadata(
                prefix="metadata",
                payload=record.metadata,
                direct_keys=self._DIRECT_METADATA_KEYS,
            )
        )
        metadata.update(
            self._build_scoped_chromadb_metadata(
                prefix="chunk",
                payload=record.chunk_metadata,
                direct_keys=self._DIRECT_CHUNK_METADATA_KEYS,
            )
        )
        metadata.update(
            self._build_scoped_chromadb_metadata(
                prefix="document",
                payload=record.document_metadata,
                direct_keys=self._DIRECT_DOCUMENT_METADATA_KEYS,
            )
        )

        if len(metadata) > self._CHROMADB_METADATA_FIELD_LIMIT:
            raise ValueError(
                "ChromaDB metadata payload exceeded the configured field budget "
                f"of {self._CHROMADB_METADATA_FIELD_LIMIT} keys."
            )

        return metadata

    def _build_scoped_chromadb_metadata(
        self,
        prefix: str,
        payload: Dict[str, Any],
        direct_keys: Sequence[str],
    ) -> Dict[str, Any]:
        """
        Build one scoped ChromaDB metadata payload under a bounded key budget.

        Parameters
        ----------
        prefix : str
            Prefix used to isolate metadata scopes.

        payload : Dict[str, Any]
            Raw metadata payload associated with one scope.

        direct_keys : Sequence[str]
            Keys that remain individually queryable in ChromaDB.

        Returns
        -------
        Dict[str, Any]
            Flat scoped metadata that keeps important keys filterable and
            serializes the remaining payload into one compact extras field.
        """

        scoped_metadata: Dict[str, Any] = {}
        remaining_payload = dict(payload)

        for key in direct_keys:
            if key not in remaining_payload:
                continue

            normalized_value = self._normalize_metadata_value(remaining_payload.pop(key))
            if normalized_value is None:
                continue

            scoped_metadata[f"{prefix}_{self._normalize_metadata_key(key)}"] = normalized_value

        extras_payload = self._normalize_metadata_extras_payload(remaining_payload)
        if extras_payload:
            scoped_metadata[f"{prefix}_extras"] = json.dumps(
                extras_payload,
                ensure_ascii=False,
                sort_keys=True,
            )

        return scoped_metadata

    def _normalize_metadata_extras_payload(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize metadata fields that are not stored as direct ChromaDB keys.

        Parameters
        ----------
        payload : Dict[str, Any]
            Remaining metadata fields from one logical scope.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable mapping of preserved overflow metadata.
        """

        normalized_payload: Dict[str, Any] = {}

        for key, value in payload.items():
            normalized_key = self._normalize_metadata_key(key)
            if not normalized_key:
                continue

            normalized_value = self._normalize_metadata_json_value(value)
            if normalized_value is None:
                continue

            normalized_payload[normalized_key] = normalized_value

        return normalized_payload

    def _normalize_metadata_json_value(self, value: Any) -> Any:
        """
        Convert metadata values into JSON-friendly overflow payload values.

        Parameters
        ----------
        value : Any
            Raw metadata value.

        Returns
        -------
        Any
            JSON-serializable value, or `None` when it should be skipped.
        """

        if isinstance(value, bool):
            return value
        if isinstance(value, (str, int, float)):
            return value
        if value is None:
            return None

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, list):
            normalized_items: List[Any] = []

            for item in value:
                normalized_item = self._normalize_metadata_json_value(item)
                if normalized_item is None:
                    continue
                normalized_items.append(normalized_item)

            return normalized_items

        if isinstance(value, dict):
            normalized_mapping: Dict[str, Any] = {}

            for nested_key, nested_value in value.items():
                normalized_nested_key = self._normalize_metadata_key(nested_key)
                if not normalized_nested_key:
                    continue

                normalized_nested_value = self._normalize_metadata_json_value(
                    nested_value
                )
                if normalized_nested_value is None:
                    continue

                normalized_mapping[normalized_nested_key] = normalized_nested_value

            return normalized_mapping

        return str(value)

    def _flatten_metadata(
        self,
        prefix: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Flatten one nested metadata payload into ChromaDB-safe scalar values.

        Parameters
        ----------
        prefix : str
            Prefix used to avoid key collisions between metadata scopes.

        payload : Dict[str, Any]
            Metadata payload to flatten.

        Returns
        -------
        Dict[str, Any]
            Flat scalar mapping compatible with ChromaDB metadata fields.
        """

        flattened_payload: Dict[str, Any] = {}

        for key, value in payload.items():
            normalized_key = self._normalize_metadata_key(key)
            if not normalized_key:
                continue

            target_key = f"{prefix}_{normalized_key}"
            normalized_value = self._normalize_metadata_value(value)
            if normalized_value is None:
                continue

            flattened_payload[target_key] = normalized_value

        return flattened_payload

    def _normalize_metadata_key(self, key: Any) -> str:
        """
        Normalize one metadata key before it is persisted in ChromaDB.

        Parameters
        ----------
        key : Any
            Raw metadata key candidate.

        Returns
        -------
        str
            Lowercase normalized key, or an empty string when invalid.
        """

        if not isinstance(key, str):
            return ""

        normalized_key = key.strip().lower()
        if not normalized_key:
            return ""

        return normalized_key.replace(" ", "_")

    def _normalize_metadata_value(self, value: Any) -> Any:
        """
        Convert metadata values into ChromaDB-compatible scalar representations.

        Parameters
        ----------
        value : Any
            Raw metadata value.

        Returns
        -------
        Any
            Scalar value safe to persist, or `None` when it should be skipped.
        """

        if isinstance(value, bool):
            return value
        if isinstance(value, (str, int, float)):
            return value
        if value is None:
            return None

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, list):
            normalized_items = [
                item for item in value if isinstance(item, (str, int, float, bool))
            ]
            return json.dumps(normalized_items, ensure_ascii=False)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)

        return str(value)

    def _resolve_run_directory(
        self,
        run_id: str,
        embedding_records: Sequence[EmbeddingVectorRecord] = (),
        manifest: Optional[EmbeddingRunManifest] = None,
        replace_existing_strategy_output: bool = False,
    ) -> Path:
        """
        Build the directory used to store one embedding run local artifacts.

        Parameters
        ----------
        run_id : str
            Stable run identifier used to build the output path.

        embedding_records : Sequence[EmbeddingVectorRecord]
            Records used to infer the active strategy when available.

        manifest : Optional[EmbeddingRunManifest]
            Manifest used to infer the active strategy when available.

        replace_existing_strategy_output : bool
            When `True`, remove previous local artifacts for the same strategy
            before creating the new run directory.

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

        if replace_existing_strategy_output:
            self._clear_strategy_output_root(strategy_name)

        run_directory = self.output_root / strategy_name / normalized_run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory

    def _clear_strategy_output_root(self, strategy_name: str) -> None:
        """
        Remove any previously persisted local artifacts for one strategy.

        Parameters
        ----------
        strategy_name : str
            Normalized strategy name whose existing artifacts must be removed.
        """

        strategy_output_root = self.output_root / strategy_name
        if not strategy_output_root.exists():
            return

        if strategy_output_root.is_dir():
            shutil.rmtree(strategy_output_root)
            return

        strategy_output_root.unlink()

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

    def _build_storage_summary_payload(
        self,
        embedding_records: Sequence[EmbeddingVectorRecord],
        manifest: EmbeddingRunManifest,
        strategy_name: str,
        deleted_record_count: int,
        upserted_record_count: int,
        run_directory: Path,
    ) -> Dict[str, Any]:
        """
        Build the local audit payload describing one ChromaDB persistence run.

        Parameters
        ----------
        embedding_records : Sequence[EmbeddingVectorRecord]
            Records persisted for the current run.

        manifest : EmbeddingRunManifest
            Manifest describing the current embedding run.

        strategy_name : str
            Active strategy name associated with the run.

        deleted_record_count : int
            Number of older strategy records deleted before upsert.

        upserted_record_count : int
            Number of records upserted into ChromaDB.

        run_directory : Path
            Local artifact directory for the run.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable storage audit payload.
        """

        return {
            "run_id": manifest.run_id,
            "strategy": strategy_name,
            "storage_backend": "chromadb",
            "storage_scope": self._normalize_chromadb_mode(),
            "storage_collection": self.chromadb_collection_name,
            "storage_location": self._build_storage_location(),
            "output_path": str(run_directory),
            "deleted_record_count": deleted_record_count,
            "upserted_record_count": upserted_record_count,
            "record_count": len(embedding_records),
            "record_ids": [
                self._resolve_storage_record_id(record)
                for record in embedding_records
            ],
        }

    def _manifest_to_dict(
        self,
        manifest: EmbeddingRunManifest,
        run_directory: Path,
        strategy_name: str,
        upserted_record_count: int,
    ) -> Dict[str, Any]:
        """
        Convert one run manifest into the persisted JSON structure.

        Parameters
        ----------
        manifest : EmbeddingRunManifest
            Manifest to serialize.

        run_directory : Path
            Run directory created for this manifest.

        strategy_name : str
            Active strategy associated with the manifest.

        upserted_record_count : int
            Number of records successfully upserted into ChromaDB.

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
            "strategy": strategy_name,
            "timestamp": manifest.generated_at_utc,
            "generated_at_utc": manifest.generated_at_utc,
            "input_root": manifest.input_root,
            "output_root": manifest.output_root,
            "auxiliary_output_root": str(run_directory),
            "input_text_field": manifest.input_text_field,
            "batch_size": manifest.batch_size,
            "record_count": manifest.record_count,
            "persisted_record_count": upserted_record_count,
            "source_files": list(manifest.source_files),
            "source_path": source_path,
            "output_path": str(run_directory),
            "storage_backend": "chromadb",
            "storage_scope": self._normalize_chromadb_mode(),
            "storage_collection": self.chromadb_collection_name,
            "storage_location": self._build_storage_location(),
            "metadata": manifest.metadata,
        }

    def _build_storage_location(self) -> str:
        """
        Build a human-readable storage location string for audit artifacts.

        Returns
        -------
        str
            Storage location summary suitable for manifests and audit files.
        """

        chromadb_mode = self._normalize_chromadb_mode()
        if chromadb_mode == "persistent":
            return str(self.chromadb_persist_directory)

        return (
            "tenant="
            f"{self.chromadb_cloud_tenant};database={self.chromadb_cloud_database};"
            f"host={self.chromadb_cloud_host};port={self.chromadb_cloud_port}"
        )

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


LocalEmbeddingStorage = ChromaEmbeddingStorage
