from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _normalize_mapping(value: Any) -> Dict[str, Any]:
    """
    Normalize one optional metadata payload into a detached dictionary.

    Parameters
    ----------
    value : Any
        Candidate metadata payload.

    Returns
    -------
    Dict[str, Any]
        Detached dictionary when the payload is mapping-like, otherwise an
        empty dictionary.
    """

    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_string_list(value: Any) -> List[str]:
    """
    Normalize one optional list payload into a clean string list.

    Parameters
    ----------
    value : Any
        Candidate list payload.

    Returns
    -------
    List[str]
        Ordered non-empty string values.
    """

    if not isinstance(value, list):
        return []

    normalized_values: List[str] = []

    for item in value:
        if not isinstance(item, str):
            continue

        normalized_item = item.strip()
        if normalized_item:
            normalized_values.append(normalized_item)

    return normalized_values


@dataclass(slots=True)
class EmbeddingInputRecord:
    """
    Internal input model used by the embedding pipeline.

    Design goals
    ------------
    This model keeps the embedding stage explicit and traceable by storing:
    - the stable chunk identity
    - the source document identity
    - the text selected for embedding
    - flexible structural metadata preserved from chunking
    """

    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    hierarchy_path: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    record_id: str = ""
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    document_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize derived identity and metadata fields after initialization.
        """

        self.metadata = _normalize_mapping(self.metadata)
        self.hierarchy_path = _normalize_string_list(self.hierarchy_path)
        self.record_id = self.record_id.strip() or self.chunk_id
        self.source_file = self.source_file.strip()
        self.chunk_metadata = self._resolve_chunk_metadata()
        self.document_metadata = self._resolve_document_metadata()

    def _resolve_chunk_metadata(self) -> Dict[str, Any]:
        """
        Build the chunk-scoped metadata payload stored with the input record.

        Returns
        -------
        Dict[str, Any]
            Detached metadata describing the chunk itself.
        """

        if self.chunk_metadata:
            return _normalize_mapping(self.chunk_metadata)

        resolved_chunk_metadata: Dict[str, Any] = {}

        if self.hierarchy_path:
            resolved_chunk_metadata["hierarchy_path"] = list(self.hierarchy_path)
        if self.page_start is not None:
            resolved_chunk_metadata["page_start"] = self.page_start
        if self.page_end is not None:
            resolved_chunk_metadata["page_end"] = self.page_end

        for metadata_key in (
            "strategy",
            "chunk_index",
            "chunk_kind",
            "chunk_sequence",
            "section_path",
            "section_title",
            "article_number",
            "article_title",
        ):
            metadata_value = self.metadata.get(metadata_key)
            if metadata_value is not None:
                resolved_chunk_metadata[metadata_key] = metadata_value

        return resolved_chunk_metadata

    def _resolve_document_metadata(self) -> Dict[str, Any]:
        """
        Build the document-scoped metadata payload stored with the input record.

        Returns
        -------
        Dict[str, Any]
            Detached metadata describing the source document.
        """

        if self.document_metadata:
            return _normalize_mapping(self.document_metadata)

        resolved_document_metadata: Dict[str, Any] = {"doc_id": self.doc_id}

        if self.source_file:
            resolved_document_metadata["source_file"] = self.source_file

        for metadata_key in (
            "document_id",
            "document_title",
            "document_type",
            "source_document_path",
            "source_file_name",
            "language",
            "jurisdiction",
        ):
            metadata_value = self.metadata.get(metadata_key)
            if metadata_value is not None:
                resolved_document_metadata[metadata_key] = metadata_value

        return resolved_document_metadata

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the record into a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            Standard Python dictionary representation of the input record.
        """

        return asdict(self)


@dataclass(slots=True)
class EmbeddingVectorRecord:
    """
    Internal output model used to store embedding results.

    Design goals
    ------------
    This record preserves the source chunk identity together with:
    - the generated vector
    - the provider/model used to generate it
    - carry-through metadata needed for filtering or inspection
    """

    chunk_id: str
    doc_id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    provider: str = ""
    source_file: str = ""
    text: str = ""
    record_id: str = ""
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    storage_record_id: str = ""
    storage_backend: str = ""
    storage_collection: str = ""

    def __post_init__(self) -> None:
        """
        Normalize derived identity and metadata fields after initialization.
        """

        self.metadata = _normalize_mapping(self.metadata)
        self.record_id = self.record_id.strip() or self.chunk_id
        self.storage_record_id = self.storage_record_id.strip() or self.record_id
        self.source_file = self.source_file.strip()
        self.provider = self.provider.strip()
        self.model = self.model.strip()
        self.storage_backend = self.storage_backend.strip()
        self.storage_collection = self.storage_collection.strip()
        self.chunk_metadata = self._resolve_chunk_metadata()
        self.document_metadata = self._resolve_document_metadata()

    def _resolve_chunk_metadata(self) -> Dict[str, Any]:
        """
        Build the chunk-scoped metadata payload stored with the vector record.

        Returns
        -------
        Dict[str, Any]
            Detached metadata describing the embedded chunk.
        """

        if self.chunk_metadata:
            return _normalize_mapping(self.chunk_metadata)

        resolved_chunk_metadata: Dict[str, Any] = {"chunk_id": self.chunk_id}

        for metadata_key in (
            "strategy",
            "hierarchy_path",
            "page_start",
            "page_end",
            "chunk_index",
            "chunk_kind",
            "chunk_sequence",
            "section_path",
            "section_title",
            "article_number",
            "article_title",
        ):
            metadata_value = self.metadata.get(metadata_key)
            if metadata_value is not None:
                resolved_chunk_metadata[metadata_key] = metadata_value

        return resolved_chunk_metadata

    def _resolve_document_metadata(self) -> Dict[str, Any]:
        """
        Build the document-scoped metadata payload stored with the vector record.

        Returns
        -------
        Dict[str, Any]
            Detached metadata describing the source document.
        """

        if self.document_metadata:
            return _normalize_mapping(self.document_metadata)

        resolved_document_metadata: Dict[str, Any] = {"doc_id": self.doc_id}

        if self.source_file:
            resolved_document_metadata["source_file"] = self.source_file

        for metadata_key in (
            "document_id",
            "document_title",
            "document_type",
            "source_document_path",
            "source_file_name",
            "language",
            "jurisdiction",
        ):
            metadata_value = self.metadata.get(metadata_key)
            if metadata_value is not None:
                resolved_document_metadata[metadata_key] = metadata_value

        return resolved_document_metadata

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the vector record into a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            Standard Python dictionary representation of the vector record.
        """

        return asdict(self)


@dataclass(slots=True)
class EmbeddingRunManifest:
    """
    Internal manifest model describing one embedding execution.

    Design goals
    ------------
    The manifest captures the configuration and output summary needed to:
    - audit what was generated
    - inspect where the data came from
    - support later visualization or reload flows
    """

    run_id: str
    provider: str
    model: str
    input_root: str
    output_root: str
    input_text_field: str
    batch_size: int
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    record_count: int = 0
    source_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    storage_backend: str = ""
    storage_scope: str = ""
    storage_collection: str = ""
    storage_location: str = ""
    auxiliary_output_root: str = ""

    def __post_init__(self) -> None:
        """
        Normalize run-level storage fields after dataclass initialization.
        """

        self.provider = self.provider.strip()
        self.model = self.model.strip()
        self.input_root = self.input_root.strip()
        self.output_root = self.output_root.strip()
        self.input_text_field = self.input_text_field.strip()
        self.source_files = _normalize_string_list(self.source_files)
        self.metadata = _normalize_mapping(self.metadata)
        self.storage_backend = self.storage_backend.strip()
        self.storage_scope = self.storage_scope.strip()
        self.storage_collection = self.storage_collection.strip()
        self.storage_location = self.storage_location.strip()
        self.auxiliary_output_root = (
            self.auxiliary_output_root.strip() or self.output_root
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the manifest into a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            Standard Python dictionary representation of the manifest.
        """

        return asdict(self)
