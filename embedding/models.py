from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the manifest into a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            Standard Python dictionary representation of the manifest.
        """

        return asdict(self)
