from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class PageText:
    """
    Raw or normalized page-level text.

    Why this model exists
    ---------------------
    The pipeline starts with page-oriented extraction and normalization.

    Keeping the page number attached to the text is important because later
    stages need page provenance to populate:
    - page_start
    - page_end
    - chunk traceability
    - debugging artifacts
    """

    page_number: int
    text: str


@dataclass(slots=True)
class DocumentMetadata:
    """
    High-level document metadata.

    Typical responsibilities
    ------------------------
    This object stores document-level information that is known early in the
    pipeline, usually from the file system or from lightweight classification.

    Examples:
    - stable document identifier
    - original file name
    - document title
    - source path
    - optional extra metadata such as institution, year, status, etc.

    Design note
    -----------
    The flexible `metadata` dictionary is intentionally kept so the pipeline
    can evolve without forcing constant schema changes.
    """

    doc_id: str
    file_name: str
    title: str
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuralNode:
    """
    A logical node in the parsed document structure.

    Typical node examples
    ---------------------
    - FRONT_MATTER
    - PREAMBLE
    - ANNEX
    - CHAPTER
    - SECTION_CONTAINER
    - ARTICLE
    - SECTION
    - LETTERED_ITEM

    Design principles
    -----------------
    1. Node text should contain only the text that belongs to that node.
    2. Contextual identity should live mostly in metadata and structural fields.
    3. The node should remain serializable and easy to inspect in JSON exports.

    New structural support fields
    -----------------------------
    - node_id:
        Optional stable identifier for the node itself.
        This is useful for future graph navigation, debugging, and linking.

    - parent_node_id:
        Optional identifier of the parent node.
        This is useful when consumers want explicit upward navigation without
        having to reconstruct it from the full tree.

    - hierarchy_path:
        Ordered path of structural labels from root to the node.
        Example:
            ["DOCUMENT", "CHAPTER:CAP_I", "ARTICLE:ART_5", "SECTION:2"]

        This improves traceability and future filtering.
    """

    node_type: str
    label: str
    title: str = ""
    text: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    node_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    hierarchy_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["StructuralNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the node recursively into a plain dictionary.

        Why this helper exists
        ----------------------
        JSON export and debugging become much easier when the full node tree can
        be converted into standard Python dictionaries.

        Returns
        -------
        Dict[str, Any]
            A recursive dataclass-based dictionary representation of the node.
        """
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    """
    Final chunk object stored as JSON and rendered into the inspection DOCX.

    Design goals
    ------------
    A chunk should be:
    - semantically coherent
    - easy to trace back to the source
    - easy to inspect manually
    - rich enough for downstream retrieval and filtering

    Core text fields
    ----------------
    - text:
        Clean chunk text intended for inspection and normal downstream usage.

    - text_for_embedding:
        Optional enriched chunk text intended specifically for embeddings.
        This allows the system to keep the visible chunk text clean while still
        including lightweight structural context when needed, for example:
            "Artigo 5.º - Revisão de prova\\n\\n<chunk body>"

        When this field is empty, downstream code may safely fall back to `text`.

    Structural traceability fields
    ------------------------------
    - source_node_type:
        The structural type from which the chunk was primarily created.
        Examples:
        - ARTICLE
        - SECTION
        - LETTERED_ITEM
        - PREAMBLE
        - FRONT_MATTER

    - source_node_label:
        The structural label from which the chunk was primarily created.
        Examples:
        - ART_5
        - 2
        - b
        - PREAMBLE

    - hierarchy_path:
        Ordered structural path associated with the chunk.
        This enables future filtering, search boosting, and debugging.

    Chunk navigation fields
    -----------------------
    - prev_chunk_id / next_chunk_id:
        Optional neighbor links for traversal and future chunk expansion.

    Quality / explanation fields
    ----------------------------
    - chunk_reason:
        A short explanation of why the chunk exists in its current shape.
        Examples:
        - "direct_article"
        - "grouped_sections"
        - "grouped_lettered_items"
        - "fallback_paragraph_split"
        - "preamble_group"

    - char_count:
        Character count of the visible chunk text.
        Useful for diagnostics and quick quality checks.
    """

    chunk_id: str
    doc_id: str
    strategy: str
    text: str
    page_start: Optional[int]
    page_end: Optional[int]

    # Optional enriched text used for embeddings or retrieval-specific flows.
    text_for_embedding: str = ""

    # Optional explicit structural provenance.
    source_node_type: str = ""
    source_node_label: str = ""

    # Optional structural path from document root to the source region.
    hierarchy_path: List[str] = field(default_factory=list)

    # Optional explanation of how / why this chunk was produced.
    chunk_reason: str = ""

    # Lightweight diagnostics.
    char_count: int = 0

    # Optional neighbor links for future traversal or chunk expansion.
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # Flexible extension metadata for domain-specific enrichment.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the chunk into a plain dictionary.

        Why this helper exists
        ----------------------
        Exporters and diagnostics are simpler when chunks can be converted to
        regular dictionaries in a consistent way.

        Returns
        -------
        Dict[str, Any]
            A dataclass-based dictionary representation of the chunk.
        """
        return asdict(self)