from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# Early extraction models
# ============================================================================
#
# These models represent the earliest stages of the pipeline.
# They are intentionally focused on extraction fidelity and traceability,
# not yet on legal interpretation or chunk generation.
#
# Architectural note
# ------------------
# The project is evolving from:
#     PDF -> plain text
#
# toward:
#     PDF -> structured intermediate extraction -> parsed JSON tree -> chunks
#
# Therefore these extraction models are essential to preserve enough structure
# for the parser to build the target master dictionary style output.
# ============================================================================


@dataclass(slots=True)
class PageText:
    """
    Legacy lightweight page-level text model.

    Why this model still exists
    ---------------------------
    Older parts of the pipeline may still expect a minimal representation
    consisting only of:
    - page number
    - page text

    This model is intentionally kept for backward compatibility while the
    pipeline transitions toward richer intermediate extraction objects.

    Important note
    --------------
    New extraction/parsing flows should prefer `ExtractedDocument` and related
    models instead of relying only on PageText.
    """

    page_number: int
    text: str


@dataclass(slots=True)
class BoundingBox:
    """
    Lightweight bounding box container.

    Coordinates
    -----------
    The values follow the common PDF coordinate convention:
    (x0, y0, x1, y1)

    Why this model matters
    ----------------------
    Bounding boxes are useful for:
    - header/footer heuristics
    - title detection
    - block grouping
    - debugging extraction issues
    """

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(slots=True)
class ExtractedLine:
    """
    One extracted line reconstructed from a PDF page.

    Why line-level data matters
    ---------------------------
    In legal/regulatory PDFs, many structural signals appear on isolated lines:
    - "CAPÍTULO I"
    - "Artigo 3.º"
    - article titles
    - annex labels
    - index entries

    Preserving lines makes downstream structure parsing much more robust.
    """

    text: str
    bbox: Optional[BoundingBox] = None
    block_index: Optional[int] = None
    line_index: Optional[int] = None


@dataclass(slots=True)
class ExtractedBlock:
    """
    One extracted text block from a page.

    Why preserve blocks?
    --------------------
    Blocks help retain a useful compromise between full layout and plain text.
    They are especially valuable when trying to distinguish:
    - title blocks
    - body text blocks
    - page header/footer regions
    - signature areas
    - index-like regions
    """

    block_index: int
    text: str
    bbox: Optional[BoundingBox] = None
    lines: List[ExtractedLine] = field(default_factory=list)
    source_mode: str = "unknown"


@dataclass(slots=True)
class PageExtractionCandidate:
    """
    One extraction candidate for a given page.

    Why this model exists
    ---------------------
    A single page may be extracted through several modes, for example:
    - dict
    - blocks
    - text
    - ocr

    Different modes can produce very different quality on the same PDF page.
    This model allows the pipeline to compare candidates explicitly before
    selecting the best one.
    """

    source_mode: str
    text: str
    quality_score: float
    blocks: List[ExtractedBlock] = field(default_factory=list)
    corruption_flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedPage:
    """
    Final selected extraction result for one page.

    This model stores:
    - selected page text
    - extraction mode used
    - quality score
    - structured blocks
    - quality/corruption flags

    Why this matters
    ----------------
    The parser should consume this richer representation instead of relying
    solely on flattened text.
    """

    page_number: int
    text: str
    selected_mode: str
    quality_score: float
    blocks: List[ExtractedBlock] = field(default_factory=list)
    corruption_flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedDocument:
    """
    Structured extraction result for a full PDF document.

    Important distinction
    ---------------------
    This is NOT the final parsed legal/regulatory JSON tree.

    It is an intermediate model that preserves enough extraction fidelity for
    the structure parser to later identify:
    - preamble
    - annexes
    - chapters
    - articles
    - internal numbering
    """

    source_path: str
    page_count: int
    pages: List[ExtractedPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """
        Return a concatenated text view of the extracted document.

        Why keep this?
        --------------
        Some debugging and legacy flows still benefit from a full-text view.
        However, it must be treated as a derived convenience view, not as the
        canonical extraction product.
        """
        return "\n\n".join(page.text for page in self.pages if page.text)


# ============================================================================
# Parsed structure models
# ============================================================================
#
# These models represent the stage after extraction and normalization, where
# the pipeline starts interpreting the document as a legal/regulatory structure.
# ============================================================================


@dataclass(slots=True)
class DocumentMetadata:
    """
    High-level document metadata.

    Typical responsibilities
    ------------------------
    This object stores document-level information known early or inferred
    during parsing.

    Examples
    --------
    - stable document identifier
    - original file name
    - document title
    - source path
    - institution
    - year
    - status
    - extra domain-specific metadata

    Design note
    -----------
    The flexible `metadata` dictionary is intentionally preserved so the schema
    can evolve without frequent breaking changes.
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
    - DOCUMENT
    - FRONT_MATTER
    - PREAMBLE
    - ANNEX
    - CHAPTER
    - ARTICLE
    - SECTION
    - LETTERED_ITEM

    Design principles
    -----------------
    1. Node text should contain only the text belonging to that node
    2. Structural identity should live mostly in metadata and hierarchy fields
    3. The node should remain easy to serialize and inspect
    4. The node should support tree navigation and diagnostic exports

    Structural identity fields
    --------------------------
    - node_id:
        Optional stable identifier for the node itself

    - parent_node_id:
        Optional identifier of the parent node

    - hierarchy_path:
        Ordered path of structural labels from root to the node
        Example:
            ["DOCUMENT:DOCUMENT", "CHAPTER:CAP_I", "ARTICLE:ART_5"]
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

        Returns
        -------
        Dict[str, Any]
            Standard Python dictionary representation of the full node tree.
        """
        return asdict(self)


# ============================================================================
# Chunking models
# ============================================================================
#
# These models represent the stage after the document structure is already
# understood and the system starts generating retrieval units.
#
# This is NOT the current focus of the project, but the models remain here
# because later stages still depend on them.
# ============================================================================


@dataclass(slots=True)
class Chunk:
    """
    Final chunk object stored as JSON and rendered into inspection artifacts.

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
        Clean chunk text intended for inspection and normal downstream usage

    - text_for_embedding:
        Optional enriched text intended specifically for embedding generation
        or retrieval-specific workflows

    Structural traceability fields
    ------------------------------
    - source_node_type:
        Structural type from which the chunk was primarily created

    - source_node_label:
        Structural label from which the chunk was primarily created

    - hierarchy_path:
        Ordered structural path associated with the chunk

    Chunk navigation fields
    -----------------------
    - prev_chunk_id / next_chunk_id:
        Optional neighbor links for traversal and future expansion

    Quality / explanation fields
    ----------------------------
    - chunk_reason:
        Short explanation of why the chunk exists in its current shape

    - char_count:
        Character count of the visible chunk text
    """

    chunk_id: str
    doc_id: str
    strategy: str
    text: str
    page_start: Optional[int]
    page_end: Optional[int]

    text_for_embedding: str = ""

    source_node_type: str = ""
    source_node_label: str = ""

    hierarchy_path: List[str] = field(default_factory=list)
    chunk_reason: str = ""
    char_count: int = 0

    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Apply lightweight consistency defaults after initialization.

        Current consistency rules
        -------------------------
        - If text_for_embedding is empty, default it to visible text
        - If char_count is zero but text exists, derive it from visible text
        """
        if not self.text_for_embedding and self.text:
            self.text_for_embedding = self.text

        if self.char_count == 0 and self.text:
            self.char_count = len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the chunk into a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            Standard Python dictionary representation of the chunk.
        """
        return asdict(self)