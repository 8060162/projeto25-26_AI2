from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class PageText:
    """
    Raw or normalized page-level text.

    We keep the page number attached so that later stages can propagate
    page_start and page_end into structural nodes and chunks.
    """

    page_number: int
    text: str


@dataclass(slots=True)
class DocumentMetadata:
    """
    High-level metadata extracted from the file system and optionally enriched
    later by parsing or custom document classification logic.
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

    Examples:
    - PREAMBLE
    - ANNEX
    - CHAPTER
    - ARTICLE
    - SECTION

    Important principle:
    The node text should contain only the textual content that belongs to
    that node, while contextual information (article number, parent label,
    document part, etc.) should live in metadata whenever possible.
    """

    node_type: str
    label: str
    title: str = ""
    text: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["StructuralNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the node recursively using dataclass conversion.
        """
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    """
    Final chunk object stored as JSON and rendered into the inspection DOCX.

    The chunk text should be as clean and semantically coherent as possible.
    The metadata should remain rich enough to support:
    - traceability
    - filtering
    - document navigation
    - future citation features
    """

    chunk_id: str
    doc_id: str
    strategy: str
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)