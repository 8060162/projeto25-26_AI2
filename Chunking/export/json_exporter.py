from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode


class JsonExporter:
    """
    Export structured information into readable JSON files.

    Why this exporter exists
    ------------------------
    JSON artifacts are useful for:
    - debugging
    - downstream integration
    - regression comparison
    - manual inspection in editors and scripts

    Important architectural distinction
    -----------------------------------
    This exporter now supports two different JSON output families:

    1. Generic debugging exports
       These reflect the internal tree and are useful for inspection.

    2. Canonical master-dictionary-style exports
       These are closer to the target JSON structure used by the project
       as a domain-facing structured representation of regulations.

    Design goals
    ------------
    - keep output deterministic and readable
    - preserve rich structural traceability
    - remain robust while the project evolves
    - separate internal debug shape from canonical domain shape
    """

    # ------------------------------------------------------------------
    # Public chunk export
    # ------------------------------------------------------------------

    def write_chunks(self, chunks: Iterable[Chunk], output_path: Path) -> None:
        """
        Export chunks into a readable JSON file.

        Parameters
        ----------
        chunks : Iterable[Chunk]
            Final chunk objects to export.

        output_path : Path
            Destination JSON path.
        """
        chunk_payload = [
            self._chunk_to_dict(chunk)
            for chunk in chunks
        ]

        self._write_json(
            payload=chunk_payload,
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # Public generic structure export
    # ------------------------------------------------------------------

    def write_structure(self, root: StructuralNode, output_path: Path) -> None:
        """
        Export the parsed structural tree into a readable generic JSON file.

        Why this method exists
        ----------------------
        This is primarily a debugging/export-of-internals method.
        It reflects the internal `StructuralNode` tree directly and is useful
        for inspection and regression comparison.

        Parameters
        ----------
        root : StructuralNode
            Root node of the parsed structure tree.

        output_path : Path
            Destination JSON path.
        """
        self._write_json(
            payload=self._node_to_dict(root),
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # Public canonical/master-style export
    # ------------------------------------------------------------------

    def write_master_dictionary(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
        output_path: Path,
    ) -> None:
        """
        Export the parsed structure into a canonical master-dictionary-style JSON.

        Why this method matters
        -----------------------
        The project does not ultimately want only a generic debug tree.
        It wants a domain-facing JSON representation similar to the target
        master dictionary, where the document is organized into:
        - document metadata
        - structured content tree
        - stable semantic keys such as PREAMBULO, CAP_I, ART_1, etc.

        Output shape
        ------------
        The exported JSON follows this general pattern:

            {
              "<DOC_ID>": {
                "doc_id": "<DOC_ID>",
                "metadata": {...},
                "estrutura": {...}
              }
            }

        Parameters
        ----------
        document_metadata : DocumentMetadata
            High-level metadata for the source document.

        root : StructuralNode
            Root structural tree produced by the parser.

        output_path : Path
            Destination JSON path.
        """
        master_payload = {
            document_metadata.doc_id: {
                "doc_id": document_metadata.doc_id,
                "metadata": self._build_master_metadata(document_metadata),
                "estrutura": self._build_master_structure(root),
            }
        }

        self._write_json(
            payload=master_payload,
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # Generic serialization helpers
    # ------------------------------------------------------------------

    def _write_json(self, payload: Any, output_path: Path) -> None:
        """
        Write JSON payload to disk with stable formatting.

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

    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Convert one chunk into a JSON-serializable dictionary.

        Why this helper exists
        ----------------------
        Explicit serialization keeps the exported shape stable and readable.
        It also makes the exporter more resilient while the internal dataclasses
        continue evolving.

        Parameters
        ----------
        chunk : Chunk
            Chunk to serialize.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable chunk payload.
        """
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "strategy": chunk.strategy,
            "text": chunk.text,
            "text_for_embedding": getattr(chunk, "text_for_embedding", ""),
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "source_node_type": getattr(chunk, "source_node_type", ""),
            "source_node_label": getattr(chunk, "source_node_label", ""),
            "hierarchy_path": list(getattr(chunk, "hierarchy_path", []) or []),
            "chunk_reason": getattr(chunk, "chunk_reason", ""),
            "char_count": getattr(chunk, "char_count", len(chunk.text or "")),
            "prev_chunk_id": getattr(chunk, "prev_chunk_id", None),
            "next_chunk_id": getattr(chunk, "next_chunk_id", None),
            "metadata": getattr(chunk, "metadata", {}) or {},
        }

    def _node_to_dict(self, node: StructuralNode) -> Dict[str, Any]:
        """
        Convert one structural node into a JSON-serializable dictionary.

        Why this helper exists
        ----------------------
        We intentionally control the output shape instead of blindly dumping
        dataclasses, because structure exports are used heavily for debugging
        and inspection.

        Parameters
        ----------
        node : StructuralNode
            Node to serialize.

        Returns
        -------
        Dict[str, Any]
            Recursive dictionary representation of the node.
        """
        return {
            "node_type": node.node_type,
            "label": node.label,
            "title": node.title,
            "text": node.text,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "node_id": getattr(node, "node_id", None),
            "parent_node_id": getattr(node, "parent_node_id", None),
            "hierarchy_path": list(getattr(node, "hierarchy_path", []) or []),
            "metadata": getattr(node, "metadata", {}) or {},
            "children": [self._node_to_dict(child) for child in node.children],
        }

    # ------------------------------------------------------------------
    # Canonical/master-style mapping
    # ------------------------------------------------------------------

    def _build_master_metadata(self, document_metadata: DocumentMetadata) -> Dict[str, Any]:
        """
        Build the document-level metadata payload for the canonical export.

        Why this helper exists
        ----------------------
        The master-style JSON should expose a stable document-level metadata
        object, distinct from the internal parser tree.

        Parameters
        ----------
        document_metadata : DocumentMetadata
            Source document metadata.

        Returns
        -------
        Dict[str, Any]
            Canonical metadata dictionary.
        """
        return {
            "file_name": document_metadata.file_name,
            "title": document_metadata.title,
            "source_path": document_metadata.source_path,
            **(document_metadata.metadata or {}),
        }

    def _build_master_structure(self, root: StructuralNode) -> Dict[str, Any]:
        """
        Convert the generic StructuralNode tree into the canonical structure map.

        Important design choice
        -----------------------
        The master dictionary format is key-oriented rather than purely list-based.
        Therefore this method converts children into named structure entries such as:
        - PREAMBULO
        - CAP_I
        - ART_1
        - NUM_1
        - AL_a

        Parameters
        ----------
        root : StructuralNode
            Root DOCUMENT node.

        Returns
        -------
        Dict[str, Any]
            Canonical structure payload.
        """
        structure: Dict[str, Any] = {}

        for child in root.children:
            key = self._master_key_for_node(child)
            structure[key] = self._master_node_to_dict(child)

        return structure

    def _master_node_to_dict(self, node: StructuralNode) -> Dict[str, Any]:
        """
        Convert one structural node into canonical master-style JSON.

        Mapping philosophy
        ------------------
        The resulting payload should be domain-facing and easy to inspect.
        It should keep:
        - title, when present
        - content, when present
        - page information
        - useful metadata
        - recursively mapped children

        Parameters
        ----------
        node : StructuralNode
            Node to convert.

        Returns
        -------
        Dict[str, Any]
            Canonical node payload.
        """
        payload: Dict[str, Any] = {}

        if node.title:
            payload["titulo"] = node.title

        if node.text:
            payload["conteudo"] = node.text

        # Preserve page information in a practical way.
        if node.page_start is not None and node.page_end is not None:
            if node.page_start == node.page_end:
                payload["pagina"] = node.page_start
            else:
                payload["pagina_inicio"] = node.page_start
                payload["pagina_fim"] = node.page_end
        elif node.page_start is not None:
            payload["pagina"] = node.page_start

        # Preserve useful node metadata without exposing too much internal noise.
        node_metadata = self._filtered_node_metadata(node)
        if node_metadata:
            payload["metadata"] = node_metadata

        # Recursively map children.
        for child in node.children:
            child_key = self._master_key_for_node(child)
            payload[child_key] = self._master_node_to_dict(child)

        return payload

    def _filtered_node_metadata(self, node: StructuralNode) -> Dict[str, Any]:
        """
        Filter node metadata for canonical export.

        Why filtering exists
        --------------------
        Internal parser metadata can become verbose and implementation-specific.
        The canonical export should preserve useful metadata, but avoid turning
        the master JSON into a dump of parser internals.

        Parameters
        ----------
        node : StructuralNode
            Node whose metadata should be filtered.

        Returns
        -------
        Dict[str, Any]
            Filtered metadata dictionary.
        """
        raw_metadata = getattr(node, "metadata", {}) or {}
        allowed_keys = {
            "document_part",
            "chapter_number",
            "chapter_title",
            "container_type",
            "container_number",
            "container_title",
            "article_number",
            "article_title",
            "annex_label",
            "annex_title",
            "raw_section_label",
            "parent_type",
            "parent_label",
            "parent_title",
            "article_label",
        }

        filtered = {
            key: value
            for key, value in raw_metadata.items()
            if key in allowed_keys and value not in ("", None)
        }

        return filtered

    def _master_key_for_node(self, node: StructuralNode) -> str:
        """
        Build the canonical key for a structural node.

        Examples
        --------
        - PREAMBLE      -> PREAMBULO
        - FRONT_MATTER  -> FRONT_MATTER
        - CHAPTER CAP_I -> CAP_I
        - ARTICLE ART_1 -> ART_1
        - SECTION 1     -> NUM_1
        - SECTION 2.1   -> NUM_2_1
        - LETTERED_ITEM a -> AL_a

        Parameters
        ----------
        node : StructuralNode
            Structural node.

        Returns
        -------
        str
            Canonical dictionary key.
        """
        if node.node_type == "PREAMBLE":
            return "PREAMBULO"

        if node.node_type == "FRONT_MATTER":
            return "FRONT_MATTER"

        if node.node_type == "CHAPTER":
            return node.label

        if node.node_type == "ANNEX":
            return self._sanitize_structure_key(node.label)

        if node.node_type == "SECTION_CONTAINER":
            return self._sanitize_structure_key(node.label)

        if node.node_type == "ARTICLE":
            return node.label

        if node.node_type == "SECTION":
            normalized = self._sanitize_structure_key(node.label)
            return f"NUM_{normalized}"

        if node.node_type == "LETTERED_ITEM":
            normalized = self._sanitize_structure_key(node.label.lower())
            return f"AL_{normalized}"

        return self._sanitize_structure_key(node.label or node.node_type)

    def _sanitize_structure_key(self, value: str) -> str:
        """
        Sanitize a structural key so it is safe and stable in JSON objects.

        Why this helper exists
        ----------------------
        Some structural labels may contain spaces, punctuation, or symbols
        that are inconvenient as dictionary keys. This helper converts them
        into a stable underscore-based form.

        Parameters
        ----------
        value : str
            Raw key-like value.

        Returns
        -------
        str
            Sanitized key.
        """
        sanitized = value.strip()
        sanitized = re.sub(r"[^\w]+", "_", sanitized, flags=re.UNICODE)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("_")
        return sanitized