from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from Chunking.chunking.models import Chunk, StructuralNode


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
    """

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
        output_path.write_text(
            json.dumps(
                [chunk.to_dict() for chunk in chunks],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_structure(self, root: StructuralNode, output_path: Path) -> None:
        """
        Export the parsed structural tree into a readable JSON file.

        Parameters
        ----------
        root : StructuralNode
            Root node of the parsed structure tree.
        output_path : Path
            Destination JSON path.
        """
        output_path.write_text(
            json.dumps(
                self._node_to_dict(root),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _node_to_dict(self, node: StructuralNode) -> dict:
        """
        Convert one structural node into a JSON-serializable dictionary.

        Why this helper exists
        ----------------------
        We intentionally control the output shape here instead of blindly
        serializing everything elsewhere, because structure exports are used
        heavily for debugging and inspection.

        Parameters
        ----------
        node : StructuralNode
            Node to serialize.

        Returns
        -------
        dict
            Recursive dictionary representation of the node.
        """
        return {
            "node_type": node.node_type,
            "label": node.label,
            "title": node.title,
            "text": node.text,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "node_id": node.node_id,
            "parent_node_id": node.parent_node_id,
            "hierarchy_path": node.hierarchy_path,
            "metadata": node.metadata,
            "children": [self._node_to_dict(child) for child in node.children],
        }