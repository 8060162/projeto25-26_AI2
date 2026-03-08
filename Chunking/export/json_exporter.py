from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from Chunking.chunking.models import Chunk, StructuralNode


class JsonExporter:
    """Export structured information into readable JSON files."""

    def write_chunks(self, chunks: Iterable[Chunk], output_path: Path) -> None:
        output_path.write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_structure(self, root: StructuralNode, output_path: Path) -> None:
        output_path.write_text(
            json.dumps(self._node_to_dict(root), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _node_to_dict(self, node: StructuralNode) -> dict:
        return {
            "node_type": node.node_type,
            "label": node.label,
            "title": node.title,
            "text": node.text,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "metadata": node.metadata,
            "children": [self._node_to_dict(child) for child in node.children],
        }
