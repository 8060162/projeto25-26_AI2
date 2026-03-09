from __future__ import annotations

from typing import List

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_article_smart import ArticleSmartChunkingStrategy
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.chunking.strategy_structure_first import StructureFirstChunkingStrategy


class HybridChunkingStrategy(BaseChunkingStrategy):
    """
    Hybrid strategy.

    Decision principle:
    - use structure-first when the parser produced a sufficiently rich
      hierarchy inside the regulation body
    - fall back to article-smart when the structure is weaker

    Why this exists:
    - some documents are highly regular and benefit from strict structure-first
    - others are only partially parsed and need a more resilient strategy
    """

    name = "hybrid"

    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        article_count = self._count_node_type(root, "ARTICLE")
        section_count = self._count_node_type(root, "SECTION")
        lettered_count = self._count_node_type(root, "LETTERED_ITEM")
        annex_count = self._count_node_type(root, "ANNEX")
        chapter_count = self._count_node_type(root, "CHAPTER")

        # Prefer structure-first only when there is evidence that the parser
        # captured meaningful hierarchy inside articles, not merely container
        # nodes like ANNEX or CHAPTER.
        has_good_article_backbone = article_count >= 3
        has_internal_structure = (section_count >= 2) or (lettered_count >= 3)
        has_container_structure = (annex_count >= 1) or (chapter_count >= 1)

        use_structure_first = (
            has_good_article_backbone
            and (
                has_internal_structure
                or (has_container_structure and article_count >= 6)
            )
        )

        if use_structure_first:
            return StructureFirstChunkingStrategy(self.settings).build_chunks(
                document_metadata,
                root,
            )

        return ArticleSmartChunkingStrategy(self.settings).build_chunks(
            document_metadata,
            root,
        )

    def _count_node_type(self, node: StructuralNode, node_type: str) -> int:
        """
        Recursively count nodes of a given type.
        """
        count = 1 if node.node_type == node_type else 0
        for child in node.children:
            count += self._count_node_type(child, node_type)
        return count