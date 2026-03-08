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
      hierarchy of articles and/or sections
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

        # -------------------------------------------------------------
        # Heuristic selection.
        #
        # We choose structure-first when:
        # - the parser found a reasonable number of articles, and
        # - there is at least some additional hierarchy (sections,
        #   lettered items, annexes, or chapters)
        #
        # Otherwise we use article-smart, which is more forgiving.
        # -------------------------------------------------------------
        has_good_article_backbone = article_count >= 3
        has_additional_structure = (
            (section_count >= 2)
            or (lettered_count >= 3)
            or (annex_count >= 1)
            or (chapter_count >= 1)
        )

        if has_good_article_backbone and has_additional_structure:
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