from __future__ import annotations

from typing import List

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_article_smart import ArticleSmartChunkingStrategy
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.chunking.strategy_structure_first import StructureFirstChunkingStrategy


class HybridChunkingStrategy(BaseChunkingStrategy):
    """
    Hybrid chunking strategy.

    Decision principle:
    - use the structure-first strategy when the parser produced a sufficiently
      rich and trustworthy hierarchy inside the regulation body
    - fall back to the article-smart strategy when the parsed structure is
      weaker, partial, or likely too shallow to justify aggressive
      structure-first chunking

    Why this strategy exists:
    - some documents are highly regular and benefit from stricter
      structure-preserving chunking
    - others are only partially parsed and need a more resilient strategy
      that still produces useful output
    """

    name = "hybrid"

    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        """
        Choose the most suitable chunking strategy for the current document.

        Heuristic overview:
        - ARTICLE count tells us whether there is a stable legal backbone
        - SECTION / LETTERED_ITEM counts tell us whether the parser captured
          meaningful internal structure inside articles
        - CHAPTER / ANNEX counts are weaker signals because they are useful
          containers, but by themselves do not guarantee that article content
          was segmented well enough for strict structure-first chunking

        Selection logic:
        - prefer structure-first when there is a good article backbone and:
            * meaningful internal article structure exists, or
            * container structure exists together with a larger article volume
        - otherwise use article-smart, which is more forgiving and robust

        This makes the hybrid strategy safer on noisy PDFs:
        it avoids choosing structure-first merely because a document contains
        chapters or annexes, when internal article segmentation is still weak.
        """
        article_count = self._count_node_type(root, "ARTICLE")
        section_count = self._count_node_type(root, "SECTION")
        lettered_count = self._count_node_type(root, "LETTERED_ITEM")
        annex_count = self._count_node_type(root, "ANNEX")
        chapter_count = self._count_node_type(root, "CHAPTER")

        # -----------------------------------------------------------------
        # Signal 1:
        # determine whether the document contains a sufficiently strong
        # article backbone.
        #
        # Why threshold >= 3:
        # a couple of detected articles may still reflect weak parsing or
        # partial extraction. Three or more articles usually indicates that
        # the parser found a meaningful legal structure.
        # -----------------------------------------------------------------
        has_good_article_backbone = article_count >= 3

        # -----------------------------------------------------------------
        # Signal 2:
        # determine whether the parser captured internal structure inside
        # articles, such as numbered sections or legal alíneas.
        #
        # These are the strongest signals for using structure-first because
        # they directly support smaller and semantically coherent chunks.
        # -----------------------------------------------------------------
        has_internal_structure = (
            section_count >= 2
            or lettered_count >= 3
        )

        # -----------------------------------------------------------------
        # Signal 3:
        # determine whether the parser captured larger structural containers
        # such as chapters or annexes.
        #
        # These are useful signals, but they are weaker than SECTION or
        # LETTERED_ITEM detection because they do not necessarily mean the
        # article body was segmented well.
        # -----------------------------------------------------------------
        has_container_structure = (
            annex_count >= 1
            or chapter_count >= 1
        )

        # -----------------------------------------------------------------
        # Final decision:
        #
        # Use structure-first when:
        # - the document clearly has an article backbone, and
        # - either:
        #   1) it has internal article structure, or
        #   2) it has container structure and enough articles to justify
        #      trusting the parsed hierarchy more strongly
        #
        # The second branch is intentionally stricter because container-only
        # structure is not enough on its own for small documents.
        # -----------------------------------------------------------------
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
        Recursively count how many nodes of a given type exist in the tree.

        Why recursion is used:
        parser output may vary in hierarchy depth, so counting only immediate
        children would be fragile and incomplete.
        """
        count = 1 if node.node_type == node_type else 0

        for child in node.children:
            count += self._count_node_type(child, node_type)

        return count