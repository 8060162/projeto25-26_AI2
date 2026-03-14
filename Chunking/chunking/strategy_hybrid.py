from __future__ import annotations

from typing import Dict, List, Tuple

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_article_smart import ArticleSmartChunkingStrategy
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.chunking.strategy_structure_first import StructureFirstChunkingStrategy


class HybridChunkingStrategy(BaseChunkingStrategy):
    """
    Hybrid chunking strategy.

    Decision principle
    ------------------
    - use the structure-first strategy when the parser produced a sufficiently
      rich and trustworthy hierarchy
    - fall back to the article-smart strategy when the parsed structure is
      weaker, shallower, or likely too incomplete for strong structure-first
      chunking

    Why this strategy exists
    ------------------------
    Some documents are highly regular and benefit from stricter
    structure-preserving chunking, while others are only partially parsed
    and need a more forgiving strategy.

    Design philosophy
    -----------------
    The hybrid strategy should remain pragmatic and lightweight:
    - no expensive scoring system
    - no dependency on external diagnostics
    - but more intelligent than simple node counting

    Compared with simpler implementations
    -------------------------------------
    This version makes the decision using richer structural signals:
    - total article count
    - article title coverage
    - number of articles with internal structure
    - total internal structure volume
    - presence of larger structural containers
    - density of article text

    This makes the hybrid strategy more robust on noisy or partially parsed PDFs.
    """

    name = "hybrid"

    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        """
        Choose the most suitable chunking strategy for the current document.

        Heuristic overview
        ------------------
        We derive a few lightweight structural quality signals:

        1. Article backbone quality
           A document needs a meaningful set of ARTICLE nodes to justify
           structure-first chunking.

        2. Internal structure coverage
           We check whether a useful proportion of articles contains internal
           structure such as SECTION or LETTERED_ITEM nodes.

        3. Article title coverage
           Good title coverage is a weak but useful signal that parsing quality
           is reasonable.

        4. Container structure
           CHAPTER, ANNEX, and SECTION_CONTAINER nodes provide supporting
           evidence, but they are weaker than article-internal structure.

        5. Article text density
           A parser that found many article shells but very little article text
           should not strongly push the decision toward structure-first.

        Decision policy
        ---------------
        Prefer structure-first when:
        - the document has a solid article backbone, and
        - internal structure coverage is meaningful

        Otherwise:
        - use article-smart, which is more forgiving and resilient

        Parameters
        ----------
        document_metadata : DocumentMetadata
            Source document metadata.

        root : StructuralNode
            Parsed structural tree.

        Returns
        -------
        List[Chunk]
            Final chunk list built by the selected strategy.
        """
        # ------------------------------------------------------------------
        # Respect the pipeline configuration.
        #
        # If the hybrid strategy is disabled in settings, fall back to the
        # article-smart strategy because it is the safer default.
        # ------------------------------------------------------------------
        if not self.settings.enable_hybrid_strategy:
            return ArticleSmartChunkingStrategy(self.settings).build_chunks(
                document_metadata,
                root,
            )

        selected_strategy_name, _signals = self._select_strategy(root)

        if selected_strategy_name == "structure_first":
            return StructureFirstChunkingStrategy(self.settings).build_chunks(
                document_metadata,
                root,
            )

        return ArticleSmartChunkingStrategy(self.settings).build_chunks(
            document_metadata,
            root,
        )

    def _select_strategy(
        self,
        root: StructuralNode,
    ) -> Tuple[str, Dict[str, float | int | bool]]:
        """
        Select the most appropriate concrete strategy for the current document.

        Why this helper exists
        ----------------------
        Separating strategy selection from chunk generation makes the hybrid
        implementation easier to:
        - read
        - test
        - debug
        - audit

        Parameters
        ----------
        root : StructuralNode
            Parsed structural tree.

        Returns
        -------
        Tuple[str, Dict[str, float | int | bool]]
            A pair containing:
            - the selected strategy name
            - the structural decision signals used
        """
        articles = self._iter_nodes_by_type(root, "ARTICLE")
        article_count = len(articles)

        section_count = self._count_node_type(root, "SECTION")
        lettered_count = self._count_node_type(root, "LETTERED_ITEM")
        annex_count = self._count_node_type(root, "ANNEX")
        chapter_count = self._count_node_type(root, "CHAPTER")
        section_container_count = self._count_node_type(root, "SECTION_CONTAINER")

        titled_article_count = sum(
            1
            for article in articles
            if (article.title or "").strip()
        )

        non_empty_article_count = sum(
            1
            for article in articles
            if (article.text or "").strip()
        )

        articles_with_internal_structure = sum(
            1
            for article in articles
            if self._article_has_internal_structure(article)
        )

        # ------------------------------------------------------------------
        # Signal 1:
        # a meaningful article backbone
        #
        # Why threshold >= 3:
        # a couple of detected articles may still reflect weak parsing or
        # partial extraction. Three or more articles usually indicates that
        # the parser found a meaningful legal backbone.
        # ------------------------------------------------------------------
        has_good_article_backbone = article_count >= 3

        # ------------------------------------------------------------------
        # Signal 2:
        # article title coverage
        #
        # Why this matters:
        # article titles are not mandatory for every document, but when the
        # parser consistently captures them, it usually indicates healthier
        # line boundaries and better structural recognition.
        # ------------------------------------------------------------------
        article_title_coverage = (
            titled_article_count / article_count
            if article_count > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # Signal 3:
        # article-level internal structure coverage
        #
        # This is stronger than raw SECTION / LETTERED_ITEM counts because it
        # asks a more useful question:
        #
        #   "How many articles actually benefited from internal parsing?"
        # ------------------------------------------------------------------
        article_internal_structure_coverage = (
            articles_with_internal_structure / article_count
            if article_count > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # Signal 4:
        # total internal structure volume
        #
        # This remains useful as supporting evidence, especially for longer
        # documents with many sectioned provisions.
        # ------------------------------------------------------------------
        has_meaningful_internal_structure_volume = (
            section_count >= 2
            or lettered_count >= 3
        )

        # ------------------------------------------------------------------
        # Signal 5:
        # supporting container structure
        #
        # Containers are helpful, but weaker than article-internal structure.
        # They should only push the decision toward structure-first when the
        # rest of the tree already looks reasonably healthy.
        # ------------------------------------------------------------------
        has_container_structure = (
            annex_count >= 1
            or chapter_count >= 1
            or section_container_count >= 1
        )

        # ------------------------------------------------------------------
        # Signal 6:
        # article text density
        #
        # Why this matters:
        # a parser may find article headers but fail to attach much body text.
        # In that case, the document may appear structurally rich while still
        # being a poor candidate for structure-first chunking.
        # ------------------------------------------------------------------
        article_text_density = (
            non_empty_article_count / article_count
            if article_count > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # Final decision policy
        #
        # Prefer structure-first when:
        # - article backbone is good, and
        # - article text density is healthy, and
        # - either:
        #   1) internal structure coverage is meaningfully high, or
        #   2) internal structure volume is present with decent title coverage,
        #   3) or container structure exists together with larger article volume
        #
        # Why these thresholds are intentionally moderate:
        # the structure-first strategy should be chosen only when there is
        # enough evidence that parser output is useful, but not so rarely that
        # the hybrid strategy becomes pointless.
        # ------------------------------------------------------------------
        use_structure_first = (
            has_good_article_backbone
            and article_text_density >= 0.65
            and (
                article_internal_structure_coverage >= 0.35
                or (
                    has_meaningful_internal_structure_volume
                    and article_title_coverage >= 0.40
                )
                or (
                    has_container_structure
                    and article_count >= 6
                    and article_title_coverage >= 0.30
                    and article_text_density >= 0.80
                )
            )
        )

        decision_signals: Dict[str, float | int | bool] = {
            "article_count": article_count,
            "section_count": section_count,
            "lettered_count": lettered_count,
            "annex_count": annex_count,
            "chapter_count": chapter_count,
            "section_container_count": section_container_count,
            "titled_article_count": titled_article_count,
            "non_empty_article_count": non_empty_article_count,
            "articles_with_internal_structure": articles_with_internal_structure,
            "has_good_article_backbone": has_good_article_backbone,
            "article_title_coverage": article_title_coverage,
            "article_internal_structure_coverage": article_internal_structure_coverage,
            "has_meaningful_internal_structure_volume": has_meaningful_internal_structure_volume,
            "has_container_structure": has_container_structure,
            "article_text_density": article_text_density,
            "use_structure_first": use_structure_first,
        }

        if use_structure_first:
            return "structure_first", decision_signals

        return "article_smart", decision_signals

    def _iter_nodes_by_type(
        self,
        node: StructuralNode,
        node_type: str,
    ) -> List[StructuralNode]:
        """
        Collect all nodes of a given type from the structural tree.

        Why this helper exists
        ----------------------
        The hybrid decision needs more than raw counts. It also needs direct
        access to ARTICLE nodes so it can inspect properties such as:
        - whether the article has a title
        - whether the article has internal structure
        - whether the article contains real text

        Parameters
        ----------
        node : StructuralNode
            Current node.

        node_type : str
            Node type to collect.

        Returns
        -------
        List[StructuralNode]
            Matching nodes found recursively.
        """
        matches: List[StructuralNode] = []

        if node.node_type == node_type:
            matches.append(node)

        for child in node.children:
            matches.extend(self._iter_nodes_by_type(child, node_type))

        return matches

    def _count_node_type(self, node: StructuralNode, node_type: str) -> int:
        """
        Recursively count how many nodes of a given type exist in the tree.

        Why recursion is used
        ---------------------
        Parser output may vary in hierarchy depth, so counting only immediate
        children would be fragile and incomplete.

        Parameters
        ----------
        node : StructuralNode
            Current node.

        node_type : str
            Node type to count.

        Returns
        -------
        int
            Total count of matching nodes.
        """
        count = 1 if node.node_type == node_type else 0

        for child in node.children:
            count += self._count_node_type(child, node_type)

        return count

    def _article_has_internal_structure(self, article: StructuralNode) -> bool:
        """
        Decide whether an article has meaningful internal parsed structure.

        Current definition
        ------------------
        An article is considered internally structured when it contains at least
        one child of type:
        - SECTION
        - LETTERED_ITEM

        Why this helper matters
        -----------------------
        The hybrid decision should not rely only on global SECTION and
        LETTERED_ITEM counts. It is more useful to know how many articles
        actually benefited from internal parsing.

        Parameters
        ----------
        article : StructuralNode
            Article node to inspect.

        Returns
        -------
        bool
            True when the article contains meaningful internal structure.
        """
        for child in article.children:
            if child.node_type in {"SECTION", "LETTERED_ITEM"} and child.text.strip():
                return True

        return False