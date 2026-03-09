from __future__ import annotations

from typing import Any, Dict, Iterator, List

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.utils.text import normalize_block_whitespace, split_paragraphs


class ArticleSmartChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy centered on ARTICLE nodes.

    Strategy philosophy:
    - treat each article as the primary legal retrieval unit
    - keep short articles intact whenever possible
    - for larger articles, prefer already parsed internal structure
      such as numbered SECTION nodes
    - if numbered sections are unavailable, fall back to LETTERED_ITEM nodes
    - if the parser did not extract usable internal structure, fall back to
      paragraph grouping

    Why this strategy is a strong default for legal and regulatory documents:
    - articles usually represent stable normative units
    - article metadata is highly interpretable downstream
    - internal numbering often maps to coherent sub-rules
    - the strategy remains resilient even when parsing is only partially
      successful

    Important:
    this strategy assumes the structure parser already did the heavy work of
    identifying ARTICLE / SECTION / LETTERED_ITEM nodes. The role here is
    to convert that structure into retrieval-friendly chunks while preserving
    semantic coherence and useful metadata.
    """

    name = "article_smart"

    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        """
        Build chunks from a parsed document tree.

        Processing order:
        1. Export PREAMBLE content first, separately from the regulation body
        2. Process each ARTICLE node in document order
        3. Prefer whole-article chunks for short articles
        4. Prefer grouped SECTION chunks for longer articles
        5. Fall back to grouped LETTERED_ITEM chunks when sections do not exist
        6. Fall back to paragraph grouping when no finer structure is available

        Important:
        chunk ids are generated in a stable sequential order for easier
        debugging and DOCX/JSON inspection.
        """
        chunks: List[Chunk] = []
        sequence = 1

        # -----------------------------------------------------------------
        # 1) Chunk the preamble / dispatch separately if present.
        #
        # We intentionally keep preamble text isolated from the regulation
        # body because:
        # - it has different semantic meaning
        # - it often contains approval / revocation context
        # - it should not be mixed with normative article content
        # -----------------------------------------------------------------
        for preamble in self._iter_nodes_by_type(root, "PREAMBLE"):
            preamble_text = normalize_block_whitespace(preamble.text)
            if not preamble_text:
                continue

            preamble_groups = self._paragraph_grouping(preamble_text)

            # Fallback:
            # if paragraph grouping returns nothing for any reason, keep the
            # full preamble as a single chunk rather than dropping it.
            if not preamble_groups:
                preamble_groups = [preamble_text]

            for group_text in preamble_groups:
                chunks.append(
                    self._make_chunk(
                        sequence=sequence,
                        document_metadata=document_metadata,
                        text=group_text,
                        page_start=preamble.page_start,
                        page_end=preamble.page_end,
                        metadata={
                            "node_type": preamble.node_type,
                            "label": preamble.label,
                            "document_part": preamble.metadata.get("document_part"),
                            "source_span_type": "preamble",
                        },
                    )
                )
                sequence += 1

        # -----------------------------------------------------------------
        # 2) Chunk article content.
        # -----------------------------------------------------------------
        for article in self._iter_nodes_by_type(root, "ARTICLE"):
            article_text = normalize_block_whitespace(article.text)
            if not article_text:
                continue

            article_meta = self._article_metadata(article)

            # -----------------------------------------------------------------
            # Case A:
            # Keep short articles as a single chunk.
            #
            # Why:
            # - a short article is often already a complete legal unit
            # - preserving the whole article improves interpretability
            # - unnecessary splitting reduces retrieval quality
            # -----------------------------------------------------------------
            if len(article_text) <= self.settings.target_chunk_chars:
                chunks.append(
                    self._make_chunk(
                        sequence=sequence,
                        document_metadata=document_metadata,
                        text=article_text,
                        page_start=article.page_start,
                        page_end=article.page_end,
                        metadata={
                            **article_meta,
                            "source_span_type": "article",
                        },
                    )
                )
                sequence += 1
                continue

            # -----------------------------------------------------------------
            # Case B:
            # Large article with SECTION nodes available.
            #
            # This is the preferred split path because numbered sections usually
            # correspond to meaningful internal rules.
            # -----------------------------------------------------------------
            section_children = [
                child
                for child in article.children
                if child.node_type == "SECTION" and child.text.strip()
            ]

            if section_children:
                section_groups = self._group_sections(section_children)

                for group in section_groups:
                    group_text = normalize_block_whitespace(
                        "\n\n".join(
                            section.text
                            for section in group
                            if section.text.strip()
                        )
                    )

                    if not group_text:
                        continue

                    group_labels = [section.label for section in group]

                    # ---------------------------------------------------------
                    # If grouped sections are still too large, fall back to
                    # paragraph grouping inside that grouped span.
                    #
                    # We keep the article + section metadata so the chunk
                    # remains structurally traceable even after fallback split.
                    # ---------------------------------------------------------
                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        paragraph_groups = self._paragraph_grouping(group_text)

                        if not paragraph_groups:
                            paragraph_groups = [group_text]

                        for paragraph_group in paragraph_groups:
                            chunks.append(
                                self._make_chunk(
                                    sequence=sequence,
                                    document_metadata=document_metadata,
                                    text=paragraph_group,
                                    page_start=article.page_start,
                                    page_end=article.page_end,
                                    metadata={
                                        **article_meta,
                                        "section_labels": group_labels,
                                        "source_span_type": "article_section_group_paragraph_split",
                                    },
                                )
                            )
                            sequence += 1
                    else:
                        chunks.append(
                            self._make_chunk(
                                sequence=sequence,
                                document_metadata=document_metadata,
                                text=group_text,
                                page_start=article.page_start,
                                page_end=article.page_end,
                                metadata={
                                    **article_meta,
                                    "section_labels": group_labels,
                                    "source_span_type": "article_section_group",
                                },
                            )
                        )
                        sequence += 1

                continue

            # -----------------------------------------------------------------
            # Case C:
            # No SECTION nodes, but LETTERED_ITEM nodes are available.
            #
            # This helps documents whose article structure is based on alíneas
            # such as "a)", "b)", "c)" rather than numbered sections.
            # -----------------------------------------------------------------
            lettered_children = [
                child
                for child in article.children
                if child.node_type == "LETTERED_ITEM" and child.text.strip()
            ]

            if lettered_children:
                lettered_groups = self._group_lettered_items(lettered_children)

                for group in lettered_groups:
                    group_text = normalize_block_whitespace(
                        "\n\n".join(
                            item.text
                            for item in group
                            if item.text.strip()
                        )
                    )

                    if not group_text:
                        continue

                    group_labels = [item.label for item in group]

                    # ---------------------------------------------------------
                    # In most cases grouped lettered items should remain small
                    # enough. If not, we still have the paragraph fallback to
                    # avoid producing oversized retrieval units.
                    # ---------------------------------------------------------
                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        paragraph_groups = self._paragraph_grouping(group_text)

                        if not paragraph_groups:
                            paragraph_groups = [group_text]

                        for paragraph_group in paragraph_groups:
                            chunks.append(
                                self._make_chunk(
                                    sequence=sequence,
                                    document_metadata=document_metadata,
                                    text=paragraph_group,
                                    page_start=article.page_start,
                                    page_end=article.page_end,
                                    metadata={
                                        **article_meta,
                                        "lettered_labels": group_labels,
                                        "source_span_type": "article_lettered_group_paragraph_split",
                                    },
                                )
                            )
                            sequence += 1
                    else:
                        chunks.append(
                            self._make_chunk(
                                sequence=sequence,
                                document_metadata=document_metadata,
                                text=group_text,
                                page_start=article.page_start,
                                page_end=article.page_end,
                                metadata={
                                    **article_meta,
                                    "lettered_labels": group_labels,
                                    "source_span_type": "article_lettered_group",
                                },
                            )
                        )
                        sequence += 1

                continue

            # -----------------------------------------------------------------
            # Case D:
            # No usable internal structure.
            #
            # This is the safety fallback for imperfect parsing or documents
            # whose internal formatting is too weak/inconsistent.
            # -----------------------------------------------------------------
            paragraph_groups = self._paragraph_grouping(article_text)

            if not paragraph_groups:
                paragraph_groups = [article_text]

            for paragraph_group in paragraph_groups:
                chunks.append(
                    self._make_chunk(
                        sequence=sequence,
                        document_metadata=document_metadata,
                        text=paragraph_group,
                        page_start=article.page_start,
                        page_end=article.page_end,
                        metadata={
                            **article_meta,
                            "source_span_type": "article_paragraph_group",
                        },
                    )
                )
                sequence += 1

        return chunks

    def _iter_nodes_by_type(
        self,
        node: StructuralNode,
        node_type: str,
    ) -> Iterator[StructuralNode]:
        """
        Recursively yield all nodes of a given type.

        Why recursion is used here:
        the tree layout may vary slightly depending on parser success
        and document structure. A recursive traversal is more robust than
        assuming a fixed path such as DOCUMENT -> CHAPTER -> ARTICLE.
        """
        if node.node_type == node_type:
            yield node

        for child in node.children:
            yield from self._iter_nodes_by_type(child, node_type)

    def _group_sections(self, sections: List[StructuralNode]) -> List[List[StructuralNode]]:
        """
        Group adjacent SECTION nodes into chunk-sized bundles.

        Grouping policy:
        - preserve original order
        - try to stay near target_chunk_chars
        - avoid flushing too early when the current group is still too small
        - merge very small trailing groups into the previous group

        Why grouping is necessary:
        some sections are too small to stand alone as useful retrieval units.
        Grouping adjacent sections often preserves meaning better than splitting
        too aggressively.
        """
        if not sections:
            return []

        groups: List[List[StructuralNode]] = []
        current_group: List[StructuralNode] = []

        for section in sections:
            if not current_group:
                current_group = [section]
                continue

            current_text = "\n\n".join(
                item.text for item in current_group if item.text.strip()
            )
            candidate_text = "\n\n".join(
                item.text for item in (current_group + [section]) if item.text.strip()
            )

            current_len = len(current_text)
            candidate_len = len(candidate_text)

            should_flush = (
                candidate_len > self.settings.target_chunk_chars
                and current_len >= self.settings.min_chunk_chars
            )

            if should_flush:
                groups.append(current_group)
                current_group = [section]
            else:
                current_group.append(section)

        if current_group:
            groups.append(current_group)

        # Merge an undersized trailing group into the previous group.
        if len(groups) >= 2:
            last_group_text = "\n\n".join(
                item.text for item in groups[-1] if item.text.strip()
            )
            if len(last_group_text) < self.settings.min_chunk_chars:
                groups[-2].extend(groups[-1])
                groups.pop()

        return groups

    def _group_lettered_items(self, items: List[StructuralNode]) -> List[List[StructuralNode]]:
        """
        Group adjacent LETTERED_ITEM nodes into chunk-sized bundles.

        This mirrors section grouping, but operates on a smaller structural
        unit. It is useful for articles whose internal logic is expressed as
        legal alíneas instead of numbered sections.
        """
        if not items:
            return []

        groups: List[List[StructuralNode]] = []
        current_group: List[StructuralNode] = []

        for item in items:
            if not current_group:
                current_group = [item]
                continue

            current_text = "\n\n".join(
                node.text for node in current_group if node.text.strip()
            )
            candidate_text = "\n\n".join(
                node.text for node in (current_group + [item]) if node.text.strip()
            )

            should_flush = (
                len(candidate_text) > self.settings.target_chunk_chars
                and len(current_text) >= self.settings.min_chunk_chars
            )

            if should_flush:
                groups.append(current_group)
                current_group = [item]
            else:
                current_group.append(item)

        if current_group:
            groups.append(current_group)

        # Merge an undersized trailing group into the previous group.
        if len(groups) >= 2:
            last_group_text = "\n\n".join(
                node.text for node in groups[-1] if node.text.strip()
            )
            if len(last_group_text) < self.settings.min_chunk_chars:
                groups[-2].extend(groups[-1])
                groups.pop()

        return groups

    def _paragraph_grouping(self, text: str) -> List[str]:
        """
        Group paragraphs into chunk-sized blocks.

        This method is the generic fallback when:
        - article sections are unavailable
        - grouped sections are still too large
        - grouped lettered items are still too large
        - preamble content needs subdivision

        Design goal:
        keep paragraph boundaries whenever possible, because paragraph
        boundaries are usually safer semantic split points than raw
        character slicing.
        """
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return []

        groups: List[str] = []
        current: List[str] = []

        for paragraph in paragraphs:
            candidate = "\n\n".join(current + [paragraph]).strip()

            should_flush = (
                current
                and len(candidate) > self.settings.target_chunk_chars
                and len("\n\n".join(current)) >= self.settings.min_chunk_chars
            )

            if should_flush:
                groups.append("\n\n".join(current))
                current = [paragraph]
            else:
                current.append(paragraph)

        if current:
            groups.append("\n\n".join(current))

        # Merge a very small trailing paragraph group into the previous group.
        if len(groups) >= 2 and len(groups[-1]) < self.settings.min_chunk_chars:
            groups[-2] = f"{groups[-2]}\n\n{groups[-1]}".strip()
            groups.pop()

        return groups

    def _article_metadata(self, node: StructuralNode) -> Dict[str, Any]:
        """
        Build article-level metadata for chunk export.

        Important:
        metadata is the primary carrier of structural identity, while the
        chunk text remains as clean as possible.

        This allows downstream consumers to understand:
        - which article the chunk came from
        - the article number and title
        - the parent structural container
        - whether the content belongs to the regulation body or annexes
        """
        return {
            "node_type": node.node_type,
            "label": node.label,
            "article_title": node.title,
            "article_number": node.metadata.get("article_number"),
            "parent_type": node.metadata.get("parent_type"),
            "parent_label": node.metadata.get("parent_label"),
            "parent_title": node.metadata.get("parent_title"),
            "document_part": node.metadata.get("document_part"),
        }

    def _make_chunk(
        self,
        sequence: int,
        document_metadata: DocumentMetadata,
        text: str,
        page_start: int | None,
        page_end: int | None,
        metadata: Dict[str, Any],
    ) -> Chunk:
        """
        Build a final Chunk object.

        Important:
        - chunk text is normalized before export
        - chunk ids remain stable and sequential
        - metadata preserves source structure and source span type

        The chunk id format is intentionally deterministic because it helps:
        - JSON inspection
        - DOCX inspection
        - regression testing
        - debugging across pipeline runs
        """
        return Chunk(
            chunk_id=f"{document_metadata.doc_id}_chunk_{sequence:04d}",
            doc_id=document_metadata.doc_id,
            strategy=self.name,
            text=normalize_block_whitespace(text),
            page_start=page_start,
            page_end=page_end,
            metadata=metadata,
        )