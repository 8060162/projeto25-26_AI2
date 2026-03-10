from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.utils.text import normalize_block_whitespace, split_paragraphs


class ArticleSmartChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy centered on ARTICLE nodes.

    Strategy philosophy
    -------------------
    - treat each article as the primary legal retrieval unit
    - keep short articles intact whenever possible
    - for larger articles, prefer already parsed internal structure such as
      numbered SECTION nodes
    - if numbered sections are unavailable, fall back to LETTERED_ITEM nodes
    - if the parser did not extract usable internal structure, fall back to
      paragraph grouping
    - keep FRONT_MATTER and PREAMBLE separate from normative article content

    Why this strategy is a strong default
    -------------------------------------
    For legal and regulatory documents, articles usually represent stable and
    meaningful normative units. This makes article-centered chunking a strong
    default for retrieval, navigation, and metadata interpretation.

    Important design note
    ---------------------
    This strategy assumes the structure parser already identified nodes such as:
    - FRONT_MATTER
    - PREAMBLE
    - ARTICLE
    - SECTION
    - LETTERED_ITEM

    The role here is to convert that structure into retrieval-friendly chunks
    while preserving semantic coherence and rich structural traceability.
    """

    name = "article_smart"

    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        """
        Build chunks from a parsed document tree.

        Processing order
        ----------------
        1. Export FRONT_MATTER separately when present
        2. Export PREAMBLE separately when present
        3. Process ARTICLE nodes in document order
        4. Prefer whole-article chunks for short articles
        5. Prefer grouped SECTION chunks for larger articles
        6. Fall back to grouped LETTERED_ITEM chunks when sections do not exist
        7. Fall back to paragraph grouping when no finer structure is available
        8. Link neighboring chunks for future traversal / expansion

        Why stable sequence numbering matters
        -------------------------------------
        Deterministic chunk ids make:
        - JSON inspection easier
        - DOCX inspection easier
        - regression comparison easier
        - debugging much easier

        Parameters
        ----------
        document_metadata : DocumentMetadata
            High-level metadata for the current source document.
        root : StructuralNode
            Parsed structural tree for the document.

        Returns
        -------
        List[Chunk]
            Final chunk list for the selected strategy.
        """
        chunks: List[Chunk] = []
        sequence = 1

        # -----------------------------------------------------------------
        # 1) Chunk front matter separately.
        #
        # Front matter often contains:
        # - cover information
        # - institutional branding
        # - dispatch headers
        # - index-like content
        #
        # This material should not be mixed with normative article chunks.
        # -----------------------------------------------------------------
        for front_matter in self._iter_nodes_by_type(root, "FRONT_MATTER"):
            front_text = normalize_block_whitespace(front_matter.text)
            if not front_text:
                continue

            front_groups = self._paragraph_grouping(front_text)
            if not front_groups:
                front_groups = [front_text]

            for group_text in front_groups:
                chunks.append(
                    self._make_chunk(
                        sequence=sequence,
                        document_metadata=document_metadata,
                        text=group_text,
                        page_start=front_matter.page_start,
                        page_end=front_matter.page_end,
                        source_node=front_matter,
                        chunk_reason="front_matter_group",
                        metadata={
                            "document_part": front_matter.metadata.get("document_part"),
                            "source_span_type": "front_matter",
                        },
                    )
                )
                sequence += 1

        # -----------------------------------------------------------------
        # 2) Chunk preamble / dispatch separately.
        #
        # The preamble often carries contextual value such as:
        # - approval context
        # - amendment context
        # - revocation context
        #
        # It should remain separate from normative article content.
        # -----------------------------------------------------------------
        for preamble in self._iter_nodes_by_type(root, "PREAMBLE"):
            preamble_text = normalize_block_whitespace(preamble.text)
            if not preamble_text:
                continue

            preamble_groups = self._paragraph_grouping(preamble_text)
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
                        source_node=preamble,
                        chunk_reason="preamble_group",
                        metadata={
                            "document_part": preamble.metadata.get("document_part"),
                            "source_span_type": "preamble",
                        },
                    )
                )
                sequence += 1

        # -----------------------------------------------------------------
        # 3) Chunk article content.
        # -----------------------------------------------------------------
        for article in self._iter_nodes_by_type(root, "ARTICLE"):
            article_text = normalize_block_whitespace(article.text)
            if not article_text:
                continue

            article_meta = self._article_metadata(article)

            # -------------------------------------------------------------
            # Case A:
            # Keep short articles as a single chunk.
            #
            # Why:
            # - a short article is often already a complete legal unit
            # - preserving the whole article improves interpretability
            # - unnecessary splitting reduces retrieval quality
            # -------------------------------------------------------------
            if len(article_text) <= self.settings.target_chunk_chars:
                chunks.append(
                    self._make_chunk(
                        sequence=sequence,
                        document_metadata=document_metadata,
                        text=article_text,
                        page_start=article.page_start,
                        page_end=article.page_end,
                        source_node=article,
                        chunk_reason="direct_article",
                        metadata={
                            **article_meta,
                            "source_span_type": "article",
                        },
                    )
                )
                sequence += 1
                continue

            # -------------------------------------------------------------
            # Case B:
            # Large article with SECTION nodes available.
            #
            # Preferred split path because numbered sections usually correspond
            # to meaningful internal legal rules.
            # -------------------------------------------------------------
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
                    hierarchy_path = self._merge_hierarchy_path(
                        article.hierarchy_path,
                        [f"SECTION:{label}" for label in group_labels],
                    )

                    # If grouped sections are still too large, use paragraph
                    # fallback while preserving article + section traceability.
                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        paragraph_groups = self._paragraph_grouping(group_text)
                        if not paragraph_groups:
                            paragraph_groups = [group_text]

                        for part_index, paragraph_group in enumerate(paragraph_groups, start=1):
                            chunks.append(
                                self._make_chunk(
                                    sequence=sequence,
                                    document_metadata=document_metadata,
                                    text=paragraph_group,
                                    page_start=article.page_start,
                                    page_end=article.page_end,
                                    source_node=article,
                                    hierarchy_path=hierarchy_path,
                                    chunk_reason="grouped_sections_paragraph_split",
                                    metadata={
                                        **article_meta,
                                        "section_labels": group_labels,
                                        "part_index": part_index,
                                        "part_count": len(paragraph_groups),
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
                                source_node=article,
                                hierarchy_path=hierarchy_path,
                                chunk_reason="grouped_sections",
                                metadata={
                                    **article_meta,
                                    "section_labels": group_labels,
                                    "source_span_type": "article_section_group",
                                },
                            )
                        )
                        sequence += 1

                continue

            # -------------------------------------------------------------
            # Case C:
            # No SECTION nodes, but LETTERED_ITEM nodes are available.
            #
            # This supports articles whose logic is mainly expressed through
            # alíneas such as "a)", "b)", "c)".
            # -------------------------------------------------------------
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
                    hierarchy_path = self._merge_hierarchy_path(
                        article.hierarchy_path,
                        [f"LETTERED_ITEM:{label}" for label in group_labels],
                    )

                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        paragraph_groups = self._paragraph_grouping(group_text)
                        if not paragraph_groups:
                            paragraph_groups = [group_text]

                        for part_index, paragraph_group in enumerate(paragraph_groups, start=1):
                            chunks.append(
                                self._make_chunk(
                                    sequence=sequence,
                                    document_metadata=document_metadata,
                                    text=paragraph_group,
                                    page_start=article.page_start,
                                    page_end=article.page_end,
                                    source_node=article,
                                    hierarchy_path=hierarchy_path,
                                    chunk_reason="grouped_lettered_items_paragraph_split",
                                    metadata={
                                        **article_meta,
                                        "lettered_labels": group_labels,
                                        "part_index": part_index,
                                        "part_count": len(paragraph_groups),
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
                                source_node=article,
                                hierarchy_path=hierarchy_path,
                                chunk_reason="grouped_lettered_items",
                                metadata={
                                    **article_meta,
                                    "lettered_labels": group_labels,
                                    "source_span_type": "article_lettered_group",
                                },
                            )
                        )
                        sequence += 1

                continue

            # -------------------------------------------------------------
            # Case D:
            # No usable internal structure.
            #
            # This is the safety fallback for imperfect parsing or documents
            # whose internal formatting is too weak/inconsistent.
            # -------------------------------------------------------------
            paragraph_groups = self._paragraph_grouping(article_text)
            if not paragraph_groups:
                paragraph_groups = [article_text]

            for part_index, paragraph_group in enumerate(paragraph_groups, start=1):
                chunks.append(
                    self._make_chunk(
                        sequence=sequence,
                        document_metadata=document_metadata,
                        text=paragraph_group,
                        page_start=article.page_start,
                        page_end=article.page_end,
                        source_node=article,
                        chunk_reason="fallback_paragraph_split",
                        metadata={
                            **article_meta,
                            "part_index": part_index,
                            "part_count": len(paragraph_groups),
                            "source_span_type": "article_paragraph_group",
                        },
                    )
                )
                sequence += 1

        self._link_neighbor_chunks(chunks)
        return chunks

    def _iter_nodes_by_type(
        self,
        node: StructuralNode,
        node_type: str,
    ) -> Iterator[StructuralNode]:
        """
        Recursively yield all nodes of a given type.

        Why recursion is used
        ---------------------
        The tree layout may vary depending on parser success and document
        structure. A recursive traversal is safer than assuming a fixed path
        such as DOCUMENT -> CHAPTER -> ARTICLE.

        Parameters
        ----------
        node : StructuralNode
            Current node.
        node_type : str
            Node type to yield.

        Yields
        ------
        Iterator[StructuralNode]
            Matching nodes in traversal order.
        """
        if node.node_type == node_type:
            yield node

        for child in node.children:
            yield from self._iter_nodes_by_type(child, node_type)

    def _group_sections(self, sections: List[StructuralNode]) -> List[List[StructuralNode]]:
        """
        Group adjacent SECTION nodes into chunk-sized bundles.

        Grouping policy
        ---------------
        - preserve original order
        - stay reasonably close to target_chunk_chars
        - avoid flushing too early when the current group is still too small
        - merge very small trailing groups into the previous group

        Why grouping is useful
        ----------------------
        Some sections are too small to stand alone as useful retrieval units.
        Grouping adjacent sections often preserves meaning better than splitting
        too aggressively.

        Parameters
        ----------
        sections : List[StructuralNode]
            Ordered section nodes from an article.

        Returns
        -------
        List[List[StructuralNode]]
            Grouped section bundles.
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

        This mirrors section grouping but operates on a smaller structural unit.

        Parameters
        ----------
        items : List[StructuralNode]
            Ordered lettered item nodes.

        Returns
        -------
        List[List[StructuralNode]]
            Grouped lettered-item bundles.
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
        - preamble or front matter needs subdivision

        Design principle
        ----------------
        Keep paragraph boundaries whenever possible because paragraph boundaries
        are usually safer semantic split points than raw character slicing.

        Parameters
        ----------
        text : str
            Input text to subdivide.

        Returns
        -------
        List[str]
            Paragraph-based chunk groups.
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

        if len(groups) >= 2 and len(groups[-1]) < self.settings.min_chunk_chars:
            groups[-2] = f"{groups[-2]}\n\n{groups[-1]}".strip()
            groups.pop()

        return groups

    def _article_metadata(self, node: StructuralNode) -> Dict[str, Any]:
        """
        Build article-level metadata for chunk export.

        Important design choice
        -----------------------
        Metadata is the primary carrier of structural identity, while chunk text
        remains as clean as possible.

        Parameters
        ----------
        node : StructuralNode
            Article node.

        Returns
        -------
        Dict[str, Any]
            Article-level metadata useful downstream.
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

    def _merge_hierarchy_path(
        self,
        base_path: Sequence[str],
        extra_items: Sequence[str],
    ) -> List[str]:
        """
        Build a merged hierarchy path for a derived chunk.

        Why this helper exists
        ----------------------
        A chunk may come from an article but also represent a grouped set of
        sections or lettered items. This helper makes it easy to preserve the
        base structural path while appending chunk-level substructure hints.

        Parameters
        ----------
        base_path : Sequence[str]
            Base hierarchy path from the source node.
        extra_items : Sequence[str]
            Extra structural labels relevant to the chunk.

        Returns
        -------
        List[str]
            Combined hierarchy path.
        """
        result = list(base_path) if base_path else []
        for item in extra_items:
            if item:
                result.append(item)
        return result

    def _build_text_for_embedding(
        self,
        source_node: StructuralNode,
        visible_text: str,
    ) -> str:
        """
        Build enriched embedding text from clean visible chunk text.

        Why this helper exists
        ----------------------
        In legal retrieval, the chunk body alone may be too context-poor.
        A short contextual prefix such as article label and title often improves
        retrieval quality while still keeping the visible chunk text clean.

        Example
        -------
        Visible text:
            "1 — O estudante ..."

        Embedding text:
            "Artigo 5.º - Revisão de prova\\n\\n1 — O estudante ..."

        Parameters
        ----------
        source_node : StructuralNode
            Main structural source of the chunk.
        visible_text : str
            Clean visible chunk text.

        Returns
        -------
        str
            Context-enriched text suitable for embedding.
        """
        visible_text = normalize_block_whitespace(visible_text)
        if not visible_text:
            return ""

        header_parts: List[str] = []

        if source_node.node_type == "ARTICLE":
            article_number = source_node.metadata.get("article_number")
            if article_number:
                header_parts.append(f"Artigo {article_number}")
            elif source_node.label:
                header_parts.append(source_node.label)

            if source_node.title:
                header_parts.append(source_node.title)

        elif source_node.node_type in {"PREAMBLE", "FRONT_MATTER"}:
            header_parts.append(source_node.node_type.replace("_", " ").title())

        else:
            if source_node.label:
                header_parts.append(source_node.label)
            if source_node.title:
                header_parts.append(source_node.title)

        header = " - ".join(part for part in header_parts if part).strip()
        if not header:
            return visible_text

        return f"{header}\n\n{visible_text}".strip()

    def _make_chunk(
        self,
        sequence: int,
        document_metadata: DocumentMetadata,
        text: str,
        page_start: int | None,
        page_end: int | None,
        source_node: StructuralNode,
        chunk_reason: str,
        metadata: Dict[str, Any],
        hierarchy_path: List[str] | None = None,
    ) -> Chunk:
        """
        Build a final Chunk object.

        Important behavior
        ------------------
        - visible chunk text is normalized before export
        - enriched embedding text is generated separately
        - chunk ids remain deterministic and sequential
        - source structure is promoted into explicit chunk fields
        - metadata remains flexible for future enrichment

        Parameters
        ----------
        sequence : int
            Stable sequence number in the current document.
        document_metadata : DocumentMetadata
            Source document metadata.
        text : str
            Visible chunk text.
        page_start : int | None
            Chunk page start.
        page_end : int | None
            Chunk page end.
        source_node : StructuralNode
            Main structural origin of the chunk.
        chunk_reason : str
            Explanation of why the chunk exists in this form.
        metadata : Dict[str, Any]
            Additional metadata payload.
        hierarchy_path : List[str] | None
            Optional custom hierarchy path. When omitted, the source node path
            is used.

        Returns
        -------
        Chunk
            Final chunk object ready for export.
        """
        visible_text = normalize_block_whitespace(text)
        effective_hierarchy_path = hierarchy_path or list(source_node.hierarchy_path)

        return Chunk(
            chunk_id=f"{document_metadata.doc_id}_chunk_{sequence:04d}",
            doc_id=document_metadata.doc_id,
            strategy=self.name,
            text=visible_text,
            text_for_embedding=self._build_text_for_embedding(
                source_node=source_node,
                visible_text=visible_text,
            ),
            page_start=page_start,
            page_end=page_end,
            source_node_type=source_node.node_type,
            source_node_label=source_node.label,
            hierarchy_path=effective_hierarchy_path,
            chunk_reason=chunk_reason,
            char_count=len(visible_text),
            metadata=metadata,
        )

    def _link_neighbor_chunks(self, chunks: List[Chunk]) -> None:
        """
        Link neighboring chunks using prev_chunk_id and next_chunk_id.

        Why this helper exists
        ----------------------
        Neighbor links are useful for:
        - future chunk expansion during retrieval
        - navigation in inspection tools
        - reconstructing local context

        Parameters
        ----------
        chunks : List[Chunk]
            Chunk list in final document order.
        """
        for index, chunk in enumerate(chunks):
            previous_chunk = chunks[index - 1] if index > 0 else None
            next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None

            chunk.prev_chunk_id = previous_chunk.chunk_id if previous_chunk else None
            chunk.next_chunk_id = next_chunk.chunk_id if next_chunk else None