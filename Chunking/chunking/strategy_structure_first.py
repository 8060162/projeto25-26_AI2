from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.utils.text import normalize_block_whitespace, split_paragraphs


class StructureFirstChunkingStrategy(BaseChunkingStrategy):
    """
    Structure-first chunking strategy.

    Strategy philosophy
    -------------------
    - trust the parser as much as possible
    - prefer the smallest meaningful structural unit already recognized
    - use SECTION groups when numbered sections exist
    - otherwise use LETTERED_ITEM groups when legal alíneas exist
    - otherwise keep the ARTICLE whole if it is reasonably sized
    - otherwise split the ARTICLE conservatively by paragraphs
    - always keep FRONT_MATTER and PREAMBLE separate from the regulation body

    Why this strategy exists
    ------------------------
    Some legal and regulatory documents are highly regular and benefit more
    from structure-preserving chunking than from broader article-centered
    grouping.

    Important design note
    ---------------------
    Compared with the article-smart strategy, this strategy is intentionally
    more loyal to parser output and more willing to preserve finer structural
    boundaries when they exist.
    """

    name = "structure_first"

    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        """
        Build chunks from the parsed document tree.

        Processing order
        ----------------
        1. Export FRONT_MATTER content separately
        2. Export PREAMBLE content separately
        3. Process ARTICLE nodes in traversal order
        4. Prefer SECTION-group chunks when sections exist
        5. Otherwise prefer LETTERED-group chunks
        6. Otherwise keep the ARTICLE whole if small enough
        7. Otherwise split the ARTICLE into paragraph-based ARTICLE_PART chunks
        8. Link neighboring chunks for traversal and later expansion

        Why stable chunk order matters
        ------------------------------
        Deterministic chunk ids make:
        - JSON inspection easier
        - DOCX inspection easier
        - regression comparison easier
        - debugging easier

        Parameters
        ----------
        document_metadata : DocumentMetadata
            Source document metadata.
        root : StructuralNode
            Parsed structural tree.

        Returns
        -------
        List[Chunk]
            Final chunk list for the strategy.
        """
        chunks: List[Chunk] = []
        sequence = 1

        # -----------------------------------------------------------------
        # 1) Handle FRONT_MATTER nodes separately.
        #
        # Front matter often contains:
        # - institutional branding
        # - title-page content
        # - dispatch headings
        # - index-like material
        #
        # This content should remain separate from normative article chunks.
        # -----------------------------------------------------------------
        for front_matter in self._iter_nodes_by_type(root, "FRONT_MATTER"):
            front_text = normalize_block_whitespace(front_matter.text)
            if not front_text:
                continue

            front_groups = self._paragraph_grouping(front_text)
            if not front_groups:
                front_groups = [front_text]

            for group_text in front_groups:
                chunk = self._make_chunk(
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
                if chunk is not None:
                    chunks.append(chunk)
                    sequence += 1

        # -----------------------------------------------------------------
        # 2) Handle PREAMBLE nodes separately.
        #
        # The preamble / dispatch usually contains contextual information that
        # should not be mixed with the normative body.
        # -----------------------------------------------------------------
        for preamble in self._iter_nodes_by_type(root, "PREAMBLE"):
            preamble_text = normalize_block_whitespace(preamble.text)
            if not preamble_text:
                continue

            preamble_groups = self._paragraph_grouping(preamble_text)
            if not preamble_groups:
                preamble_groups = [preamble_text]

            for group_text in preamble_groups:
                chunk = self._make_chunk(
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
                if chunk is not None:
                    chunks.append(chunk)
                    sequence += 1

        # -----------------------------------------------------------------
        # 3) Handle ARTICLE nodes.
        #
        # We recursively traverse the tree rather than assuming a fixed layout
        # because parser output may vary slightly depending on the document.
        # -----------------------------------------------------------------
        for article in self._iter_nodes_by_type(root, "ARTICLE"):
            article_text = normalize_block_whitespace(article.text)
            if not article_text:
                continue

            article_meta: Dict[str, Any] = self._article_metadata(article)

            # -------------------------------------------------------------
            # Preferred path A:
            # Use SECTION nodes when available.
            #
            # Why:
            # numbered sections usually represent meaningful legal sub-rules
            # and therefore make excellent retrieval units.
            # -------------------------------------------------------------
            section_children = [
                child
                for child in article.children
                if child.node_type == "SECTION" and child.text.strip()
            ]

            if section_children:
                section_groups = self._group_sections(section_children)

                for section_group in section_groups:
                    group_text = normalize_block_whitespace(
                        "\n\n".join(
                            section.text
                            for section in section_group
                            if section.text.strip()
                        )
                    )

                    if not group_text:
                        continue

                    section_labels = [section.label for section in section_group]
                    hierarchy_path = self._merge_hierarchy_path(
                        article.hierarchy_path,
                        [f"SECTION:{label}" for label in section_labels],
                    )

                    # If a grouped section block is still too large, split it
                    # conservatively by paragraph while preserving group metadata.
                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        paragraph_groups = self._paragraph_grouping(group_text)
                        if not paragraph_groups:
                            paragraph_groups = [group_text]

                        for part_index, paragraph_group in enumerate(
                            paragraph_groups,
                            start=1,
                        ):
                            chunk = self._make_chunk(
                                sequence=sequence,
                                document_metadata=document_metadata,
                                text=paragraph_group,
                                page_start=article.page_start,
                                page_end=article.page_end,
                                source_node=article,
                                hierarchy_path=hierarchy_path,
                                chunk_reason="section_group_paragraph_split",
                                metadata={
                                    **article_meta,
                                    "node_type": "SECTION_GROUP",
                                    "section_labels": section_labels,
                                    "group_size": len(section_group),
                                    "part_index": part_index,
                                    "part_count": len(paragraph_groups),
                                    "source_span_type": "section_group_paragraph_split",
                                },
                            )
                            if chunk is not None:
                                chunks.append(chunk)
                                sequence += 1
                    else:
                        chunk = self._make_chunk(
                            sequence=sequence,
                            document_metadata=document_metadata,
                            text=group_text,
                            page_start=article.page_start,
                            page_end=article.page_end,
                            source_node=article,
                            hierarchy_path=hierarchy_path,
                            chunk_reason="section_group",
                            metadata={
                                **article_meta,
                                "node_type": "SECTION_GROUP",
                                "section_labels": section_labels,
                                "group_size": len(section_group),
                                "source_span_type": "section_group",
                            },
                        )
                        if chunk is not None:
                            chunks.append(chunk)
                            sequence += 1

                continue

            # -------------------------------------------------------------
            # Preferred path B:
            # If there are no SECTION nodes, try LETTERED_ITEM nodes.
            #
            # Why:
            # some legal documents express internal structure mainly through
            # alíneas such as "a)", "b)", "c)".
            # -------------------------------------------------------------
            lettered_children = [
                child
                for child in article.children
                if child.node_type == "LETTERED_ITEM" and child.text.strip()
            ]

            if lettered_children:
                lettered_groups = self._group_lettered_items(lettered_children)

                for lettered_group in lettered_groups:
                    group_text = normalize_block_whitespace(
                        "\n\n".join(
                            item.text
                            for item in lettered_group
                            if item.text.strip()
                        )
                    )

                    if not group_text:
                        continue

                    lettered_labels = [item.label for item in lettered_group]
                    hierarchy_path = self._merge_hierarchy_path(
                        article.hierarchy_path,
                        [f"LETTERED_ITEM:{label}" for label in lettered_labels],
                    )

                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        paragraph_groups = self._paragraph_grouping(group_text)
                        if not paragraph_groups:
                            paragraph_groups = [group_text]

                        for part_index, paragraph_group in enumerate(
                            paragraph_groups,
                            start=1,
                        ):
                            chunk = self._make_chunk(
                                sequence=sequence,
                                document_metadata=document_metadata,
                                text=paragraph_group,
                                page_start=article.page_start,
                                page_end=article.page_end,
                                source_node=article,
                                hierarchy_path=hierarchy_path,
                                chunk_reason="lettered_group_paragraph_split",
                                metadata={
                                    **article_meta,
                                    "node_type": "LETTERED_GROUP",
                                    "lettered_labels": lettered_labels,
                                    "group_size": len(lettered_group),
                                    "part_index": part_index,
                                    "part_count": len(paragraph_groups),
                                    "source_span_type": "lettered_group_paragraph_split",
                                },
                            )
                            if chunk is not None:
                                chunks.append(chunk)
                                sequence += 1
                    else:
                        chunk = self._make_chunk(
                            sequence=sequence,
                            document_metadata=document_metadata,
                            text=group_text,
                            page_start=article.page_start,
                            page_end=article.page_end,
                            source_node=article,
                            hierarchy_path=hierarchy_path,
                            chunk_reason="lettered_group",
                            metadata={
                                **article_meta,
                                "node_type": "LETTERED_GROUP",
                                "lettered_labels": lettered_labels,
                                "group_size": len(lettered_group),
                                "source_span_type": "lettered_group",
                            },
                        )
                        if chunk is not None:
                            chunks.append(chunk)
                            sequence += 1

                continue

            # -------------------------------------------------------------
            # Preferred path C:
            # No smaller structure exists, so keep the ARTICLE whole if it is
            # already reasonably sized.
            # -------------------------------------------------------------
            if len(article_text) <= self.settings.target_chunk_chars:
                chunk = self._make_chunk(
                    sequence=sequence,
                    document_metadata=document_metadata,
                    text=article_text,
                    page_start=article.page_start,
                    page_end=article.page_end,
                    source_node=article,
                    chunk_reason="direct_article",
                    metadata={
                        **article_meta,
                        "node_type": "ARTICLE",
                        "source_span_type": "article",
                    },
                )
                if chunk is not None:
                    chunks.append(chunk)
                    sequence += 1
            else:
                # ---------------------------------------------------------
                # Final fallback:
                # split an oversized article conservatively by paragraph.
                # ---------------------------------------------------------
                parts = self._split_large_text(article_text)

                for part_index, part_text in enumerate(parts, start=1):
                    chunk = self._make_chunk(
                        sequence=sequence,
                        document_metadata=document_metadata,
                        text=part_text,
                        page_start=article.page_start,
                        page_end=article.page_end,
                        source_node=article,
                        chunk_reason="article_part",
                        metadata={
                            **article_meta,
                            "node_type": "ARTICLE_PART",
                            "part_index": part_index,
                            "part_count": len(parts),
                            "source_span_type": "article_part",
                        },
                    )
                    if chunk is not None:
                        chunks.append(chunk)
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
        It makes the strategy robust against small parser variations in the
        hierarchy layout.

        Parameters
        ----------
        node : StructuralNode
            Current node.
        node_type : str
            Target node type.

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

        Design goals
        ------------
        - preserve section order
        - stay reasonably close to the configured target size
        - avoid producing very small chunks
        - merge undersized trailing groups into the previous group

        Why grouping matters
        --------------------
        Some numbered sections are too small to stand alone as useful retrieval
        units, but adjacent sections often remain semantically compatible.

        Parameters
        ----------
        sections : List[StructuralNode]
            Ordered section nodes.

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

            should_flush = (
                len(candidate_text) > self.settings.target_chunk_chars
                and len(current_text) >= self.settings.min_chunk_chars
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

        This mirrors section grouping but applies to legal alíneas.

        Parameters
        ----------
        items : List[StructuralNode]
            Ordered lettered-item nodes.

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

    def _split_large_text(self, text: str) -> List[str]:
        """
        Split a large text block conservatively by paragraph boundaries.

        This method is only used when:
        - the parser did not provide smaller usable structure
        - the full article is too large to keep as a single chunk

        Design principle
        ----------------
        Paragraph boundaries are safer semantic split points than raw character
        slicing.

        Parameters
        ----------
        text : str
            Oversized text block.

        Returns
        -------
        List[str]
            Paragraph-based parts.
        """
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return [text]

        parts: List[str] = []
        current: List[str] = []

        for paragraph in paragraphs:
            candidate = "\n\n".join(current + [paragraph]).strip()

            should_flush = (
                current
                and len(candidate) > self.settings.target_chunk_chars
                and len("\n\n".join(current)) >= self.settings.min_chunk_chars
            )

            if should_flush:
                parts.append("\n\n".join(current))
                current = [paragraph]
            else:
                current.append(paragraph)

        if current:
            parts.append("\n\n".join(current))

        if len(parts) >= 2 and len(parts[-1]) < self.settings.min_chunk_chars:
            parts[-2] = f"{parts[-2]}\n\n{parts[-1]}".strip()
            parts.pop()

        return parts

    def _paragraph_grouping(self, text: str) -> List[str]:
        """
        Group paragraph blocks into chunk-sized bundles.

        This helper is mainly used for:
        - FRONT_MATTER subdivision
        - PREAMBLE subdivision
        - oversized SECTION_GROUP subdivision
        - oversized LETTERED_GROUP subdivision

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        List[str]
            Paragraph-group bundles.
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

    def _article_metadata(self, article: StructuralNode) -> Dict[str, Any]:
        """
        Build article-level metadata for chunk export.

        Parameters
        ----------
        article : StructuralNode
            Article node.

        Returns
        -------
        Dict[str, Any]
            Article-level metadata payload.
        """
        return {
            "parent_type": article.metadata.get("parent_type"),
            "parent_label": article.metadata.get("parent_label"),
            "parent_title": article.metadata.get("parent_title"),
            "document_part": article.metadata.get("document_part"),
            "article_label": article.label,
            "article_title": article.title,
            "article_number": article.metadata.get("article_number"),
        }

    def _merge_hierarchy_path(
        self,
        base_path: Sequence[str],
        extra_items: Sequence[str],
    ) -> List[str]:
        """
        Merge a base hierarchy path with chunk-level structural hints.

        Why this helper exists
        ----------------------
        A chunk may come from an article but represent a grouped set of sections
        or lettered items. This helper preserves the source path while adding
        derived structural hints relevant to the chunk itself.

        Parameters
        ----------
        base_path : Sequence[str]
            Base source-node path.
        extra_items : Sequence[str]
            Additional structural items for the current chunk.

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
        Build enriched embedding text from visible chunk text.

        Why this helper exists
        ----------------------
        A chunk body alone may be too context-poor for retrieval. A small
        structural prefix such as article number and title often improves
        embedding quality without polluting the visible text.

        Parameters
        ----------
        source_node : StructuralNode
            Main structural source node.
        visible_text : str
            Visible chunk text.

        Returns
        -------
        str
            Enriched embedding text.
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
    ) -> Chunk | None:
        """
        Build the final Chunk object.

        Important behavior
        ------------------
        - text is normalized before export
        - empty chunks are discarded defensively
        - chunk ids remain deterministic and sequential
        - source structure is promoted into explicit chunk fields

        Parameters
        ----------
        sequence : int
            Stable chunk sequence number.
        document_metadata : DocumentMetadata
            Source document metadata.
        text : str
            Visible chunk text.
        page_start : int | None
            Chunk page start.
        page_end : int | None
            Chunk page end.
        source_node : StructuralNode
            Main structural source node.
        chunk_reason : str
            Explanation of how the chunk was created.
        metadata : Dict[str, Any]
            Flexible extra metadata.
        hierarchy_path : List[str] | None
            Optional custom hierarchy path.

        Returns
        -------
        Chunk | None
            Final chunk or None when the normalized text is empty.
        """
        normalized_text = normalize_block_whitespace(text)
        if not normalized_text:
            return None

        effective_hierarchy_path = hierarchy_path or list(source_node.hierarchy_path)

        return Chunk(
            chunk_id=f"{document_metadata.doc_id}_chunk_{sequence:04d}",
            doc_id=document_metadata.doc_id,
            strategy=self.name,
            text=normalized_text,
            text_for_embedding=self._build_text_for_embedding(
                source_node=source_node,
                visible_text=normalized_text,
            ),
            page_start=page_start,
            page_end=page_end,
            source_node_type=source_node.node_type,
            source_node_label=source_node.label,
            hierarchy_path=effective_hierarchy_path,
            chunk_reason=chunk_reason,
            char_count=len(normalized_text),
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
            Final chunk list in document order.
        """
        for index, chunk in enumerate(chunks):
            previous_chunk = chunks[index - 1] if index > 0 else None
            next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None

            chunk.prev_chunk_id = previous_chunk.chunk_id if previous_chunk else None
            chunk.next_chunk_id = next_chunk.chunk_id if next_chunk else None