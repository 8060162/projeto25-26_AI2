from __future__ import annotations

from typing import Any, Dict, Iterator, List

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.utils.text import normalize_block_whitespace, split_paragraphs


class StructureFirstChunkingStrategy(BaseChunkingStrategy):
    """
    Structure-first chunking strategy.

    Strategy philosophy:
    - trust the parser as much as possible
    - prefer the smallest meaningful structural unit already recognized
    - use SECTION groups when numbered sections exist
    - otherwise use LETTERED_ITEM groups when legal alíneas exist
    - otherwise keep the ARTICLE whole if it is reasonably sized
    - otherwise split the ARTICLE conservatively by paragraphs
    - always keep PREAMBLE content separate from the regulation body

    Why this strategy exists:
    some legal and regulatory documents are highly regular and benefit from
    structure-preserving chunking more than from generic paragraph splitting.

    Important:
    this strategy is intentionally conservative. It is designed for documents
    where the parser has already extracted a good hierarchy and where preserving
    structural fidelity is more important than aggressive balancing of chunk
    sizes.
    """

    name = "structure_first"

    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        """
        Build chunks from the parsed document tree.

        Processing order:
        1. Export PREAMBLE content first and separately
        2. Process ARTICLE nodes in traversal order
        3. Prefer SECTION_GROUP chunks when sections exist
        4. Otherwise prefer LETTERED_GROUP chunks
        5. Otherwise keep the ARTICLE whole if small enough
        6. Otherwise split the ARTICLE into paragraph-based ARTICLE_PART chunks

        Important:
        chunk sequence numbers are generated in a stable order so that JSON
        inspection, DOCX inspection, and regression testing remain easier.
        """
        chunks: List[Chunk] = []
        sequence = 1

        # -----------------------------------------------------------------
        # 1) Handle PREAMBLE nodes separately.
        #
        # Rationale:
        # the preamble / dispatch usually contains approval, revocation,
        # consultation, or contextual information that should not be mixed
        # with the normative body of the regulation.
        # -----------------------------------------------------------------
        for preamble in self._iter_nodes_by_type(root, "PREAMBLE"):
            preamble_text = normalize_block_whitespace(preamble.text)
            if not preamble_text:
                continue

            preamble_groups = self._paragraph_grouping(preamble_text)

            # Safe fallback:
            # if no grouping was produced, keep the entire preamble as one
            # chunk rather than losing it.
            if not preamble_groups:
                preamble_groups = [preamble_text]

            for group_text in preamble_groups:
                chunk = self._chunk(
                    sequence=sequence,
                    document_metadata=document_metadata,
                    text=group_text,
                    page_start=preamble.page_start,
                    page_end=preamble.page_end,
                    metadata={
                        "node_type": "PREAMBLE",
                        "label": preamble.label,
                        "document_part": preamble.metadata.get("document_part"),
                        "source_span_type": "preamble",
                    },
                )
                if chunk is not None:
                    chunks.append(chunk)
                    sequence += 1

        # -----------------------------------------------------------------
        # 2) Handle ARTICLE nodes.
        #
        # We recursively traverse the tree rather than assuming a fixed
        # DOCUMENT -> CHAPTER -> ARTICLE layout, because parser output may
        # vary depending on the document.
        # -----------------------------------------------------------------
        for article in self._iter_nodes_by_type(root, "ARTICLE"):
            article_text = normalize_block_whitespace(article.text)
            if not article_text:
                continue

            article_meta: Dict[str, Any] = {
                "parent_type": article.metadata.get("parent_type"),
                "parent_label": article.metadata.get("parent_label"),
                "parent_title": article.metadata.get("parent_title"),
                "document_part": article.metadata.get("document_part"),
                "article_label": article.label,
                "article_title": article.title,
                "article_number": article.metadata.get("article_number"),
            }

            # -------------------------------------------------------------
            # Preferred path A:
            # use SECTION nodes when available.
            #
            # Why:
            # numbered sections usually represent meaningful internal legal
            # subdivisions and are therefore excellent retrieval units.
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

                    # -----------------------------------------------------
                    # If a grouped section block is still too large, split it
                    # conservatively by paragraph while preserving the group
                    # metadata.
                    # -----------------------------------------------------
                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        paragraph_groups = self._paragraph_grouping(group_text)

                        # Safe fallback:
                        # if paragraph grouping fails for any reason, keep the
                        # original grouped section text as a single export unit.
                        if not paragraph_groups:
                            paragraph_groups = [group_text]

                        for part_index, paragraph_group in enumerate(
                            paragraph_groups,
                            start=1,
                        ):
                            chunk = self._chunk(
                                sequence=sequence,
                                document_metadata=document_metadata,
                                text=paragraph_group,
                                page_start=article.page_start,
                                page_end=article.page_end,
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
                        chunk = self._chunk(
                            sequence=sequence,
                            document_metadata=document_metadata,
                            text=group_text,
                            page_start=article.page_start,
                            page_end=article.page_end,
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
            # if there are no SECTION nodes, try LETTERED_ITEM nodes.
            #
            # Why:
            # some legal documents use alíneas such as "a)", "b)", "c)"
            # instead of numbered internal sections.
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

                    # -----------------------------------------------------
                    # In most cases lettered-item groups are small enough,
                    # but we still protect against accidental oversize output.
                    # -----------------------------------------------------
                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        paragraph_groups = self._paragraph_grouping(group_text)

                        if not paragraph_groups:
                            paragraph_groups = [group_text]

                        for part_index, paragraph_group in enumerate(
                            paragraph_groups,
                            start=1,
                        ):
                            chunk = self._chunk(
                                sequence=sequence,
                                document_metadata=document_metadata,
                                text=paragraph_group,
                                page_start=article.page_start,
                                page_end=article.page_end,
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
                        chunk = self._chunk(
                            sequence=sequence,
                            document_metadata=document_metadata,
                            text=group_text,
                            page_start=article.page_start,
                            page_end=article.page_end,
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
            # no smaller structure exists, so keep the ARTICLE whole if it
            # is already reasonably sized.
            # -------------------------------------------------------------
            if len(article_text) <= self.settings.target_chunk_chars:
                chunk = self._chunk(
                    sequence=sequence,
                    document_metadata=document_metadata,
                    text=article_text,
                    page_start=article.page_start,
                    page_end=article.page_end,
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
                    chunk = self._chunk(
                        sequence=sequence,
                        document_metadata=document_metadata,
                        text=part_text,
                        page_start=article.page_start,
                        page_end=article.page_end,
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

        return chunks

    def _iter_nodes_by_type(
        self,
        node: StructuralNode,
        node_type: str,
    ) -> Iterator[StructuralNode]:
        """
        Recursively yield all nodes of a given type.

        Why recursion is used:
        it makes the strategy robust against small parser variations in the
        hierarchy layout.
        """
        if node.node_type == node_type:
            yield node

        for child in node.children:
            yield from self._iter_nodes_by_type(child, node_type)

    def _group_sections(self, sections: List[StructuralNode]) -> List[List[StructuralNode]]:
        """
        Group adjacent SECTION nodes into chunk-sized bundles.

        Design goals:
        - preserve section order
        - stay reasonably close to the configured target size
        - avoid producing very small chunks
        - merge undersized trailing groups into the previous group

        Why grouping matters:
        some numbered sections are too small to stand alone as useful
        retrieval units, but adjacent sections often remain semantically
        compatible.
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

        # Merge a very small trailing group into the previous group.
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

        # Merge a very small trailing group into the previous group.
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
        - the parser did not provide smaller usable structure, and
        - the full article is too large to keep as a single chunk

        Design principle:
        paragraph boundaries are safer semantic split points than raw
        character slicing.
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

        # Merge a very small trailing part into the previous part.
        if len(parts) >= 2 and len(parts[-1]) < self.settings.min_chunk_chars:
            parts[-2] = f"{parts[-2]}\n\n{parts[-1]}".strip()
            parts.pop()

        return parts

    def _paragraph_grouping(self, text: str) -> List[str]:
        """
        Group paragraph blocks into chunk-sized bundles.

        This helper is mainly used for:
        - PREAMBLE subdivision
        - oversized SECTION_GROUP subdivision
        - oversized LETTERED_GROUP subdivision
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

        # Merge a very small trailing group into the previous one.
        if len(groups) >= 2 and len(groups[-1]) < self.settings.min_chunk_chars:
            groups[-2] = f"{groups[-2]}\n\n{groups[-1]}".strip()
            groups.pop()

        return groups

    def _chunk(
        self,
        sequence: int,
        document_metadata: DocumentMetadata,
        text: str,
        page_start: int | None,
        page_end: int | None,
        metadata: Dict[str, Any],
    ) -> Chunk | None:
        """
        Build the final Chunk object.

        Important:
        - text is normalized before export
        - empty chunks are discarded defensively
        - chunk ids remain deterministic and sequential

        Returning None for empty normalized text makes the strategy robust
        against parser noise or accidental empty group generation.
        """
        normalized_text = normalize_block_whitespace(text)
        if not normalized_text:
            return None

        return Chunk(
            chunk_id=f"{document_metadata.doc_id}_chunk_{sequence:04d}",
            doc_id=document_metadata.doc_id,
            strategy=self.name,
            text=normalized_text,
            page_start=page_start,
            page_end=page_end,
            metadata=metadata,
        )