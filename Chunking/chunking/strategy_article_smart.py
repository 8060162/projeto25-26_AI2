from __future__ import annotations

from typing import Iterator, List

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.utils.text import normalize_block_whitespace, split_paragraphs


class ArticleSmartChunkingStrategy(BaseChunkingStrategy):
    """
    Article-aware chunking strategy.

    Core idea:
    - Use ARTICLE as the primary semantic unit.
    - If an article is short enough, keep it as one chunk.
    - If an article is long, prefer splitting by SECTION nodes already
      extracted by the structure parser.
    - If needed, use LETTERED_ITEM nodes as a secondary structure layer.
    - If there is still no usable structure, fall back to paragraph grouping.

    Why this fits regulatory documents well:
    - articles are usually stable and legally meaningful
    - section numbering often maps to coherent sub-rules
    - metadata remains easy to interpret downstream
    - the strategy stays resilient when structure is only partial
    """

    name = "article_smart"

    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        sequence = 1

        # -------------------------------------------------------------
        # 1) Chunk the preamble / dispatch separately if present.
        #
        # We intentionally keep it apart from the regulation body.
        # -------------------------------------------------------------
        for preamble in self._iter_nodes_by_type(root, "PREAMBLE"):
            if not preamble.text.strip():
                continue

            preamble_groups = self._paragraph_grouping(preamble.text)
            if not preamble_groups:
                preamble_groups = [preamble.text]

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

        # -------------------------------------------------------------
        # 2) Chunk article content.
        # -------------------------------------------------------------
        for article in self._iter_nodes_by_type(root, "ARTICLE"):
            article_text = normalize_block_whitespace(article.text)
            if not article_text:
                continue

            article_meta = self._article_metadata(article)

            # ---------------------------------------------------------
            # Case A:
            # Short article -> keep as a single chunk.
            #
            # We keep this simple because a short article is often already
            # semantically complete and legally meaningful.
            # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # Case B:
            # Larger article -> prefer SECTION nodes if available.
            # ---------------------------------------------------------
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

                    # -------------------------------------------------
                    # If a grouped section block is still too large,
                    # try paragraph grouping as a safe fallback.
                    # -------------------------------------------------
                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        for paragraph_group in self._paragraph_grouping(group_text):
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

            # ---------------------------------------------------------
            # Case C:
            # No SECTION nodes -> try LETTERED_ITEM nodes.
            #
            # This can help articles that use "a)", "b)", "c)" structure
            # without formal numbered sections.
            # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # Case D:
            # No usable structure -> paragraph-based fallback.
            # ---------------------------------------------------------
            for paragraph_group in self._paragraph_grouping(article_text):
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
        """
        if node.node_type == node_type:
            yield node

        for child in node.children:
            yield from self._iter_nodes_by_type(child, node_type)

    def _group_sections(self, sections: List[StructuralNode]) -> List[List[StructuralNode]]:
        """
        Group section nodes into chunk-sized bundles.

        Grouping policy:
        - try to stay around target_chunk_chars
        - avoid creating very small final groups when possible
        - preserve original section order

        This is intentionally simple and predictable.
        """
        if not sections:
            return []

        groups: List[List[StructuralNode]] = []
        current_group: List[StructuralNode] = []

        for section in sections:
            if not current_group:
                current_group = [section]
                continue

            current_text = "\n\n".join(item.text for item in current_group if item.text.strip())
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

        # Merge a very small trailing group into the previous one.
        if len(groups) >= 2:
            last_group_text = "\n\n".join(item.text for item in groups[-1] if item.text.strip())
            if len(last_group_text) < self.settings.min_chunk_chars:
                groups[-2].extend(groups[-1])
                groups.pop()

        return groups

    def _group_lettered_items(self, items: List[StructuralNode]) -> List[List[StructuralNode]]:
        """
        Group lettered items into chunk-sized bundles.

        This mirrors section grouping, but on a smaller structural unit.
        """
        if not items:
            return []

        groups: List[List[StructuralNode]] = []
        current_group: List[StructuralNode] = []

        for item in items:
            if not current_group:
                current_group = [item]
                continue

            current_text = "\n\n".join(node.text for node in current_group if node.text.strip())
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
            last_group_text = "\n\n".join(node.text for node in groups[-1] if node.text.strip())
            if len(last_group_text) < self.settings.min_chunk_chars:
                groups[-2].extend(groups[-1])
                groups.pop()

        return groups

    def _paragraph_grouping(self, text: str) -> List[str]:
        """
        Group paragraphs into chunk-sized blocks.

        This is the fallback when article sections are unavailable or when
        a section group is still too large.
        """
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return []

        groups: List[str] = []
        current: List[str] = []

        for paragraph in paragraphs:
            candidate = "\n\n".join(current + [paragraph]).strip()

            if (
                current
                and len(candidate) > self.settings.target_chunk_chars
                and len("\n\n".join(current)) >= self.settings.min_chunk_chars
            ):
                groups.append("\n\n".join(current))
                current = [paragraph]
            else:
                current.append(paragraph)

        if current:
            groups.append("\n\n".join(current))

        # Merge a very small trailing paragraph group into the previous one.
        if len(groups) >= 2 and len(groups[-1]) < self.settings.min_chunk_chars:
            groups[-2] = f"{groups[-2]}\n\n{groups[-1]}".strip()
            groups.pop()

        return groups

    def _article_metadata(self, node: StructuralNode) -> dict:
        """
        Build rich article-level metadata without polluting the chunk text.
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
        metadata: dict,
    ) -> Chunk:
        """
        Build a final chunk object.

        Chunk text is normalized, but metadata remains the main vehicle for:
        - article identity
        - parent structure
        - source span type
        - page range
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