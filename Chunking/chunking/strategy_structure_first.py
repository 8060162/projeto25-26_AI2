from __future__ import annotations

from typing import Iterator, List

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.utils.text import normalize_block_whitespace, split_paragraphs


class StructureFirstChunkingStrategy(BaseChunkingStrategy):
    """
    More structure-aware chunking strategy.

    Core principle:
    - prefer the smallest meaningful legal block already recognized
      by the parser
    - SECTION nodes are preferred when they exist
    - LETTERED_ITEM nodes are used when sections do not exist
    - otherwise ARTICLE nodes are used
    - PREAMBLE is kept separate from the regulation body
    - metadata should remain rich even when text stays clean

    This strategy is intentionally conservative:
    it trusts the parsed structure more than generic paragraph splitting.
    """

    name = "structure_first"

    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        sequence = 1

        for preamble in self._iter_nodes_by_type(root, "PREAMBLE"):
            if not preamble.text.strip():
                continue

            preamble_groups = self._paragraph_grouping(preamble.text)
            if not preamble_groups:
                preamble_groups = [preamble.text]

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

        for article in self._iter_nodes_by_type(root, "ARTICLE"):
            article_text = normalize_block_whitespace(article.text)
            if not article_text:
                continue

            article_meta = {
                "parent_type": article.metadata.get("parent_type"),
                "parent_label": article.metadata.get("parent_label"),
                "parent_title": article.metadata.get("parent_title"),
                "document_part": article.metadata.get("document_part"),
                "article_label": article.label,
                "article_title": article.title,
                "article_number": article.metadata.get("article_number"),
            }

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

                    if len(group_text) > self.settings.hard_max_chunk_chars:
                        paragraph_groups = self._paragraph_grouping(group_text)
                        for part_index, paragraph_group in enumerate(paragraph_groups, start=1):
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
        if node.node_type == node_type:
            yield node

        for child in node.children:
            yield from self._iter_nodes_by_type(child, node_type)

    def _group_sections(self, sections: List[StructuralNode]) -> List[List[StructuralNode]]:
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
            last_group_text = "\n\n".join(item.text for item in groups[-1] if item.text.strip())
            if len(last_group_text) < self.settings.min_chunk_chars:
                groups[-2].extend(groups[-1])
                groups.pop()

        return groups

    def _group_lettered_items(self, items: List[StructuralNode]) -> List[List[StructuralNode]]:
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

    def _split_large_text(self, text: str) -> List[str]:
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return [text]

        parts: List[str] = []
        current: List[str] = []

        for paragraph in paragraphs:
            candidate = "\n\n".join(current + [paragraph]).strip()

            if (
                current
                and len(candidate) > self.settings.target_chunk_chars
                and len("\n\n".join(current)) >= self.settings.min_chunk_chars
            ):
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

        return groups

    def _chunk(
        self,
        sequence: int,
        document_metadata: DocumentMetadata,
        text: str,
        page_start: int | None,
        page_end: int | None,
        metadata: dict,
    ) -> Chunk | None:
        """
        Build a final chunk object with stable chunk ids.
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