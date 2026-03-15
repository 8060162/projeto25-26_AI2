from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.utils.text import normalize_block_whitespace, split_paragraphs


# ============================================================================
# Local cleanup regexes
# ============================================================================
#
# Why define cleanup rules here?
# ------------------------------
# Even if extraction, normalization, and parsing already cleaned most of the
# document, the chunking strategy still needs a final safety layer.
#
# The final chunk text should be as clean as possible and should contain:
# - semantic content
#
# and should avoid carrying:
# - page counters
# - editorial lines
# - article headers
# - numbered or lettered prefixes when those are already captured in metadata
# - residual article titles accidentally merged into body text
# ============================================================================

INLINE_PAGE_COUNTER_RE = re.compile(r"^\s*\d+\s*\|\s*\d+\s*$")
LOOSE_PAGE_COUNTER_RE = re.compile(r"^\s*\d+\s*\|\s*\d+\s*[\-–—.]?\s*$")
LEADING_PAGE_MARKER_RE = re.compile(r"^\s*Pág\.\s*\d+\s*", re.IGNORECASE)

DR_EDITORIAL_RE = re.compile(
    r"^\s*(N\.?\s*º|PARTE\s+[A-Z]|Diário da República)\b",
    re.IGNORECASE,
)

ARTICLE_HEADER_RE = re.compile(
    r"^\s*(?:ARTIGO|ART\.?)\s+\d+(?:\.\d+)?\s*(?:\.?\s*[ºo°])?\s*(?:[—–\-:]\s*.*)?$",
    re.IGNORECASE,
)

NUMBERED_ITEM_PREFIX_RE = re.compile(
    r"^\s*(?:n\.?\s*[ºo]\s*)?\d+(?:\.\d+)*(?:\.\s+|\)\s+|\s+[—–\-]\s+|\s+)",
    re.IGNORECASE,
)

LETTERED_ITEM_PREFIX_RE = re.compile(r"^\s*[a-z]\)\s+", re.IGNORECASE)

UPPERCASE_HEADING_RE = re.compile(
    r"^[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ0-9 /().,\-–—ºª]{3,}$"
)

PROSE_START_RE = re.compile(
    r"^\s*(?:O|A|Os|As|No|Na|Nos|Nas|Em|Para|Por|Quando|Sempre|Caso|Se|Nos\s+termos|Deve|Devem|Pode|Podem|É|São)\b",
    re.IGNORECASE,
)


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
    - keep PREAMBLE separate from normative article content
    - avoid chunking FRONT_MATTER by default

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
        1. Skip FRONT_MATTER by default
        2. Export PREAMBLE separately when present
        3. Process ARTICLE nodes in document order
        4. Prefer whole-article chunks for short articles
        5. Prefer grouped SECTION chunks for larger articles
        6. Fall back to grouped LETTERED_ITEM chunks when sections do not exist
        7. Fall back to paragraph grouping when no finer structure is available
        8. Link neighboring chunks when enabled

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
        # 1) FRONT_MATTER
        #
        # Current policy:
        # do not create retrieval chunks from FRONT_MATTER by default.
        #
        # Why?
        # ----
        # FRONT_MATTER typically contains:
        # - institutional branding
        # - cover/title-page content
        # - dispatch headings
        # - index-like content
        #
        # These are usually poor retrieval units and tend to introduce noise.
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # 2) PREAMBLE
        #
        # The preamble often carries useful contextual or normative information
        # such as approval context, amendment context, or introductory clauses.
        # It should remain separate from article chunks.
        # -----------------------------------------------------------------
        for preamble in self._iter_nodes_by_type(root, "PREAMBLE"):
            preamble_text = self._clean_chunk_text(
                preamble.text,
                remove_structural_prefixes=False,
                source_node=preamble,
            )
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
                    source_node_type_override="PREAMBLE",
                    source_node_label_override=preamble.label,
                    chunk_reason="preamble_group",
                    metadata={
                        **self._base_document_metadata(document_metadata),
                        "document_part": preamble.metadata.get("document_part"),
                        "source_span_type": "preamble",
                        "source_node_id": preamble.node_id,
                        "parent_node_id": preamble.parent_node_id,
                    },
                )
                if chunk is not None:
                    chunks.append(chunk)
                    sequence += 1

        # -----------------------------------------------------------------
        # 3) ARTICLE content
        # -----------------------------------------------------------------
        for article in self._iter_nodes_by_type(root, "ARTICLE"):
            article_text = self._clean_chunk_text(
                article.text,
                remove_structural_prefixes=False,
                source_node=article,
            )
            if not article_text:
                continue

            article_meta = self._article_metadata(
                document_metadata=document_metadata,
                article=article,
            )

            # -------------------------------------------------------------
            # Case A:
            # Keep short articles as a single chunk.
            #
            # Why:
            # - a short article is often already a complete legal unit
            # - preserving the whole article improves interpretability
            # - unnecessary splitting can reduce retrieval quality
            # -------------------------------------------------------------
            if len(article_text) <= self.settings.target_chunk_chars:
                chunk = self._make_chunk(
                    sequence=sequence,
                    document_metadata=document_metadata,
                    text=article_text,
                    page_start=article.page_start,
                    page_end=article.page_end,
                    source_node=article,
                    source_node_type_override="ARTICLE",
                    source_node_label_override=article.label,
                    chunk_reason="direct_article",
                    metadata={
                        **article_meta,
                        "source_span_type": "article",
                    },
                )
                if chunk is not None:
                    chunks.append(chunk)
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
                    group_text = self._join_cleaned_node_texts(
                        nodes=group,
                        remove_structural_prefixes=True,
                    )
                    if not group_text:
                        continue

                    group_labels = [section.label for section in group]
                    hierarchy_path = self._merge_hierarchy_path(
                        getattr(article, "hierarchy_path", []),
                        [f"SECTION:{label}" for label in group_labels],
                    )
                    group_page_start, group_page_end = self._group_page_range(group)

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
                                page_start=group_page_start,
                                page_end=group_page_end,
                                source_node=article,
                                source_node_type_override="SECTION_GROUP",
                                source_node_label_override=",".join(group_labels),
                                hierarchy_path=hierarchy_path,
                                chunk_reason="grouped_sections_paragraph_split",
                                metadata={
                                    **article_meta,
                                    "section_labels": group_labels,
                                    "part_index": part_index,
                                    "part_count": len(paragraph_groups),
                                    "source_span_type": "article_section_group_paragraph_split",
                                    "source_node_ids": [section.node_id for section in group],
                                    "group_page_start": group_page_start,
                                    "group_page_end": group_page_end,
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
                            page_start=group_page_start,
                            page_end=group_page_end,
                            source_node=article,
                            source_node_type_override="SECTION_GROUP",
                            source_node_label_override=",".join(group_labels),
                            hierarchy_path=hierarchy_path,
                            chunk_reason="grouped_sections",
                            metadata={
                                **article_meta,
                                "section_labels": group_labels,
                                "source_span_type": "article_section_group",
                                "source_node_ids": [section.node_id for section in group],
                                "group_page_start": group_page_start,
                                "group_page_end": group_page_end,
                            },
                        )
                        if chunk is not None:
                            chunks.append(chunk)
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
                    group_text = self._join_cleaned_node_texts(
                        nodes=group,
                        remove_structural_prefixes=True,
                    )
                    if not group_text:
                        continue

                    group_labels = [item.label for item in group]
                    hierarchy_path = self._merge_hierarchy_path(
                        getattr(article, "hierarchy_path", []),
                        [f"LETTERED_ITEM:{label}" for label in group_labels],
                    )
                    group_page_start, group_page_end = self._group_page_range(group)

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
                                page_start=group_page_start,
                                page_end=group_page_end,
                                source_node=article,
                                source_node_type_override="LETTERED_GROUP",
                                source_node_label_override=",".join(group_labels),
                                hierarchy_path=hierarchy_path,
                                chunk_reason="grouped_lettered_items_paragraph_split",
                                metadata={
                                    **article_meta,
                                    "lettered_labels": group_labels,
                                    "part_index": part_index,
                                    "part_count": len(paragraph_groups),
                                    "source_span_type": "article_lettered_group_paragraph_split",
                                    "source_node_ids": [item.node_id for item in group],
                                    "group_page_start": group_page_start,
                                    "group_page_end": group_page_end,
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
                            page_start=group_page_start,
                            page_end=group_page_end,
                            source_node=article,
                            source_node_type_override="LETTERED_GROUP",
                            source_node_label_override=",".join(group_labels),
                            hierarchy_path=hierarchy_path,
                            chunk_reason="grouped_lettered_items",
                            metadata={
                                **article_meta,
                                "lettered_labels": group_labels,
                                "source_span_type": "article_lettered_group",
                                "source_node_ids": [item.node_id for item in group],
                                "group_page_start": group_page_start,
                                "group_page_end": group_page_end,
                            },
                        )
                        if chunk is not None:
                            chunks.append(chunk)
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
                chunk = self._make_chunk(
                    sequence=sequence,
                    document_metadata=document_metadata,
                    text=paragraph_group,
                    page_start=article.page_start,
                    page_end=article.page_end,
                    source_node=article,
                    source_node_type_override="ARTICLE_PART",
                    source_node_label_override=article.label,
                    chunk_reason="fallback_paragraph_split",
                    metadata={
                        **article_meta,
                        "part_index": part_index,
                        "part_count": len(paragraph_groups),
                        "source_span_type": "article_paragraph_group",
                    },
                )
                if chunk is not None:
                    chunks.append(chunk)
                    sequence += 1

        if self.settings.enable_chunk_neighbor_links:
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

            current_text = self._join_cleaned_node_texts(
                nodes=current_group,
                remove_structural_prefixes=True,
            )
            candidate_text = self._join_cleaned_node_texts(
                nodes=current_group + [section],
                remove_structural_prefixes=True,
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

        # Merge an undersized trailing group into the previous one when possible.
        if len(groups) >= 2:
            last_group_text = self._join_cleaned_node_texts(
                nodes=groups[-1],
                remove_structural_prefixes=True,
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

            current_text = self._join_cleaned_node_texts(
                nodes=current_group,
                remove_structural_prefixes=True,
            )
            candidate_text = self._join_cleaned_node_texts(
                nodes=current_group + [item],
                remove_structural_prefixes=True,
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
            last_group_text = self._join_cleaned_node_texts(
                nodes=groups[-1],
                remove_structural_prefixes=True,
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
        - preamble needs subdivision

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

        # Merge a very small trailing group into the previous one.
        if len(groups) >= 2 and len(groups[-1]) < self.settings.min_chunk_chars:
            groups[-2] = f"{groups[-2]}\n\n{groups[-1]}".strip()
            groups.pop()

        return groups

    def _clean_chunk_text(
        self,
        text: str,
        remove_structural_prefixes: bool,
        source_node: Optional[StructuralNode] = None,
    ) -> str:
        """
        Apply final visible-text cleanup before chunk export.

        Why this helper exists
        ----------------------
        The parser intentionally preserves structure-rich text.
        However, visible chunk text should be cleaner and should avoid carrying
        structure markers that are better represented in metadata.

        Cleanup goals
        -------------
        - remove obvious residual page/editorial noise
        - optionally remove numbered or lettered legal prefixes
        - remove residual article title at the beginning of the text
        - normalize whitespace conservatively
        - preserve semantic body text

        Parameters
        ----------
        text : str
            Input text.

        remove_structural_prefixes : bool
            When True, remove prefixes such as:
            - "1. "
            - "2) "
            - "a) "

        source_node : Optional[StructuralNode]
            Structural source node, when available. This allows the cleanup
            layer to defensively remove a residual article title prefix that
            should live in metadata rather than visible chunk text.

        Returns
        -------
        str
            Cleaned chunk text.
        """
        if not text:
            return ""

        kept_lines: List[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            line = LEADING_PAGE_MARKER_RE.sub("", line).strip()
            if not line:
                continue

            if INLINE_PAGE_COUNTER_RE.match(line):
                continue

            if LOOSE_PAGE_COUNTER_RE.match(line):
                continue

            if DR_EDITORIAL_RE.match(line):
                continue

            if ARTICLE_HEADER_RE.match(line):
                continue

            # Remove numbering/alínea prefixes only when the structural label
            # is already preserved in metadata.
            if remove_structural_prefixes:
                line = NUMBERED_ITEM_PREFIX_RE.sub("", line).strip()
                line = LETTERED_ITEM_PREFIX_RE.sub("", line).strip()

            if not line:
                continue

            kept_lines.append(line)

        cleaned = "\n".join(kept_lines)
        cleaned = normalize_block_whitespace(cleaned).strip()

        # Defensive final cleanup:
        # if the parser still left the article title at the beginning of the
        # visible text, remove it here so chunk text remains content-focused.
        cleaned = self._remove_residual_title_prefix(
            cleaned_text=cleaned,
            source_node=source_node,
        )

        # Remove a short uppercase heading residue at the beginning when it
        # clearly behaves like a title fragment rather than body text.
        cleaned = self._remove_leading_heading_residue(cleaned)

        return cleaned.strip()

    def _remove_residual_title_prefix(
        self,
        cleaned_text: str,
        source_node: Optional[StructuralNode],
    ) -> str:
        """
        Remove a residual source-node title prefix from the visible chunk text.

        Why this helper exists
        ----------------------
        Even after parser fixes, some edge cases may still leave:

            "ÂMBITO O presente regulamento ..."
            "PAGAMENTO FORA DE PRAZO O não pagamento ..."

        at the beginning of ARTICLE text.

        That title should live in metadata, not in the visible chunk text.

        Parameters
        ----------
        cleaned_text : str
            Already cleaned candidate visible text.

        source_node : Optional[StructuralNode]
            Structural source node.

        Returns
        -------
        str
            Text with the residual title prefix removed when it can be done
            safely.
        """
        if not cleaned_text or source_node is None:
            return cleaned_text

        title = (source_node.title or "").strip()
        if not title:
            return cleaned_text

        # Normalize both sides conservatively before comparing.
        normalized_title = normalize_block_whitespace(title).strip()
        normalized_text = normalize_block_whitespace(cleaned_text).strip()

        if not normalized_title or not normalized_text:
            return normalized_text

        # Direct prefix removal:
        # "ÂMBITO O presente regulamento ..."
        if normalized_text.lower().startswith(normalized_title.lower() + " "):
            stripped = normalized_text[len(normalized_title):].strip()
            if stripped:
                return stripped

        # Also handle the case where title occupies the first line exactly.
        lines = normalized_text.splitlines()
        if lines:
            first_line = lines[0].strip()
            if first_line.lower() == normalized_title.lower():
                remaining = "\n".join(lines[1:]).strip()
                if remaining:
                    return remaining

        return normalized_text

    def _remove_leading_heading_residue(self, text: str) -> str:
        """
        Remove a short uppercase heading-like fragment at the beginning.

        Why this helper exists
        ----------------------
        Some edge cases still produce text that starts with a heading fragment
        that is not meaningful as visible chunk content.

        Safety behavior
        ---------------
        This helper is intentionally conservative:
        - it only examines the first line
        - it only removes clearly heading-like uppercase fragments
        - it only removes them when followed by ordinary prose

        Parameters
        ----------
        text : str
            Candidate visible chunk text.

        Returns
        -------
        str
            Cleaned text.
        """
        if not text:
            return ""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        first_line = lines[0]
        if (
            UPPERCASE_HEADING_RE.match(first_line)
            and len(first_line.split()) <= 8
            and len(lines) >= 2
            and PROSE_START_RE.match(lines[1])
        ):
            return "\n".join(lines[1:]).strip()

        return text.strip()

    def _join_cleaned_node_texts(
        self,
        nodes: Sequence[StructuralNode],
        remove_structural_prefixes: bool,
    ) -> str:
        """
        Join multiple node texts after applying visible-text cleanup.

        Parameters
        ----------
        nodes : Sequence[StructuralNode]
            Nodes whose text should be merged.

        remove_structural_prefixes : bool
            Whether numbered / lettered prefixes should be stripped.

        Returns
        -------
        str
            Clean merged text.
        """
        parts: List[str] = []

        for node in nodes:
            cleaned = self._clean_chunk_text(
                node.text,
                remove_structural_prefixes=remove_structural_prefixes,
                source_node=node,
            )
            if cleaned:
                parts.append(cleaned)

        return normalize_block_whitespace("\n\n".join(parts)).strip()

    def _group_page_range(
        self,
        nodes: Sequence[StructuralNode],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Compute the real page range for a grouped chunk.

        Why this helper exists
        ----------------------
        Grouped section/alínea chunks should not blindly inherit the full
        article page range when a narrower and more accurate group span is
        available.

        Parameters
        ----------
        nodes : Sequence[StructuralNode]
            Nodes included in the chunk.

        Returns
        -------
        Tuple[Optional[int], Optional[int]]
            page_start, page_end
        """
        starts = [node.page_start for node in nodes if node.page_start is not None]
        ends = [node.page_end for node in nodes if node.page_end is not None]

        page_start = min(starts) if starts else None
        page_end = max(ends) if ends else page_start

        return page_start, page_end

    def _base_document_metadata(
        self,
        document_metadata: DocumentMetadata,
    ) -> Dict[str, Any]:
        """
        Build basic document-level metadata shared by all chunks.

        Parameters
        ----------
        document_metadata : DocumentMetadata
            Source document metadata.

        Returns
        -------
        Dict[str, Any]
            Base metadata dictionary.
        """
        return {
            "document_title": document_metadata.title,
            "source_file_name": document_metadata.file_name,
            "source_path": document_metadata.source_path,
        }

    def _article_metadata(
        self,
        document_metadata: DocumentMetadata,
        article: StructuralNode,
    ) -> Dict[str, Any]:
        """
        Build article-level metadata for chunk export.

        Important design choice
        -----------------------
        Metadata is the primary carrier of structural identity, while chunk text
        remains as clean as possible.

        Parameters
        ----------
        document_metadata : DocumentMetadata
            Source document metadata.

        article : StructuralNode
            Article node.

        Returns
        -------
        Dict[str, Any]
            Article-level metadata useful downstream.
        """
        return {
            **self._base_document_metadata(document_metadata),
            "node_type": article.node_type,
            "label": article.label,
            "article_title": article.title,
            "article_number": article.metadata.get("article_number"),
            "parent_type": article.metadata.get("parent_type"),
            "parent_label": article.metadata.get("parent_label"),
            "parent_title": article.metadata.get("parent_title"),
            "document_part": article.metadata.get("document_part"),
            "source_node_id": article.node_id,
            "parent_node_id": article.parent_node_id,
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
        metadata: Optional[Dict[str, Any]] = None,
        source_node_type_override: Optional[str] = None,
    ) -> str:
        """
        Build enriched embedding text from clean visible chunk text.

        Why this helper exists
        ----------------------
        In legal retrieval, the chunk body alone may be too context-poor.
        A short contextual prefix such as article label and title often improves
        retrieval quality while still keeping the visible chunk text clean.

        Parameters
        ----------
        source_node : StructuralNode
            Main structural source of the chunk.

        visible_text : str
            Clean visible chunk text.

        metadata : Optional[Dict[str, Any]]
            Chunk metadata, useful for group-specific context.

        source_node_type_override : Optional[str]
            Optional logical chunk source type, such as:
            - SECTION_GROUP
            - LETTERED_GROUP
            - ARTICLE_PART

        Returns
        -------
        str
            Context-enriched text suitable for embedding.
        """
        visible_text = normalize_block_whitespace(visible_text)
        if not visible_text:
            return ""

        header_parts: List[str] = []
        metadata = metadata or {}
        effective_type = source_node_type_override or source_node.node_type

        article_number = source_node.metadata.get("article_number")
        article_title = source_node.title or metadata.get("article_title") or ""

        if article_number:
            header_parts.append(f"Artigo {article_number}")
        elif source_node.label:
            header_parts.append(source_node.label)

        if article_title:
            header_parts.append(article_title)

        if effective_type == "SECTION_GROUP":
            section_labels = metadata.get("section_labels") or []
            if section_labels:
                header_parts.append("Secções " + ", ".join(section_labels))

        if effective_type == "LETTERED_GROUP":
            lettered_labels = metadata.get("lettered_labels") or []
            if lettered_labels:
                header_parts.append("Alíneas " + ", ".join(lettered_labels))

        if effective_type == "ARTICLE_PART":
            part_index = metadata.get("part_index")
            part_count = metadata.get("part_count")
            if part_index is not None and part_count is not None:
                header_parts.append(f"Parte {part_index}/{part_count}")

        if effective_type == "PREAMBLE":
            header_parts = ["Preamble"]

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
        source_node_type_override: str | None = None,
        source_node_label_override: str | None = None,
    ) -> Chunk | None:
        """
        Build a final Chunk object.

        Important behavior
        ------------------
        - visible chunk text is normalized before export
        - enriched embedding text is generated separately
        - empty chunks are discarded defensively
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

        source_node_type_override : str | None
            Explicit source-node type for the chunk payload.

        source_node_label_override : str | None
            Explicit source-node label for the chunk payload.

        Returns
        -------
        Chunk | None
            Final chunk object or None when visible text is empty.
        """
        visible_text = normalize_block_whitespace(text)
        if not visible_text:
            return None

        effective_hierarchy_path = list(
            hierarchy_path or getattr(source_node, "hierarchy_path", [])
        )

        text_for_embedding = (
            self._build_text_for_embedding(
                source_node=source_node,
                visible_text=visible_text,
                metadata=metadata,
                source_node_type_override=source_node_type_override,
            )
            if self.settings.include_text_for_embedding
            else visible_text
        )

        return Chunk(
            chunk_id=f"{document_metadata.doc_id}_chunk_{sequence:04d}",
            doc_id=document_metadata.doc_id,
            strategy=self.name,
            text=visible_text,
            text_for_embedding=text_for_embedding,
            page_start=page_start,
            page_end=page_end,
            source_node_type=source_node_type_override or getattr(source_node, "node_type", ""),
            source_node_label=source_node_label_override or getattr(source_node, "label", ""),
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