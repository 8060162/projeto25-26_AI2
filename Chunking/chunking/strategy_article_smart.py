from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.config.patterns import (
    COVER_NOISE_RE,
    FRONT_MATTER_RE,
    INLINE_PAGE_COUNTER_LINE_RE,
    LETTERED_ITEM_PREFIX_RE,
    LEADING_PAGE_MARKER_RE,
    LOOSE_PAGE_COUNTER_LINE_RE,
    NUMBERED_ITEM_PREFIX_RE,
    PROSE_START_RE,
    SIGNATURE_LINE_RE,
    SUSPICIOUS_GARBLED_LINE_RE,
    UPPERCASE_HEAVY_RE,
)
from Chunking.utils.text import (
    fold_editorial_text,
    has_suspicious_truncated_ending,
    normalize_block_whitespace,
    split_paragraphs,
)


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

DR_EDITORIAL_RE = re.compile(
    r"^\s*(N\.?\s*º|PARTE\s+[A-Z]|Diário da República)\b",
    re.IGNORECASE,
)

ARTICLE_HEADER_RE = re.compile(
    r"^\s*(?:ARTIGO|ART\.?)\s+\d+(?:\.\d+)?\s*(?:\.?\s*[ºo°])?\s*(?:[—–\-:]\s*.*)?$",
    re.IGNORECASE,
)

TITLE_SEPARATOR_RE = re.compile(r"\s*\|\s*")

LINE_NUMBERED_SPLIT_RE = re.compile(
    r"^\s*(?:n\.?\s*[ºo]\s*)?\d+(?:\.\d+)*(?:\.\s+|\)\s+|\s+[—–\-]\s+|\s+)",
    re.IGNORECASE,
)

LINE_LETTERED_SPLIT_RE = re.compile(r"^\s*[a-z]\)\s+", re.IGNORECASE)

INLINE_NUMBERED_SPLIT_RE = re.compile(
    r"(?:n\.?\s*[ºo]\s*)?\d+(?:\.\d+)*(?:\.\s+|\)\s+|\s+[—–\-]\s+)",
    re.IGNORECASE,
)

INLINE_LETTERED_SPLIT_RE = re.compile(r"[a-z]\)\s+", re.IGNORECASE)

ACCESS_FOOTNOTE_RE = re.compile(
    r"^\s*(?:\(?\d+\)?\s+)?"
    r"(?:Acess[íi]vel|Dispon[íi]vel|Publicado|Publicada|Publicados|Publicadas)\b",
    re.IGNORECASE,
)

FOOTNOTE_URL_RE = re.compile(
    r"^\s*(?:\(?\d+\)?\s+)?(?:cf\.\s+|ver\s+)?(?:.*(?:https?://|www\.))",
    re.IGNORECASE,
)

INLINE_HEADING_WITH_PROSE_RE = re.compile(
    r"^(?P<heading>[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ0-9 /().,\-–—ºª]{5,80})\s+"
    r"(?P<body>(?:O|A|Os|As|No|Na|Nos|Nas|Em|Para|Por|Quando|Sempre|Caso|Se|"
    r"Nos\s+termos|Deve|Devem|Pode|Podem|É|São)\b.*)$"
)

PERSON_NAME_LINE_RE = re.compile(
    r"^\s*[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç]+"
    r"(?:\s+[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç]+){0,3}"
    r"\.?\s*$"
)

TITLE_FRAGMENT_LINE_RE = re.compile(
    r"^\s*(?:Regulamento|Despacho|Anexo)\b(?:\s+.*)?$",
    re.IGNORECASE,
)

TRUNCATED_TAIL_CONNECTOR_RE = re.compile(
    r"(?:\b(?:e|ou|que|de|do|da|dos|das|para|por)\s*)+$",
    re.IGNORECASE,
)

LEADING_WEAK_CONTINUATION_RE = re.compile(
    r"^\s*(?:"
    r"(?:e|ou|que|de|do|da|dos|das|para|por|com|sem|em|na|no|nas|nos)\b"
    r"|[a-zà-ÿ]"
    r")",
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

            preamble_chunk_reason = "preamble_group"
            if len(preamble_text) <= self.settings.target_chunk_chars:
                preamble_groups = [preamble_text]
            else:
                preamble_groups, split_mode = self._split_oversized_text(
                    preamble_text
                )
                preamble_groups = self._apply_split_overlap(
                    preamble_groups,
                    preamble_text,
                )
                preamble_chunk_reason = f"preamble_{split_mode}"

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
                    chunk_reason=preamble_chunk_reason,
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
                section_groups = self._refine_structural_groups_for_export(
                    article=article,
                    groups=section_groups,
                )

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

                    if len(group_text) > self.settings.target_chunk_chars:
                        split_candidate_text = self._join_cleaned_node_texts(
                            nodes=group,
                            remove_structural_prefixes=False,
                        )
                        paragraph_groups, split_mode = self._split_grouped_node_texts(
                            nodes=group,
                            source_node=article,
                        )

                        paragraph_groups = self._apply_split_overlap(
                            paragraph_groups,
                            split_candidate_text,
                        )

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
                                chunk_reason=f"grouped_sections_{split_mode}",
                                metadata={
                                    **article_meta,
                                    "section_labels": group_labels,
                                    "part_index": part_index,
                                    "part_count": len(paragraph_groups),
                                    "source_span_type": (
                                        f"article_section_group_{split_mode}"
                                    ),
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
                lettered_groups = self._refine_structural_groups_for_export(
                    article=article,
                    groups=lettered_groups,
                )

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

                    if len(group_text) > self.settings.target_chunk_chars:
                        split_candidate_text = self._join_cleaned_node_texts(
                            nodes=group,
                            remove_structural_prefixes=False,
                        )
                        paragraph_groups, split_mode = self._split_grouped_node_texts(
                            nodes=group,
                            source_node=article,
                        )

                        paragraph_groups = self._apply_split_overlap(
                            paragraph_groups,
                            split_candidate_text,
                        )

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
                                chunk_reason=f"grouped_lettered_items_{split_mode}",
                                metadata={
                                    **article_meta,
                                    "lettered_labels": group_labels,
                                    "part_index": part_index,
                                    "part_count": len(paragraph_groups),
                                    "source_span_type": (
                                        f"article_lettered_group_{split_mode}"
                                    ),
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
            paragraph_groups, split_mode = self._split_oversized_text(article_text)
            paragraph_groups = self._apply_split_overlap(
                paragraph_groups,
                article_text,
            )

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
                    chunk_reason=f"fallback_{split_mode}",
                    metadata={
                        **article_meta,
                        "part_index": part_index,
                        "part_count": len(paragraph_groups),
                        "source_span_type": f"article_{split_mode}",
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
                merged_text = self._join_cleaned_node_texts(
                    nodes=groups[-2] + groups[-1],
                    remove_structural_prefixes=True,
                )
                if len(merged_text) <= self.settings.target_chunk_chars:
                    groups[-2].extend(groups[-1])
                    groups.pop()

        return groups

    def _refine_structural_groups_for_export(
        self,
        article: StructuralNode,
        groups: Sequence[Sequence[StructuralNode]],
    ) -> List[List[StructuralNode]]:
        """
        Avoid multi-node grouped exports when article structure is incomplete.

        Why this helper exists
        ----------------------
        Parser integrity metadata already exposes when an article capture is
        structurally incomplete. In that situation, grouping multiple SECTION
        or LETTERED_ITEM nodes can crystallize weak definitions or damaged
        continuations into one apparently valid chunk. Export should stay
        conservative and fall back to single-node groups.

        Parameters
        ----------
        article : StructuralNode
            Parent article for the candidate groups.

        groups : Sequence[Sequence[StructuralNode]]
            Candidate grouped structural nodes.

        Returns
        -------
        List[List[StructuralNode]]
            Groups safe enough for final export.
        """
        if not groups:
            return []

        normalized_groups = [list(group) for group in groups if group]

        normalized_groups = self._reattach_weak_group_lead_ins(
            article=article,
            groups=normalized_groups,
        )

        if not article.metadata.get("is_structurally_incomplete", False):
            return normalized_groups

        refined_groups: List[List[StructuralNode]] = []
        for group in normalized_groups:
            normalized_group = [node for node in group if node is not None]
            if not normalized_group:
                continue

            if len(normalized_group) == 1:
                refined_groups.append(normalized_group)
                continue

            if self._should_preserve_incomplete_structural_group(
                article=article,
                group=normalized_group,
            ):
                refined_groups.append(normalized_group)
                continue

            for node in normalized_group:
                refined_groups.append([node])

        return refined_groups

    def _reattach_weak_group_lead_ins(
        self,
        article: StructuralNode,
        groups: Sequence[Sequence[StructuralNode]],
    ) -> List[List[StructuralNode]]:
        """
        Reattach weak introductory structural groups to the continuation group.

        Why this helper exists
        ----------------------
        Some articles still produce a short numbered or lettered lead-in that
        ends with a visible continuation cue such as ``:`` or ``ou:`` while the
        real substantive content starts in the immediately following structural
        unit. Exporting that lead-in as a standalone group creates a weak chunk,
        even when the parser already preserved the right nearby nodes.

        Parameters
        ----------
        article : StructuralNode
            Parent article for the candidate groups.

        groups : Sequence[Sequence[StructuralNode]]
            Candidate grouped structural nodes.

        Returns
        -------
        List[List[StructuralNode]]
            Groups with weak lead-ins reattached when safe.
        """
        if len(groups) < 2:
            return [list(group) for group in groups if group]

        refined_groups: List[List[StructuralNode]] = [list(groups[0])]

        for next_group in groups[1:]:
            normalized_next_group = [node for node in next_group if node is not None]
            if not normalized_next_group:
                continue

            previous_group = refined_groups[-1]
            if self._should_reattach_weak_group_lead_in(
                article=article,
                current_group=previous_group,
                next_group=normalized_next_group,
            ):
                previous_group.extend(normalized_next_group)
                continue

            refined_groups.append(normalized_next_group)

        return refined_groups

    def _should_reattach_weak_group_lead_in(
        self,
        article: StructuralNode,
        current_group: Sequence[StructuralNode],
        next_group: Sequence[StructuralNode],
    ) -> bool:
        """
        Decide whether a weak structural lead-in should stay with the next group.

        Parameters
        ----------
        article : StructuralNode
            Parent article for the candidate groups.

        current_group : Sequence[StructuralNode]
            Current structural group under review.

        next_group : Sequence[StructuralNode]
            Immediately following structural group.

        Returns
        -------
        bool
            True when the current group should be merged into the next one.
        """
        if not current_group or not next_group:
            return False

        trailing_node = current_group[-1]
        leading_node = next_group[0]
        trailing_text = self._clean_chunk_text(
            trailing_node.text,
            remove_structural_prefixes=True,
            source_node=article,
        )
        if not self._looks_like_weak_group_lead_in(trailing_text):
            return False

        if not self._is_structural_group_continuation(
            trailing_node=trailing_node,
            leading_node=leading_node,
        ):
            return False

        merged_visible_text = self._join_cleaned_node_texts(
            nodes=list(current_group) + list(next_group),
            remove_structural_prefixes=True,
        )
        if not merged_visible_text:
            return False

        return len(merged_visible_text) <= self.settings.hard_max_chunk_chars

    def _looks_like_weak_group_lead_in(self, text: str) -> bool:
        """
        Detect whether a grouped structural unit behaves like a weak lead-in.

        Parameters
        ----------
        text : str
            Final visible text for the candidate trailing node.

        Returns
        -------
        bool
            True when the text looks introductory and semantically incomplete.
        """
        normalized_text = normalize_block_whitespace(text).strip()
        if not normalized_text:
            return False

        if not normalized_text.endswith(":"):
            return False

        return len(normalized_text) < self.settings.target_chunk_chars

    def _is_structural_group_continuation(
        self,
        trailing_node: StructuralNode,
        leading_node: StructuralNode,
    ) -> bool:
        """
        Decide whether two neighboring structural nodes form a continuation pair.

        Parameters
        ----------
        trailing_node : StructuralNode
            Last node of the current group.

        leading_node : StructuralNode
            First node of the next group.

        Returns
        -------
        bool
            True when the next node continues the previous structural lead-in.
        """
        if trailing_node.node_type != leading_node.node_type:
            return False

        trailing_label = str(trailing_node.label).strip()
        leading_label = str(leading_node.label).strip()
        if not trailing_label or not leading_label:
            return False

        if trailing_node.node_type == "SECTION":
            return leading_label.startswith(f"{trailing_label}.")

        if trailing_node.node_type == "LETTERED_ITEM":
            return (
                len(trailing_label) == 1
                and len(leading_label) == 1
                and trailing_label.isalpha()
                and leading_label.isalpha()
                and ord(leading_label.lower()) == ord(trailing_label.lower()) + 1
            )

        return False

    def _should_preserve_incomplete_structural_group(
        self,
        article: StructuralNode,
        group: Sequence[StructuralNode],
    ) -> bool:
        """
        Decide whether an incomplete multi-node group is still safe to export.

        Parameters
        ----------
        article : StructuralNode
            Parent article for the candidate group.

        group : Sequence[StructuralNode]
            Candidate grouped structural nodes.

        Returns
        -------
        bool
            True when keeping the multi-node group is safer than splitting it.
        """
        if len(group) < 2:
            return False

        first_node = group[0]
        second_node = group[1]
        first_text = self._clean_chunk_text(
            first_node.text,
            remove_structural_prefixes=True,
            source_node=article,
        )
        if not self._looks_like_weak_group_lead_in(first_text):
            return False

        return self._is_structural_group_continuation(
            trailing_node=first_node,
            leading_node=second_node,
        )

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
                merged_text = self._join_cleaned_node_texts(
                    nodes=groups[-2] + groups[-1],
                    remove_structural_prefixes=True,
                )
                if len(merged_text) <= self.settings.target_chunk_chars:
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
        split_target_chars = self._split_target_chars()

        for paragraph in paragraphs:
            candidate = "\n\n".join(current + [paragraph]).strip()

            should_flush = (
                current
                and len(candidate) > split_target_chars
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
            merged_text = f"{groups[-2]}\n\n{groups[-1]}".strip()
            if len(merged_text) <= split_target_chars:
                groups[-2] = merged_text
                groups.pop()

        return groups

    def _split_oversized_text(self, text: str) -> Tuple[List[str], str]:
        """
        Split oversized text using paragraphs first, then legal split cues.

        Why this helper exists
        ----------------------
        Paragraph grouping is the preferred fallback because paragraph breaks
        are usually safe semantic boundaries. However, some long legal blocks
        survive as a single paragraph even when clear numbering or alinea cues
        are still visible inside the text.

        Parameters
        ----------
        text : str
            Candidate text to split.

        Returns
        -------
        Tuple[List[str], str]
            Chunk parts and the split mode used.
        """
        paragraph_groups = self._paragraph_grouping(text)
        if not paragraph_groups:
            return [text], "paragraph_split"

        if len(paragraph_groups) > 1:
            return self._repair_weak_split_boundaries(paragraph_groups), "paragraph_split"

        only_group = paragraph_groups[0].strip()
        if len(only_group) <= self._split_target_chars():
            return paragraph_groups, "paragraph_split"

        legal_groups = self._split_by_legal_signals(only_group)
        if len(legal_groups) >= 2:
            return self._repair_weak_split_boundaries(legal_groups), "legal_signal_split"

        return self._repair_weak_split_boundaries(paragraph_groups), "paragraph_split"

    def _split_by_legal_signals(self, text: str) -> List[str]:
        """
        Split text using visible legal numbering or alinea cues.

        Safety behavior
        ---------------
        This helper remains conservative:
        - existing line-based list structure is preferred
        - inline splitting only triggers on trustworthy boundary cues
        - final parts are regrouped to stay near chunk-size targets

        Parameters
        ----------
        text : str
            Candidate oversized text.

        Returns
        -------
        List[str]
            Deterministically split parts, or the original text when no
            trustworthy legal boundaries are found.
        """
        if not text:
            return []

        line_blocks = self._split_line_based_legal_blocks(text)
        if len(line_blocks) >= 2:
            regrouped_line_blocks = self._group_text_parts(line_blocks)
            if self._is_usable_split_result(regrouped_line_blocks):
                return regrouped_line_blocks

        clause_blocks = self._split_clause_like_lines(text)
        if len(clause_blocks) >= 2:
            regrouped_clause_blocks = self._group_text_parts(clause_blocks)
            if self._is_usable_split_result(regrouped_clause_blocks):
                return regrouped_clause_blocks

        inline_blocks = self._split_inline_legal_blocks(text)
        if len(inline_blocks) >= 2:
            regrouped_inline_blocks = self._group_text_parts(inline_blocks)
            if self._is_usable_split_result(regrouped_inline_blocks):
                return regrouped_inline_blocks

        return [text.strip()]

    def _is_usable_split_result(self, parts: Sequence[str]) -> bool:
        """
        Decide whether a candidate split result is safe to export.

        Parameters
        ----------
        parts : Sequence[str]
            Candidate split parts after regrouping.

        Returns
        -------
        bool
            True when the result yields multiple non-empty parts and each part
            remains within the hard chunk ceiling.
        """
        normalized_parts = [
            normalize_block_whitespace(part).strip()
            for part in parts
            if normalize_block_whitespace(part).strip()
        ]
        if len(normalized_parts) < 2:
            return False

        return all(
            len(part) <= self.settings.hard_max_chunk_chars
            for part in normalized_parts
        )

    def _split_line_based_legal_blocks(self, text: str) -> List[str]:
        """
        Split text when numbered items already start on separate lines.

        Parameters
        ----------
        text : str
            Candidate text.

        Returns
        -------
        List[str]
            Extracted line-based legal blocks.
        """
        blocks: List[str] = []
        current_lines: List[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            starts_new_block = (
                bool(current_lines)
                and (
                    LINE_NUMBERED_SPLIT_RE.match(line) is not None
                    or LINE_LETTERED_SPLIT_RE.match(line) is not None
                )
            )

            if starts_new_block:
                block_text = normalize_block_whitespace("\n".join(current_lines)).strip()
                if block_text:
                    blocks.append(block_text)
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            block_text = normalize_block_whitespace("\n".join(current_lines)).strip()
            if block_text:
                blocks.append(block_text)

        return blocks

    def _split_clause_like_lines(self, text: str) -> List[str]:
        """
        Split clause-like legal lines when numbering is absent but boundaries remain.

        Why this helper exists
        ----------------------
        Some regulations encode lists as one introductory line followed by
        clause lines separated only by line breaks and punctuation such as
        ":" or ";". Those lines are still meaningful legal boundaries even
        when numbering or letter markers are missing.

        Parameters
        ----------
        text : str
            Candidate oversized text.

        Returns
        -------
        List[str]
            Clause-like blocks when the pattern looks trustworthy.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return []

        blocks: List[str] = []
        current_lines: List[str] = []
        boundary_count = 0

        for line in lines:
            starts_new_block = False

            if current_lines:
                previous_line = current_lines[-1].strip()
                starts_new_block = self._is_clause_line_boundary(
                    previous_line=previous_line,
                    current_line=line,
                )

            if starts_new_block:
                block_text = normalize_block_whitespace("\n".join(current_lines)).strip()
                if block_text:
                    blocks.append(block_text)
                current_lines = [line]
                boundary_count += 1
            else:
                current_lines.append(line)

        if current_lines:
            block_text = normalize_block_whitespace("\n".join(current_lines)).strip()
            if block_text:
                blocks.append(block_text)

        if boundary_count < 2:
            return []

        return blocks

    def _is_clause_line_boundary(
        self,
        previous_line: str,
        current_line: str,
    ) -> bool:
        """
        Decide whether one line starts a new clause-like legal block.

        Parameters
        ----------
        previous_line : str
            Previous non-empty line in the current candidate block.

        current_line : str
            Current line being inspected.

        Returns
        -------
        bool
            True when the transition behaves like a clause boundary.
        """
        if not previous_line or not current_line:
            return False

        if previous_line[-1] not in {":", ";", "."}:
            return False

        if len(current_line) < 24:
            return False

        if self._is_standalone_article_header_line(current_line):
            return False

        if LINE_NUMBERED_SPLIT_RE.match(current_line) is not None:
            return False

        if LINE_LETTERED_SPLIT_RE.match(current_line) is not None:
            return False

        first_character = current_line[0]
        if not (first_character.isupper() or first_character.isdigit()):
            return False

        return True

    def _split_inline_legal_blocks(self, text: str) -> List[str]:
        """
        Split one long block using inline numbering or alinea signals.

        Parameters
        ----------
        text : str
            Candidate text.

        Returns
        -------
        List[str]
            Inline-derived legal blocks.
        """
        normalized_text = normalize_block_whitespace(text).strip()
        if not normalized_text:
            return []

        split_points: List[int] = []

        for pattern in (INLINE_NUMBERED_SPLIT_RE, INLINE_LETTERED_SPLIT_RE):
            for match in pattern.finditer(normalized_text):
                start_index = match.start()
                if self._is_valid_inline_legal_split(
                    normalized_text,
                    start_index,
                ):
                    split_points.append(start_index)

        unique_split_points = sorted(set(split_points))
        if not unique_split_points:
            return [normalized_text]

        parts: List[str] = []
        previous_index = 0

        for split_index in unique_split_points:
            part = normalized_text[previous_index:split_index].strip()
            if part:
                parts.append(part)
            previous_index = split_index

        trailing_part = normalized_text[previous_index:].strip()
        if trailing_part:
            parts.append(trailing_part)

        return parts

    def _is_valid_inline_legal_split(
        self,
        text: str,
        split_index: int,
    ) -> bool:
        """
        Decide whether an inline numbering cue is a trustworthy split point.

        Parameters
        ----------
        text : str
            Full candidate text.

        split_index : int
            Candidate start index of the legal cue.

        Returns
        -------
        bool
            True when the boundary looks structurally meaningful.
        """
        if split_index <= 0:
            return False

        prefix = text[:split_index].rstrip()
        if not prefix:
            return False

        previous_char = prefix[-1]
        return previous_char in {":", ";", "\n"}

    def _group_text_parts(self, parts: Sequence[str]) -> List[str]:
        """
        Group already split text parts into chunk-sized bundles.

        Parameters
        ----------
        parts : Sequence[str]
            Ordered split candidates.

        Returns
        -------
        List[str]
            Regrouped chunk-sized bundles.
        """
        if not parts:
            return []

        groups: List[str] = []
        current_parts: List[str] = []
        split_target_chars = self._split_target_chars()

        for part in parts:
            cleaned_part = normalize_block_whitespace(part).strip()
            if not cleaned_part:
                continue

            candidate = "\n\n".join(current_parts + [cleaned_part]).strip()
            current_text = "\n\n".join(current_parts).strip()

            should_flush = bool(current_parts) and (
                len(candidate) > self.settings.hard_max_chunk_chars
                or (
                    len(candidate) > split_target_chars
                    and len(current_text) >= self.settings.min_chunk_chars
                )
            )

            if should_flush:
                groups.append(current_text)
                current_parts = [cleaned_part]
            else:
                current_parts.append(cleaned_part)

        if current_parts:
            groups.append("\n\n".join(current_parts).strip())

        if len(groups) >= 2 and len(groups[-1]) < self.settings.min_chunk_chars:
            merged_text = f"{groups[-2]}\n\n{groups[-1]}".strip()
            if len(merged_text) <= split_target_chars:
                groups[-2] = merged_text
                groups.pop()

        return self._repair_weak_split_boundaries(groups)

    def _split_target_chars(self) -> int:
        """
        Reserve headroom for visible overlap when split chunks need continuity.

        Returns
        -------
        int
            Effective target size for split-only grouping decisions.
        """
        if self.settings.overlap_chars <= 0:
            return self.settings.target_chunk_chars

        reserved_overlap = min(self.settings.overlap_chars, 160)
        reserved_separator = 2
        reserved_padding = 24
        effective_target = (
            self.settings.hard_max_chunk_chars
            - reserved_overlap
            - reserved_separator
            - reserved_padding
        )

        return max(
            self.settings.min_chunk_chars,
            min(self.settings.target_chunk_chars, effective_target),
        )

    def _split_grouped_node_texts(
        self,
        nodes: Sequence[StructuralNode],
        source_node: StructuralNode,
    ) -> Tuple[List[str], str]:
        """
        Split grouped structural nodes without losing legal split markers early.

        Why this helper exists
        ----------------------
        Grouped SECTION or LETTERED_ITEM chunks may still exceed the target
        size. Their final visible text intentionally removes structural
        prefixes, but the split heuristics need those markers to recover
        trustworthy boundaries before the final cleanup step.

        Parameters
        ----------
        nodes : Sequence[StructuralNode]
            Adjacent structural nodes that currently form one grouped chunk.

        source_node : StructuralNode
            Parent article used for final visible-text cleanup.

        Returns
        -------
        Tuple[List[str], str]
            Final cleaned chunk parts and the split mode used.
        """
        visible_text = self._join_cleaned_node_texts(
            nodes=nodes,
            remove_structural_prefixes=True,
        )
        if not visible_text:
            return [], "paragraph_split"

        if len(visible_text) <= self.settings.target_chunk_chars:
            return [visible_text], "paragraph_split"

        split_candidate_text = self._join_cleaned_node_texts(
            nodes=nodes,
            remove_structural_prefixes=False,
        )
        if not split_candidate_text:
            return [visible_text], "paragraph_split"

        raw_parts, split_mode = self._split_oversized_text(split_candidate_text)
        cleaned_parts: List[str] = []

        for raw_part in raw_parts:
            cleaned_part = self._clean_chunk_text(
                raw_part,
                remove_structural_prefixes=True,
                source_node=source_node,
            )
            if cleaned_part:
                cleaned_parts.append(cleaned_part)

        if not cleaned_parts:
            return [visible_text], "paragraph_split"

        return self._repair_weak_split_boundaries(cleaned_parts), split_mode

    def _repair_weak_split_boundaries(self, parts: Sequence[str]) -> List[str]:
        """
        Merge neighboring split parts when the boundary is semantically weak.

        Why this helper exists
        ----------------------
        Size-based grouping and legal-marker splits can still create awkward
        boundaries where one part ends like a truncated fragment or the next
        part starts like a weak continuation. This helper reattaches those
        cases conservatively when doing so stays within the hard size ceiling.

        Parameters
        ----------
        parts : Sequence[str]
            Ordered split parts after initial grouping.

        Returns
        -------
        List[str]
            Split parts with weak boundaries merged when safe.
        """
        normalized_parts = [
            normalize_block_whitespace(part).strip()
            for part in parts
            if normalize_block_whitespace(part).strip()
        ]
        if len(normalized_parts) < 2:
            return normalized_parts

        repaired_parts: List[str] = [normalized_parts[0]]

        for current_part in normalized_parts[1:]:
            previous_part = repaired_parts[-1]
            boundary_is_weak = (
                has_suspicious_truncated_ending(previous_part)
                or self._starts_with_weak_continuation(current_part)
            )
            joiner = " " if boundary_is_weak else "\n\n"
            candidate = f"{previous_part}{joiner}{current_part}".strip()

            if (
                len(candidate) <= self.settings.hard_max_chunk_chars
                and boundary_is_weak
            ):
                repaired_parts[-1] = candidate
                continue

            repaired_parts.append(current_part)

        return repaired_parts

    def _starts_with_weak_continuation(self, text: str) -> bool:
        """
        Detect whether one split part starts like a dangling continuation.

        Parameters
        ----------
        text : str
            Candidate split part.

        Returns
        -------
        bool
            True when the part opens with a weak continuation fragment.
        """
        normalized_text = normalize_block_whitespace(text).strip()
        if not normalized_text:
            return False

        if self._part_starts_with_legal_marker(normalized_text):
            return False

        return LEADING_WEAK_CONTINUATION_RE.match(normalized_text) is not None

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

            if INLINE_PAGE_COUNTER_LINE_RE.match(line):
                continue

            if LOOSE_PAGE_COUNTER_LINE_RE.match(line):
                continue

            if DR_EDITORIAL_RE.match(line):
                continue

            if self._is_standalone_article_header_line(line):
                continue

            if self._looks_like_chunk_footnote_line(line):
                continue

            if self._looks_like_garbled_chunk_line(line):
                continue

            # Remove numbering/alínea prefixes only when the structural label
            # is already preserved in metadata.
            if remove_structural_prefixes:
                line = NUMBERED_ITEM_PREFIX_RE.sub("", line).strip()
                line = LETTERED_ITEM_PREFIX_RE.sub("", line).strip()

            if not line:
                continue

            kept_lines.append(line)

        kept_lines = self._trim_trailing_garbled_chunk_block(kept_lines)
        kept_lines = self._trim_trailing_chunk_residue_lines(kept_lines)

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
        cleaned = self._remove_inline_heading_residue(cleaned)

        return cleaned.strip()

    def _trim_trailing_chunk_residue_lines(self, lines: List[str]) -> List[str]:
        """
        Remove short trailing sign-off or title residue from visible chunk text.

        Parameters
        ----------
        lines : List[str]
            Already filtered visible lines.

        Returns
        -------
        List[str]
            Lines with conservative tail cleanup applied.
        """
        if not lines:
            return []

        end_index = len(lines)
        tail_removed_count = 0
        strong_tail_signal_count = 0

        while end_index > 0 and self._looks_like_chunk_tail_residue_line(
            lines[end_index - 1]
        ):
            if self._is_strong_chunk_tail_residue_line(lines[end_index - 1]):
                strong_tail_signal_count += 1

            end_index -= 1
            tail_removed_count += 1

        if tail_removed_count == 0 or strong_tail_signal_count == 0:
            return lines

        trimmed_lines = list(lines[:end_index])
        if not trimmed_lines:
            return []

        trimmed_last_line = self._trim_truncated_tail_connector(trimmed_lines[-1])
        if trimmed_last_line:
            trimmed_lines[-1] = trimmed_last_line
        else:
            trimmed_lines.pop()

        return trimmed_lines

    def _looks_like_chunk_tail_residue_line(self, line: str) -> bool:
        """
        Detect weak trailing lines that behave like sign-off or title residue.

        Parameters
        ----------
        line : str
            Candidate trailing visible line.

        Returns
        -------
        bool
            True when the line behaves more like residue than body prose.
        """
        if not line:
            return False

        stripped = normalize_block_whitespace(line).strip()
        if not stripped:
            return False

        if self._is_strong_chunk_tail_residue_line(stripped):
            return True

        if PROSE_START_RE.match(stripped) and len(stripped.split()) >= 4:
            return False

        return (
            len(stripped.split()) <= 6
            and UPPERCASE_HEAVY_RE.match(stripped) is not None
            and stripped[-1] not in {".", ",", ";", ":"}
        )

    def _is_strong_chunk_tail_residue_line(self, line: str) -> bool:
        """
        Detect strong tail-residue signals safe to trim at chunk end.

        Parameters
        ----------
        line : str
            Candidate trailing visible line.

        Returns
        -------
        bool
            True when the line strongly behaves like sign-off or title residue.
        """
        if not line:
            return False

        stripped = normalize_block_whitespace(line).strip()
        if not stripped:
            return False

        if SIGNATURE_LINE_RE.match(stripped):
            return True

        if COVER_NOISE_RE.match(stripped):
            return True

        if FRONT_MATTER_RE.match(stripped) and len(stripped.split()) <= 10:
            return True

        if PERSON_NAME_LINE_RE.match(stripped):
            return True

        if TITLE_FRAGMENT_LINE_RE.match(stripped) and len(stripped.split()) <= 8:
            return True

        folded_line = fold_editorial_text(stripped)
        if folded_line in {"regulamento", "regulamento de", "regulamento do"}:
            return True

        return False

    def _trim_truncated_tail_connector(self, line: str) -> str:
        """
        Remove a dangling connector left behind after tail-residue trimming.

        Parameters
        ----------
        line : str
            Last remaining visible line.

        Returns
        -------
        str
            Line with a weak truncated connector removed conservatively.
        """
        if not line:
            return ""

        stripped = normalize_block_whitespace(line).strip()
        if not stripped:
            return ""

        if stripped.endswith((".", ";", ":", "?", "!")):
            return stripped

        return TRUNCATED_TAIL_CONNECTOR_RE.sub("", stripped).strip()

    def _is_standalone_article_header_line(self, line: str) -> bool:
        """
        Decide whether one line is a real article header, not a body citation.

        Parameters
        ----------
        line : str
            Candidate visible line.

        Returns
        -------
        bool
            True when the line behaves like a standalone article header.
        """
        normalized_line = normalize_block_whitespace(line).strip()
        if not normalized_line:
            return False

        if ARTICLE_HEADER_RE.match(normalized_line) is None:
            return False

        suffix_match = re.match(
            r"^\s*(?:ARTIGO|ART\.?)\s+\d+(?:\.\d+)?\s*(?:\.?\s*[ºo°])?\s*(?:[—–\-:]\s*(.*))?$",
            normalized_line,
            re.IGNORECASE,
        )
        if suffix_match is None:
            return False

        suffix = (suffix_match.group(1) or "").strip()
        if not suffix:
            return True

        # Standalone headers may carry a short title suffix, but citation-heavy
        # spans such as "artigo 46.o -B do Decreto-Lei ..." belong to body text.
        if len(suffix.split()) > 8:
            return False

        if suffix[-1] in {".", ",", ";"}:
            return False

        return True

    def _looks_like_chunk_footnote_line(self, line: str) -> bool:
        """
        Detect footnote-style residue that should not remain in final chunk text.

        Parameters
        ----------
        line : str
            Candidate visible line.

        Returns
        -------
        bool
            True when the line behaves like an editorial footnote.
        """
        if not line:
            return False

        return bool(
            ACCESS_FOOTNOTE_RE.match(line)
            or FOOTNOTE_URL_RE.match(line)
        )

    def _looks_like_garbled_chunk_line(self, line: str) -> bool:
        """
        Detect strongly suspicious garbled residue in final chunk cleanup.

        Parameters
        ----------
        line : str
            Candidate visible line.

        Returns
        -------
        bool
            True when the line looks clearly corrupted.
        """
        if not line:
            return False

        if len(line) < 10:
            return False

        if SUSPICIOUS_GARBLED_LINE_RE.match(line):
            return True

        total_len = len(line)
        alpha_count = sum(1 for ch in line if ch.isalpha())
        whitespace_count = sum(1 for ch in line if ch.isspace())
        symbol_like_count = sum(
            1
            for ch in line
            if not ch.isalnum() and not ch.isspace()
        )

        alpha_ratio = alpha_count / max(total_len, 1)
        symbol_ratio = symbol_like_count / max(total_len, 1)

        if alpha_ratio < 0.20 and symbol_ratio > 0.35 and whitespace_count <= 1:
            return True

        return False

    def _trim_trailing_garbled_chunk_block(self, lines: List[str]) -> List[str]:
        """
        Remove a trailing multi-line garbled block from visible chunk text.

        Parameters
        ----------
        lines : List[str]
            Already filtered visible lines.

        Returns
        -------
        List[str]
            Lines with a weak trailing garbled block removed when safe.
        """
        if not lines:
            return []

        start_index = len(lines)
        while start_index > 0 and self._looks_like_trailing_garbled_chunk_line(
            lines[start_index - 1]
        ):
            start_index -= 1

        candidate = lines[start_index:]
        if len(candidate) < 4:
            return lines

        strong_signal_count = sum(
            1
            for line in candidate
            if self._is_strong_trailing_garbled_chunk_line(line)
        )
        short_line_count = sum(1 for line in candidate if len(line.split()) <= 2)
        previous_line = lines[start_index - 1] if start_index > 0 else ""

        if short_line_count < len(candidate) - 1:
            return lines

        if strong_signal_count < 2:
            return lines

        if not previous_line.endswith(":") and len(candidate) < 6:
            return lines

        return lines[:start_index]

    def _looks_like_trailing_garbled_chunk_line(self, line: str) -> bool:
        """
        Detect short weak lines that may belong to a garbled trailing block.

        Parameters
        ----------
        line : str
            Candidate trailing visible line.

        Returns
        -------
        bool
            True when the line behaves like weak garbled residue.
        """
        if not line:
            return False

        stripped = normalize_block_whitespace(line).strip()
        if not stripped or len(stripped) > 24:
            return False

        if self._is_standalone_article_header_line(stripped):
            return False

        if PROSE_START_RE.match(stripped) and len(stripped.split()) >= 3:
            return False

        if self._is_strong_trailing_garbled_chunk_line(stripped):
            return True

        return len(stripped.split()) <= 1 and not stripped.endswith(".")

    def _is_strong_trailing_garbled_chunk_line(self, line: str) -> bool:
        """
        Detect strong garbling signals inside a short trailing residue line.

        Parameters
        ----------
        line : str
            Candidate trailing visible line.

        Returns
        -------
        bool
            True when the line has strong OCR-like corruption signals.
        """
        if not line:
            return False

        stripped = normalize_block_whitespace(line).strip()
        if not stripped:
            return False

        if len(stripped) <= 2:
            return True

        if re.fullmatch(r"[_=+\-*/\\|(){}\[\]]+", stripped):
            return True

        if re.fullmatch(r"\d+(?:\s+\d+){0,2}", stripped):
            return True

        alpha_chars = [character for character in stripped if character.isalpha()]
        if not alpha_chars:
            return False

        vowel_ratio = (
            sum(
                1
                for character in alpha_chars
                if character.lower() in "aeiouáàâãéèêíìîóòôõúùû"
            )
            / len(alpha_chars)
        )
        uppercase_ratio = (
            sum(1 for character in alpha_chars if character.isupper()) / len(alpha_chars)
        )
        lowercase_ratio = (
            sum(1 for character in alpha_chars if character.islower()) / len(alpha_chars)
        )

        return (
            (len(alpha_chars) <= 3 and len(stripped.split()) <= 1)
            or (uppercase_ratio >= 0.85 and vowel_ratio <= 0.30)
            or (
                0.20 <= uppercase_ratio <= 0.80
                and 0.20 <= lowercase_ratio <= 0.80
                and vowel_ratio <= 0.28
            )
        )

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
            UPPERCASE_HEAVY_RE.match(first_line)
            and len(first_line.split()) <= 8
            and len(lines) >= 2
            and PROSE_START_RE.match(lines[1])
        ):
            return "\n".join(lines[1:]).strip()

        return text.strip()

    def _remove_inline_heading_residue(self, text: str) -> str:
        """
        Remove heading residue glued to prose on the first visible line.

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

        match = INLINE_HEADING_WITH_PROSE_RE.match(lines[0])
        if match is None:
            return text.strip()

        heading = match.group("heading").strip()
        body = match.group("body").strip()

        if len(heading.split()) > 8:
            return text.strip()

        if self._is_standalone_article_header_line(heading):
            return text.strip()

        lines[0] = body
        return "\n".join(lines).strip()

    def _apply_split_overlap(
        self,
        parts: Sequence[str],
        source_text: str,
    ) -> List[str]:
        """
        Apply real overlap to split outputs so later parts keep context.

        Parameters
        ----------
        parts : Sequence[str]
            Ordered split parts after cleanup.

        source_text : str
            Unsplitted source text used to recover introductory context.

        Returns
        -------
        List[str]
            Split parts with visible overlap when it is safe and useful.
        """
        normalized_parts = [
            normalize_block_whitespace(part).strip()
            for part in parts
            if normalize_block_whitespace(part).strip()
        ]
        if len(normalized_parts) < 2:
            return normalized_parts

        if self.settings.overlap_chars <= 0:
            return normalized_parts

        intro_context = self._extract_intro_overlap(source_text)
        overlapped_parts: List[str] = [normalized_parts[0]]

        for index in range(1, len(normalized_parts)):
            current_part = normalized_parts[index]
            previous_part = normalized_parts[index - 1]
            overlap_prefix = ""

            if self._part_starts_with_legal_marker(current_part):
                overlap_prefix = intro_context

            if not overlap_prefix:
                overlap_prefix = self._extract_trailing_overlap(previous_part)

            overlapped_parts.append(
                self._prepend_overlap_prefix(
                    current_part=current_part,
                    overlap_prefix=overlap_prefix,
                )
            )

        return overlapped_parts

    def _extract_intro_overlap(self, source_text: str) -> str:
        """
        Extract a short introductory context to repeat across legal splits.

        Parameters
        ----------
        source_text : str
            Full split source text.

        Returns
        -------
        str
            Introductory context trimmed to overlap size.
        """
        normalized_text = normalize_block_whitespace(source_text).strip()
        if not normalized_text:
            return ""

        split_points: List[int] = []
        for pattern in (INLINE_NUMBERED_SPLIT_RE, INLINE_LETTERED_SPLIT_RE):
            for match in pattern.finditer(normalized_text):
                split_index = match.start()
                if self._is_valid_inline_legal_split(normalized_text, split_index):
                    split_points.append(split_index)

        if not split_points:
            return ""

        intro_text = normalized_text[: min(split_points)].strip()
        if not intro_text:
            return ""

        if intro_text[-1] not in {":", ";"}:
            return ""

        return self._trim_overlap_text(intro_text)

    def _extract_trailing_overlap(self, text: str) -> str:
        """
        Extract a short trailing context window from the previous chunk part.

        Parameters
        ----------
        text : str
            Previous split part.

        Returns
        -------
        str
            Trailing overlap text.
        """
        paragraphs = split_paragraphs(text)
        candidate_blocks: List[str] = []

        if paragraphs:
            candidate_blocks.append(paragraphs[-1])

        candidate_blocks.append(text)

        seen_candidates = set()
        for block in candidate_blocks:
            normalized_block = normalize_block_whitespace(block).strip()
            if not normalized_block or normalized_block in seen_candidates:
                continue

            seen_candidates.add(normalized_block)

            if (
                len(normalized_block) <= self.settings.overlap_chars
                and self._is_coherent_overlap_prefix(normalized_block)
            ):
                return normalized_block

            trailing_sentence = self._extract_trailing_sentence(normalized_block)
            if (
                trailing_sentence
                and len(trailing_sentence) <= self.settings.overlap_chars
                and self._is_coherent_overlap_prefix(trailing_sentence)
            ):
                return trailing_sentence

        return ""

    def _trim_overlap_text(self, text: str) -> str:
        """
        Trim overlap text to a safe size while preserving readable boundaries.

        Parameters
        ----------
        text : str
            Candidate overlap text.

        Returns
        -------
        str
            Trimmed overlap text.
        """
        cleaned_text = normalize_block_whitespace(text).strip()
        if not cleaned_text:
            return ""

        if len(cleaned_text) <= self.settings.overlap_chars:
            return cleaned_text

        trimmed = cleaned_text[-self.settings.overlap_chars :].strip()
        first_space = trimmed.find(" ")
        if first_space > 0 and first_space < len(trimmed) - 1:
            trimmed = trimmed[first_space + 1 :].strip()

        return trimmed

    def _extract_trailing_sentence(self, text: str) -> str:
        """
        Extract the last sentence-like unit when it fits overlap safely.

        Parameters
        ----------
        text : str
            Candidate overlap source text.

        Returns
        -------
        str
            Last sentence-like unit, or an empty string when none is reliable.
        """
        normalized_text = normalize_block_whitespace(text).strip()
        if not normalized_text:
            return ""

        sentence_boundaries = list(re.finditer(r"[.!?;:]\s+", normalized_text))
        if not sentence_boundaries:
            return ""

        trailing_sentence = normalized_text[sentence_boundaries[-1].end() :].strip()
        return trailing_sentence

    def _is_coherent_overlap_prefix(self, text: str) -> bool:
        """
        Decide whether overlap text starts cleanly enough to be shown again.

        Parameters
        ----------
        text : str
            Candidate overlap prefix.

        Returns
        -------
        bool
            True when the overlap behaves like a visible context prefix.
        """
        normalized_text = normalize_block_whitespace(text).strip()
        if not normalized_text:
            return False

        first_character = normalized_text[0]
        if first_character in {",", ";", ":", "-", "—", "–", ")", "]"}:
            return False

        if normalized_text[:1].islower():
            return False

        return True

    def _part_starts_with_legal_marker(self, text: str) -> bool:
        """
        Decide whether a split part starts with a legal list marker.

        Parameters
        ----------
        text : str
            Candidate split part.

        Returns
        -------
        bool
            True when the text begins with numbered or lettered cues.
        """
        if not text:
            return False

        return bool(
            LINE_NUMBERED_SPLIT_RE.match(text)
            or LINE_LETTERED_SPLIT_RE.match(text)
        )

    def _prepend_overlap_prefix(
        self,
        current_part: str,
        overlap_prefix: str,
    ) -> str:
        """
        Prepend overlap text to one split part without breaking size ceilings.

        Parameters
        ----------
        current_part : str
            Current split part.

        overlap_prefix : str
            Context to prepend.

        Returns
        -------
        str
            Current part with overlap applied when safe.
        """
        normalized_part = normalize_block_whitespace(current_part).strip()
        normalized_prefix = normalize_block_whitespace(overlap_prefix).strip()

        if not normalized_prefix:
            return normalized_part

        if normalized_part.startswith(normalized_prefix):
            return normalized_part

        candidate = f"{normalized_prefix}\n\n{normalized_part}".strip()
        if len(candidate) <= self.settings.hard_max_chunk_chars:
            return candidate

        available_chars = self.settings.hard_max_chunk_chars - len(normalized_part) - 2
        if available_chars < 24:
            return normalized_part

        reduced_prefix = self._trim_overlap_text(normalized_prefix[-available_chars:])
        if not reduced_prefix:
            return normalized_part

        candidate = f"{reduced_prefix}\n\n{normalized_part}".strip()
        if len(candidate) <= self.settings.hard_max_chunk_chars:
            return candidate

        return normalized_part

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
            "is_structurally_incomplete": article.metadata.get(
                "is_structurally_incomplete",
                False,
            ),
            "truncation_signals": list(article.metadata.get("truncation_signals", [])),
            "integrity_warnings": list(article.metadata.get("integrity_warnings", [])),
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

    def _build_meta_text(
        self,
        source_node: StructuralNode,
        visible_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_node_type_override: Optional[str] = None,
    ) -> str:
        """
        Build optional structure-enriched text from clean visible chunk text.

        Why this helper exists
        ----------------------
        In legal retrieval, the chunk body alone may be too context-poor.
        A short contextual prefix such as article label and title can help
        manual inspection while still keeping the visible chunk text clean.

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
            Context-enriched text suitable for auxiliary inspection.
        """
        visible_text = normalize_block_whitespace(visible_text)
        if not visible_text:
            return ""

        header_parts: List[str] = []
        metadata = metadata or {}
        effective_type = source_node_type_override or source_node.node_type

        article_number = source_node.metadata.get("article_number")
        article_title = self._sanitize_embedding_header_text(
            source_node.title or metadata.get("article_title") or ""
        )

        if article_number:
            self._append_unique_header_part(
                header_parts,
                f"Artigo {article_number}",
                keep_article_header=True,
            )
        elif source_node.label:
            self._append_unique_header_part(header_parts, source_node.label)

        if article_title:
            self._append_unique_header_part(header_parts, article_title)

        if effective_type == "SECTION_GROUP":
            section_labels = metadata.get("section_labels") or []
            if section_labels:
                self._append_unique_header_part(
                    header_parts,
                    "Secções " + ", ".join(section_labels),
                )

        if effective_type == "LETTERED_GROUP":
            lettered_labels = metadata.get("lettered_labels") or []
            if lettered_labels:
                self._append_unique_header_part(
                    header_parts,
                    "Alíneas " + ", ".join(lettered_labels),
                )

        if effective_type == "ARTICLE_PART":
            part_index = metadata.get("part_index")
            part_count = metadata.get("part_count")
            if part_index is not None and part_count is not None:
                self._append_unique_header_part(
                    header_parts,
                    f"Parte {part_index}/{part_count}",
                )

        if effective_type == "PREAMBLE":
            header_parts = ["Preamble"]

        header = " - ".join(part for part in header_parts if part).strip()
        if not header:
            return visible_text

        return f"{header}\n\n{visible_text}".strip()

    def _sanitize_embedding_header_text(
        self,
        text: str,
        keep_article_header: bool = False,
    ) -> str:
        """
        Remove structural and editorial pollution from embedding header text.

        Why this helper exists
        ----------------------
        Parser improvements should already keep titles clean, but embedding
        text must remain defensive because polluted titles or line residues can
        still degrade retrieval quality.

        Parameters
        ----------
        text : str
            Candidate header text.

        keep_article_header : bool
            When True, preserve an explicit article header provided by the
            strategy itself.

        Returns
        -------
        str
            Sanitized one-line header text.
        """
        if not text:
            return ""

        cleaned_parts: List[str] = []

        for raw_line in text.splitlines():
            line = TITLE_SEPARATOR_RE.sub(" ", raw_line.strip())
            line = LEADING_PAGE_MARKER_RE.sub("", line).strip()

            if not line:
                continue

            if INLINE_PAGE_COUNTER_LINE_RE.match(line):
                continue

            if LOOSE_PAGE_COUNTER_LINE_RE.match(line):
                continue

            if DR_EDITORIAL_RE.match(line):
                continue

            if not keep_article_header and self._is_standalone_article_header_line(line):
                continue

            cleaned_parts.append(line)

        return normalize_block_whitespace(" ".join(cleaned_parts)).strip()

    def _append_unique_header_part(
        self,
        header_parts: List[str],
        candidate_part: str,
        keep_article_header: bool = False,
    ) -> None:
        """
        Append a sanitized header part only when it adds new information.

        Parameters
        ----------
        header_parts : List[str]
            Mutable list of current header parts.

        candidate_part : str
            Candidate part to append.

        keep_article_header : bool
            When True, preserve an explicit article header part.
        """
        sanitized_part = self._sanitize_embedding_header_text(
            candidate_part,
            keep_article_header=keep_article_header,
        )
        if not sanitized_part:
            return

        candidate_key = sanitized_part.casefold()
        for existing_part in header_parts:
            if existing_part.casefold() == candidate_key:
                return

        header_parts.append(sanitized_part)

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
        - optional meta text is generated separately
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

        effective_source_node_type = source_node_type_override or getattr(
            source_node,
            "node_type",
            "",
        )
        effective_source_node_label = source_node_label_override or getattr(
            source_node,
            "label",
            "",
        )

        if self._should_drop_unsafe_chunk(
            visible_text=visible_text,
            source_node=source_node,
            metadata=metadata,
            source_node_type=effective_source_node_type,
            source_node_label=effective_source_node_label,
        ):
            return None

        effective_hierarchy_path = list(
            hierarchy_path or getattr(source_node, "hierarchy_path", [])
        )

        meta_text = (
            self._build_meta_text(
                source_node=source_node,
                visible_text=visible_text,
                metadata=metadata,
                source_node_type_override=source_node_type_override,
            )
            if self.settings.include_meta_text
            else ""
        )

        return Chunk(
            chunk_id=f"{document_metadata.doc_id}_chunk_{sequence:04d}",
            doc_id=document_metadata.doc_id,
            strategy=self.name,
            text=visible_text,
            meta_text=meta_text,
            page_start=page_start,
            page_end=page_end,
            source_node_type=effective_source_node_type,
            source_node_label=effective_source_node_label,
            hierarchy_path=effective_hierarchy_path,
            chunk_reason=chunk_reason,
            char_count=len(visible_text),
            metadata=metadata,
        )

    def _should_drop_unsafe_chunk(
        self,
        visible_text: str,
        source_node: StructuralNode,
        metadata: Dict[str, Any],
        source_node_type: str,
        source_node_label: str,
    ) -> bool:
        """
        Decide whether a chunk should be dropped as structurally unsafe.

        Why this helper exists
        ----------------------
        Task-level cleanup should not fabricate missing text. When the parser
        already signals that an article is structurally incomplete and the final
        visible text still ends like a damaged fragment, exporting the chunk
        creates a semantically weak retrieval unit. In that narrow case the
        strategy should refuse the export.

        Parameters
        ----------
        visible_text : str
            Final candidate visible chunk text.

        source_node : StructuralNode
            Main structural origin of the chunk.

        metadata : Dict[str, Any]
            Chunk metadata payload.

        source_node_type : str
            Effective exported source-node type for the chunk.

        source_node_label : str
            Effective exported source-node label for the chunk.

        Returns
        -------
        bool
            True when the chunk should be dropped.
        """
        if not visible_text:
            return True

        if not metadata.get("is_structurally_incomplete", False):
            return False

        if (
            source_node_type == "ARTICLE"
            and source_node.node_type == "ARTICLE"
            and has_suspicious_truncated_ending(visible_text)
        ):
            return True

        return self._is_orphaned_structural_continuation_chunk(
            metadata=metadata,
            source_node_type=source_node_type,
            source_node_label=source_node_label,
        )

    def _is_orphaned_structural_continuation_chunk(
        self,
        metadata: Dict[str, Any],
        source_node_type: str,
        source_node_label: str,
    ) -> bool:
        """
        Detect chunk exports that still represent orphaned structural continuation.

        Parameters
        ----------
        metadata : Dict[str, Any]
            Chunk metadata payload.

        source_node_type : str
            Effective exported source-node type.

        source_node_label : str
            Effective exported source-node label.

        Returns
        -------
        bool
            True when the export is still an orphaned continuation unit.
        """
        integrity_warnings = set(metadata.get("integrity_warnings", []))
        if not integrity_warnings:
            return False

        if source_node_type == "SECTION_GROUP":
            section_labels = self._extract_structural_labels(
                metadata.get("section_labels"),
                source_node_label,
            )
            if (
                "possible_orphaned_numbered_continuation" in integrity_warnings
                and section_labels
                and all(
                    self._is_subordinate_numbered_label(label)
                    for label in section_labels
                )
            ):
                return True

        if source_node_type == "LETTERED_GROUP":
            lettered_labels = self._extract_structural_labels(
                metadata.get("lettered_labels"),
                source_node_label,
            )
            if (
                "possible_broken_lettered_enumeration" in integrity_warnings
                and lettered_labels
                and self._is_non_initial_lettered_label(lettered_labels[0])
            ):
                return True

        return False

    def _extract_structural_labels(
        self,
        labels: Any,
        fallback_label: str,
    ) -> List[str]:
        """
        Normalize grouped structural labels from metadata or export fallback.

        Parameters
        ----------
        labels : Any
            Candidate labels payload from chunk metadata.

        fallback_label : str
            Exported label string used when metadata is absent.

        Returns
        -------
        List[str]
            Normalized non-empty labels.
        """
        if isinstance(labels, Sequence) and not isinstance(labels, str):
            return [str(label).strip() for label in labels if str(label).strip()]

        if not fallback_label:
            return []

        return [
            part.strip()
            for part in str(fallback_label).split(",")
            if part.strip()
        ]

    def _is_subordinate_numbered_label(self, label: str) -> bool:
        """
        Decide whether a numbered label represents a subordinate continuation.

        Parameters
        ----------
        label : str
            Candidate structural label.

        Returns
        -------
        bool
            True when the label behaves like a decimal continuation.
        """
        normalized_label = str(label).strip()
        if not normalized_label:
            return False

        return re.match(r"^\d+\.\d+$", normalized_label) is not None

    def _is_non_initial_lettered_label(self, label: str) -> bool:
        """
        Decide whether a lettered label starts after the expected first item.

        Parameters
        ----------
        label : str
            Candidate lettered label.

        Returns
        -------
        bool
            True when the label starts after ``a``.
        """
        normalized_label = str(label).strip().lower()
        if not normalized_label:
            return False

        return len(normalized_label) == 1 and normalized_label.isalpha() and normalized_label != "a"

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
