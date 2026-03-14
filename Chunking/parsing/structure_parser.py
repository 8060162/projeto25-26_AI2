from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple, Union

from Chunking.chunking.models import StructuralNode
from Chunking.cleaning.normalizer import NormalizedDocument, NormalizedPage


# ============================================================================
# Structure-aware regexes for Portuguese legal / regulatory documents
# ============================================================================
#
# These patterns intentionally remain local to this module so the parser stays
# self-contained and easy to evolve.
#
# Design philosophy
# -----------------
# This parser should prefer conservative structure detection over aggressive
# guessing. In legal/regulatory documents, false positives are often more
# harmful than missing a small structural cue.
# ============================================================================

# Accept article header variants such as:
# - "ARTIGO 1º"
# - "Artigo 1.º"
# - "Artigo 1"
# - "Art. 1"
# - "Artigo 1 - Título"
# - "Artigo 1: Título"
ARTICLE_HEADER_RE = re.compile(
    r"^\s*(?:ARTIGO|ART\.?)\s+(\d+(?:\.\d+)?)\s*(?:\.?\s*[ºo°])?\s*(?:[—–\-:]\s*(.*))?\s*$",
    re.IGNORECASE,
)

# Accept chapter headers such as:
# - "CAPÍTULO I"
# - "CAPITULO IV"
# - "CAPÍTULO IV - Título"
CHAPTER_HEADER_RE = re.compile(
    r"^\s*CAP[ÍI]TULO\s+([IVXLCDM\d]+)\s*(?:[—–\-:]\s*(.*))?\s*$",
    re.IGNORECASE,
)

# Accept section/subsection/title headers such as:
# - "SECÇÃO I"
# - "SUBSECÇÃO II"
# - "TÍTULO III"
SECTION_CONTAINER_RE = re.compile(
    r"^\s*(SECÇÃO|SUBSECÇÃO|TÍTULO)\s+([IVXLCDM\dA-Z]+)\s*(?:[—–\-:]\s*(.*))?\s*$",
    re.IGNORECASE,
)

# Accept annex headers such as:
# - "ANEXO"
# - "ANEXO I"
# - "ANEXO A - Título"
ANNEX_HEADER_RE = re.compile(
    r"^\s*(ANEXO(?:\s+[IVXLCDM\dA-Z]+)?)\s*(?:[—–\-:]\s*(.*))?\s*$",
    re.IGNORECASE,
)

# Accept numbered legal structural blocks such as:
# - "1. Texto..."
# - "2) Texto..."
# - "2.1. Texto..."
# - "2.1 — Texto..."
# - "2.1 - Texto..."
# - "n.º 1 ..."
# - "N.º 2 ..."
#
# Important:
# this pattern is intentionally stricter than "any number at line start".
NUMBERED_BLOCK_RE = re.compile(
    r"(?m)^(?P<label>(?:n\.?\s*[ºo]\s*)?\d+(?:\.\d+)*)(?:\.\s+|\)\s+|\s+[—–\-]\s+|\s+)"
)

# Accept legal-style lettered items such as:
# - "a) ..."
# - "b) ..."
LETTERED_BLOCK_RE = re.compile(
    r"(?m)^(?P<label>[a-z])\)\s+",
    re.IGNORECASE,
)

# Uppercase-heavy lines often behave like headings/titles.
UPPERCASE_HEAVY_RE = re.compile(
    r"^[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ0-9 /().,\-–—ºª]+$"
)

# Detect obvious body-like openings.
BODY_START_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*[\.\)]?\s+|n\.?\s*[ºo]\s*\d+\s+|[a-z]\)\s+|O\s|A\s|Os\s|As\s|Nos\s|Na\s|No\s|Em\s|Para\s|Quando\s|Sempre\s|Considera-se\s)",
    re.IGNORECASE,
)

# Detect lines that strongly look like front matter rather than body.
FRONT_MATTER_RE = re.compile(
    r"^\s*(POLITÉCNICO DO PORTO|P\.PORTO|DESPACHO|REGULAMENTO\s+N\.?\s*º|REGULAMENTO\s+Nº|ÍNDICE)\b",
    re.IGNORECASE,
)


class StructureParser:
    """
    Parse normalized text into a navigable structural tree.

    High-level purpose
    ------------------
    This parser is the bridge between:
        normalized document text
    and:
        a structured legal/regulatory tree

    It does NOT yet generate the final master-dictionary JSON structure, but
    it should produce a tree rich enough for that export step to be deterministic
    and clean.

    Design goals
    ------------
    - Be pragmatic rather than academically perfect
    - Be robust for Portuguese legal and regulatory PDFs
    - Preserve enough hierarchy to support export into a canonical JSON tree
    - Enrich metadata so final chunk text can stay clean later
    - Prefer conservative structure detection over risky inference

    Important note
    --------------
    This parser assumes that the normalizer already preserved meaningful line
    boundaries. The parser relies heavily on those boundaries to detect:
    - chapters
    - section containers
    - annexes
    - articles
    - titles
    - numbered sections
    - lettered items
    """

    def __init__(self) -> None:
        """
        Initialize parser state.

        Why this exists
        ---------------
        The parser creates explicit structural identity metadata such as:
        - node_id
        - parent_node_id
        - hierarchy_path

        Keeping an internal counter makes identifiers stable within one parsing
        run and easy to inspect in JSON outputs.
        """
        self._node_sequence = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        normalized_input: Union[NormalizedDocument, Sequence[Tuple[int, str]], Sequence[NormalizedPage]],
    ) -> StructuralNode:
        """
        Parse normalized content into a structural tree.

        Supported inputs
        ----------------
        This method accepts:
        - NormalizedDocument
        - List[NormalizedPage]
        - legacy list of tuples: (page_number, page_text)

        High-level output shape
        -----------------------
        DOCUMENT
          ├── FRONT_MATTER
          ├── PREAMBLE
          ├── ANNEX / CHAPTER / SECTION_CONTAINER
          │     ├── ARTICLE
          │     │     ├── SECTION
          │     │     ├── LETTERED_ITEM
          │     │     └── ...
          │     └── ARTICLE
          └── ...

        Important behavior
        ------------------
        - Content before the first article is no longer treated as one
          undifferentiated block.
        - It is split into:
            * FRONT_MATTER: cover, branding, title, index-like opening material
            * PREAMBLE: genuine introductory normative prose
        - Articles are attached to the current structural container
        - ARTICLE.text remains the canonical fallback text even when SECTION and
          LETTERED_ITEM children are later derived from it

        Parameters
        ----------
        normalized_input : Union[NormalizedDocument, Sequence[Tuple[int, str]], Sequence[NormalizedPage]]
            Normalized input document/pages.

        Returns
        -------
        StructuralNode
            Root DOCUMENT node containing the parsed tree.
        """
        pages_text = self._coerce_pages_text(normalized_input)

        root = self._make_node(
            node_type="DOCUMENT",
            label="DOCUMENT",
            title="",
            text="",
            page_start=None,
            page_end=None,
            metadata={
                "document_part": "document_root",
            },
            parent=None,
        )

        lines = self._collect_lines(pages_text)

        # Container stack design
        # ----------------------
        # The stack always contains the currently active non-article containers.
        #
        # Example:
        #   DOCUMENT -> CHAPTER -> SECTION_CONTAINER
        #
        # ARTICLE nodes are deliberately NOT kept in this stack, because article
        # routing is already handled separately via `current_article`.
        container_stack: List[StructuralNode] = [root]
        current_article: Optional[StructuralNode] = None

        pre_article_lines: List[Tuple[int, str]] = []
        first_article_seen = False
        index = 0

        while index < len(lines):
            page_number, line = lines[index]

            # ---------------------------------------------------------
            # Chapter detection
            # ---------------------------------------------------------
            chapter_match = CHAPTER_HEADER_RE.match(line)
            if chapter_match:
                chapter_number = chapter_match.group(1).strip()
                inline_title = (chapter_match.group(2) or "").strip()

                consumed_title, next_index = self._consume_following_title_lines(
                    lines=lines,
                    start_index=index + 1,
                    max_lines=2,
                )

                chapter_title = inline_title or consumed_title

                # Chapters attach directly under DOCUMENT.
                container_stack = [root]
                current_parent = container_stack[-1]

                chapter_node = self._make_node(
                    node_type="CHAPTER",
                    label=f"CAP_{chapter_number}",
                    title=chapter_title,
                    text="",
                    page_start=page_number,
                    page_end=page_number,
                    metadata={
                        "chapter_number": chapter_number,
                        "chapter_title": chapter_title,
                        "document_part": "regulation_body",
                    },
                    parent=current_parent,
                )

                current_parent.children.append(chapter_node)
                container_stack.append(chapter_node)
                current_article = None
                index = next_index
                continue

            # ---------------------------------------------------------
            # Section-container detection
            #
            # These are structural containers such as:
            # - SECÇÃO I
            # - SUBSECÇÃO II
            # - TÍTULO III
            #
            # They attach below the nearest currently active non-article
            # container.
            # ---------------------------------------------------------
            section_container_match = SECTION_CONTAINER_RE.match(line)
            if section_container_match:
                container_type = section_container_match.group(1).strip().upper()
                container_number = section_container_match.group(2).strip()
                inline_title = (section_container_match.group(3) or "").strip()

                consumed_title, next_index = self._consume_following_title_lines(
                    lines=lines,
                    start_index=index + 1,
                    max_lines=2,
                )

                container_title = inline_title or consumed_title
                current_parent = container_stack[-1]

                container_node = self._make_node(
                    node_type="SECTION_CONTAINER",
                    label=f"{container_type}_{container_number}",
                    title=container_title,
                    text="",
                    page_start=page_number,
                    page_end=page_number,
                    metadata={
                        "container_type": container_type,
                        "container_number": container_number,
                        "container_title": container_title,
                        "document_part": current_parent.metadata.get(
                            "document_part",
                            "regulation_body",
                        ),
                        "parent_type": current_parent.node_type,
                        "parent_label": current_parent.label,
                        "parent_title": current_parent.title,
                    },
                    parent=current_parent,
                )

                current_parent.children.append(container_node)
                container_stack.append(container_node)
                current_article = None
                index = next_index
                continue

            # ---------------------------------------------------------
            # Annex detection
            # ---------------------------------------------------------
            annex_match = ANNEX_HEADER_RE.match(line)
            if annex_match:
                annex_label = annex_match.group(1).strip()
                inline_title = (annex_match.group(2) or "").strip()

                consumed_title, next_index = self._consume_following_title_lines(
                    lines=lines,
                    start_index=index + 1,
                    max_lines=3,
                )

                annex_title = inline_title or consumed_title

                # Annexes attach directly under DOCUMENT.
                container_stack = [root]
                current_parent = container_stack[-1]

                annex_node = self._make_node(
                    node_type="ANNEX",
                    label=annex_label,
                    title=annex_title,
                    text="",
                    page_start=page_number,
                    page_end=page_number,
                    metadata={
                        "annex_label": annex_label,
                        "annex_title": annex_title,
                        "document_part": "regulation_annex",
                    },
                    parent=current_parent,
                )

                current_parent.children.append(annex_node)
                container_stack.append(annex_node)
                current_article = None
                index = next_index
                continue

            # ---------------------------------------------------------
            # Article detection
            # ---------------------------------------------------------
            article_match = ARTICLE_HEADER_RE.match(line)
            if article_match:
                if not first_article_seen and pre_article_lines:
                    self._attach_pre_article_content(root, pre_article_lines)

                first_article_seen = True

                article_number = article_match.group(1).strip()
                inline_title = (article_match.group(2) or "").strip()

                consumed_title, next_index = self._consume_following_title_lines(
                    lines=lines,
                    start_index=index + 1,
                    max_lines=2,
                )

                article_title = inline_title or consumed_title
                current_parent = container_stack[-1]

                current_article = self._make_node(
                    node_type="ARTICLE",
                    label=f"ART_{article_number}",
                    title=article_title,
                    text="",
                    page_start=page_number,
                    page_end=page_number,
                    metadata={
                        "article_number": article_number,
                        "article_title": article_title,
                        "parent_type": current_parent.node_type,
                        "parent_label": current_parent.label,
                        "parent_title": current_parent.title,
                        "document_part": current_parent.metadata.get(
                            "document_part",
                            "regulation_body",
                        ),
                    },
                    parent=current_parent,
                )

                current_parent.children.append(current_article)
                index = next_index
                continue

            # ---------------------------------------------------------
            # Body text routing
            # ---------------------------------------------------------
            if not first_article_seen:
                pre_article_lines.append((page_number, line))
                index += 1
                continue

            # After the first article, free text is attached to the current
            # article when one is active; otherwise to the current container.
            target = current_article or container_stack[-1]
            self._append_text_to_node(
                node=target,
                line=line,
                page_number=page_number,
            )
            index += 1

        if not first_article_seen and pre_article_lines:
            self._attach_pre_article_content(root, pre_article_lines)

        self._post_process_nodes(root)
        self._infer_missing_page_ranges(root)
        return root

    # ------------------------------------------------------------------
    # Input coercion / line collection
    # ------------------------------------------------------------------

    def _coerce_pages_text(
        self,
        normalized_input: Union[NormalizedDocument, Sequence[Tuple[int, str]], Sequence[NormalizedPage]],
    ) -> List[Tuple[int, str]]:
        """
        Normalize supported parser inputs into a list of page tuples.

        Why this helper exists
        ----------------------
        The parser is being migrated from the older contract:
            List[tuple[int, str]]
        toward richer normalized models.

        This helper preserves compatibility during that transition.

        Parameters
        ----------
        normalized_input : Union[NormalizedDocument, Sequence[Tuple[int, str]], Sequence[NormalizedPage]]
            Supported parser input variants.

        Returns
        -------
        List[Tuple[int, str]]
            Normalized list of (page_number, page_text) tuples.
        """
        if isinstance(normalized_input, NormalizedDocument):
            return [(page.page_number, page.text) for page in normalized_input.pages]

        if normalized_input and isinstance(normalized_input[0], NormalizedPage):  # type: ignore[index]
            return [(page.page_number, page.text) for page in normalized_input]  # type: ignore[union-attr]

        return list(normalized_input)  # type: ignore[arg-type]

    def _collect_lines(self, pages_text: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """
        Flatten page texts into an ordered list of (page_number, line).

        Why page numbers remain attached
        --------------------------------
        Page provenance is required later for:
        - page_start
        - page_end
        - child-derived page ranges
        - JSON export metadata
        - eventual chunk traceability

        Parameters
        ----------
        pages_text : List[Tuple[int, str]]
            Page tuples from the normalization stage.

        Returns
        -------
        List[Tuple[int, str]]
            Ordered flattened line tuples.
        """
        lines: List[Tuple[int, str]] = []

        for page_number, page_text in pages_text:
            for raw_line in page_text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                lines.append((page_number, line))

        return lines

    # ------------------------------------------------------------------
    # Pre-article handling
    # ------------------------------------------------------------------

    def _attach_pre_article_content(
        self,
        root: StructuralNode,
        pre_article_lines: List[Tuple[int, str]],
    ) -> None:
        """
        Split pre-article material into FRONT_MATTER and PREAMBLE nodes.

        Why this helper exists
        ----------------------
        Earlier implementations often treated everything before the first
        article as PREAMBLE. That behaves poorly on real institutional PDFs
        because opening pages frequently contain:
        - cover/title material
        - dispatch headers
        - institutional branding
        - index/table-of-contents material

        Those regions are not equivalent to genuine normative preamble prose.

        Current strategy
        ----------------
        - classify obvious front-matter lines using lightweight heuristics
        - preserve the remaining prose as PREAMBLE
        - keep both nodes separate for downstream export and later chunking

        Parameters
        ----------
        root : StructuralNode
            Root DOCUMENT node.

        pre_article_lines : List[Tuple[int, str]]
            All lines seen before the first detected article.
        """
        if not pre_article_lines:
            return

        front_lines: List[Tuple[int, str]] = []
        preamble_lines: List[Tuple[int, str]] = []

        for page_number, line in pre_article_lines:
            if self._looks_like_front_matter_line(line):
                front_lines.append((page_number, line))
            else:
                preamble_lines.append((page_number, line))

        if front_lines:
            front_text = self._normalize_node_text(
                "\n".join(line for _, line in front_lines)
            )
            if front_text:
                front_node = self._make_node(
                    node_type="FRONT_MATTER",
                    label="FRONT_MATTER",
                    title="",
                    text=front_text,
                    page_start=front_lines[0][0],
                    page_end=front_lines[-1][0],
                    metadata={
                        "document_part": "front_matter",
                    },
                    parent=root,
                )
                root.children.insert(0, front_node)

        if preamble_lines:
            preamble_text = self._normalize_node_text(
                "\n".join(line for _, line in preamble_lines)
            )
            if preamble_text:
                preamble_node = self._make_node(
                    node_type="PREAMBLE",
                    label="PREAMBLE",
                    title="",
                    text=preamble_text,
                    page_start=preamble_lines[0][0],
                    page_end=preamble_lines[-1][0],
                    metadata={
                        "document_part": "dispatch_or_intro",
                    },
                    parent=root,
                )

                # Keep PREAMBLE after FRONT_MATTER when both exist.
                insert_index = (
                    1
                    if root.children and root.children[0].node_type == "FRONT_MATTER"
                    else 0
                )
                root.children.insert(insert_index, preamble_node)

    def _looks_like_front_matter_line(self, line: str) -> bool:
        """
        Decide whether a line likely belongs to front matter rather than
        normative preamble prose.

        Typical positive examples
        -------------------------
        - "POLITÉCNICO DO PORTO"
        - "DESPACHO P.PORTO/..."
        - "REGULAMENTO n.º 633/2024"
        - "ÍNDICE"

        Safety philosophy
        -----------------
        This heuristic should remain lightweight and conservative.
        It is better to leave some front matter inside PREAMBLE than to
        aggressively reclassify genuine normative prose.

        Parameters
        ----------
        line : str
            Candidate line.

        Returns
        -------
        bool
            True when the line looks like front matter.
        """
        if not line:
            return False

        if FRONT_MATTER_RE.match(line):
            return True

        if UPPERCASE_HEAVY_RE.match(line) and len(line.split()) <= 8:
            return True

        return False

    # ------------------------------------------------------------------
    # Title detection
    # ------------------------------------------------------------------

    def _consume_following_title_lines(
        self,
        lines: List[Tuple[int, str]],
        start_index: int,
        max_lines: int,
    ) -> Tuple[str, int]:
        """
        Consume a small number of title-like lines following a structural header.

        Why this exists
        ---------------
        In many legal PDFs, the title appears on the line immediately after the
        structural marker instead of inline.

        Example
        -------
            Artigo 5.º
            Revisão de prova

        Important safety behavior
        -------------------------
        The logic remains conservative to avoid swallowing the first body
        sentence into metadata.

        Parameters
        ----------
        lines : List[Tuple[int, str]]
            Full flattened line stream.

        start_index : int
            Starting index after the structural marker.

        max_lines : int
            Maximum number of title lines to consume.

        Returns
        -------
        Tuple[str, int]
            Consumed title text and the next unread index.
        """
        collected: List[str] = []
        index = start_index

        while index < len(lines) and len(collected) < max_lines:
            _, line = lines[index]

            if not self._is_probable_title_line(
                line=line,
                collected_count=len(collected),
            ):
                break

            collected.append(line)
            index += 1

        title = " | ".join(collected).strip()
        return title, index

    def _is_probable_title_line(self, line: str, collected_count: int = 0) -> bool:
        """
        Decide whether a line is likely to be a title rather than body text.

        Positive examples
        -----------------
        - "ÂMBITO"
        - "Definições"
        - "Condições para realização da inscrição"

        Negative examples
        -----------------
        - "1. O presente regulamento ..."
        - "n.º 1 O estudante ..."
        - "a) O estudante ..."
        - "O presente regulamento aplica-se ..."

        Design philosophy
        -----------------
        Title detection must remain conservative.
        A missed title is preferable to swallowing normative prose into metadata.

        Parameters
        ----------
        line : str
            Candidate line.

        collected_count : int, default=0
            How many title lines have already been consumed.

        Returns
        -------
        bool
            True when the line looks title-like.
        """
        if not line:
            return False

        if ARTICLE_HEADER_RE.match(line):
            return False

        if CHAPTER_HEADER_RE.match(line):
            return False

        if ANNEX_HEADER_RE.match(line):
            return False

        if SECTION_CONTAINER_RE.match(line):
            return False

        if BODY_START_RE.match(line):
            return False

        if LETTERED_BLOCK_RE.match(line):
            return False

        if len(line) > 120:
            return False

        word_count = len(line.split())
        if word_count > 12:
            return False

        lowered = line.lower()

        body_starters = (
            "o presente",
            "a presente",
            "cada ",
            "considera-se",
            "nos termos",
            "mediante ",
            "deverá ",
            "deverão ",
            "em cada ",
            "sempre que ",
            "quando ",
            "para efeitos",
            "podem ",
            "pode ",
            "serão ",
            "é ",
            "são ",
            "compete ",
            "aplica-se ",
        )
        if lowered.startswith(body_starters):
            return False

        if line.endswith(".") or line.endswith(";") or line.endswith(":"):
            return False

        if UPPERCASE_HEAVY_RE.match(line):
            return True

        if collected_count == 0 and word_count <= 8 and line[:1].isupper():
            return True

        if collected_count == 1 and word_count <= 6 and line[:1].isupper():
            return True

        return False

    # ------------------------------------------------------------------
    # Node text accumulation
    # ------------------------------------------------------------------

    def _append_text_to_node(
        self,
        node: StructuralNode,
        line: str,
        page_number: int,
    ) -> None:
        """
        Append one line of text to a node and update its page range.

        Why this helper exists
        ----------------------
        It keeps text accumulation and page-range tracking centralized and
        predictable.

        Parameters
        ----------
        node : StructuralNode
            Target node.

        line : str
            Line to append.

        page_number : int
            Source page number.
        """
        if node.page_start is None:
            node.page_start = page_number

        node.page_end = page_number
        node.text = f"{node.text}\n{line}".strip()

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process_nodes(self, node: StructuralNode) -> None:
        """
        Perform recursive post-processing over the structural tree.

        Tasks
        -----
        - normalize node text
        - recursively post-process children
        - derive SECTION children from article text when appropriate
        - derive LETTERED_ITEM children from article/section text when appropriate

        Important implementation choice
        -------------------------------
        ARTICLE nodes keep their full canonical text even after SECTION or
        LETTERED_ITEM children are extracted. This is intentional because:
        - the export layer may still need the full article content
        - later chunking may want full-article fallback
        - child nodes are explicit structure enrichment, not destructive splits

        Parameters
        ----------
        node : StructuralNode
            Current node in recursive traversal.
        """
        if node.text:
            node.text = self._normalize_node_text(node.text)

        for child in node.children:
            self._post_process_nodes(child)

        if node.node_type == "ARTICLE" and node.text:
            section_children = self._extract_numbered_children(node)
            node.children.extend(section_children)

            # Only fall back to direct lettered extraction when numbered
            # structure was not found.
            if not section_children:
                node.children.extend(self._extract_lettered_children(node))

        if node.node_type == "SECTION" and node.text:
            node.children.extend(self._extract_lettered_children(node))

    def _infer_missing_page_ranges(self, node: StructuralNode) -> None:
        """
        Infer missing page ranges recursively from child nodes.

        Why this helper exists
        ----------------------
        Some containers may initially have only partial page information.
        This method repairs that by deriving the broadest plausible page range
        from descendants.

        Parameters
        ----------
        node : StructuralNode
            Node whose page range should be repaired if necessary.
        """
        for child in node.children:
            self._infer_missing_page_ranges(child)

        child_starts = [child.page_start for child in node.children if child.page_start is not None]
        child_ends = [child.page_end for child in node.children if child.page_end is not None]

        if node.page_start is None and child_starts:
            node.page_start = min(child_starts)

        if node.page_end is None and child_ends:
            node.page_end = max(child_ends)

    def _normalize_node_text(self, text: str) -> str:
        """
        Normalize node text while preserving structural line boundaries.

        Why this helper remains light
        -----------------------------
        The parser still benefits from retained newline structure, and later
        export/chunking may also rely on it. Therefore aggressive flattening
        does not belong here.

        Parameters
        ----------
        text : str
            Raw node text.

        Returns
        -------
        str
            Conservatively normalized text.
        """
        text = text.strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Numbered structure extraction
    # ------------------------------------------------------------------

    def _extract_numbered_children(self, article: StructuralNode) -> List[StructuralNode]:
        """
        Split article text into numbered SECTION children.

        Supported structural examples
        -----------------------------
        - "1. ..."
        - "2) ..."
        - "2.1. ..."
        - "2.1 - ..."
        - "2.1 — ..."
        - "n.º 1 ..."
        - "N.º 2 ..."

        Important safety behavior
        -------------------------
        This method rejects weak or implausible matches so that ordinary body
        lines beginning with numbers are not mistaken for structural sections.

        Parameters
        ----------
        article : StructuralNode
            Article node whose text should be inspected.

        Returns
        -------
        List[StructuralNode]
            Derived SECTION nodes.
        """
        text = article.text.strip()
        if not text:
            return []

        matches = list(NUMBERED_BLOCK_RE.finditer(text))
        if len(matches) < 2:
            return []

        labels = [self._normalize_numbered_label(match.group("label")) for match in matches]
        if not self._has_plausible_numbered_structure(labels):
            return []

        children: List[StructuralNode] = []

        for match_index, match in enumerate(matches):
            raw_label = match.group("label")
            normalized_label = self._normalize_numbered_label(raw_label)

            start = match.start()
            end = (
                matches[match_index + 1].start()
                if match_index + 1 < len(matches)
                else len(text)
            )
            block_text = text[start:end].strip()

            section_node = self._make_node(
                node_type="SECTION",
                label=normalized_label,
                title="",
                text=self._normalize_node_text(block_text),
                page_start=article.page_start,
                page_end=article.page_end,
                metadata={
                    "article_label": article.label,
                    "article_number": article.metadata.get("article_number"),
                    "article_title": article.title,
                    "document_part": article.metadata.get("document_part"),
                    "parent_type": "ARTICLE",
                    "parent_label": article.label,
                    "raw_section_label": raw_label,
                },
                parent=article,
            )

            children.append(section_node)

        return children

    def _normalize_numbered_label(self, label: str) -> str:
        """
        Normalize a numeric structural label into a cleaner comparison format.

        Examples
        --------
        - "n.º 1" -> "1"
        - "Nº 2"  -> "2"
        - "2.1"   -> "2.1"

        Parameters
        ----------
        label : str
            Raw regex label.

        Returns
        -------
        str
            Normalized numeric label.
        """
        cleaned = re.sub(r"(?i)^n\.?\s*[ºo]\s*", "", label).strip()
        cleaned = re.sub(r"\s+", "", cleaned)
        return cleaned

    def _has_plausible_numbered_structure(self, labels: List[str]) -> bool:
        """
        Validate whether extracted numeric labels look like genuine structure.

        Why this validation exists
        --------------------------
        Regex-only matching is not enough for legal text. Without a plausibility
        check, the parser may incorrectly interpret lines such as:
        - "60 créditos ECTS ..."
        - "24 meses após ..."
        as numbered sections.

        Acceptance rule
        ---------------
        - at least two labels must exist
        - labels should form a plausible increasing sequence
        - simple numeric labels such as 1, 2, 3 are strongly preferred
        - decimal labels such as 2.1, 2.2 are accepted when coherent

        Parameters
        ----------
        labels : List[str]
            Normalized numeric labels.

        Returns
        -------
        bool
            True when the sequence looks structurally plausible.
        """
        if len(labels) < 2:
            return False

        previous_parts: Optional[List[int]] = None
        plausible_transitions = 0

        for label in labels:
            try:
                current_parts = [int(part) for part in label.split(".")]
            except ValueError:
                return False

            if previous_parts is None:
                previous_parts = current_parts
                continue

            if self._is_plausible_label_transition(previous_parts, current_parts):
                plausible_transitions += 1

            previous_parts = current_parts

        required = max(1, len(labels) - 2)
        return plausible_transitions >= required

    def _is_plausible_label_transition(
        self,
        previous_parts: List[int],
        current_parts: List[int],
    ) -> bool:
        """
        Decide whether a transition between two numeric labels is structurally plausible.

        Accepted examples
        -----------------
        - 1   -> 2
        - 2   -> 3
        - 2   -> 2.1
        - 2.1 -> 2.2
        - 2.9 -> 3
        - 3.1 -> 4

        Rejected examples
        -----------------
        - 2   -> 60
        - 2.1 -> 24
        - 3.1 -> 9.7
        """
        if len(previous_parts) == len(current_parts):
            if previous_parts[:-1] == current_parts[:-1]:
                return 1 <= (current_parts[-1] - previous_parts[-1]) <= 2

            if len(previous_parts) == 1:
                return 1 <= (current_parts[0] - previous_parts[0]) <= 2

            return False

        if len(current_parts) == len(previous_parts) + 1:
            return current_parts[:-1] == previous_parts and current_parts[-1] == 1

        if len(current_parts) + 1 == len(previous_parts):
            if len(current_parts) == 1:
                return 0 <= (current_parts[0] - previous_parts[0]) <= 1
            return False

        return False

    # ------------------------------------------------------------------
    # Lettered structure extraction
    # ------------------------------------------------------------------

    def _extract_lettered_children(self, node: StructuralNode) -> List[StructuralNode]:
        """
        Split text into lettered items.

        Supported examples
        ------------------
        - "a) ..."
        - "b) ..."

        Why this is useful
        ------------------
        Even when lettered items do not become standalone retrieval units yet,
        preserving them as explicit structure improves metadata quality and
        later export/chunking flexibility.

        Parameters
        ----------
        node : StructuralNode
            ARTICLE or SECTION node to inspect.

        Returns
        -------
        List[StructuralNode]
            Derived LETTERED_ITEM nodes.
        """
        text = node.text.strip()
        if not text:
            return []

        matches = list(LETTERED_BLOCK_RE.finditer(text))
        if len(matches) < 2:
            return []

        children: List[StructuralNode] = []

        for match_index, match in enumerate(matches):
            label = match.group("label").lower()
            start = match.start()
            end = (
                matches[match_index + 1].start()
                if match_index + 1 < len(matches)
                else len(text)
            )
            block_text = text[start:end].strip()

            lettered_node = self._make_node(
                node_type="LETTERED_ITEM",
                label=label,
                title="",
                text=self._normalize_node_text(block_text),
                page_start=node.page_start,
                page_end=node.page_end,
                metadata={
                    "parent_type": node.node_type,
                    "parent_label": node.label,
                    "document_part": node.metadata.get("document_part"),
                    "article_label": node.metadata.get(
                        "article_label",
                        node.label if node.node_type == "ARTICLE" else None,
                    ),
                    "article_number": node.metadata.get("article_number"),
                    "article_title": node.metadata.get(
                        "article_title",
                        node.title,
                    ),
                },
                parent=node,
            )

            children.append(lettered_node)

        return children

    # ------------------------------------------------------------------
    # Node factory
    # ------------------------------------------------------------------

    def _make_node(
        self,
        node_type: str,
        label: str,
        title: str,
        text: str,
        page_start: Optional[int],
        page_end: Optional[int],
        metadata: dict,
        parent: Optional[StructuralNode],
    ) -> StructuralNode:
        """
        Create one StructuralNode with explicit structural identity fields.

        Why this helper exists
        ----------------------
        The project benefits from richer structural traceability:
        - stable node identifiers
        - explicit parent linkage
        - hierarchy paths for export, debugging, and later chunking

        Parameters
        ----------
        node_type : str
            Node type such as DOCUMENT, CHAPTER, ARTICLE, SECTION.

        label : str
            Human-readable structural label.

        title : str
            Optional title.

        text : str
            Optional node text.

        page_start : Optional[int]
            Starting page.

        page_end : Optional[int]
            Ending page.

        metadata : dict
            Arbitrary node metadata.

        parent : Optional[StructuralNode]
            Parent node, if any.

        Returns
        -------
        StructuralNode
            Newly created node.
        """
        self._node_sequence += 1
        node_id = f"node_{self._node_sequence:05d}"

        parent_node_id = parent.node_id if parent else None

        hierarchy_path = list(parent.hierarchy_path) if parent else []
        hierarchy_path.append(f"{node_type}:{label}")

        return StructuralNode(
            node_type=node_type,
            label=label,
            title=title,
            text=text,
            page_start=page_start,
            page_end=page_end,
            node_id=node_id,
            parent_node_id=parent_node_id,
            hierarchy_path=hierarchy_path,
            metadata=metadata,
            children=[],
        )