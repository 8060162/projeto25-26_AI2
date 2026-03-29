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

# Detect numbered blocks that survived inline inside one long body span.
#
# Examples:
# - "Definições: 1. ... 2. ..."
# - "Considera-se: 1) ... 2) ..."
INLINE_NUMBERED_BLOCK_RE = re.compile(
    r"(?P<label>(?:n\.?\s*[ºo]\s*)?\d+(?:\.\d+)*)(?:\.\s+|\)\s+|\s+[—–\-]\s+|\s+)"
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

# Detect a likely inline article title followed by body text on the same line.
#
# Example:
# - "ÂMBITO O presente regulamento aplica-se..."
# - "PAGAMENTO FORA DE PRAZO O não pagamento..."
#
# Important:
# this heuristic is intentionally conservative and is only meant to recover
# cases where normalization/extraction merged:
#   title line + first body line
INLINE_TITLE_BODY_SPLIT_RE = re.compile(
    r"^\s*"
    r"(?P<title>[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ0-9 /().,\-–—ºª]{1,80}?)"
    r"\s+"
    r"(?P<body>(?:O|A|Os|As|No|Na|Nos|Nas|Em|Para|Por|Quando|Sempre|Caso|Se|Nos\s+termos|Deve|Devem|Pode|Podem|É|São)\b.*)$",
    re.IGNORECASE,
)

# Detect likely body starts inside an article header suffix so the parser can
# recover the clean title and move the remaining text back into article body.
HEADER_SUFFIX_BODY_START_RE = re.compile(
    r"(?:^|\s)(?P<body>"
    r"(?:n\.?\s*[ºo]\s*)?\d+(?:\.\d+)*(?:\.\s+|\)\s+|\s+[—–\-]\s+|\s+)"
    r"|[a-z]\)\s+"
    r"|(?:O|A|Os|As|No|Na|Nos|Nas|Em|Para|Por|Quando|Sempre|Caso|Se|Nos\s+termos|Deve|Devem|Pode|Podem|É|São)\b"
    r")",
    re.IGNORECASE,
)

# Detect lines that strongly look like front matter rather than body.
FRONT_MATTER_RE = re.compile(
    r"^\s*(POLITÉCNICO DO PORTO|P\.PORTO|DESPACHO|REGULAMENTO\s+N\.?\s*º|REGULAMENTO\s+Nº|ÍNDICE)\b",
    re.IGNORECASE,
)

# Detect strong preamble openings commonly found in legal dispatch material.
PREAMBLE_OPENING_RE = re.compile(
    r"^\s*(?:"
    r"Considerando(?:\s+que)?|"
    r"Determino|"
    r"Nos\s+termos|"
    r"Ao\s+abrigo|"
    r"Tendo\s+em\s+conta|"
    r"Em\s+cumprimento|"
    r"Sob\s+proposta"
    r")\b",
    re.IGNORECASE,
)

# Detect standalone institutional title lines that usually belong to a cover or
# title block rather than to genuine preamble prose.
INSTITUTIONAL_TITLE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"Escola\s+Superior\b.*|"
    r"Instituto\s+Politécnico\b.*|"
    r"Politécnico\s+do\s+Porto\b.*|"
    r"P\.PORTO\b.*"
    r")$",
    re.IGNORECASE,
)

# Detect short person-name lines commonly leaked from sign-off blocks.
PERSON_NAME_LINE_RE = re.compile(
    r"^\s*[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç]+"
    r"(?:\s+[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç]+){1,4}\s*$"
)

# Detect short title fragments often leaked between sign-off blocks and the
# first real regulation heading.
REGULATION_TITLE_FRAGMENT_RE = re.compile(
    r"^\s*(?:"
    r"Regulamento\b.*|"
    r"Despacho\b.*|"
    r"Estatutos?\b.*|"
    r"Princ[ií]pios\s+gerais\b.*|"
    r"Disposi[cç][õo]es\s+gerais\b.*"
    r")$",
    re.IGNORECASE,
)

# Detect dangling connector endings that often indicate a preamble line was cut
# by leaked title/signature residue immediately after it.
TRUNCATED_PREAMBLE_TAIL_RE = re.compile(
    r"(?:\s+(?:e\s+que|de\s+|do\s+|da\s+|dos\s+|das\s+|de|do|da|dos|das))\s*$",
    re.IGNORECASE,
)

# Detect lines that should almost never survive as meaningful preamble content.
#
# These are not necessarily "bad extraction", but they are often administrative
# residues that should not pollute PREAMBLE.
#
# Important:
# keep this intentionally narrow. Terms such as "REGULAMENTO" and "DESPACHO"
# may still be legitimate front-matter content and should usually be classified,
# not discarded.
NON_NORMATIVE_PRE_ARTICLE_RE = re.compile(
    r"^\s*(O\s+PRESIDENTE\s+DO\s+POLITÉCNICO|PRESIDENTE\s+DO\s+POLITÉCNICO)\s*$",
    re.IGNORECASE,
)

# Detect clearly suspicious / broken lines that should not become structural text.
#
# This intentionally targets obvious corruption rather than subtle OCR noise.
SUSPICIOUS_GARBLED_LINE_RE = re.compile(
    r"^[^A-Za-zÀ-ÿ]{0,3}(?:[\*\+\-/=<>\\\[\]\{\}_`~]{2,}|[0-9\W]{12,})$"
)

# Detect page-counter-like or synthetic page residues that sometimes survive
# normalization and later pollute PREAMBLE or article bodies.
PAGE_COUNTER_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")

# Detect citation-heavy header suffixes that actually belong to body prose.
#
# Example:
# - "B do Decreto-Lei n.o 74/2006, de 24 de março"
#
# These fragments may appear after "artigo 46.o -" inside citations and must
# not be promoted to article titles.
CITATION_HEAVY_SUFFIX_RE = re.compile(
    r"\b(?:Decreto|Lei|Regulamento|Estatutos?|Código|RJIES)\b.*"
    r"(?:\bn\.?\s*[ºo]\b|\b\d{1,4}/\d{2,4}\b|\bde\s+\d{1,2}\s+de\b|,)",
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
    - be pragmatic rather than academically perfect
    - be robust for Portuguese legal and regulatory PDFs
    - preserve enough hierarchy to support export into a canonical JSON tree
    - enrich metadata so final chunk text can stay clean later
    - prefer conservative structure detection over risky inference

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
        normalized_input: Union[
            NormalizedDocument,
            Sequence[Tuple[int, str]],
            Sequence[NormalizedPage],
        ],
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
        - content before the first article is no longer treated as one
          undifferentiated block
        - it is split into:
            * FRONT_MATTER: cover, branding, title, index-like opening material
            * PREAMBLE: genuine introductory normative prose
        - articles are attached to the current structural container
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
        body_started = False
        index = 0

        while index < len(lines):
            page_number, line = lines[index]

            # ---------------------------------------------------------
            # Chapter detection
            # ---------------------------------------------------------
            chapter_match = CHAPTER_HEADER_RE.match(line)
            if chapter_match:
                if not body_started and pre_article_lines:
                    self._attach_pre_article_content(root, pre_article_lines)
                    pre_article_lines = []

                body_started = True
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
            # ---------------------------------------------------------
            section_container_match = SECTION_CONTAINER_RE.match(line)
            if section_container_match:
                if not body_started and pre_article_lines:
                    self._attach_pre_article_content(root, pre_article_lines)
                    pre_article_lines = []

                body_started = True
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
                if not body_started and pre_article_lines:
                    self._attach_pre_article_content(root, pre_article_lines)
                    pre_article_lines = []

                body_started = True
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
            article_match = self._match_article_header(line)
            if article_match:
                if not body_started and pre_article_lines:
                    self._attach_pre_article_content(root, pre_article_lines)
                    pre_article_lines = []

                body_started = True

                article_number = article_match.group(1).strip()
                inline_title = (article_match.group(2) or "").strip()

                consumed_title, next_index = self._consume_following_title_lines(
                    lines=lines,
                    start_index=index + 1,
                    max_lines=2,
                )

                article_title = inline_title or consumed_title
                header_body = ""

                if inline_title:
                    split_header = self._split_header_suffix_title_and_body(inline_title)
                    if split_header is not None:
                        article_title, header_body = split_header

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

                if header_body:
                    self._append_text_to_node(
                        node=current_article,
                        line=header_body,
                        page_number=page_number,
                    )

                # -----------------------------------------------------
                # Recovery heuristic:
                # sometimes extraction/normalization merges:
                #
                #   <article title line>
                #   <first body line>
                #
                # into:
                #
                #   "<TITLE> O presente regulamento ..."
                #
                # If the article still has no title, inspect the first unread
                # line and try to split it into:
                # - article title
                # - first body line
                # -----------------------------------------------------
                if not current_article.title and next_index < len(lines):
                    inline_title_result = self._split_inline_title_and_body(
                        lines[next_index][1]
                    )

                    if inline_title_result is not None:
                        recovered_title, recovered_body = inline_title_result

                        current_article.title = recovered_title
                        current_article.metadata["article_title"] = recovered_title

                        if recovered_body:
                            self._append_text_to_node(
                                node=current_article,
                                line=recovered_body,
                                page_number=lines[next_index][0],
                            )

                        index = next_index + 1
                        continue

                index = next_index
                continue

            # ---------------------------------------------------------
            # Body text routing
            # ---------------------------------------------------------
            if not body_started:
                if self._should_keep_pre_article_line(line):
                    pre_article_lines.append((page_number, line))
                index += 1
                continue

            # Ignore obvious synthetic residues that should not pollute the
            # post-article structural tree.
            if PAGE_COUNTER_RE.match(line):
                index += 1
                continue

            if self._looks_like_garbled_line(line):
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

        if not body_started and pre_article_lines:
            self._attach_pre_article_content(root, pre_article_lines)

        self._post_process_nodes(root)
        self._infer_missing_page_ranges(root)
        return root

    # ------------------------------------------------------------------
    # Input coercion / line collection
    # ------------------------------------------------------------------

    def _coerce_pages_text(
        self,
        normalized_input: Union[
            NormalizedDocument,
            Sequence[Tuple[int, str]],
            Sequence[NormalizedPage],
        ],
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

        preamble_lines = self._trim_preamble_tail_residue(preamble_lines)

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

                insert_index = (
                    1
                    if root.children and root.children[0].node_type == "FRONT_MATTER"
                    else 0
                )
                root.children.insert(insert_index, preamble_node)

    def _should_keep_pre_article_line(self, line: str) -> bool:
        """
        Decide whether a line seen before the first article should remain
        eligible for FRONT_MATTER / PREAMBLE classification.

        Why this helper exists
        ----------------------
        The pre-article region is especially vulnerable to contamination from:
        - cover residues
        - page counters
        - broken extraction artifacts
        - short administrative fragments
        - synthetic page leftovers

        The goal here is not to clean the document aggressively, but to avoid
        obviously non-structural noise from polluting the tree.

        Parameters
        ----------
        line : str
            Candidate line before the first article.

        Returns
        -------
        bool
            True when the line should be retained for further classification.
        """
        if not line or not line.strip():
            return False

        stripped = line.strip()

        if PAGE_COUNTER_RE.match(stripped):
            return False

        if NON_NORMATIVE_PRE_ARTICLE_RE.match(stripped):
            return False

        if self._looks_like_garbled_line(stripped):
            return False

        return True

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

        if self._looks_like_institutional_title_line(line):
            return True

        # Short uppercase-heavy lines are often title-page / cover / index
        # material, but intentionally limit this heuristic so legitimate
        # long structural prose is not reclassified too aggressively.
        if UPPERCASE_HEAVY_RE.match(line) and len(line.split()) <= 8:
            return True

        return False

    def _looks_like_institutional_title_line(self, line: str) -> bool:
        """
        Detect standalone institutional title lines near the document start.

        Why this helper exists
        ----------------------
        Some legal PDFs begin with title-page residue such as school or
        institution names written as standalone heading lines. Those lines are
        structurally closer to FRONT_MATTER than to genuine preamble prose.

        Parameters
        ----------
        line : str
            Candidate line.

        Returns
        -------
        bool
            True when the line behaves like a standalone institutional heading.
        """
        if not line:
            return False

        stripped = line.strip()
        if not stripped:
            return False

        if PREAMBLE_OPENING_RE.match(stripped):
            return False

        if BODY_START_RE.match(stripped):
            return False

        if not INSTITUTIONAL_TITLE_LINE_RE.match(stripped):
            return False

        if len(stripped.split()) > 18:
            return False

        if any(char.isdigit() for char in stripped):
            return False

        if "," in stripped or ";" in stripped or ":" in stripped:
            return False

        return True

    def _trim_preamble_tail_residue(
        self,
        preamble_lines: List[Tuple[int, str]],
    ) -> List[Tuple[int, str]]:
        """
        Remove non-normative residue leaked at the end of the preamble region.

        Why this helper exists
        ----------------------
        The last pre-body lines sometimes contain sign-off names, roles, or
        partial regulation titles. Those fragments should not survive inside
        PREAMBLE because they harm retrieval text and blur the body boundary.

        Parameters
        ----------
        preamble_lines : List[Tuple[int, str]]
            Candidate preamble lines after front-matter classification.

        Returns
        -------
        List[Tuple[int, str]]
            Preamble lines with trailing residue removed conservatively.
        """
        if not preamble_lines:
            return []

        trimmed_lines = list(preamble_lines)
        removed_tail = False

        while trimmed_lines and self._looks_like_preamble_tail_residue(
            trimmed_lines[-1][1]
        ):
            trimmed_lines.pop()
            removed_tail = True

        if removed_tail and trimmed_lines:
            page_number, last_line = trimmed_lines[-1]
            cleaned_last_line = TRUNCATED_PREAMBLE_TAIL_RE.sub("", last_line).strip()

            if cleaned_last_line:
                trimmed_lines[-1] = (page_number, cleaned_last_line)
            else:
                trimmed_lines.pop()

        return trimmed_lines

    def _looks_like_preamble_tail_residue(self, line: str) -> bool:
        """
        Detect trailing preamble lines that behave like sign-off or title noise.

        Parameters
        ----------
        line : str
            Candidate trailing line.

        Returns
        -------
        bool
            True when the line is more likely editorial residue than prose.
        """
        if not line:
            return False

        stripped = line.strip()
        if not stripped:
            return False

        if PREAMBLE_OPENING_RE.match(stripped):
            return False

        if BODY_START_RE.match(stripped):
            return False

        if self._looks_like_front_matter_line(stripped):
            return True

        if PERSON_NAME_LINE_RE.match(stripped):
            return True

        if NON_NORMATIVE_PRE_ARTICLE_RE.match(stripped):
            return True

        if REGULATION_TITLE_FRAGMENT_RE.match(stripped):
            return True

        if len(stripped.split()) <= 5 and stripped.endswith(":"):
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

        title = self._merge_title_lines(collected)
        return title, index

    def _merge_title_lines(self, lines: Sequence[str]) -> str:
        """
        Merge multiple title lines without injecting artificial separators.

        Why this helper exists
        ----------------------
        Structural titles may legitimately span multiple physical lines in the
        PDF, but those line breaks should not become synthetic markers such as
        " | " in parser output or downstream embedding text.

        Parameters
        ----------
        lines : Sequence[str]
            Title-like lines collected after one structural marker.

        Returns
        -------
        str
            Title rebuilt as ordinary readable text.
        """
        cleaned_lines = [line.strip() for line in lines if line and line.strip()]
        if not cleaned_lines:
            return ""

        return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()

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

        if self._match_article_header(line):
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

        if self._looks_like_garbled_line(line):
            return False

        if CITATION_HEAVY_SUFFIX_RE.search(line):
            return False

        if len(line) > 120:
            return False

        word_count = len(line.split())
        if word_count > 14:
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

        if collected_count == 0 and word_count <= 12 and line[:1].isupper():
            return True

        if collected_count == 1 and word_count <= 8 and line[:1].isupper():
            return True

        return False

    def _split_inline_title_and_body(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Try to split one line into an inline title and a body continuation.

        Why this helper exists
        ----------------------
        Some PDFs or normalization flows collapse:

            ÂMBITO
            O presente regulamento aplica-se ...

        into:

            ÂMBITO O presente regulamento aplica-se ...

        In those cases, consuming title lines in the normal way no longer works,
        because the title is no longer isolated on its own line.

        This helper attempts a conservative recovery of that pattern.

        Parameters
        ----------
        line : str
            Candidate line immediately following an article header.

        Returns
        -------
        Optional[Tuple[str, str]]
            (title, body) when the pattern looks trustworthy, otherwise None.
        """
        if not line:
            return None

        if self._looks_like_garbled_line(line):
            return None

        stripped_line = line.strip()
        match = INLINE_TITLE_BODY_SPLIT_RE.match(stripped_line)
        if match:
            title = (match.group("title") or "").strip()
            body = (match.group("body") or "").strip()

            if (
                title
                and body
                and len(title) >= 4
                and len(title.split()) <= 8
                and not title.endswith((",", ";", ":"))
                and not BODY_START_RE.match(title)
                and UPPERCASE_HEAVY_RE.match(title)
                and len(body.split()) >= 3
            ):
                return title, body

        for match in HEADER_SUFFIX_BODY_START_RE.finditer(stripped_line):
            body_start = match.start("body")
            if body_start <= 0:
                continue

            title = stripped_line[:body_start].strip()
            body = stripped_line[body_start:].strip()

            if not title or not body:
                continue

            if not self._is_probable_title_line(title):
                continue

            if len(body.split()) < 3:
                continue

            return title, body

        return None

    def _split_header_suffix_title_and_body(
        self,
        header_suffix: str,
    ) -> Optional[Tuple[str, str]]:
        """
        Recover title/body boundaries inside an inline article header suffix.

        Example
        -------
        "Definições 1. Para efeitos do presente regulamento..."
        -> ("Definições", "1. Para efeitos do presente regulamento...")

        Parameters
        ----------
        header_suffix : str
            Text captured after the article marker in the same header line.

        Returns
        -------
        Optional[Tuple[str, str]]
            (title, body) when the split looks trustworthy, otherwise None.
        """
        if not header_suffix:
            return None

        stripped_suffix = header_suffix.strip()
        if not stripped_suffix:
            return None

        for match in HEADER_SUFFIX_BODY_START_RE.finditer(stripped_suffix):
            body_start = match.start("body")
            if body_start <= 0:
                continue

            title = stripped_suffix[:body_start].strip()
            body = stripped_suffix[body_start:].strip()

            if not title or not body:
                continue

            if not self._is_probable_title_line(title):
                continue

            if len(body.split()) < 3:
                continue

            return title, body

        return None

    def _match_article_header(self, line: str) -> Optional[re.Match[str]]:
        """
        Match article headers only when the line looks structurally plausible.

        Why this helper exists
        ----------------------
        Legal prose often contains internal references such as:

            artigo 46.o -B do Decreto-Lei n.o 74/2006

        A regex-only match would incorrectly treat those citations as new
        article headers, truncating the current article body and creating
        spurious ARTICLE nodes.

        This helper keeps article detection conservative:
        - plain markers such as "Artigo 5.o" remain valid
        - inline title variants such as "Artigo 5 - Objeto" remain valid
        - merged header/body lines that can be cleanly split remain valid
        - citation-like body lines are rejected

        Parameters
        ----------
        line : str
            Candidate line from normalized parser input.

        Returns
        -------
        Optional[re.Match[str]]
            The regex match when the line behaves like a real article header,
            otherwise None.
        """
        if not line:
            return None

        match = ARTICLE_HEADER_RE.match(line)
        if not match:
            return None

        header_suffix = (match.group(2) or "").strip()
        if not header_suffix:
            return match

        if self._is_probable_title_line(header_suffix):
            return match

        if self._split_header_suffix_title_and_body(header_suffix) is not None:
            return match

        return None

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
            # Avoid duplicating derived children if the parser is reused or the
            # tree is post-processed more than once.
            has_sections = any(child.node_type == "SECTION" for child in node.children)
            has_lettered = any(
                child.node_type == "LETTERED_ITEM" for child in node.children
            )

            if not has_sections:
                section_children = self._extract_numbered_children(node)
                node.children.extend(section_children)
            else:
                section_children = []

            # Only fall back to direct lettered extraction when numbered
            # structure was not found and no lettered children already exist.
            if not section_children and not has_lettered:
                node.children.extend(self._extract_lettered_children(node))

        if node.node_type == "SECTION" and node.text:
            has_lettered = any(
                child.node_type == "LETTERED_ITEM" for child in node.children
            )
            if not has_lettered:
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

        child_starts = [
            child.page_start for child in node.children if child.page_start is not None
        ]
        child_ends = [
            child.page_end for child in node.children if child.page_end is not None
        ]

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

        matches = self._find_numbered_block_matches(text)
        if len(matches) < 2:
            return []

        labels = [
            self._normalize_numbered_label(match.group("label"))
            for match in matches
        ]
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

    def _find_numbered_block_matches(self, text: str) -> List[re.Match[str]]:
        """
        Find numbered structural markers using line-based and inline recovery.

        Why this helper exists
        ----------------------
        Real article bodies sometimes preserve legal numbering inside one long
        paragraph instead of keeping each item at the start of a separate line.
        The parser should recover that structure when the numbering remains
        clear, but only with conservative boundaries.

        Parameters
        ----------
        text : str
            Article text to inspect.

        Returns
        -------
        List[re.Match[str]]
            Ordered numbered-block matches suitable for SECTION extraction.
        """
        line_matches = list(NUMBERED_BLOCK_RE.finditer(text))
        if len(line_matches) >= 2:
            return line_matches

        if len(text) < 160:
            return line_matches

        inline_matches: List[re.Match[str]] = []

        for match in INLINE_NUMBERED_BLOCK_RE.finditer(text):
            if self._has_inline_numbered_boundary(text, match.start("label")):
                inline_matches.append(match)

        return inline_matches if len(inline_matches) >= 2 else line_matches

    def _has_inline_numbered_boundary(self, text: str, start_index: int) -> bool:
        """
        Validate that one inline numeric marker starts at a strong boundary.

        Accepted boundaries
        -------------------
        - start of text
        - newline
        - colon
        - semicolon

        Parameters
        ----------
        text : str
            Full article text.

        start_index : int
            Start offset of the numeric label.

        Returns
        -------
        bool
            True when the inline label begins at a conservative boundary.
        """
        if start_index <= 0:
            return True

        probe_index = start_index - 1
        while probe_index >= 0 and text[probe_index].isspace():
            probe_index -= 1

        if probe_index < 0:
            return True

        return text[probe_index] in {":", ";", "\n"}

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
    # Generic suspicious-line detection
    # ------------------------------------------------------------------

    def _looks_like_garbled_line(self, line: str) -> bool:
        """
        Decide whether a single line looks clearly corrupted or non-linguistic.

        Why this helper exists
        ----------------------
        Some PDFs contain isolated lines that are obviously unusable and should
        not contaminate structural nodes such as PREAMBLE.

        Important safety note
        ---------------------
        This helper is intentionally conservative and only targets obvious
        corruption, not minor OCR imperfections.

        Parameters
        ----------
        line : str
            Candidate line.

        Returns
        -------
        bool
            True when the line looks strongly suspicious.
        """
        if not line:
            return False

        stripped = line.strip()
        if len(stripped) < 10:
            return False

        if SUSPICIOUS_GARBLED_LINE_RE.match(stripped):
            return True

        total_len = len(stripped)
        alpha_count = sum(1 for ch in stripped if ch.isalpha())
        whitespace_count = sum(1 for ch in stripped if ch.isspace())
        symbol_like_count = sum(
            1
            for ch in stripped
            if not ch.isalnum() and not ch.isspace()
        )

        alpha_ratio = alpha_count / max(total_len, 1)
        symbol_ratio = symbol_like_count / max(total_len, 1)

        if alpha_ratio < 0.20 and symbol_ratio > 0.35 and whitespace_count <= 1:
            return True

        return False

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
