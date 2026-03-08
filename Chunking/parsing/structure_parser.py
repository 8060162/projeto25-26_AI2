from __future__ import annotations

import re
from typing import List, Optional, Tuple

from Chunking.chunking.models import StructuralNode


# -------------------------------------------------------------------------
# Structure-aware regexes for Portuguese legal / regulatory documents.
#
# We keep them inside this file so the parser becomes self-contained and
# easier to evolve without depending on another module.
# -------------------------------------------------------------------------

# Accept variants like:
# "ARTIGO 1º"
# "Artigo 1.º"
# "Artigo 1"
# "Artigo 1 - Título"
ARTICLE_HEADER_RE = re.compile(
    r"^\s*ARTIGO\s+(\d+)(?:\s*\.?\s*[ºo])?\s*(?:[—–\-:]\s*(.*))?\s*$",
    re.IGNORECASE,
)

# Accept variants like:
# "CAPÍTULO I"
# "CAPITULO IV"
# "CAPÍTULO IV - Título"
CHAPTER_HEADER_RE = re.compile(
    r"^\s*CAP[ÍI]TULO\s+([IVXLCDM\d]+)\s*(?:[—–\-:]\s*(.*))?\s*$",
    re.IGNORECASE,
)

# Accept variants like:
# "ANEXO"
# "ANEXO I"
# "ANEXO A - Título"
ANNEX_HEADER_RE = re.compile(
    r"^\s*(ANEXO(?:\s+[IVXLCDM\dA-Z]+)?)\s*(?:[—–\-:]\s*(.*))?\s*$",
    re.IGNORECASE,
)

# Accept numbered blocks like:
# "1."
# "2."
# "2.1"
# "2.1."
# "2.1 -"
# "2.1 —"
NUMBERED_BLOCK_RE = re.compile(
    r"(?m)^(?P<label>\d+(?:\.\d+)*)(?:\.)?\s*(?:[—–\-]\s*)?"
)

# Accept lettered items like:
# "a)"
# "b)"
LETTERED_BLOCK_RE = re.compile(
    r"(?m)^(?P<label>[a-z])\)\s+",
    re.IGNORECASE,
)

# Uppercase-heavy lines are often titles in these documents.
UPPERCASE_HEAVY_RE = re.compile(r"^[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ0-9 /().,\-–—ºª]+$")

# Body-like lines often start like this.
BODY_START_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*\.?\s*|[a-z]\)\s+|O\s|A\s|Os\s|As\s|Nos\s|Na\s|No\s|Considera-se\s)",
    re.IGNORECASE,
)


class StructureParser:
    """
    Parse a normalized document into a navigable structure.

    Design goals:
    - Be pragmatic rather than academically perfect.
    - Be robust for regulatory PDFs with article-based structure.
    - Preserve enough hierarchy to generate semantically coherent chunks.
    - Enrich metadata so chunk text can stay clean.
    - Be tolerant to slight formatting variations across documents.
    """

    def parse(self, pages_text: List[tuple[int, str]]) -> StructuralNode:
        """
        Parse normalized page text into a structural tree.

        Expected high-level output:
        DOCUMENT
          ├── PREAMBLE
          ├── ANNEX / CHAPTER
          │     ├── ARTICLE
          │     │     ├── SECTION
          │     │     ├── LETTERED_ITEM
          │     │     └── ...
          │     └── ARTICLE
          └── ...

        Important:
        This parser assumes the normalizer preserved line boundaries.
        """
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
        )

        lines = self._collect_lines(pages_text)

        current_container: StructuralNode = root
        current_article: Optional[StructuralNode] = None

        preamble_lines: List[str] = []
        preamble_page_start: Optional[int] = None
        preamble_page_end: Optional[int] = None

        first_article_seen = False
        index = 0

        while index < len(lines):
            page_number, line = lines[index]

            # ---------------------------------------------------------
            # Chapter detection
            # ---------------------------------------------------------
            chapter_match = CHAPTER_HEADER_RE.match(line)
            if chapter_match:
                chapter_label = chapter_match.group(1).strip()
                inline_title = (chapter_match.group(2) or "").strip()

                consumed_title, next_index = self._consume_following_title_lines(
                    lines=lines,
                    start_index=index + 1,
                    max_lines=2,
                )

                chapter_title = inline_title or consumed_title

                chapter_node = StructuralNode(
                    node_type="CHAPTER",
                    label=f"CAP_{chapter_label}",
                    title=chapter_title,
                    page_start=page_number,
                    page_end=page_number,
                    metadata={
                        "chapter_number": chapter_label,
                        "document_part": "regulation_body",
                    },
                )

                root.children.append(chapter_node)
                current_container = chapter_node
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

                annex_node = StructuralNode(
                    node_type="ANNEX",
                    label=annex_label,
                    title=annex_title,
                    page_start=page_number,
                    page_end=page_number,
                    metadata={
                        "document_part": "regulation_annex",
                    },
                )

                root.children.append(annex_node)
                current_container = annex_node
                current_article = None
                index = next_index
                continue

            # ---------------------------------------------------------
            # Article detection
            # ---------------------------------------------------------
            article_match = ARTICLE_HEADER_RE.match(line)
            if article_match:
                first_article_seen = True

                article_number = article_match.group(1).strip()
                inline_title = (article_match.group(2) or "").strip()

                consumed_title, next_index = self._consume_following_title_lines(
                    lines=lines,
                    start_index=index + 1,
                    max_lines=2,
                )

                article_title = inline_title or consumed_title

                current_article = StructuralNode(
                    node_type="ARTICLE",
                    label=f"ART_{article_number}",
                    title=article_title,
                    page_start=page_number,
                    page_end=page_number,
                    metadata={
                        "article_number": article_number,
                        "article_title": article_title,
                        "parent_type": current_container.node_type,
                        "parent_label": current_container.label,
                        "parent_title": current_container.title,
                        "document_part": current_container.metadata.get(
                            "document_part",
                            "regulation_body",
                        ),
                    },
                )

                current_container.children.append(current_article)
                index = next_index
                continue

            # ---------------------------------------------------------
            # Body text routing
            # ---------------------------------------------------------
            if not first_article_seen:
                if preamble_page_start is None:
                    preamble_page_start = page_number
                preamble_page_end = page_number
                preamble_lines.append(line)
                index += 1
                continue

            target = current_article or current_container
            self._append_text_to_node(
                node=target,
                line=line,
                page_number=page_number,
            )
            index += 1

        preamble_text = self._normalize_node_text("\n".join(preamble_lines))
        if preamble_text:
            preamble_node = StructuralNode(
                node_type="PREAMBLE",
                label="PREAMBLE",
                title="",
                text=preamble_text,
                page_start=preamble_page_start,
                page_end=preamble_page_end,
                metadata={
                    "document_part": "dispatch_or_intro",
                },
            )
            root.children.insert(0, preamble_node)

        self._post_process_nodes(root)
        return root

    def _collect_lines(self, pages_text: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """
        Flatten page texts into a single ordered list of (page_number, line).

        We intentionally keep page numbers attached to lines so we can
        propagate page_start and page_end into structural nodes.
        """
        lines: List[Tuple[int, str]] = []

        for page_number, page_text in pages_text:
            for raw_line in page_text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                lines.append((page_number, line))

        return lines

    def _consume_following_title_lines(
        self,
        lines: List[Tuple[int, str]],
        start_index: int,
        max_lines: int,
    ) -> Tuple[str, int]:
        """
        Consume a small number of title-like lines after a structural header.

        Important change:
        This version is more conservative than before and tries hard not to
        absorb the first body sentence into the title.
        """
        collected: List[str] = []
        index = start_index

        while index < len(lines) and len(collected) < max_lines:
            _, line = lines[index]

            if not self._is_probable_title_line(line, collected_count=len(collected)):
                break

            collected.append(line)
            index += 1

        title = " | ".join(collected).strip()
        return title, index

    def _is_probable_title_line(self, line: str, collected_count: int = 0) -> bool:
        """
        Heuristic for deciding whether a line looks like a title rather than body text.

        Positive examples:
        - "ÂMBITO"
        - "Definições"
        - "Condições para realização da inscrição"
        - "Regulamento P.PORTO/P-005/2023"

        Negative examples:
        - "1. O presente regulamento ..."
        - "a) O estudante ..."
        - "O presente regulamento aplica-se ..."
        """
        if not line:
            return False

        if ARTICLE_HEADER_RE.match(line):
            return False

        if CHAPTER_HEADER_RE.match(line):
            return False

        if ANNEX_HEADER_RE.match(line):
            return False

        if BODY_START_RE.match(line):
            return False

        if LETTERED_BLOCK_RE.match(line):
            return False

        # Strongly reject obvious paragraph-like lines.
        if len(line) > 120:
            return False

        if len(line.split()) > 14:
            return False

        # If it clearly looks like body prose, reject it.
        lowered = line.lower()
        body_starters = (
            "o presente",
            "a presente",
            "os/as",
            "o/a",
            "cada ",
            "considera-se",
            "nos termos",
            "mediante ",
            "deverá ",
            "deverão ",
        )
        if lowered.startswith(body_starters):
            return False

        # First consumed title line may be uppercase-heavy or short title case.
        # A second title line is only accepted if still clearly title-like.
        if line.endswith(".") or line.endswith(";") or line.endswith(":"):
            return False

        if UPPERCASE_HEAVY_RE.match(line):
            return True

        # Title case / mixed case short line can still be a valid title.
        if len(line.split()) <= 10 and collected_count == 0:
            return True

        # A second title line should be stricter.
        if len(line.split()) <= 8 and collected_count == 1:
            return True

        return False

    def _append_text_to_node(
        self,
        node: StructuralNode,
        line: str,
        page_number: int,
    ) -> None:
        """
        Append one text line to a node while keeping page range updated.
        """
        if node.page_start is None:
            node.page_start = page_number

        node.page_end = page_number
        node.text = f"{node.text}\n{line}".strip()

    def _post_process_nodes(self, node: StructuralNode) -> None:
        """
        Perform post-processing tasks:
        - normalize node text
        - extract numbered sections inside articles
        - extract lettered items inside sections/articles
        - infer page ranges from children when needed
        """
        if node.text:
            node.text = self._normalize_node_text(node.text)

        for child in node.children:
            self._post_process_nodes(child)

        if node.node_type == "ARTICLE" and node.text:
            section_children = self._extract_numbered_children(node)
            node.children.extend(section_children)

            # If no numbered sections were found, we can still try to extract
            # lettered items directly from the article body.
            if not section_children:
                node.children.extend(self._extract_lettered_children(node))

        # Extract lettered items inside sections too.
        if node.node_type == "SECTION" and node.text:
            node.children.extend(self._extract_lettered_children(node))

        if node.page_end is None and node.children:
            node.page_end = max(
                child.page_end or child.page_start or 0
                for child in node.children
            )

        if node.page_start is None and node.children:
            node.page_start = min(
                child.page_start or 10**9
                for child in node.children
            )

    def _normalize_node_text(self, text: str) -> str:
        """
        Normalize node text while preserving structural line boundaries.

        This is intentionally lighter than a generic whitespace normalizer
        because the chunker may still want to use line boundaries.
        """
        text = text.strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_numbered_children(self, article: StructuralNode) -> List[StructuralNode]:
        """
        Split article text into numbered sub-blocks.

        Supported examples:
        - "1. ..."
        - "2. ..."
        - "2.1 ..."
        - "2.2 ..."
        - "2.1 - ..."
        - "2.1 — ..."

        Why this matters:
        - section-level metadata becomes richer
        - large articles can be chunked more intelligently
        """
        text = article.text.strip()
        if not text:
            return []

        matches = list(NUMBERED_BLOCK_RE.finditer(text))

        # Only treat the text as sectioned if we found at least 2 blocks.
        # This avoids creating a useless single SECTION for articles that
        # only begin with one numbered line.
        if len(matches) < 2:
            return []

        children: List[StructuralNode] = []

        for match_index, match in enumerate(matches):
            label = match.group("label")
            start = match.start()
            end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(text)
            block_text = text[start:end].strip()

            children.append(
                StructuralNode(
                    node_type="SECTION",
                    label=label,
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
                    },
                )
            )

        return children

    def _extract_lettered_children(self, node: StructuralNode) -> List[StructuralNode]:
        """
        Split text into lettered items.

        Supported examples:
        - "a) ..."
        - "b) ..."

        Why this is useful:
        even if lettered items do not become standalone chunks immediately,
        preserving them as structure helps metadata richness and future
        chunking improvements.
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
            end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(text)
            block_text = text[start:end].strip()

            children.append(
                StructuralNode(
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
                        "article_label": node.metadata.get("article_label", node.label if node.node_type == "ARTICLE" else None),
                        "article_number": node.metadata.get("article_number"),
                        "article_title": node.metadata.get("article_title", node.title),
                    },
                )
            )

        return children