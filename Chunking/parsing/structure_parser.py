from __future__ import annotations

import re
from typing import List, Optional, Tuple

from Chunking.chunking.models import StructuralNode


# -------------------------------------------------------------------------
# Structure-aware regexes for Portuguese legal / regulatory documents.
#
# These patterns remain local to this module so the parser stays
# self-contained and easy to evolve.
#
# Important design principle:
# this parser should prefer conservative structure detection over aggressive
# guessing. In legal and regulatory documents, false positives are often
# more harmful than missing a small structural cue.
# -------------------------------------------------------------------------

# Accept article header variants such as:
# - "ARTIGO 1º"
# - "Artigo 1.º"
# - "Artigo 1"
# - "Art. 1"
# - "Artigo 1 - Título"
# - "Artigo 1: Título"
#
# Notes:
# - We accept "ARTIGO" and "ART." variants.
# - We allow optional ordinal markers such as º, .º, o, °.
# - We allow decimal article identifiers for robustness, even though most
#   regulations use integer article numbers.
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

# Accept annex headers such as:
# - "ANEXO"
# - "ANEXO I"
# - "ANEXO A - Título"
ANNEX_HEADER_RE = re.compile(
    r"^\s*(ANEXO(?:\s+[IVXLCDM\dA-Z]+)?)\s*(?:[—–\-:]\s*(.*))?\s*$",
    re.IGNORECASE,
)

# Accept numbered structural blocks such as:
# - "1. Texto..."
# - "2) Texto..."
# - "2.1. Texto..."
# - "2.1 — Texto..."
# - "2.1 - Texto..."
#
# Important:
# this pattern is intentionally stricter than a plain "number at line start".
# That stricter behavior avoids false positives such as:
# - "60 créditos ECTS"
# - "24 meses após ..."
# which are body lines rather than real structural section headers.
NUMBERED_BLOCK_RE = re.compile(
    r"(?m)^(?P<label>\d+(?:\.\d+)*)(?:\.\s+|\)\s+|\s+[—–\-]\s+)"
)

# Accept legal-style lettered items such as:
# - "a) ..."
# - "b) ..."
LETTERED_BLOCK_RE = re.compile(
    r"(?m)^(?P<label>[a-z])\)\s+",
    re.IGNORECASE,
)

# Uppercase-heavy lines often behave like titles in these documents.
UPPERCASE_HEAVY_RE = re.compile(
    r"^[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ0-9 /().,\-–—ºª]+$"
)

# Detect obvious body-like openings.
#
# These are intentionally biased toward rejecting paragraph-like lines from
# being mistaken for titles.
BODY_START_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*[\.\)]?\s+|[a-z]\)\s+|O\s|A\s|Os\s|As\s|Nos\s|Na\s|No\s|Considera-se\s)",
    re.IGNORECASE,
)


class StructureParser:
    """
    Parse normalized page text into a navigable structural tree.

    Design goals:
    - Be pragmatic rather than academically perfect.
    - Be robust for Portuguese legal and regulatory PDFs.
    - Preserve enough hierarchy to support high-quality chunking.
    - Enrich metadata so chunk text can stay clean.
    - Prefer conservative structure detection over risky inference.

    Important note:
    This parser assumes that the text normalizer has already preserved
    meaningful line boundaries. The parser relies heavily on those line
    boundaries to recognize article headers, titles, sections, and list items.
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
        - Everything before the first detected article is treated as preamble.
        - Articles are attached to the current structural container
          (DOCUMENT, CHAPTER, or ANNEX).
        - The article body is preserved as canonical fallback text even when
          derived SECTION or LETTERED_ITEM children are later extracted.
        """
        root = StructuralNode(
            node_type="DOCUMENT",
            label="DOCUMENT",
            metadata={
                "document_part": "regulation_body",
            },
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

            # After the first article, free text is attached to the current
            # article when one is active; otherwise it is attached to the
            # current container as a defensive fallback.
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

        Why page numbers remain attached:
        page-level provenance is needed later to populate:
        - page_start
        - page_end
        - child-derived page ranges
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
        Consume a small number of title-like lines following a structural header.

        Why this exists:
        in many legal PDFs, the article/chapter/annex title appears on the
        line immediately after the header instead of inline.

        Important:
        this logic intentionally stays conservative to avoid absorbing the
        first body sentence into the title.
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

        Positive examples:
        - "ÂMBITO"
        - "Definições"
        - "Condições para realização da inscrição"
        - "Regulamento P.PORTO/P-005/2023"

        Negative examples:
        - "1. O presente regulamento ..."
        - "a) O estudante ..."
        - "O presente regulamento aplica-se ..."
        - "Em cada ano letivo ..."
        - "Sempre que ..."

        Heuristic philosophy:
        title detection must be conservative. A missed title is preferable to
        swallowing the first body sentence into metadata.
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

        # Strong rejection of obvious paragraph-like lines.
        if len(line) > 120:
            return False

        word_count = len(line.split())
        if word_count > 12:
            return False

        lowered = line.lower()

        # Strongly reject body-style openings and discourse markers that
        # frequently begin normative prose.
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

        # Titles should not usually end in strong sentence punctuation.
        # A trailing colon is especially suspicious because it often signals
        # that the following lines are the body of a numbered rule.
        if line.endswith(".") or line.endswith(";") or line.endswith(":"):
            return False

        # Strong positive signal: uppercase-heavy title line.
        if UPPERCASE_HEAVY_RE.match(line):
            return True

        # Mixed-case titles are allowed, but the heuristic remains strict.
        #
        # First consumed title line:
        # - must be reasonably short
        # - should begin with uppercase
        if collected_count == 0 and word_count <= 8 and line[:1].isupper():
            return True

        # Second consumed title line:
        # - must be even shorter and still title-shaped
        if collected_count == 1 and word_count <= 6 and line[:1].isupper():
            return True

        return False

    def _append_text_to_node(
        self,
        node: StructuralNode,
        line: str,
        page_number: int,
    ) -> None:
        """
        Append one line of text to a node and keep its page range updated.
        """
        if node.page_start is None:
            node.page_start = page_number

        node.page_end = page_number
        node.text = f"{node.text}\n{line}".strip()

    def _post_process_nodes(self, node: StructuralNode) -> None:
        """
        Perform recursive post-processing.

        Tasks:
        - normalize node text
        - recursively post-process children
        - derive SECTION children from article text when appropriate
        - derive LETTERED_ITEM children from article/section text when appropriate
        - infer page ranges from children when missing

        Important implementation choice:
        ARTICLE nodes keep their full canonical text even after SECTION or
        LETTERED_ITEM children are extracted. This is intentional because the
        chunking layer may still need the full article text as a fallback.
        """
        if node.text:
            node.text = self._normalize_node_text(node.text)

        for child in node.children:
            self._post_process_nodes(child)

        if node.node_type == "ARTICLE" and node.text:
            section_children = self._extract_numbered_children(node)
            node.children.extend(section_children)

            # Only fall back to direct lettered extraction from the article
            # when no numbered section structure was found. This avoids
            # creating redundant parallel structures unnecessarily.
            if not section_children:
                node.children.extend(self._extract_lettered_children(node))

        # Section-level lettered extraction is still useful because many legal
        # provisions use numbered sections with nested alíneas.
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

        This helper remains deliberately light because:
        - the parser still benefits from newline structure
        - downstream chunking may rely on retained line boundaries
        - aggressive text flattening belongs elsewhere in the pipeline
        """
        text = text.strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_numbered_children(self, article: StructuralNode) -> List[StructuralNode]:
        """
        Split article text into numbered section-like children.

        Supported structural examples:
        - "1. ..."
        - "2. ..."
        - "2.1. ..."
        - "2.1 - ..."
        - "2.1 — ..."
        - "2) ..."

        Important safety behavior:
        this method rejects weak or implausible matches so that ordinary body
        lines beginning with numbers are not mistaken for structural sections.
        """
        text = article.text.strip()
        if not text:
            return []

        matches = list(NUMBERED_BLOCK_RE.finditer(text))

        # Do not create derived SECTION nodes unless the article clearly
        # behaves like a sectioned provision.
        if len(matches) < 2:
            return []

        labels = [match.group("label") for match in matches]
        if not self._has_plausible_numbered_structure(labels):
            return []

        children: List[StructuralNode] = []

        for match_index, match in enumerate(matches):
            label = match.group("label")
            start = match.start()
            end = (
                matches[match_index + 1].start()
                if match_index + 1 < len(matches)
                else len(text)
            )
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

    def _has_plausible_numbered_structure(self, labels: List[str]) -> bool:
        """
        Validate whether extracted numeric labels look like genuine structure.

        Why this validation exists:
        regex-only matching is not enough for legal text. Without an additional
        plausibility check, the parser may incorrectly interpret lines such as:
        - "60 créditos ECTS ..."
        - "24 meses após ..."
        as numbered sections.

        Current acceptance rule:
        - at least two labels must exist
        - labels should form a plausible increasing sequence
        - simple numeric labels such as 1, 2, 3 are strongly preferred
        - decimal labels such as 2.1, 2.2 are accepted when they remain
          coherent with adjacent labels
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

        # Require at least one plausible transition and prefer the majority
        # of transitions to behave structurally.
        required = max(1, len(labels) - 2)
        return plausible_transitions >= required

    def _is_plausible_label_transition(
        self,
        previous_parts: List[int],
        current_parts: List[int],
    ) -> bool:
        """
        Decide whether a transition between two numeric labels is structurally plausible.

        Accepted examples:
        - 1   -> 2
        - 2   -> 3
        - 2   -> 2.1
        - 2.1 -> 2.2
        - 2.9 -> 3
        - 3.1 -> 4

        Rejected examples:
        - 2   -> 60
        - 2.1 -> 24
        - 3.1 -> 9.7
        """
        # Same nesting depth: expect a small forward increment on the last part.
        if len(previous_parts) == len(current_parts):
            if previous_parts[:-1] == current_parts[:-1]:
                return 1 <= (current_parts[-1] - previous_parts[-1]) <= 2

            # Allow top-level jumps only when they remain small and simple.
            if len(previous_parts) == 1:
                return 1 <= (current_parts[0] - previous_parts[0]) <= 2

            return False

        # One level deeper: e.g. 2 -> 2.1
        if len(current_parts) == len(previous_parts) + 1:
            return current_parts[:-1] == previous_parts and current_parts[-1] == 1

        # One level shallower: e.g. 2.9 -> 3 or 3.1 -> 4
        if len(current_parts) + 1 == len(previous_parts):
            if len(current_parts) == 1:
                return 0 <= (current_parts[0] - previous_parts[0]) <= 1
            return False

        return False

    def _extract_lettered_children(self, node: StructuralNode) -> List[StructuralNode]:
        """
        Split text into lettered items.

        Supported examples:
        - "a) ..."
        - "b) ..."

        Why this is useful:
        even when lettered items do not become standalone chunks immediately,
        preserving them as explicit structure improves metadata quality and
        future chunking flexibility.
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
                )
            )

        return children