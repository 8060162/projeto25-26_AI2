from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from Chunking.chunking.models import PageText
from Chunking.utils.text import (
    join_hyphenated_linebreaks,
    normalize_block_whitespace,
    normalize_line_endings,
    strip_control_characters,
)


# -------------------------------------------------------------------------
# Regexes used by the normalizer
# -------------------------------------------------------------------------
#
# The purpose of this module is intentionally narrow and conservative.
#
# It is NOT a full legal-structure parser.
# It should:
# - remove obvious PDF / editorial / layout noise
# - preserve structural line boundaries useful to the parser
# - repair some extraction artifacts conservatively
# - avoid aggressive cleanup that could silently remove legal content
#
# This module sits between extraction and structure parsing, so it must be
# careful: once text is destroyed here, later pipeline stages cannot recover it.
# -------------------------------------------------------------------------

# Example:
# "Pág. 363"
# "Pág. 364"
#
# We strip this marker only when it appears at the beginning of the line so
# that useful text after the marker can still survive.
LEADING_PAGE_MARKER_RE = re.compile(r"^\s*Pág\.\s*\d+\s*", re.IGNORECASE)

# Example:
# "3|14"
# "10 | 14"
#
# These are classic page counters extracted from PDF footers / headers.
INLINE_PAGE_COUNTER_RE = re.compile(r"^\s*\d+\s*\|\s*\d+\s*$")

# Example:
# "316552597"
#
# Long pure numeric tails are often publication / layout noise.
PURE_NUMERIC_NOISE_RE = re.compile(r"^\s*\d{6,}\s*$")

# Typical Diário da República editorial lines or similar page furniture.
DR_EDITORIAL_RE = re.compile(
    r"^\s*(N\.?\s*º|PARTE\s+[A-Z]|Diário da República)\b",
    re.IGNORECASE,
)

# Likely signature / signing lines that should not pollute retrieval text.
SIGNATURE_LINE_RE = re.compile(
    r"^\s*(Assinado por:|Num\.?\s+de\s+Identificação:|Data:\s*\d{4}\.\d{2}\.\d{2})",
    re.IGNORECASE,
)

# Decorative / cover-like lines that often belong to title pages rather than
# normative body content.
COVER_NOISE_RE = re.compile(
    r"^\s*(DESPACHO\s+P\.PORTO\/P-|REGULAMENTO\s+P\.PORTO\/P-)\S*",
    re.IGNORECASE,
)

# Detect obvious dot-leader lines frequently used in tables of contents.
INDEX_DOT_LEADER_RE = re.compile(r"\.{3,}")

# Detect short lines ending with a page number, often TOC style.
INDEX_TRAILING_PAGE_RE = re.compile(r".+\s+\d{1,4}\s*$")

# Typical heading-like TOC entries.
INDEX_HEADING_RE = re.compile(
    r"^\s*(CAP[ÍI]TULO|ARTIGO|ANEXO|ÍNDICE|SECÇÃO|SUBSECÇÃO|TÍTULO)\b",
    re.IGNORECASE,
)

# Normalize inner whitespace for comparison and reporting.
MULTISPACE_RE = re.compile(r"\s+")

# Detect common structural markers that should usually remain on their own line.
ARTICLE_HEADER_RE = re.compile(
    r"^\s*(?:ARTIGO|ART\.?)\s+\d+(?:\.\d+)?\s*(?:\.?\s*[ºo°])?\b",
    re.IGNORECASE,
)
CHAPTER_HEADER_RE = re.compile(
    r"^\s*CAP[ÍI]TULO\s+[IVXLCDM\d]+\b",
    re.IGNORECASE,
)
ANNEX_HEADER_RE = re.compile(
    r"^\s*ANEXO(?:\s+[IVXLCDM\dA-Z]+)?\b",
    re.IGNORECASE,
)
SECTION_HEADER_RE = re.compile(
    r"^\s*(?:SECÇÃO|SUBSECÇÃO|TÍTULO)\b",
    re.IGNORECASE,
)

# Detect numbered legal items such as:
# - "1."
# - "1)"
# - "1 -"
# - "1 —"
# - "2.1."
# - "2.1 -"
NUMBERED_ITEM_RE = re.compile(
    r"^\s*\d+(?:\.\d+)*(?:\.\s+|\)\s+|\s+[—–\-]\s+)"
)

# Detect lettered legal items such as:
# - "a) ..."
# - "b) ..."
LETTERED_ITEM_RE = re.compile(r"^\s*[a-z]\)\s+", re.IGNORECASE)

# Detect likely starts of ordinary prose.
PROSE_START_RE = re.compile(
    r"^\s*(?:O|A|Os|As|No|Na|Nos|Nas|Em|Para|Por|Quando|Sempre|Caso|Se|Nos\s+termos|Deve|Devem|Pode|Podem|É|São)\b",
    re.IGNORECASE,
)

# Detect line endings that usually indicate completed sentences / boundaries.
HARD_LINE_END_RE = re.compile(r"[.;:!?]$")


@dataclass(slots=True)
class NormalizedDocument:
    """
    Intermediate normalized document representation.

    It keeps:
    - page-level cleaned text
    - a document-level consolidated body
    - a dropped-lines report useful for inspection and debugging

    Important:
    this structure intentionally remains lightweight so it does not break the
    rest of the pipeline.
    """

    pages: List[PageText]
    full_text: str
    dropped_lines_report: Dict[str, int]


class TextNormalizer:
    """
    Remove PDF layout noise before structure parsing and chunking.

    Design principles
    -----------------
    1. Stay conservative
       It is safer to keep a little noise than to remove real legal content.

    2. Preserve structural line boundaries
       The downstream parser relies on line boundaries to detect:
       - article headers
       - chapter headers
       - annex markers
       - numbered provisions
       - lettered items

    3. Repair only obvious extraction artifacts
       We do not aggressively flatten the text here.

    4. Be more careful with repeated-line detection
       Repeated lines are often headers / footers, but legal text may also
       repeat legitimate phrases. Therefore repeated-line heuristics should
       focus mainly on page margins rather than the full body.
    """

    # Only consider a small top / bottom window of each page when detecting
    # repeated layout furniture. This reduces accidental removal of valid body
    # text that happens to repeat.
    _REPEATED_LINE_MARGIN_WINDOW = 4

    # TOC / index blocks usually live near the beginning of a document.
    # We limit aggressive TOC block removal to the early pages.
    _MAX_TOC_SCAN_PAGES = 5

    def normalize(self, pages: List[PageText]) -> NormalizedDocument:
        """
        Normalize a list of page-level extracted texts.

        Processing phases
        -----------------
        1. Detect repeated short lines that likely behave like headers/footers
        2. Clean line by line, preserving structural boundaries
        3. Remove likely TOC / index blocks, mainly in early pages
        4. Reflow obvious broken prose lines conservatively
        5. Rebuild page text with shared block-level normalization
        6. Concatenate pages into a debug-friendly full_text

        Why repeated-line detection happens first
        -----------------------------------------
        Repeated page furniture is easier to recognize globally before page-by-
        page cleanup removes contextual clues.

        Parameters
        ----------
        pages : List[PageText]
            Raw extracted page texts.

        Returns
        -------
        NormalizedDocument
            Page-level normalized text plus a consolidated debug view.
        """
        repeated_lines = self._find_repeated_noise_candidates(pages)
        cleaned_pages: List[PageText] = []
        dropped_counter: Counter[str] = Counter()
        total_pages = len(pages)

        for page in pages:
            cleaned_lines: List[str] = []

            for raw_line in page.text.splitlines():
                line, drop_reason = self._clean_line(
                    raw_line=raw_line,
                    repeated_lines=repeated_lines,
                )

                if drop_reason is not None:
                    dropped_counter[drop_reason] += 1
                    continue

                if line is None:
                    continue

                cleaned_lines.append(line)

            # Remove likely TOC/index blocks, but do it conservatively and
            # mainly on early pages where tables of contents typically appear.
            cleaned_lines, block_drop_report = self._remove_index_blocks(
                lines=cleaned_lines,
                page_number=page.page_number,
                total_pages=total_pages,
            )
            dropped_counter.update(block_drop_report)

            # Reflow obvious broken prose lines while preserving structural
            # markers such as article headers and legal list items.
            cleaned_lines = self._reflow_lines(cleaned_lines)

            cleaned_text = self._finalize_page_text(cleaned_lines)

            cleaned_pages.append(
                PageText(
                    page_number=page.page_number,
                    text=cleaned_text,
                )
            )

        full_text = self._build_full_text(cleaned_pages)

        return NormalizedDocument(
            pages=cleaned_pages,
            full_text=full_text,
            dropped_lines_report=dict(dropped_counter),
        )

    def _clean_line(
        self,
        raw_line: str,
        repeated_lines: set[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Clean a single line and decide whether it should be dropped.

        Return contract
        ---------------
        - (cleaned_line, None) when the line should remain
        - (None, "reason") when the line should be dropped

        Important behavior
        ------------------
        - We strip leading page markers rather than dropping the whole line
          because valid legal content may still follow the marker.
        - We keep line-level structure intact because later parsing depends on it.
        - We do not attempt semantic interpretation here.
        """
        line = normalize_line_endings(raw_line).strip()

        if not line:
            return None, "empty_line"

        # Remove non-printable control-character noise early so later regexes
        # operate on cleaner text.
        line = strip_control_characters(line).strip()
        if not line:
            return None, "control_char_only"

        # Remove leading page markers such as "Pág. 364".
        new_line = LEADING_PAGE_MARKER_RE.sub("", line).strip()
        if new_line != line:
            line = new_line
            if not line:
                return None, "page_marker_only"

        # Remove classic page counters like "3|14".
        if INLINE_PAGE_COUNTER_RE.match(line):
            return None, "inline_page_counter"

        # Remove long numeric publication/layout tails.
        if PURE_NUMERIC_NOISE_RE.match(line):
            return None, "trailing_numeric_code"

        # Remove pure Diário da República editorial lines.
        if DR_EDITORIAL_RE.match(line) and len(line) <= 140:
            return None, "dr_editorial_line"

        # Remove signature metadata from body text.
        if SIGNATURE_LINE_RE.match(line):
            return None, "signature_line"

        # Remove obvious decorative cover lines.
        if COVER_NOISE_RE.match(line) and len(line) <= 140:
            return None, "cover_line"

        # Remove repeated page furniture lines.
        #
        # Important:
        # repeated-line candidates are now detected mainly from page margins,
        # which makes this step much safer than scanning the whole body text.
        normalized_for_compare = self._normalize_for_comparison(line)
        if normalized_for_compare in repeated_lines:
            return None, "repeated_line"

        # Remove very obvious single-line TOC entries early.
        if self._looks_like_index_line(line):
            return None, "index_like_line"

        # Normalize inner whitespace while preserving line identity.
        line = MULTISPACE_RE.sub(" ", line).strip()

        if not line:
            return None, "empty_after_cleanup"

        return line, None

    def _find_repeated_noise_candidates(self, pages: List[PageText]) -> set[str]:
        """
        Find short normalized lines repeated across page margins.

        Why this heuristic changed
        --------------------------
        The previous implementation looked across all lines on the page.
        That is risky because valid legal phrases may repeat in the body.

        The improved approach only looks at a small top/bottom window of each
        page, because repeated headers and footers almost always live there.

        Parameters
        ----------
        pages : List[PageText]
            Raw extracted pages.

        Returns
        -------
        set[str]
            Normalized repeated-line candidates likely to be page furniture.
        """
        counter: Counter[str] = Counter()
        total_pages = max(1, len(pages))

        for page in pages:
            raw_lines = [line for line in page.text.splitlines() if line.strip()]
            if not raw_lines:
                continue

            margin_lines = self._margin_lines(raw_lines)
            unique_page_lines = {
                self._normalize_for_comparison(line)
                for line in margin_lines
                if line.strip()
            }

            for line in unique_page_lines:
                if not line:
                    continue

                # Only short lines are eligible as repeated page furniture.
                if len(line) <= 120:
                    counter[line] += 1

        # Use a conservative threshold:
        # - at least two pages
        # - at least half the pages for larger documents
        threshold = max(2, total_pages // 2)

        return {
            line
            for line, count in counter.items()
            if count >= threshold
        }

    def _margin_lines(self, lines: List[str]) -> List[str]:
        """
        Return the likely page-margin lines from a page.

        Why this helper exists
        ----------------------
        Repeated headers and footers are usually near the top or bottom of the
        page. Restricting repeated-line analysis to these regions reduces the
        chance of dropping legitimate repeated legal prose.

        Parameters
        ----------
        lines : List[str]
            Non-empty lines from a page.

        Returns
        -------
        List[str]
            Combined top/bottom window lines.
        """
        if not lines:
            return []

        window = min(self._REPEATED_LINE_MARGIN_WINDOW, len(lines))
        top = lines[:window]
        bottom = lines[-window:] if len(lines) > window else []
        return top + bottom

    def _normalize_for_comparison(self, line: str) -> str:
        """
        Normalize a line specifically for repeated-line comparison.

        This helper intentionally removes only superficial layout differences:
        - line ending variation
        - control characters
        - leading page marker
        - repeated internal whitespace

        It does not attempt semantic normalization.
        """
        line = normalize_line_endings(line).strip()
        line = strip_control_characters(line).strip()
        line = LEADING_PAGE_MARKER_RE.sub("", line).strip()
        line = MULTISPACE_RE.sub(" ", line).strip()
        return line

    def _looks_like_index_line(self, line: str) -> bool:
        """
        Detect likely single-line table-of-contents entries.

        Strong signals
        --------------
        - dot leaders
        - heading-like text followed by a trailing page number

        Safety behavior
        ---------------
        This heuristic is intentionally conservative and should only catch
        obvious cases. More ambiguous TOC patterns are handled later at
        block level.
        """
        if len(line) > 220:
            return False

        if INDEX_DOT_LEADER_RE.search(line):
            return True

        if INDEX_TRAILING_PAGE_RE.match(line):
            words = line.split()

            if len(words) <= 14 and INDEX_HEADING_RE.match(line):
                return True

        return False

    def _remove_index_blocks(
        self,
        lines: List[str],
        page_number: int,
        total_pages: int,
    ) -> Tuple[List[str], Counter[str]]:
        """
        Remove likely table-of-contents blocks.

        Why block-level detection matters
        ---------------------------------
        A TOC is often easier to identify as a cluster of lines rather than as
        isolated single lines.

        Safety rules
        ------------
        - aggressive TOC block removal is mainly limited to early pages
        - candidate blocks must have enough density/size to behave like a TOC
        - short ambiguous runs are preserved

        Parameters
        ----------
        lines : List[str]
            Already cleaned lines for one page.
        page_number : int
            1-based page number.
        total_pages : int
            Total number of pages in the document.

        Returns
        -------
        Tuple[List[str], Counter[str]]
            The kept lines plus a block-level dropped-lines report.
        """
        if not lines:
            return [], Counter()

        # TOC blocks are overwhelmingly likely to appear near the beginning.
        if not self._is_early_document_page(page_number, total_pages):
            return lines, Counter()

        kept_lines: List[str] = []
        dropped_counter: Counter[str] = Counter()

        index = 0
        while index < len(lines):
            block_lines: List[str] = []

            while index < len(lines):
                line = lines[index]
                is_toc_like = self._is_toc_block_candidate(line)

                if not is_toc_like and not block_lines:
                    break

                if not is_toc_like and block_lines:
                    break

                block_lines.append(line)
                index += 1

            # Only remove blocks that are large enough and TOC-like enough.
            if block_lines and self._looks_like_real_toc_block(block_lines):
                dropped_counter["index_block_line"] += len(block_lines)
                continue

            if block_lines:
                kept_lines.extend(block_lines)
                continue

            kept_lines.append(lines[index])
            index += 1

        return kept_lines, dropped_counter

    def _is_early_document_page(self, page_number: int, total_pages: int) -> bool:
        """
        Decide whether a page belongs to the document front matter region.

        Why this matters
        ----------------
        Title pages and tables of contents usually live near the beginning.
        Restricting aggressive TOC removal to early pages prevents accidental
        deletion of legitimate structural blocks later in the body.

        Returns
        -------
        bool
            True when the page is early enough to justify TOC-specific cleanup.
        """
        max_scan_pages = min(self._MAX_TOC_SCAN_PAGES, max(1, total_pages // 3 + 1))
        return page_number <= max_scan_pages

    def _is_toc_block_candidate(self, line: str) -> bool:
        """
        Decide whether a line looks like part of a TOC block.

        Strong signals
        --------------
        - dot leaders
        - explicit structural headings such as "CAPÍTULO", "ARTIGO", "ÍNDICE"
        - short heading ending with a page number

        Important:
        this helper alone is not enough to remove text. It only marks lines as
        candidates. Final removal requires block-level evidence.
        """
        if INDEX_DOT_LEADER_RE.search(line):
            return True

        if INDEX_HEADING_RE.match(line):
            return True

        if INDEX_TRAILING_PAGE_RE.match(line) and len(line.split()) <= 16:
            return True

        return False

    def _looks_like_real_toc_block(self, block_lines: List[str]) -> bool:
        """
        Decide whether a candidate block really behaves like a TOC.

        Why this extra validation exists
        --------------------------------
        A sequence such as:

            CAPÍTULO I
            Disposições gerais
            ARTIGO 1.º
            Objeto

        may be real document content rather than a TOC.
        Therefore we require stronger TOC signals before dropping a block.

        Current acceptance logic
        ------------------------
        A block is considered TOC-like when:
        - it is large enough, and
        - it contains multiple strong TOC signals such as:
            * dot leaders
            * trailing page numbers
        """
        if len(block_lines) < 4:
            return False

        strong_signals = 0

        for line in block_lines:
            if INDEX_DOT_LEADER_RE.search(line):
                strong_signals += 2
                continue

            if INDEX_TRAILING_PAGE_RE.match(line) and len(line.split()) <= 16:
                strong_signals += 1

        return strong_signals >= 3

    def _reflow_lines(self, lines: List[str]) -> List[str]:
        """
        Reflow obvious broken prose lines conservatively.

        Why this exists
        ---------------
        PDF extraction often breaks ordinary prose into multiple short lines.
        If those lines remain separated, later paragraph grouping produces
        artificial paragraph boundaries and weaker chunks.

        Safety behavior
        ---------------
        We only merge lines when:
        - the previous line does not look like a structural marker
        - the current line does not look like a structural marker
        - the previous line does not end like a hard sentence boundary
        - the current line looks like prose continuation

        We deliberately avoid aggressive paragraph reconstruction.
        """
        if not lines:
            return []

        rebuilt: List[str] = [lines[0]]

        for current_line in lines[1:]:
            previous_line = rebuilt[-1]

            if self._should_merge_lines(previous_line, current_line):
                rebuilt[-1] = f"{previous_line} {current_line}".strip()
            else:
                rebuilt.append(current_line)

        return rebuilt

    def _should_merge_lines(self, previous_line: str, current_line: str) -> bool:
        """
        Decide whether two consecutive lines should be merged into one prose line.

        Merge-friendly example
        ----------------------
        Previous:
            "O estudante pode requerer a revisão da prova"
        Current:
            "no prazo de cinco dias úteis"

        Keep-separate example
        ---------------------
        Previous:
            "Artigo 5.º"
        Current:
            "Revisão de prova"

        or

        Previous:
            "1. O estudante ..."
        Current:
            "2. O docente ..."

        Returns
        -------
        bool
            True when the lines should be merged.
        """
        if not previous_line or not current_line:
            return False

        if self._is_structural_line(previous_line):
            return False

        if self._is_structural_line(current_line):
            return False

        if HARD_LINE_END_RE.search(previous_line):
            return False

        # If the current line looks like the beginning of ordinary prose,
        # it is often a continuation of the previous broken line.
        if PROSE_START_RE.match(current_line):
            return True

        # Also merge when the current line starts lowercase, which frequently
        # happens in wrapped legal prose.
        if current_line[:1].islower():
            return True

        # Otherwise remain conservative.
        return False

    def _is_structural_line(self, line: str) -> bool:
        """
        Decide whether a line likely carries structural meaning and should
        therefore remain on its own line.

        Structural examples
        -------------------
        - article headers
        - chapter headers
        - annex headers
        - section/subsection headers
        - numbered legal items
        - lettered legal items
        """
        return bool(
            ARTICLE_HEADER_RE.match(line)
            or CHAPTER_HEADER_RE.match(line)
            or ANNEX_HEADER_RE.match(line)
            or SECTION_HEADER_RE.match(line)
            or NUMBERED_ITEM_RE.match(line)
            or LETTERED_ITEM_RE.match(line)
        )

    def _finalize_page_text(self, cleaned_lines: List[str]) -> str:
        """
        Rebuild the cleaned page text.

        Final cleanup steps
        -------------------
        1. Rejoin the page using preserved line boundaries
        2. Repair hyphenated line breaks
        3. Apply shared conservative block normalization

        Important:
        we still preserve structure-friendly line breaks here.
        """
        if not cleaned_lines:
            return ""

        text = "\n".join(cleaned_lines)

        # Repair words split by PDF wrapping:
        # "matrí-\ncula" -> "matrícula"
        text = join_hyphenated_linebreaks(text)

        # Apply shared conservative block cleanup without flattening structure.
        text = normalize_block_whitespace(text)

        return text.strip()

    def _build_full_text(self, pages: List[PageText]) -> str:
        """
        Build a debug-friendly full document text from normalized pages.

        Design choice
        -------------
        We keep page boundaries visible only as blank separation between pages,
        not as explicit page markers inside the text body.

        This makes the full-text export easier to read while remaining useful
        for debugging normalization quality.
        """
        non_empty_pages = [page.text.strip() for page in pages if page.text.strip()]
        if not non_empty_pages:
            return ""

        return "\n\n".join(non_empty_pages).strip()