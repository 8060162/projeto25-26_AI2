from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from Chunking.chunking.models import PageText
from Chunking.utils.text import join_hyphenated_linebreaks


# -------------------------------------------------------------------------
# Regexes used by the normalizer.
#
# The goal here is not to perfectly understand the document semantics.
# The goal is to remove obvious PDF/editorial noise while preserving
# structural line breaks that are useful for the parser.
# -------------------------------------------------------------------------

# Example:
# "Pág. 363"
# "Pág. 364"
LEADING_PAGE_MARKER_RE = re.compile(r"^\s*Pág\.\s*\d+\s*", re.IGNORECASE)

# Example:
# "3|14"
# "10 | 14"
INLINE_PAGE_COUNTER_RE = re.compile(r"^\s*\d+\s*\|\s*\d+\s*$")

# Example:
# "316552597"
# Long pure numeric tails are often publication / layout noise.
PURE_NUMERIC_NOISE_RE = re.compile(r"^\s*\d{6,}\s*$")

# Typical Diário da República editorial lines or related page furniture.
DR_EDITORIAL_RE = re.compile(
    r"^\s*(N\.?\s*º|PARTE\s+[A-Z]|Diário da República)\b",
    re.IGNORECASE,
)

# Likely signature / signing lines that should not pollute chunks.
SIGNATURE_LINE_RE = re.compile(
    r"^\s*(Assinado por:|Num\.?\s+de\s+Identificação:|Data:\s*\d{4}\.\d{2}\.\d{2})",
    re.IGNORECASE,
)

# Sometimes there are pure document-cover lines that are not useful for QA.
# We keep this conservative and use it only for very specific patterns.
COVER_NOISE_RE = re.compile(
    r"^\s*(DESPACHO\s+P\.PORTO\/P-|REGULAMENTO\s+P\.PORTO\/P-)\S*",
    re.IGNORECASE,
)

# Used to detect obvious "dot leader" table-of-contents lines.
INDEX_DOT_LEADER_RE = re.compile(r"\.{3,}")

# Used to detect short "title + page number" lines.
INDEX_TRAILING_PAGE_RE = re.compile(r".+\s+\d{1,4}\s*$")

# Title-like index entries.
INDEX_HEADING_RE = re.compile(
    r"^\s*(CAP[ÍI]TULO|ARTIGO|ANEXO|ÍNDICE)\b",
    re.IGNORECASE,
)

# Normalize inner whitespace for comparison and reporting.
MULTISPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class NormalizedDocument:
    """
    Intermediate normalized document representation.

    It keeps page-level cleaned text and a document-level consolidated body.
    This is useful for debugging and for exporting intermediate snapshots.
    """

    pages: List[PageText]
    full_text: str
    dropped_lines_report: Dict[str, int]


class TextNormalizer:
    """
    Remove PDF layout noise before structure parsing and chunking.

    Important design choice:
    We DO NOT aggressively unwrap all single newlines here.

    Why?
    Because for legal / regulatory PDFs, line breaks often carry structure:
    - article headers
    - section titles
    - annex markers
    - numbered items
    - lettered items

    If we flatten the text too early, the parser loses the clues it needs.

    Another important design principle:
    We favor "safe cleanup" over "aggressive cleanup".
    In legal documents, it is usually better to keep a little noise than
    to accidentally drop real normative content.
    """

    def normalize(self, pages: List[PageText]) -> NormalizedDocument:
        """
        Normalize a list of page-level extracted texts.

        Main phases:
        1. Detect repeated short lines that likely behave like headers/footers.
        2. Clean line by line, preserving structural line boundaries.
        3. Remove likely table-of-contents blocks.
        4. Rebuild each page text with minimal whitespace cleanup.
        5. Concatenate pages into a single debug-friendly full_text.
        """
        repeated_lines = self._find_repeated_noise_candidates(pages)
        cleaned_pages: List[PageText] = []
        dropped_counter: Counter[str] = Counter()

        for page in pages:
            cleaned_lines: List[str] = []

            for raw_line in page.text.splitlines():
                line, drop_reason = self._clean_line(raw_line, repeated_lines)

                if drop_reason is not None:
                    dropped_counter[drop_reason] += 1
                    continue

                if line is None:
                    continue

                cleaned_lines.append(line)

            # ---------------------------------------------------------
            # Remove likely index blocks after line-level cleanup.
            #
            # This is done at block level because a TOC is often easier
            # to detect as a cluster of lines than one line at a time.
            # ---------------------------------------------------------
            cleaned_lines, block_drop_report = self._remove_index_blocks(cleaned_lines)
            dropped_counter.update(block_drop_report)

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

        Returns:
        - (cleaned_line, None) when the line should stay
        - (None, "reason") when the line should be dropped

        Notes:
        - We first strip page markers like "Pág. 363" from the beginning
          of the line instead of dropping the entire line, because the same
          line may still contain useful legal content after that marker.
        - We preserve line-level structure because the downstream parser
          needs it.
        """
        line = raw_line.strip()

        if not line:
            return None, "empty_line"

        # -------------------------------------------------------------
        # Remove leading page marker if present.
        # -------------------------------------------------------------
        new_line = LEADING_PAGE_MARKER_RE.sub("", line).strip()
        if new_line != line:
            line = new_line
            if not line:
                return None, "page_marker_only"

        # -------------------------------------------------------------
        # Remove inline page counters like "3|14".
        # -------------------------------------------------------------
        if INLINE_PAGE_COUNTER_RE.match(line):
            return None, "inline_page_counter"

        # -------------------------------------------------------------
        # Remove long numeric layout tails.
        # -------------------------------------------------------------
        if PURE_NUMERIC_NOISE_RE.match(line):
            return None, "trailing_numeric_code"

        # -------------------------------------------------------------
        # Remove pure Diário da República editorial lines.
        # -------------------------------------------------------------
        if DR_EDITORIAL_RE.match(line) and len(line) <= 140:
            return None, "dr_editorial_line"

        # -------------------------------------------------------------
        # Remove signature/signing metadata from body text.
        #
        # These lines are often useful as document-level metadata, but
        # they are usually harmful inside chunks.
        # -------------------------------------------------------------
        if SIGNATURE_LINE_RE.match(line):
            return None, "signature_line"

        # -------------------------------------------------------------
        # Remove some obvious cover-only lines.
        # This remains conservative on purpose.
        # -------------------------------------------------------------
        if COVER_NOISE_RE.match(line) and len(line) <= 140:
            return None, "cover_line"

        # -------------------------------------------------------------
        # Remove repeated short lines detected across multiple pages.
        # These are often headers, footers or institutional banners.
        # -------------------------------------------------------------
        normalized_for_compare = self._normalize_for_comparison(line)
        if normalized_for_compare in repeated_lines:
            return None, "repeated_line"

        # -------------------------------------------------------------
        # Remove very obvious index-like lines early.
        #
        # We still also do block-level index removal later because not all
        # TOC lines are easy to classify one by one.
        # -------------------------------------------------------------
        if self._looks_like_index_line(line):
            return None, "index_like_line"

        # -------------------------------------------------------------
        # Normalize inner whitespace, but DO NOT remove the fact that
        # this is still a separate line.
        # -------------------------------------------------------------
        line = MULTISPACE_RE.sub(" ", line).strip()

        if not line:
            return None, "empty_after_cleanup"

        return line, None

    def _find_repeated_noise_candidates(self, pages: List[PageText]) -> set[str]:
        """
        Find short normalized lines repeated across multiple pages.

        Typical examples:
        - institutional banner lines
        - repeated regulation headers
        - page furniture repeated on many pages

        Important:
        We normalize candidates before counting them so that superficial
        whitespace differences do not prevent detection.
        """
        counter: Counter[str] = Counter()
        total_pages = max(1, len(pages))

        for page in pages:
            unique_page_lines = {
                self._normalize_for_comparison(line)
                for line in page.text.splitlines()
                if line.strip()
            }

            for line in unique_page_lines:
                if not line:
                    continue

                # We only consider relatively short lines for repeated-noise
                # detection. This avoids accidentally marking legitimate long
                # content as "repeated".
                if len(line) <= 120:
                    counter[line] += 1

        threshold = max(2, total_pages // 2)

        repeated = {
            line
            for line, count in counter.items()
            if count >= threshold
        }

        return repeated

    def _normalize_for_comparison(self, line: str) -> str:
        """
        Normalize a line specifically for repeated-line detection.

        We remove very common layout artifacts here to make repeated headers
        easier to detect.
        """
        line = line.strip()
        line = LEADING_PAGE_MARKER_RE.sub("", line).strip()
        line = MULTISPACE_RE.sub(" ", line).strip()
        return line

    def _looks_like_index_line(self, line: str) -> bool:
        """
        Detect likely table-of-contents lines.

        The heuristic is intentionally conservative:
        - dot leaders are a strong signal
        - short title + trailing number is a moderate signal
        """
        if len(line) > 220:
            return False

        if INDEX_DOT_LEADER_RE.search(line):
            return True

        if INDEX_TRAILING_PAGE_RE.match(line):
            words = line.split()

            # A short heading followed by a page number often behaves like TOC.
            if len(words) <= 14 and INDEX_HEADING_RE.match(line):
                return True

        return False

    def _remove_index_blocks(self, lines: List[str]) -> Tuple[List[str], Counter[str]]:
        """
        Remove likely table-of-contents blocks.

        Why block-level detection matters:
        a TOC is often composed of many consecutive lines that individually
        might look harmless, but together clearly behave like an index.

        Heuristic:
        - scan consecutive windows of lines
        - if enough lines in the window look like TOC lines,
          drop the whole block
        """
        if not lines:
            return [], Counter()

        kept_lines: List[str] = []
        dropped_counter: Counter[str] = Counter()

        index = 0
        while index < len(lines):
            # Try to detect a contiguous block of likely TOC lines.
            block_start = index
            toc_score = 0
            block_lines: List[str] = []

            while index < len(lines):
                line = lines[index]
                is_toc_like = self._is_toc_block_candidate(line)

                if not is_toc_like and not block_lines:
                    break

                if not is_toc_like and block_lines:
                    # Stop the block when the TOC pattern is broken.
                    break

                block_lines.append(line)
                toc_score += 1
                index += 1

            # Decide whether the captured block is actually an index block.
            if block_lines and toc_score >= 4:
                dropped_counter["index_block_line"] += len(block_lines)
                continue

            # If not a TOC block, emit the original first line and continue.
            if block_lines:
                kept_lines.extend(block_lines)
                continue

            kept_lines.append(lines[index])
            index += 1

        return kept_lines, dropped_counter

    def _is_toc_block_candidate(self, line: str) -> bool:
        """
        Decide whether a line looks like part of a table of contents block.

        Strong signals:
        - dot leaders
        - "CAPÍTULO", "ARTIGO", "ÍNDICE"
        - heading + trailing page number
        """
        if INDEX_DOT_LEADER_RE.search(line):
            return True

        if INDEX_HEADING_RE.match(line):
            return True

        if INDEX_TRAILING_PAGE_RE.match(line) and len(line.split()) <= 16:
            return True

        return False

    def _finalize_page_text(self, cleaned_lines: List[str]) -> str:
        """
        Rebuild the cleaned page text.

        Important:
        We preserve line breaks because the structure parser relies on them.

        We still perform some safe cleanup:
        - join broken hyphenation across line breaks
        - trim spaces around line breaks
        - collapse many blank lines into at most one blank separator
        """
        if not cleaned_lines:
            return ""

        text = "\n".join(cleaned_lines)

        # Fix PDF line-break hyphenation:
        # "matrí-\ncula" -> "matrícula"
        text = join_hyphenated_linebreaks(text)

        # Normalize spaces before/after newlines without flattening structure.
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)

        # Collapse excessive blank lines but keep paragraph separation.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _build_full_text(self, pages: List[PageText]) -> str:
        """
        Build a debug-friendly full document text from normalized pages.

        We keep page boundaries visible only as empty-line separation between
        pages, not as explicit "Pág. X" markers inside the content.
        """
        non_empty_pages = [page.text.strip() for page in pages if page.text.strip()]
        if not non_empty_pages:
            return ""
        return "\n\n".join(non_empty_pages).strip()