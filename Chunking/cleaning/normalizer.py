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
# We keep this conservative because we do not want to drop real content.
DR_EDITORIAL_RE = re.compile(
    r"^\s*(N\.?\s*º|PARTE\s+[A-Z]|Diário da República)\b",
    re.IGNORECASE,
)

# Very simple heuristic for "table of contents / index-like" lines.
# This is intentionally not aggressive.
INDEX_DOT_LEADER_RE = re.compile(r"\.{3,}")
INDEX_TRAILING_PAGE_RE = re.compile(r".+\s+\d{1,4}\s*$")

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
    """

    def normalize(self, pages: List[PageText]) -> NormalizedDocument:
        """
        Normalize a list of page-level extracted texts.

        Main phases:
        1. Detect repeated short lines that likely behave like headers/footers.
        2. Clean line by line, preserving structural line boundaries.
        3. Rebuild each page text with minimal whitespace cleanup.
        4. Concatenate pages into a single debug-friendly full_text.
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
                    # Defensive guard. In practice we only reach this if
                    # a line was intentionally neutralized.
                    continue

                cleaned_lines.append(line)

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
            # Empty lines are not "dropped noise"; they are simply ignored here.
            return None, "empty_line"

        # -------------------------------------------------------------
        # Remove leading page marker if present.
        #
        # Example:
        # "Pág. 363 INSTITUTO POLITÉCNICO DO PORTO ..."
        # becomes
        # "INSTITUTO POLITÉCNICO DO PORTO ..."
        # -------------------------------------------------------------
        new_line = LEADING_PAGE_MARKER_RE.sub("", line).strip()
        if new_line != line:
            line = new_line
            if not line:
                return None, "page_marker_only"

        # -------------------------------------------------------------
        # Remove pure inline page counters such as "3|14".
        # These are almost always layout-only noise.
        # -------------------------------------------------------------
        if INLINE_PAGE_COUNTER_RE.match(line):
            return None, "inline_page_counter"

        # -------------------------------------------------------------
        # Remove long numeric publication/layout tails.
        # Example: "316552597"
        # -------------------------------------------------------------
        if PURE_NUMERIC_NOISE_RE.match(line):
            return None, "trailing_numeric_code"

        # -------------------------------------------------------------
        # Remove obvious DR editorial lines.
        #
        # Keep this cautious. We only drop lines that look purely editorial.
        # -------------------------------------------------------------
        if DR_EDITORIAL_RE.match(line) and len(line) <= 120:
            return None, "dr_editorial_line"

        # -------------------------------------------------------------
        # Remove repeated short lines detected across multiple pages.
        # These are often headers, footers or institutional banners.
        # -------------------------------------------------------------
        normalized_for_compare = self._normalize_for_comparison(line)
        if normalized_for_compare in repeated_lines:
            return None, "repeated_line"

        # -------------------------------------------------------------
        # Remove likely table-of-contents / index-like lines.
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

        # A line becomes a repeated-noise candidate if it appears in at least
        # 2 pages, or in half the document pages for longer documents.
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
        if len(line) > 180:
            return False

        if INDEX_DOT_LEADER_RE.search(line):
            return True

        # Example:
        # "Artigo 3.º .............. 5"
        # "Condições de inscrição 7"
        if INDEX_TRAILING_PAGE_RE.match(line):
            words = line.split()
            if len(words) <= 12:
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