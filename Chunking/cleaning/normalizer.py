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

# Some common cover / institutional title lines that often behave like
# decorative cover content rather than normative text.
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
    r"^\s*(CAP[ÍI]TULO|ARTIGO|ANEXO|ÍNDICE|SECÇÃO)\b",
    re.IGNORECASE,
)

# Normalize inner whitespace for comparison and reporting.
MULTISPACE_RE = re.compile(r"\s+")

# Footnote / note line, usually valid content and should not be removed
# automatically unless explicitly duplicated or clearly noisy.
FOOTNOTE_LINE_RE = re.compile(r"^\s*\(\d+\)\s+")

# Likely normative / structural lines. These should be protected from
# repeated-line removal when possible.
STRUCTURAL_OR_NORMATIVE_LINE_RE = re.compile(
    r"^\s*(CAP[ÍI]TULO|ARTIGO|ANEXO|SECÇÃO|\d+(?:\.\d+)*\s*[—–\-\.]|\(?\d+\)|[a-z]\))\b",
    re.IGNORECASE,
)


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
        5. Remove accidental duplicate pages/blocks.
        6. Concatenate pages into a single debug-friendly full_text.
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

            # Remove likely table-of-contents blocks after line-level cleanup.
            cleaned_lines, block_drop_report = self._remove_index_blocks(cleaned_lines)
            dropped_counter.update(block_drop_report)

            cleaned_text = self._finalize_page_text(cleaned_lines)

            cleaned_pages.append(
                PageText(
                    page_number=page.page_number,
                    text=cleaned_text,
                )
            )

        # Remove empty pages and accidental duplicate page bodies.
        deduped_pages, dedupe_report = self._deduplicate_pages(cleaned_pages)
        dropped_counter.update(dedupe_report)

        full_text = self._build_full_text(deduped_pages)

        return NormalizedDocument(
            pages=deduped_pages,
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
          line may still contain useful content after that marker.
        - We preserve line-level structure because the downstream parser
          needs it.
        """
        line = raw_line.strip()

        if not line:
            return None, "empty_line"

        # Remove leading page marker if present.
        new_line = LEADING_PAGE_MARKER_RE.sub("", line).strip()
        if new_line != line:
            line = new_line
            if not line:
                return None, "page_marker_only"

        # Remove inline page counters like "3|14".
        if INLINE_PAGE_COUNTER_RE.match(line):
            return None, "inline_page_counter"

        # Remove long numeric layout tails.
        if PURE_NUMERIC_NOISE_RE.match(line):
            return None, "trailing_numeric_code"

        # Remove pure Diário da República editorial lines.
        if DR_EDITORIAL_RE.match(line) and len(line) <= 140:
            return None, "dr_editorial_line"

        # Remove signature/signing metadata from body text.
        if SIGNATURE_LINE_RE.match(line):
            return None, "signature_line"

        # Remove some obvious cover-only lines.
        if COVER_NOISE_RE.match(line) and len(line) <= 140:
            return None, "cover_line"

        # Remove repeated short lines detected across multiple pages.
        #
        # IMPORTANT:
        # We avoid dropping lines that look normative/structural, because
        # those may legitimately repeat in legal content or extracted notes.
        normalized_for_compare = self._normalize_for_comparison(line)
        if (
            normalized_for_compare in repeated_lines
            and not self._should_protect_from_repeated_line_drop(line)
        ):
            return None, "repeated_line"

        # Remove very obvious index-like lines early.
        if self._looks_like_index_line(line):
            return None, "index_like_line"

        # Normalize inner whitespace, but DO NOT remove the fact that
        # this is still a separate line.
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
            if count >= threshold and self._is_plausible_repeated_noise(line)
        }

        return repeated

    def _is_plausible_repeated_noise(self, line: str) -> bool:
        """
        Decide whether a repeated line looks like real repeated noise rather
        than legitimate repeated legal content.

        We prefer being conservative here.
        """
        stripped = line.strip()

        if not stripped:
            return False

        # Protect clearly structural / normative lines.
        if self._should_protect_from_repeated_line_drop(stripped):
            return False

        # Very short institutional or decorative lines are plausible noise.
        if len(stripped) <= 50:
            return True

        # Uppercase-heavy repeated lines are often headers/banners.
        uppercase_ratio = self._uppercase_ratio(stripped)
        if uppercase_ratio >= 0.65 and len(stripped) <= 120:
            return True

        # Very common repeated boilerplate patterns.
        lowered = stripped.lower()
        repeated_noise_fragments = (
            "instituto politécnico do porto",
            "politécnico do porto",
            "diário da república",
        )
        if any(fragment in lowered for fragment in repeated_noise_fragments):
            return True

        return False

    def _should_protect_from_repeated_line_drop(self, line: str) -> bool:
        """
        Protect lines that should almost never be dropped merely for being repeated.
        """
        stripped = line.strip()

        if not stripped:
            return False

        if STRUCTURAL_OR_NORMATIVE_LINE_RE.match(stripped):
            return True

        if FOOTNOTE_LINE_RE.match(stripped):
            return True

        lowered = stripped.lower()
        if lowered.startswith(
            (
                "o presente ",
                "a presente ",
                "considerando",
                "determino",
                "as dúvidas",
                "disposições finais",
            )
        ):
            return True

        return False

    def _uppercase_ratio(self, text: str) -> float:
        """
        Compute an approximate uppercase ratio over alphabetic characters.
        """
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return 0.0

        uppercase_count = sum(1 for char in letters if char.isupper())
        return uppercase_count / len(letters)

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
        - short heading + trailing page number is a moderate signal
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
        - scan consecutive windows of TOC-like lines
        - only drop the block if it is long enough and dense enough
        """
        if not lines:
            return [], Counter()

        kept_lines: List[str] = []
        dropped_counter: Counter[str] = Counter()

        index = 0
        while index < len(lines):
            if not self._is_toc_block_candidate(lines[index]):
                kept_lines.append(lines[index])
                index += 1
                continue

            block_start = index
            block_lines: List[str] = []

            while index < len(lines) and self._is_toc_block_candidate(lines[index]):
                block_lines.append(lines[index])
                index += 1

            # Conservative rule:
            # only treat it as a TOC block when it is clearly long enough.
            if len(block_lines) >= 4:
                dropped_counter["index_block_line"] += len(block_lines)
                continue

            # Too short -> keep it.
            kept_lines.extend(block_lines)

        return kept_lines, dropped_counter

    def _is_toc_block_candidate(self, line: str) -> bool:
        """
        Decide whether a line looks like part of a table of contents block.

        Strong signals:
        - dot leaders
        - "CAPÍTULO", "ARTIGO", "ÍNDICE"
        - heading + trailing page number

        Important:
        We keep this narrow so we do not accidentally classify normative
        body blocks as TOC.
        """
        if INDEX_DOT_LEADER_RE.search(line):
            return True

        if INDEX_HEADING_RE.match(line) and INDEX_TRAILING_PAGE_RE.match(line):
            return True

        if INDEX_TRAILING_PAGE_RE.match(line) and len(line.split()) <= 10:
            lowered = line.lower()
            if any(
                token in lowered
                for token in ("capítulo", "capitulo", "artigo", "anexo", "índice", "indice")
            ):
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

    def _deduplicate_pages(self, pages: List[PageText]) -> Tuple[List[PageText], Counter[str]]:
        """
        Remove accidental duplicate page bodies.

        This helps with extraction glitches where the same page content
        appears twice in sequence or where a later page is an almost exact
        duplicate of an earlier one.

        We keep this conservative:
        - only exact normalized duplicates are removed
        - empty pages are ignored
        """
        kept_pages: List[PageText] = []
        dropped_counter: Counter[str] = Counter()
        seen_bodies: set[str] = set()

        for page in pages:
            normalized_body = self._normalize_page_body_for_dedup(page.text)

            if not normalized_body:
                # Keep empty pages out of the final result.
                dropped_counter["empty_page_after_cleanup"] += 1
                continue

            if normalized_body in seen_bodies:
                dropped_counter["duplicate_page_body"] += 1
                continue

            seen_bodies.add(normalized_body)
            kept_pages.append(page)

        return kept_pages, dropped_counter

    def _normalize_page_body_for_dedup(self, text: str) -> str:
        """
        Normalize a page body for exact duplicate detection.
        """
        text = text.strip()
        text = MULTISPACE_RE.sub(" ", text)
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