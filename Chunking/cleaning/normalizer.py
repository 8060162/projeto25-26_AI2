from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from Chunking.chunking.models import ExtractedDocument, ExtractedPage, PageText
from Chunking.utils.text import (
    normalize_block_whitespace,
    normalize_line_endings,
    repair_broken_hyphenation,
    strip_control_characters,
)


# ============================================================================
# Regexes used by the normalizer
# ============================================================================
#
# Important architectural note
# ----------------------------
# This module is intentionally conservative.
#
# It is NOT a legal-structure parser.
# It should:
# - remove obvious PDF / editorial / layout noise
# - preserve structural line boundaries useful to the parser
# - repair obvious extraction artifacts conservatively
# - avoid aggressive cleanup that could remove legal content
#
# This module sits between extraction and structure parsing.
# Therefore it must be careful:
# once text is destroyed here, later stages cannot recover it.
# ============================================================================

# Example:
# "Pág. 363"
# "Pág. 364"
#
# Strip this marker only when it appears at the beginning of the line so
# that useful content after the marker can still survive.
LEADING_PAGE_MARKER_RE = re.compile(r"^\s*Pág\.\s*\d+\s*", re.IGNORECASE)

# Example:
# "3|14"
# "10 | 14"
#
# Classic page counters from headers/footers.
INLINE_PAGE_COUNTER_RE = re.compile(r"^\s*\d+\s*\|\s*\d+\s*$")

# Slightly looser page counter variants.
LOOSE_PAGE_COUNTER_RE = re.compile(r"^\s*\d+\s*\|\s*\d+\s*[\-–—.]?\s*$")

# Long pure numeric tails often represent editorial or publication noise.
PURE_NUMERIC_NOISE_RE = re.compile(r"^\s*\d{6,}\s*$")

# Typical Diário da República editorial furniture.
DR_EDITORIAL_RE = re.compile(
    r"^\s*(N\.?\s*º|PARTE\s+[A-Z]|Diário da República)\b",
    re.IGNORECASE,
)

# Signature or signing metadata lines.
SIGNATURE_LINE_RE = re.compile(
    r"^\s*(Assinado por:|Num\.?\s+de\s+Identificação:|Data:\s*\d{4}\.\d{2}\.\d{2})",
    re.IGNORECASE,
)

# Decorative cover lines that often belong to title pages rather than body text.
COVER_NOISE_RE = re.compile(
    r"^\s*(DESPACHO\s+P\.PORTO\/P-|REGULAMENTO\s+P\.PORTO\/P-)\S*",
    re.IGNORECASE,
)

# Repeated institutional banner-like lines.
INSTITUTIONAL_BANNER_RE = re.compile(
    r"^\s*(POLIT[ÉE]CNICO\s+DO\s+PORTO|P\.PORTO)\s*$",
    re.IGNORECASE,
)

# Very short layout residues sometimes leaked by extraction.
#
# Examples:
# - "Vo/"
# - "V0/"
# - "CAPÍTULO |"
# - "CAPITULO |"
SHORT_LAYOUT_RESIDUE_RE = re.compile(
    r"^\s*(?:[Vv][o0]/?|CAP[ÍI]TULO\s*\|)\s*$",
    re.IGNORECASE,
)

# Common editorial date formats found in publication headers.
EDITORIAL_DATE_RE = re.compile(
    r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}\b"
)

# Publication summary lines from Diario da Republica pages.
SUMMARY_LINE_RE = re.compile(r"^\s*Sum[áa]rio:\s+", re.IGNORECASE)

# Dot leaders typical of table-of-contents lines.
INDEX_DOT_LEADER_RE = re.compile(r"\.{3,}")

# Short TOC-like lines ending with a page number.
INDEX_TRAILING_PAGE_RE = re.compile(r".+\s+\d{1,4}\s*$")

# Heading-like TOC entries.
INDEX_HEADING_RE = re.compile(
    r"^\s*(CAP[ÍI]TULO|ARTIGO|ANEXO|ÍNDICE|SECÇÃO|SUBSECÇÃO|TÍTULO)\b",
    re.IGNORECASE,
)

# Normalize inner whitespace for comparison and reporting.
MULTISPACE_RE = re.compile(r"\s+")

# Remove non-alphanumeric separators when building conservative editorial
# comparison strings.
NON_ALNUM_EDITORIAL_RE = re.compile(r"[^a-z0-9]+")

# Structural headers that should usually remain on their own line.
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

# Detect line endings that usually indicate completed sentences / hard boundaries.
HARD_LINE_END_RE = re.compile(r"[.;:!?]$")

# Detect lines that are almost entirely symbolic and therefore suspicious.
MOSTLY_SYMBOLIC_RE = re.compile(r"^[^\wÀ-ÿ]{6,}$")

# Detect highly suspicious garbled-looking lines.
SUSPICIOUS_GARBLED_LINE_RE = re.compile(
    r"^[^A-Za-zÀ-ÿ]{0,3}(?:[\*\+\-/=<>\\\[\]\{\}_`~]{2,}|[0-9\W]{12,})$"
)

# Detect footnote-style editorial notes that only carry publication/access
# residue rather than legal meaning.
ACCESS_NOTE_RE = re.compile(
    r"^\s*\(?\d+\)?\s+"
    r"(?:Acess[íi]vel|Dispon[íi]vel|Publicado|Publicada|Publicados|Publicadas)\b",
    re.IGNORECASE,
)

# Detect short signature residue lines commonly injected around publication
# closing metadata.
SIGNATURE_RESIDUE_RE = re.compile(
    r"^\s*(?:"
    r".{0,80}\bO\s+PRESIDENTE\b.*|"
    r".{0,80}\bO\s+DIRETOR\b.*|"
    r".{0,80}\bA\s+PRESIDENTE\b.*|"
    r".{0,80}\bA\s+DIRETORA\b.*|"
    r".{0,80}[—-]\s*O\s+Presidente\b.*"
    r")$",
    re.IGNORECASE,
)

# Detect date/location publication closing lines tied to sign-off metadata.
PUBLICATION_SIGNOFF_RE = re.compile(
    r"^\s*.+?,\s+\d{1,2}\s+de\s+[A-Za-zÀ-ÿ]+\s+de\s+\d{4}\s*$",
    re.IGNORECASE,
)

# Detect uppercase-heavy heading fragments that often behave like
# front-matter / index residue on early pages.
UPPERCASE_HEAVY_RE = re.compile(
    r"^[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ0-9 /().,\-–—ºª]+$"
)

# Detect likely "all-caps heading residue" that is not ordinary prose and not
# a structural header we want to preserve.
#
# This is especially useful for broken TOC/front-matter fragments such as:
# - "ESTUDANTES DE MESTRADO INSCRITOS ..."
# - "DISSERTAÇÃO, TRABALHO DE PROJETO ..."
UPPERCASE_RESIDUE_RE = re.compile(
    r"^[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ0-9 ,/().\-–—ºª]{8,}$"
)

# Detect lines that look like continued index headings or title-page headings
# but do not look like ordinary prose.
#
# This is intentionally broad, but it is only used with extra safeguards,
# mainly on early pages.
NON_PROSE_HEADING_RE = re.compile(
    r"^(?!.*[.;:!?]$)(?!\s*(?:O|A|Os|As|No|Na|Nos|Nas|Em|Para|Por|Quando|Sempre|Caso|Se)\b).+$",
    re.IGNORECASE,
)


# ============================================================================
# Normalization output models
# ============================================================================


@dataclass(slots=True)
class NormalizedPage:
    """
    Normalized page-level text plus lightweight provenance.

    Why this model exists
    ---------------------
    The pipeline is transitioning away from plain `PageText` as the only
    representation. During this transition, normalization should preserve at
    least some useful provenance from extraction, for example:
    - selected extraction mode
    - upstream quality score
    - upstream corruption flags

    This information is useful for:
    - parser confidence
    - debugging suspicious pages
    - deciding whether a page should be treated more defensively downstream
    """

    page_number: int
    text: str
    selected_mode: str = ""
    upstream_quality_score: Optional[float] = None
    upstream_flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedDocument:
    """
    Intermediate normalized document representation.

    Contents
    --------
    - page-level cleaned text
    - document-level consolidated full_text
    - dropped-lines report for inspection/debugging
    - optional page-level normalization diagnostics

    Important architectural note
    ----------------------------
    This is still NOT the final parsed legal/regulatory JSON tree.
    It is the normalized substrate that should feed the structure parser.
    """

    pages: List[NormalizedPage]
    full_text: str
    dropped_lines_report: Dict[str, int]
    page_reports: List[Dict[str, Any]] = field(default_factory=list)


class TextNormalizer:
    """
    Remove PDF layout noise before structure parsing.

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
       This module should not aggressively flatten the document.

    4. Be careful with repeated-line detection
       Repeated lines are often headers / footers, but valid legal text may
       also repeat. Therefore repeated-line heuristics should focus mainly on
       page margins rather than the full body.

    5. Remove only obviously unusable line-level garbage
       If a line looks severely corrupted or purely decorative, it should not
       survive into the parser input.

    6. Remain parser-friendly
       The goal is not to produce "pretty text".
       The goal is to produce text that is clean enough for reliable parsing
       while still retaining structural clues.
    """

    # Only consider a small top / bottom window of each page when detecting
    # repeated layout furniture.
    _REPEATED_LINE_MARGIN_WINDOW = 4

    # Tables of contents usually live near the beginning of a document.
    _MAX_TOC_SCAN_PAGES = 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        document_or_pages: Union[
            ExtractedDocument,
            Sequence[PageText],
            Sequence[ExtractedPage],
        ],
    ) -> NormalizedDocument:
        """
        Normalize page-level extracted text conservatively.

        Supported inputs
        ----------------
        This method accepts:
        - ExtractedDocument
        - List[ExtractedPage]
        - List[PageText]

        Processing phases
        -----------------
        1. Coerce the input into page-like objects
        2. Detect repeated short lines that likely behave like headers/footers
        3. Clean line by line while preserving structural boundaries
        4. Remove likely TOC/index blocks, mainly in early pages
        5. Reflow obvious broken prose lines conservatively
        6. Rebuild page text with shared block-level normalization
        7. Concatenate pages into a debug-friendly full_text

        Why repeated-line detection happens first
        -----------------------------------------
        Repeated page furniture is easier to recognize globally before page-by-
        page cleanup removes contextual clues.

        Parameters
        ----------
        document_or_pages : Union[ExtractedDocument, Sequence[PageText], Sequence[ExtractedPage]]
            Extracted document or page-like objects.

        Returns
        -------
        NormalizedDocument
            Page-level normalized text plus diagnostics.
        """
        pages = self._coerce_pages(document_or_pages)
        repeated_lines = self._find_repeated_noise_candidates(pages)

        cleaned_pages: List[NormalizedPage] = []
        dropped_counter: Counter[str] = Counter()
        page_reports: List[Dict[str, Any]] = []

        total_pages = len(pages)

        for page in pages:
            (
                page_number,
                page_text,
                selected_mode,
                upstream_quality_score,
                upstream_flags,
            ) = self._extract_page_fields(page)

            cleaned_lines: List[str] = []
            raw_lines = page_text.splitlines()

            for raw_line in raw_lines:
                line, drop_reason = self._clean_line(
                    raw_line=raw_line,
                    repeated_lines=repeated_lines,
                    page_number=page_number,
                    total_pages=total_pages,
                )

                if drop_reason is not None:
                    dropped_counter[drop_reason] += 1
                    continue

                if line is None:
                    continue

                cleaned_lines.append(line)

            cleaned_lines, split_editorial_report = self._remove_split_editorial_blocks(
                cleaned_lines
            )
            dropped_counter.update(split_editorial_report)

            # Remove likely TOC/index blocks conservatively and mainly on early pages.
            cleaned_lines, block_drop_report = self._remove_index_blocks(
                lines=cleaned_lines,
                page_number=page_number,
                total_pages=total_pages,
            )
            dropped_counter.update(block_drop_report)

            # Reflow obvious broken prose lines while preserving structural markers.
            cleaned_lines = self._reflow_lines(cleaned_lines)

            cleaned_text = self._finalize_page_text(cleaned_lines)

            cleaned_pages.append(
                NormalizedPage(
                    page_number=page_number,
                    text=cleaned_text,
                    selected_mode=selected_mode,
                    upstream_quality_score=upstream_quality_score,
                    upstream_flags=list(upstream_flags),
                )
            )

            page_reports.append(
                {
                    "page_number": page_number,
                    "selected_mode": selected_mode,
                    "raw_line_count": len(raw_lines),
                    "cleaned_line_count": len(cleaned_lines),
                    "raw_char_count": len(page_text),
                    "cleaned_char_count": len(cleaned_text),
                    "upstream_quality_score": upstream_quality_score,
                    "upstream_flags": list(upstream_flags),
                }
            )

        full_text = self._build_full_text(cleaned_pages)

        return NormalizedDocument(
            pages=cleaned_pages,
            full_text=full_text,
            dropped_lines_report=dict(dropped_counter),
            page_reports=page_reports,
        )

    # ------------------------------------------------------------------
    # Line cleaning
    # ------------------------------------------------------------------

    def _clean_line(
        self,
        raw_line: str,
        repeated_lines: set[str],
        page_number: int,
        total_pages: int,
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

        Parameters
        ----------
        raw_line : str
            Raw line extracted from one page.

        repeated_lines : set[str]
            Previously detected repeated margin lines likely to be layout noise.

        page_number : int
            Current 1-based page number.

        total_pages : int
            Total number of pages in the document.

        Returns
        -------
        Tuple[Optional[str], Optional[str]]
            Cleaned line and drop reason.
        """
        line = normalize_line_endings(raw_line).strip()

        if not line:
            return None, "empty_line"

        # Remove control-character noise early so later regexes operate on
        # cleaner text.
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

        # Remove slightly noisier page counter variants.
        if LOOSE_PAGE_COUNTER_RE.match(line):
            return None, "loose_page_counter"

        # Remove long pure numeric publication/layout tails.
        if PURE_NUMERIC_NOISE_RE.match(line):
            return None, "trailing_numeric_code"

        # Remove purely institutional banner lines.
        if INSTITUTIONAL_BANNER_RE.match(line):
            return None, "institutional_banner"

        # Remove very short layout residues.
        if SHORT_LAYOUT_RESIDUE_RE.match(line):
            return None, "short_layout_residue"

        # Remove pure Diário da República editorial lines.
        #
        # The length cap is a safety mechanism so we do not remove large lines
        # that merely begin similarly but contain meaningful text.
        if self._looks_like_dr_editorial_line(line):
            return None, "dr_editorial_line"

        # Remove publication summary lines that belong to editorial front matter.
        if self._looks_like_publication_summary_line(
            line=line,
            page_number=page_number,
            total_pages=total_pages,
        ):
            return None, "publication_summary_line"

        # Remove signature metadata lines.
        if SIGNATURE_LINE_RE.match(line):
            return None, "signature_line"

        # Remove obvious decorative cover lines.
        if COVER_NOISE_RE.match(line) and len(line) <= 160:
            return None, "cover_line"

        # Remove clearly unusable symbolic garbage lines.
        if MOSTLY_SYMBOLIC_RE.match(line):
            return None, "mostly_symbolic_line"

        # Remove strongly suspicious garbled lines.
        #
        # Important:
        # this is intentionally conservative. We only remove obvious line-level
        # garbage, not whole-page extraction failures.
        if self._looks_like_garbled_line(line):
            return None, "garbled_line"

        # Remove clearly non-semantic publication/access notes.
        if self._looks_like_non_semantic_note_line(line):
            return None, "non_semantic_note_line"

        # Remove short publication sign-off residue such as names, roles, and
        # date/location lines that do not belong to the normative body.
        if self._looks_like_signature_residue_line(line):
            return None, "signature_residue_line"

        # Remove repeated page furniture lines.
        #
        # Important:
        # repeated-line candidates are detected mainly from page margins,
        # which makes this step much safer than scanning the whole body.
        normalized_for_compare = self._normalize_for_comparison(line)
        if normalized_for_compare in repeated_lines:
            return None, "repeated_line"

        # Remove very obvious single-line TOC entries early.
        if self._looks_like_index_line(
            line=line,
            page_number=page_number,
            total_pages=total_pages,
        ):
            return None, "index_like_line"

        # Remove likely uppercase heading residue on early pages.
        #
        # This targets broken TOC/front-matter fragments that frequently leak
        # into PREAMBLE, especially in institutional PDFs.
        if self._looks_like_uppercase_front_matter_residue(
            line=line,
            page_number=page_number,
            total_pages=total_pages,
        ):
            return None, "uppercase_front_matter_residue"

        # Normalize inner whitespace while preserving the line boundary itself.
        line = MULTISPACE_RE.sub(" ", line).strip()

        if not line:
            return None, "empty_after_cleanup"

        return line, None

    # ------------------------------------------------------------------
    # Repeated line detection
    # ------------------------------------------------------------------

    def _find_repeated_noise_candidates(
        self,
        pages: List[Union[PageText, ExtractedPage]],
    ) -> set[str]:
        """
        Find short normalized lines repeated across page margins.

        Why this heuristic matters
        --------------------------
        Repeated headers and footers are often strong noise candidates.

        Why margin-only scanning?
        -------------------------
        Scanning the full page body is risky because valid legal phrases may
        repeat across articles or chapters. Restricting the analysis to top/bottom
        margin regions makes this heuristic much safer.

        Parameters
        ----------
        pages : List[Union[PageText, ExtractedPage]]
            Page-like objects.

        Returns
        -------
        set[str]
            Normalized repeated-line candidates likely to be page furniture.
        """
        counter: Counter[str] = Counter()
        total_pages = max(1, len(pages))

        for page in pages:
            _, page_text, _, _, _ = self._extract_page_fields(page)

            raw_lines = [line for line in page_text.splitlines() if line.strip()]
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

                # Only relatively short lines are eligible as repeated furniture.
                if len(line) <= 140:
                    counter[line] += 1

        threshold = max(2, total_pages // 2)

        repeated = {
            line
            for line, count in counter.items()
            if count >= threshold
        }

        # Additional safeguard:
        # ignore lines that look too content-heavy or too long.
        return {
            line
            for line in repeated
            if len(line.split()) <= 12
        }

    def _margin_lines(self, lines: List[str]) -> List[str]:
        """
        Return likely page-margin lines from a page.

        Why this helper exists
        ----------------------
        Repeated headers and footers usually live near the top or bottom of a
        page. Restricting repeated-line analysis to these areas reduces false
        positives.

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

    def _looks_like_dr_editorial_line(self, line: str) -> bool:
        """
        Detect short standalone Diário da República editorial residue.

        Why this helper exists
        ----------------------
        OCR and PDF extraction often degrade official publication headers into
        variants such as:
        - "DIARIO DA REPUBLICA"
        - "DIARIO o DA REPUBLICA | 07-06-2024"
        - "N.o 109 | 07-06-2024"

        The cleanup must remain conservative. Legal prose that merely mentions
        the Diário da República should not be removed.

        Parameters
        ----------
        line : str
            Candidate line.

        Returns
        -------
        bool
            True when the line behaves like short editorial page furniture.
        """
        if not line or len(line) > 180:
            return False

        if DR_EDITORIAL_RE.match(line):
            return True

        folded_line = self._fold_editorial_text(line)
        if not folded_line:
            return False

        tokens = folded_line.split()
        if len(tokens) > 14:
            return False

        has_publication_marker = (
            "republica" in tokens
            and "diario" in tokens
            and abs(tokens.index("republica") - tokens.index("diario")) <= 4
        )
        if not has_publication_marker:
            return False

        has_header_shape = (
            "|" in line
            or EDITORIAL_DATE_RE.search(line) is not None
            or "parte" in tokens
            or any(token.startswith("n") and len(token) <= 3 for token in tokens)
        )
        if not has_header_shape:
            return False

        return not PROSE_START_RE.match(line)

    def _looks_like_publication_summary_line(
        self,
        line: str,
        page_number: int,
        total_pages: int,
    ) -> bool:
        """
        Detect short Diario da Republica summary lines on early pages.

        Safety behavior
        ---------------
        The rule is limited to the document front region because genuine body
        text may legitimately contain the word "sumario" later.
        """
        if not self._is_early_document_page(page_number, total_pages):
            return False

        if not SUMMARY_LINE_RE.match(line):
            return False

        return len(line.split()) <= 24

    def _looks_like_non_semantic_note_line(self, line: str) -> bool:
        """
        Detect publication/access notes that should not survive normalization.

        This helper intentionally targets only strongly non-semantic residue,
        such as web access notes or short publication metadata notes.
        """
        if not line or len(line) > 280:
            return False

        folded_line = self._fold_editorial_text(line)
        if not folded_line:
            return False

        if ACCESS_NOTE_RE.match(line):
            return True

        if folded_line.startswith("nota "):
            has_publication_signal = (
                "diario" in folded_line
                or "republica" in folded_line
                or "publicado" in folded_line
                or "publicada" in folded_line
                or "https" in folded_line
                or "www" in folded_line
            )
            if has_publication_signal and len(line.split()) <= 28:
                return True

        return False

    def _looks_like_signature_residue_line(self, line: str) -> bool:
        """
        Detect short sign-off residue that belongs to publication metadata.

        The normalizer removes only lines that look like dates, roles, or
        names attached to sign-off furniture rather than legal provisions.
        """
        if not line or len(line) > 180:
            return False

        if SIGNATURE_RESIDUE_RE.match(line):
            return True

        if PUBLICATION_SIGNOFF_RE.match(line):
            folded_line = self._fold_editorial_text(line)
            return (
                "instituto" in folded_line
                or "porto" in folded_line
                or "presidente" in folded_line
                or "diretor" in folded_line
                or "diretora" in folded_line
            )

        return False

    def _remove_split_editorial_blocks(
        self,
        lines: List[str],
    ) -> Tuple[List[str], Counter[str]]:
        """
        Remove short multi-line editorial residue that escaped line cleanup.

        Why this helper exists
        ----------------------
        Some publication headers survive as adjacent short lines instead of one
        standalone line, for example:
        - "DIARIO o"
        - "DA REPUBLICA |"

        Line-by-line cleanup may keep each fragment because neither line alone
        is conclusive enough. Recombining only very short local windows keeps
        the rule conservative while removing this recurring noise.

        Parameters
        ----------
        lines : List[str]
            Already cleaned lines for one page.

        Returns
        -------
        Tuple[List[str], Counter[str]]
            Kept lines plus a block-level dropped-lines report.
        """
        if not lines:
            return [], Counter()

        kept_lines: List[str] = []
        dropped_counter: Counter[str] = Counter()
        index = 0

        while index < len(lines):
            removed = False

            for window_size in (3, 2):
                if index + window_size > len(lines):
                    continue

                candidate_lines = lines[index : index + window_size]
                if not self._looks_like_split_dr_editorial_block(candidate_lines):
                    continue

                dropped_counter["dr_editorial_block_line"] += len(candidate_lines)
                index += window_size
                removed = True
                break

            if removed:
                continue

            kept_lines.append(lines[index])
            index += 1

        return kept_lines, dropped_counter

    def _looks_like_split_dr_editorial_block(self, lines: Sequence[str]) -> bool:
        """
        Decide whether a short adjacent line window is editorial residue.

        Parameters
        ----------
        lines : Sequence[str]
            Adjacent page lines to inspect together.

        Returns
        -------
        bool
            True when the window behaves like split publication header noise.
        """
        if len(lines) < 2 or len(lines) > 3:
            return False

        stripped_lines = [line.strip() for line in lines if line and line.strip()]
        if len(stripped_lines) != len(lines):
            return False

        if any(len(line) > 80 for line in stripped_lines):
            return False

        combined = " ".join(stripped_lines)
        folded_combined = self._fold_editorial_text(combined)
        if not folded_combined:
            return False

        if "diario" not in folded_combined and "republica" not in folded_combined:
            return False

        return self._looks_like_dr_editorial_line(combined)

    def _fold_editorial_text(self, text: str) -> str:
        """
        Fold a short line into an ASCII-like comparison form.

        Why this helper exists
        ----------------------
        Editorial headers often suffer from OCR noise in accents, casing, and
        separators. This helper removes only those superficial differences so
        conservative line classification can compare stable tokens.

        Parameters
        ----------
        text : str
            Raw candidate text.

        Returns
        -------
        str
            Lowercased accent-folded comparison string.
        """
        normalized_text = unicodedata.normalize("NFKD", text)
        without_marks = "".join(
            character
            for character in normalized_text
            if not unicodedata.combining(character)
        )
        lowered_text = without_marks.lower()
        collapsed_text = NON_ALNUM_EDITORIAL_RE.sub(" ", lowered_text)
        return MULTISPACE_RE.sub(" ", collapsed_text).strip()

    # ------------------------------------------------------------------
    # TOC / index handling
    # ------------------------------------------------------------------

    def _looks_like_index_line(
        self,
        line: str,
        page_number: int,
        total_pages: int,
    ) -> bool:
        """
        Detect likely single-line table-of-contents entries.

        Strong signals
        --------------
        - dot leaders
        - heading-like text followed by a trailing page number
        - uppercase heading residue on very early pages
        - short non-prose heading fragments that clearly do not behave like body text

        Safety behavior
        ---------------
        This heuristic remains conservative. Stronger cleanup is still mainly
        limited to the document front region.

        Parameters
        ----------
        line : str
            Candidate line.

        page_number : int
            Current page number.

        total_pages : int
            Total page count.

        Returns
        -------
        bool
            True when the line looks strongly index-like.
        """
        if len(line) > 220:
            return False

        if INDEX_DOT_LEADER_RE.search(line):
            return True

        if INDEX_TRAILING_PAGE_RE.match(line):
            words = line.split()
            if len(words) <= 14 and INDEX_HEADING_RE.match(line):
                return True

        # Additional heuristic:
        # on very early pages, short uppercase heading fragments are often
        # residual TOC/front-matter content rather than body text.
        if self._is_early_document_page(page_number, total_pages):
            if (
                UPPERCASE_RESIDUE_RE.match(line)
                and NON_PROSE_HEADING_RE.match(line)
                and not self._is_structural_line(line)
                and 4 <= len(line.split()) <= 14
            ):
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

        if not self._is_early_document_page(page_number, total_pages):
            return lines, Counter()

        kept_lines: List[str] = []
        dropped_counter: Counter[str] = Counter()

        index = 0
        while index < len(lines):
            block_lines: List[str] = []

            while index < len(lines):
                line = lines[index]
                is_toc_like = self._is_toc_block_candidate(
                    line=line,
                    page_number=page_number,
                    total_pages=total_pages,
                )

                if not is_toc_like and not block_lines:
                    break

                if not is_toc_like and block_lines:
                    break

                block_lines.append(line)
                index += 1

            # Only remove blocks that are sufficiently TOC-like.
            if block_lines and self._looks_like_real_toc_block(
                block_lines=block_lines,
                page_number=page_number,
                total_pages=total_pages,
            ):
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
        Decide whether a page belongs to the document front-matter region.

        Why this matters
        ----------------
        Title pages and tables of contents usually live near the beginning.
        Restricting aggressive TOC removal to early pages prevents accidental
        deletion of legitimate blocks later in the body.

        Returns
        -------
        bool
            True when the page is early enough to justify TOC-specific cleanup.
        """
        max_scan_pages = min(self._MAX_TOC_SCAN_PAGES, max(1, total_pages // 3 + 1))
        return page_number <= max_scan_pages

    def _is_toc_block_candidate(
        self,
        line: str,
        page_number: int,
        total_pages: int,
    ) -> bool:
        """
        Decide whether a line looks like part of a TOC block.

        Strong signals
        --------------
        - dot leaders
        - explicit structural headings such as "CAPÍTULO", "ARTIGO", "ÍNDICE"
        - short heading ending with a page number
        - uppercase heading residue on early pages

        Important
        ---------
        This helper alone is not sufficient to remove text.
        Final removal requires block-level evidence.
        """
        if INDEX_DOT_LEADER_RE.search(line):
            return True

        if INDEX_HEADING_RE.match(line):
            return True

        if INDEX_TRAILING_PAGE_RE.match(line) and len(line.split()) <= 16:
            return True

        if self._is_early_document_page(page_number, total_pages):
            if (
                UPPERCASE_RESIDUE_RE.match(line)
                and NON_PROSE_HEADING_RE.match(line)
                and not self._is_structural_line(line)
                and 4 <= len(line.split()) <= 18
            ):
                return True

        return False

    def _looks_like_real_toc_block(
        self,
        block_lines: List[str],
        page_number: int,
        total_pages: int,
    ) -> bool:
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
            * repeated uppercase heading residue on early pages

        Parameters
        ----------
        block_lines : List[str]
            Candidate block lines.

        page_number : int
            Current page number.

        total_pages : int
            Total page count.

        Returns
        -------
        bool
            True when the block strongly behaves like a TOC.
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
                continue

            if (
                self._is_early_document_page(page_number, total_pages)
                and UPPERCASE_RESIDUE_RE.match(line)
                and NON_PROSE_HEADING_RE.match(line)
                and not self._is_structural_line(line)
                and 4 <= len(line.split()) <= 18
            ):
                strong_signals += 1

        return strong_signals >= 3

    def _looks_like_uppercase_front_matter_residue(
        self,
        line: str,
        page_number: int,
        total_pages: int,
    ) -> bool:
        """
        Detect uppercase heading residue that should not survive into the parser.

        Why this helper exists
        ----------------------
        Some institutional PDFs contain broken front-matter or index fragments
        that are:
        - all uppercase
        - short/medium length
        - non-prose
        - not legal structural markers
        - mostly concentrated on early pages

        These lines frequently contaminate PREAMBLE if left untouched.

        Safety behavior
        ---------------
        This heuristic is deliberately restricted:
        - only early pages
        - excludes recognized structural headers
        - excludes ordinary prose starts
        - excludes long body-like text

        Parameters
        ----------
        line : str
            Candidate line.

        page_number : int
            Current page number.

        total_pages : int
            Total page count.

        Returns
        -------
        bool
            True when the line looks like front-matter residue.
        """
        if not self._is_early_document_page(page_number, total_pages):
            return False

        if self._is_structural_line(line):
            return False

        if PROSE_START_RE.match(line):
            return False

        if len(line) > 180:
            return False

        word_count = len(line.split())
        if word_count < 4 or word_count > 18:
            return False

        if not UPPERCASE_RESIDUE_RE.match(line):
            return False

        if not NON_PROSE_HEADING_RE.match(line):
            return False

        # Avoid dropping short legitimate mixed-content lines that merely
        # contain some uppercase words but are not truly heading-like.
        alpha_count = sum(1 for ch in line if ch.isalpha())
        upper_alpha_count = sum(1 for ch in line if ch.isalpha() and ch.isupper())
        if alpha_count == 0:
            return False

        upper_ratio = upper_alpha_count / alpha_count
        if upper_ratio < 0.85:
            return False

        return True

    # ------------------------------------------------------------------
    # Prose reflow
    # ------------------------------------------------------------------

    def _reflow_lines(self, lines: List[str]) -> List[str]:
        """
        Reflow obvious broken prose lines conservatively.

        Why this exists
        ---------------
        PDF extraction often breaks ordinary prose into multiple short lines.
        If those lines remain separated, the parser may inherit artificial
        paragraph boundaries that do not exist semantically.

        Safety behavior
        ---------------
        We only merge lines when:
        - the previous line does not look structural
        - the current line does not look structural
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

        if PROSE_START_RE.match(current_line):
            return True

        # Also merge when the current line starts lowercase, which frequently
        # happens in wrapped legal prose.
        if current_line[:1].islower():
            return True

        return False

    def _is_structural_line(self, line: str) -> bool:
        """
        Decide whether a line likely carries structural meaning.

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

    # ------------------------------------------------------------------
    # Garbling detection
    # ------------------------------------------------------------------

    def _looks_like_garbled_line(self, line: str) -> bool:
        """
        Decide whether a single line looks clearly corrupted or non-linguistic.

        Why this helper exists
        ----------------------
        Some problematic PDFs produce isolated lines that are obviously unusable,
        even when the full document does not need OCR fallback.

        Important safety note
        ---------------------
        This helper must remain conservative.
        We only drop lines that look strongly suspicious at line level.

        Parameters
        ----------
        line : str
            Candidate line.

        Returns
        -------
        bool
            True when the line looks clearly garbled.
        """
        if not line:
            return False

        if len(line) < 10:
            return False

        if SUSPICIOUS_GARBLED_LINE_RE.match(line):
            return True

        c1_control_count = sum(1 for ch in line if 0x80 <= ord(ch) <= 0x9F)
        if c1_control_count >= 1:
            compact_line = MULTISPACE_RE.sub("", line)
            if len(compact_line) >= 18:
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

        # Strongly suspicious when a line has very little alphabetic content,
        # many symbols, and almost no visible word separation.
        if alpha_ratio < 0.20 and symbol_ratio > 0.35 and whitespace_count <= 1:
            return True

        # Some corrupted publication lines still contain many letters, but they
        # collapse into unusually long compact strings mixed with punctuation or
        # mojibake markers. Treat those as garbled only when the line keeps
        # little visible word structure.
        compact_tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", line)
        longest_compact_token = max((len(token) for token in compact_tokens), default=0)
        if (
            longest_compact_token >= 28
            and whitespace_count <= 3
            and symbol_ratio > 0.08
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Final page/document rebuilding
    # ------------------------------------------------------------------

    def _finalize_page_text(self, cleaned_lines: List[str]) -> str:
        """
        Rebuild the cleaned page text.

        Final cleanup steps
        -------------------
        1. Rejoin the page using preserved line boundaries
        2. Repair conservative broken hyphenation artifacts
        3. Apply shared conservative block normalization

        Important
        ---------
        We still preserve structure-friendly line breaks here.
        """
        if not cleaned_lines:
            return ""

        text = "\n".join(cleaned_lines)

        # Reuse the shared conservative Task 1 repair sequence so normalized
        # parser input no longer carries broken line-break or inline hyphenation.
        text = repair_broken_hyphenation(text)

        # Apply shared conservative block cleanup without flattening structure.
        text = normalize_block_whitespace(text)

        return text.strip()

    def _build_full_text(self, pages: List[NormalizedPage]) -> str:
        """
        Build a debug-friendly full document text from normalized pages.

        Design choice
        -------------
        We keep page boundaries visible only as blank separation between pages,
        not as explicit page markers inside the body text.

        This makes the debug export easier to read while still preserving a
        useful notion of page separation.
        """
        non_empty_pages = [page.text.strip() for page in pages if page.text.strip()]
        if not non_empty_pages:
            return ""

        return "\n\n".join(non_empty_pages).strip()

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def _coerce_pages(
        self,
        document_or_pages: Union[
            ExtractedDocument,
            Sequence[PageText],
            Sequence[ExtractedPage],
        ],
    ) -> List[Union[PageText, ExtractedPage]]:
        """
        Normalize the input into a list of page-like objects.

        Why this helper exists
        ----------------------
        The codebase is currently transitioning from the legacy `PageText`
        model to the richer `ExtractedDocument` / `ExtractedPage` model.
        This helper allows normalization to support both during the migration.
        """
        if isinstance(document_or_pages, ExtractedDocument):
            return list(document_or_pages.pages)

        return list(document_or_pages)

    def _extract_page_fields(
        self,
        page: Union[PageText, ExtractedPage],
    ) -> Tuple[int, str, str, Optional[float], List[str]]:
        """
        Extract normalized fields from either PageText or ExtractedPage.

        Returns
        -------
        Tuple[int, str, str, Optional[float], List[str]]
            Normalized fields:
            - page_number
            - text
            - selected_mode
            - upstream_quality_score
            - upstream_flags
        """
        if isinstance(page, ExtractedPage):
            return (
                page.page_number,
                page.text,
                page.selected_mode,
                page.quality_score,
                list(page.corruption_flags),
            )

        return (
            page.page_number,
            page.text,
            "",
            None,
            [],
        )
