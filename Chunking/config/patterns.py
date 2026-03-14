"""
Regex and text patterns shared across the pipeline.

Why this module exists
----------------------
These patterns are intentionally centralized here so the team can tune or
extend common low-level behavior without editing multiple modules.

Design principle
----------------
This module should contain only:
- shared
- generic
- low-level
patterns.

Document-specific structural rules that are tightly coupled to the legal
structure parser may still live inside the parser module when that keeps the
logic clearer and safer.
"""

from __future__ import annotations

import re


# -------------------------------------------------------------------------
# Generic layout / cleanup patterns
# -------------------------------------------------------------------------

# Page counters commonly found in institutional PDFs, such as:
# "3|14"
# "10 | 14"
PAGE_COUNTER_RE = re.compile(r"\b\d+\s*\|\s*\d+\b")

# Loose full-line page-counter pattern.
#
# This is useful for cleanup logic that wants to identify lines consisting
# almost entirely of page counters and little else.
#
# Example matches:
# "3|14"
# "10 | 14"
# "3 | 14 -"
LOOSE_PAGE_COUNTER_LINE_RE = re.compile(
    r"^\s*\d+\s*\|\s*\d+\s*[\-–—.]?\s*$"
)

# Leading page marker.
#
# Example matches:
# "Pág. 3"
# "Pág. 27"
LEADING_PAGE_MARKER_RE = re.compile(
    r"^\s*Pág\.\s*\d+\s*",
    flags=re.IGNORECASE,
)

# Long pure numeric lines often correspond to publication / layout noise.
#
# Example matches:
# "316552597"
PURE_NUMERIC_NOISE_RE = re.compile(r"^\s*\d{6,}\s*$")

# Repeated horizontal spacing cleanup.
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Excessive blank-line cleanup.
#
# We keep up to one blank separator and collapse anything larger.
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Detect dot leaders frequently used in indices / tables of contents.
#
# Example matches:
# "........"
INDEX_DOT_LEADER_RE = re.compile(r"\.{3,}")

# Detect heading-like lines ending with a page number.
#
# Example matches:
# "CAPÍTULO I 3"
# "ANEXO A 12"
INDEX_TRAILING_PAGE_RE = re.compile(r".+\s+\d{1,4}\s*$")

# Detect highly symbolic lines that are usually extraction garbage.
MOSTLY_SYMBOLIC_RE = re.compile(r"^[^\wÀ-ÿ]{6,}$")

# Detect suspicious garbled lines made mostly of symbols and punctuation.
#
# Example matches:
# "*+,-*-.-/-1/2*-34+*4/-5/-1/6-/"
SUSPICIOUS_GARBLED_LINE_RE = re.compile(
    r"^[^A-Za-zÀ-ÿ]{0,3}(?:[\*\+\-/=<>\\\[\]\{\}_`~]{2,}|[0-9\W]{12,})$"
)


# -------------------------------------------------------------------------
# Shared structural heading patterns
# -------------------------------------------------------------------------
#
# These patterns are intentionally broader than the stricter patterns used
# inside the dedicated structure parser.
#
# They are useful for:
# - exploratory preprocessing
# - diagnostics
# - lightweight structural hints in utility code
# -------------------------------------------------------------------------

# Example matches:
# "CAPÍTULO I"
# "CAPITULO IV - Disposições Gerais"
CHAPTER_RE = re.compile(
    r"^(CAP[IÍ]TULO\s+[IVXLCDM0-9]+)\s*(?:[-–—:]\s*)?(.*)$",
    flags=re.IGNORECASE,
)

# Example matches:
# "SECÇÃO I"
# "SUBSECÇÃO II"
# "TÍTULO III - Disposições Gerais"
SECTION_CONTAINER_RE = re.compile(
    r"^(SECÇÃO|SUBSECÇÃO|TÍTULO)\s+([IVXLCDM0-9A-Z]+)\s*(?:[-–—:]\s*)?(.*)$",
    flags=re.IGNORECASE,
)

# Example matches:
# "ARTIGO 1"
# "ART. 2.º"
# "Artigo 10 - Âmbito"
#
# Note:
# this shared pattern accepts decimal numbering as well because some corpora
# use article-like local numbering in appendices or converted exports.
ARTICLE_RE = re.compile(
    r"^(?:ARTIGO|ART\.?)[\s\-]*([0-9]+(?:\.[0-9]+)?)\s*(?:\.º|º|o)?\s*(.*)$",
    flags=re.IGNORECASE,
)

# Example matches:
# "ANEXO"
# "ANEXO I"
# "ANEXO A - Tabela"
ANNEX_RE = re.compile(
    r"^(ANEXO(?:\s+[IVXLCDM0-9A-Z\-]+)?)\s*(?:[-–—:]\s*)?(.*)$",
    flags=re.IGNORECASE,
)

# Generic heading-like lines often found in tables of contents or legal
# structure markers.
#
# Example matches:
# "CAPÍTULO I"
# "ARTIGO 2"
# "ANEXO A"
# "SECÇÃO II"
INDEX_HEADING_RE = re.compile(
    r"^(CAP[IÍ]TULO|SECÇÃO|SUBSECÇÃO|TÍTULO|ARTIGO|ANEXO|ÍNDICE)\b",
    flags=re.IGNORECASE,
)

# Uppercase-heavy lines often behave like headings or titles.
#
# This pattern is intentionally generic and should not be treated as proof
# of structure on its own.
UPPERCASE_HEAVY_RE = re.compile(
    r"^[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ0-9 /().,\-–—ºª]+$"
)


# -------------------------------------------------------------------------
# Shared list / sub-structure patterns
# -------------------------------------------------------------------------

# Numbered blocks inside articles / sections:
# "1."
# "2."
# "2.1."
# "2.1"
# "1)"
# "1 -"
# "1 —"
# "n.º 1"
#
# This pattern is intentionally generic and line-based.
NUMBERED_BLOCK_RE = re.compile(
    r"(?m)^((?:n\.?\s*[ºo]\s*)?\d+(?:\.\d+)*)"
    r"(?:\.\s+|\)\s+|\s+[—–\-]\s+|\s+)",
    flags=re.IGNORECASE,
)

# Legal-style alíneas:
# "a)"
# "b)"
# "c)"
LETTER_ITEM_RE = re.compile(
    r"(?m)^([a-z])\)\s+",
    flags=re.IGNORECASE,
)

# Detect probable prose openings.
#
# This is useful when deciding whether a line likely begins ordinary sentence
# content rather than a structural heading.
PROSE_START_RE = re.compile(
    r"^\s*(?:O|A|Os|As|No|Na|Nos|Nas|Em|Para|Por|Quando|Sempre|Caso|Se|Nos\s+termos|Deve|Devem|Pode|Podem|É|São)\b",
    flags=re.IGNORECASE,
)

# Detect line endings that usually behave like hard boundaries.
HARD_LINE_END_RE = re.compile(r"[.;:!?]$")


# -------------------------------------------------------------------------
# Front-matter / institutional hints
# -------------------------------------------------------------------------
#
# These patterns are useful as weak signals for:
# - cover/title pages
# - dispatch headers
# - institutional banners
# - front matter vs normative body differentiation
# -------------------------------------------------------------------------

# Typical institutional / dispatch-like openings.
#
# Example matches:
# "POLITÉCNICO DO PORTO"
# "P.PORTO"
# "DESPACHO ..."
# "REGULAMENTO n.º 123/2024"
# "ÍNDICE"
FRONT_MATTER_RE = re.compile(
    r"^\s*(POLITÉCNICO DO PORTO|P\.PORTO|DESPACHO|REGULAMENTO\s+N\.?\s*º|REGULAMENTO\s+Nº|ÍNDICE)\b",
    flags=re.IGNORECASE,
)

# Institutional banner lines that often behave like repeated headers.
INSTITUTIONAL_BANNER_RE = re.compile(
    r"^\s*(POLIT[ÉE]CNICO\s+DO\s+PORTO|P\.PORTO)\s*$",
    flags=re.IGNORECASE,
)

# Diário da República / editorial lines.
DR_EDITORIAL_RE = re.compile(
    r"^\s*(N\.?\s*º|PARTE\s+[A-Z]|Diário da República)\b",
    flags=re.IGNORECASE,
)

# Likely signature lines that usually should not remain in retrieval text.
SIGNATURE_LINE_RE = re.compile(
    r"^\s*(Assinado por:|Num\.?\s+de\s+Identificação:|Data:\s*\d{4}\.\d{2}\.\d{2})",
    flags=re.IGNORECASE,
)

# Decorative / cover-like regulation lines.
COVER_NOISE_RE = re.compile(
    r"^\s*(DESPACHO\s+P\.PORTO\/P-|REGULAMENTO\s+P\.PORTO\/P-)\S*",
    flags=re.IGNORECASE,
)


# -------------------------------------------------------------------------
# Table-of-contents / index hints
# -------------------------------------------------------------------------
#
# This is not meant to be a definitive TOC detector.
# It is only a lightweight shared hint pattern for modules that want a quick
# signal that a line may belong to an index-like block.
# -------------------------------------------------------------------------

# Example matches:
# "CAPÍTULO I ........ 3"
# "ARTIGO 2 5"
# "ANEXO A pág. 12"
INDEX_HINT_RE = re.compile(
    r"(?i)(cap[ií]tulo|secção|subsecção|título|artigo|anexo).{0,80}(p[aá]g|\b\d+\b)"
)