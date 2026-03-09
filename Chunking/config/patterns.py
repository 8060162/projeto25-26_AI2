"""
Regex and text patterns used by the pipeline.

These patterns are intentionally centralized here so that the team can tune
or extend them without needing to modify parsing or chunking code directly.

Important design principle:
this module should contain shared, low-level patterns only.
Document-specific parsing rules that are tightly coupled to the legal
structure parser may still live inside the parser module when that keeps the
logic clearer and more maintainable.
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

# Repeated horizontal spacing cleanup.
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Excessive blank-line cleanup.
#
# We keep up to one blank separator and collapse anything larger.
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


# -------------------------------------------------------------------------
# Shared structural heading patterns
# -------------------------------------------------------------------------
#
# These patterns are generic and intentionally broader than the stricter ones
# used by the dedicated structure parser.
#
# They are useful for:
# - exploratory preprocessing
# - generic diagnostics
# - lightweight structure detection in utilities
# -------------------------------------------------------------------------

# Example matches:
# "CAPÍTULO I"
# "CAPITULO IV - Disposições Gerais"
CHAPTER_RE = re.compile(
    r"^(CAP[IÍ]TULO\s+[IVXLCDM0-9]+)\s*(?:[-–—:]\s*)?(.*)$",
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


# -------------------------------------------------------------------------
# Shared list / sub-structure patterns
# -------------------------------------------------------------------------

# Numbered blocks inside articles / sections:
# "1."
# "2."
# "2.1."
# "2.1"
#
# This pattern is intentionally generic and line-based.
NUMBERED_BLOCK_RE = re.compile(
    r"(?m)^(\d+(?:\.\d+)*)\.?\s+",
)

# Legal-style alíneas:
# "a)"
# "b)"
# "c)"
LETTER_ITEM_RE = re.compile(
    r"(?m)^([a-z])\)\s+",
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
INDEX_HINT_RE = re.compile(
    r"(?i)(cap[ií]tulo|artigo|anexo).{0,80}(p[aá]g|\b\d+\b)"
)