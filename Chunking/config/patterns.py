"""
Regex and text patterns used by the pipeline.

These patterns are intentionally centralized here so they can be tuned
without changing multiple parser or normalizer modules.

Important:
This file should only exist if it is treated as a real source of truth.
If parser and normalizer keep their own independent regexes, this module
should be removed to avoid drift.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------
# Generic cleanup patterns
# ---------------------------------------------------------------------

PAGE_COUNTER_RE = re.compile(r"^\s*\d+\s*\|\s*\d+\s*$")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# ---------------------------------------------------------------------
# Structural headings
# ---------------------------------------------------------------------

CHAPTER_RE = re.compile(
    r"^\s*CAP[ÍI]TULO\s+([IVXLCDM0-9]+)\s*(?:[-–—:]\s*(.*))?\s*$",
    flags=re.IGNORECASE,
)

ARTICLE_RE = re.compile(
    r"^\s*(?:ARTIGO|ART\.?)\s+([0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*\.?\s*[ºo])?\s*(?:[-–—:]\s*(.*))?\s*$",
    flags=re.IGNORECASE,
)

ANNEX_RE = re.compile(
    r"^\s*(ANEXO(?:\s+[IVXLCDM0-9A-Z\-]+)?)\s*(?:[-–—:]\s*(.*))?\s*$",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------
# Internal structural blocks
# ---------------------------------------------------------------------

NUMBERED_BLOCK_RE = re.compile(
    r"(?m)^(?P<label>\d+(?:\.\d+)*)(?:\.)?\s*(?:[—–\-]\s*)?"
)

LETTER_ITEM_RE = re.compile(
    r"(?m)^(?P<label>[a-z])\)\s+",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------
# TOC / index hints
# ---------------------------------------------------------------------

INDEX_DOT_LEADER_RE = re.compile(r"\.{3,}")

INDEX_TRAILING_PAGE_RE = re.compile(
    r"^.+\s+\d{1,4}\s*$"
)

INDEX_HEADING_RE = re.compile(
    r"^\s*(CAP[ÍI]TULO|ARTIGO|ANEXO|ÍNDICE)\b",
    flags=re.IGNORECASE,
)