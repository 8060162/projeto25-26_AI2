"""
Regex and text patterns used by the pipeline.

These patterns are intentionally centralized here so that the team can tune
or extend them without needing to modify parsing or chunking code directly.
"""

import re

# Page counters commonly found in institutional PDFs, such as "3|14".
PAGE_COUNTER_RE = re.compile(r"\b\d+\s*\|\s*\d+\b")

# Whitespace cleanup.
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Common chapter / article / annex patterns.
CHAPTER_RE = re.compile(
    r"^(CAP[IÍ]TULO\s+[IVXLCDM0-9]+)\s*(?:[-–—:]\s*)?(.*)$",
    flags=re.IGNORECASE,
)

ARTICLE_RE = re.compile(
    r"^(?:ARTIGO|ART\.?)[\s\-]*([0-9]+(?:\.[0-9]+)?)\s*(?:\.º|º|o)?\s*(.*)$",
    flags=re.IGNORECASE,
)

ANNEX_RE = re.compile(
    r"^(ANEXO\s+[IVXLCDM0-9A-Z\-]*)\s*(?:[-–—:]\s*)?(.*)$",
    flags=re.IGNORECASE,
)

# Numbered blocks inside articles: 1. 2. 2.1 2.2 etc.
NUMBERED_BLOCK_RE = re.compile(
    r"(?m)^(\d+(?:\.\d+)*)\.\s+",
)

# Legal-style alíneas: a) b) c)
LETTER_ITEM_RE = re.compile(
    r"(?m)^([a-z])\)\s+",
    flags=re.IGNORECASE,
)

# Detect likely table of contents / index lines.
INDEX_HINT_RE = re.compile(
    r"(?i)(cap[ií]tulo|artigo|anexo).{0,80}(p[aá]g|\b\d+\b)"
)
