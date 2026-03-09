from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from Chunking.config.patterns import MULTI_NEWLINE_RE, MULTI_SPACE_RE


# -------------------------------------------------------------------------
# Character-level cleanup patterns
# -------------------------------------------------------------------------
#
# These helpers are intentionally conservative.
#
# The pipeline processes Portuguese legal / regulatory PDF text, so we must
# avoid destructive cleanup that could silently remove meaningful content.
#
# The goals of this module are:
# - remove obvious control-character noise introduced by PDF extraction
# - normalize Unicode in a predictable and stable way
# - preserve structural newlines for downstream parsing
# - keep text readable for inspection, chunking, and embeddings
# -------------------------------------------------------------------------

# Match Unicode control characters while preserving:
# - newline (\n)
# - carriage return (\r), which is normalized later
# - tab (\t), which is later collapsed into ordinary spacing
#
# Why this matters:
# PDF extraction frequently injects invisible bytes such as:
# \x00, \x03, \x10, \x1f, etc.
#
# Those characters later pollute:
# - article titles
# - chunk text
# - metadata
# - keyword search
# - inspection exports
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Soft hyphen is frequently introduced by OCR / PDF extraction and is almost
# never useful in the final normalized representation.
SOFT_HYPHEN_RE = re.compile("\u00AD")

# Non-breaking spaces often appear in extracted PDF text and should behave
# like ordinary spaces inside the pipeline.
NBSP_RE = re.compile(r"[\u00A0\u2007\u202F]")

# Match repeated horizontal whitespace.
#
# We keep this local pattern because some helpers need explicit control over
# spaces and tabs independently of the shared project-level patterns.
HORIZONTAL_WS_RE = re.compile(r"[ \t]+")

# Used for conservative repair of awkward spaces before punctuation.
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

# Used for conservative cleanup of spacing after opening brackets.
SPACE_AFTER_OPENING_BRACKET_RE = re.compile(r"([(\[{])\s+")

# Used for conservative cleanup of spacing before closing brackets.
SPACE_BEFORE_CLOSING_BRACKET_RE = re.compile(r"\s+([)\]}])")

# Join words broken across a line by PDF wrapping, for example:
# "regula-\nmento" -> "regulamento"
#
# This must remain conservative:
# we only join when both sides look alphabetic.
HYPHENATED_LINEBREAK_RE = re.compile(r"([A-Za-zÀ-ÿ])-\n([A-Za-zÀ-ÿ])")

# Some extracted PDFs also break lines inside words without a hyphen.
# Repairing those cases safely is much harder and much riskier.
# We deliberately do not attempt aggressive word fusion here.


def slugify_file_stem(name: str) -> str:
    """
    Convert a file stem into a safe and stable document identifier.

    Design goals:
    - keep identifiers filesystem-safe
    - keep identifiers log/debug friendly
    - avoid unbounded length
    - remain deterministic

    Example:
        "Despacho n.º 7088/2023" -> "Despacho_n_7088_2023"

    Notes:
    - We intentionally keep ASCII alphanumerics plus underscore only.
    - This helper is meant for identifiers, not for user-facing labels.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug[:120] if slug else "document"


def strip_control_characters(text: str) -> str:
    """
    Remove non-printable control characters that commonly appear in PDF text.

    Why this helper exists:
    extracted PDF text often contains hidden control bytes that:
    - break regex matching
    - pollute chunk text
    - produce unreadable inspection outputs
    - degrade embeddings and retrieval quality

    Important:
    we preserve structural newline handling by not removing line breaks here.
    """
    if not text:
        return ""

    return CONTROL_CHARS_RE.sub("", text)


def normalize_unicode_text(text: str) -> str:
    """
    Apply conservative Unicode normalization suitable for legal text.

    Steps:
    - normalize compatibility variants with NFKC
    - remove soft hyphen
    - convert non-breaking spaces to ordinary spaces

    Why NFKC:
    it reduces many extraction inconsistencies while generally preserving the
    visible meaning of the text.

    Important:
    we intentionally do not lowercase text or remove accents because those may
    be meaningful for:
    - display
    - legal references
    - user inspection
    - downstream matching against official wording
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = SOFT_HYPHEN_RE.sub("", text)
    text = NBSP_RE.sub(" ", text)
    return text


def normalize_line_endings(text: str) -> str:
    """
    Normalize all line endings to Unix-style newlines.

    Why this matters:
    PDF extraction and cross-platform files may contain:
    - CRLF (Windows)
    - CR (legacy/macOS old style)
    - LF

    The parser and chunker assume line ending behavior is consistent.
    """
    if not text:
        return ""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_inline_whitespace(text: str) -> str:
    """
    Collapse repeated horizontal whitespace while preserving line boundaries.

    This function is intentionally line-aware:
    - it cleans each line independently
    - it does not flatten structural newlines
    - it keeps the text suitable for later structure parsing

    Example:
        "Artigo   1   \\n   Âmbito" -> "Artigo 1\\nÂmbito"
    """
    if not text:
        return ""

    lines = [
        HORIZONTAL_WS_RE.sub(" ", line).strip()
        for line in text.splitlines()
    ]
    return "\n".join(lines)


def normalize_punctuation_spacing(text: str) -> str:
    """
    Perform conservative punctuation spacing cleanup.

    This helper fixes common PDF extraction artifacts such as:
    - extra spaces before punctuation
    - extra spaces after opening brackets
    - extra spaces before closing brackets

    It is intentionally conservative and does not try to rewrite legal citation
    style or typography conventions.
    """
    if not text:
        return ""

    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = SPACE_AFTER_OPENING_BRACKET_RE.sub(r"\1", text)
    text = SPACE_BEFORE_CLOSING_BRACKET_RE.sub(r"\1", text)
    return text


def normalize_block_whitespace(text: str) -> str:
    """
    Normalize a block of text while preserving document structure.

    This is the main high-level cleanup helper used before exporting chunks
    or preparing text for display / debugging.

    Cleanup order matters:
    1. Normalize Unicode variants
    2. Normalize line endings
    3. Remove control characters
    4. Normalize horizontal whitespace
    5. Normalize punctuation spacing
    6. Collapse excessive blank lines

    Important:
    this function does NOT unwrap all single newlines.
    The structure parser depends on those line boundaries.
    """
    if not text:
        return ""

    text = normalize_unicode_text(text)
    text = normalize_line_endings(text)
    text = strip_control_characters(text)
    text = normalize_inline_whitespace(text)
    text = normalize_punctuation_spacing(text)

    # Collapse excessive blank lines while preserving paragraph separation.
    text = MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def join_hyphenated_linebreaks(text: str) -> str:
    """
    Repair words split by PDF line wrapping.

    Example:
        "regula-\\nmento" -> "regulamento"

    Design choice:
    this remains intentionally conservative because over-aggressive word joining
    can corrupt valid legal text.

    We only join when:
    - the left side ends with a letter
    - the right side starts with a letter

    We do not attempt broader OCR reconstruction in this helper.
    """
    if not text:
        return ""

    return HYPHENATED_LINEBREAK_RE.sub(r"\1\2", text)


def unwrap_single_newlines(text: str) -> str:
    """
    Convert most single newlines into spaces while preserving paragraph breaks.

    This helper is useful only in contexts where:
    - structure parsing has already finished, or
    - a text block is known to behave like paragraph prose

    It should NOT be used before structural parsing of legal documents,
    because article headers, section labels, and list items often depend
    on original line boundaries.

    Example:
        "Linha 1\\nLinha 2\\n\\nParágrafo 2"
        -> "Linha 1 Linha 2\\n\\nParágrafo 2"
    """
    if not text:
        return ""

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    rebuilt: List[str] = []

    for paragraph in paragraphs:
        rebuilt.append(re.sub(r"\s*\n\s*", " ", paragraph).strip())

    return "\n\n".join(rebuilt)


def split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraph-like blocks.

    Current rule:
    - blank-line separation defines paragraph boundaries

    Why this is intentionally simple:
    in legal and regulatory corpora, paragraph splitting must be predictable.
    More aggressive heuristics often damage section / list structure.

    Expected usage:
    - chunk fallback grouping
    - preamble subdivision
    - oversized article subdivision

    Important:
    input should already be reasonably normalized.
    """
    if not text:
        return []

    text = normalize_line_endings(text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)

    return [part.strip() for part in text.split("\n\n") if part.strip()]


def first_non_empty(items: Iterable[str]) -> str:
    """
    Return the first non-empty, non-whitespace item from an iterable.

    This helper is useful when choosing among:
    - inline titles
    - consumed title lines
    - fallback labels

    Returns an empty string when no usable value exists.
    """
    for item in items:
        if item and item.strip():
            return item.strip()

    return ""