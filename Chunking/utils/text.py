from __future__ import annotations

import re
from typing import Iterable, List

from Chunking.config.patterns import MULTI_NEWLINE_RE, MULTI_SPACE_RE


def slugify_file_stem(name: str) -> str:
    """Convert a file name into a safe document identifier."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug[:120] if slug else "document"


def normalize_inline_whitespace(text: str) -> str:
    """
    Collapse repeated spaces while preserving line boundaries.
    """
    lines = [MULTI_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(lines)


def normalize_block_whitespace(text: str) -> str:
    """Clean repeated spaces and excessive blank lines."""
    text = normalize_inline_whitespace(text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def join_hyphenated_linebreaks(text: str) -> str:
    """
    Fix words broken by PDF line wrapping, such as:
    regula-
    mento

    This is intentionally conservative.
    """
    return re.sub(r"([A-Za-zÀ-ÿ])\-\n([A-Za-zÀ-ÿ])", r"\1\2", text)


def unwrap_single_newlines(text: str) -> str:
    """
    Convert most single line breaks into spaces, while preserving paragraph
    breaks represented by blank lines.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    rebuilt: List[str] = []
    for paragraph in paragraphs:
        rebuilt.append(re.sub(r"\s*\n\s*", " ", paragraph).strip())
    return "\n\n".join(rebuilt)


def split_paragraphs(text: str) -> List[str]:
    """Split text into meaningful paragraph blocks."""
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def first_non_empty(items: Iterable[str]) -> str:
    """Return the first non-empty item from an iterable."""
    for item in items:
        if item and item.strip():
            return item.strip()
    return ""
