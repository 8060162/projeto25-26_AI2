from __future__ import annotations

from embedding.models import EmbeddingInputRecord
from Chunking.utils.text import (
    join_hyphenated_linebreaks,
    normalize_block_whitespace,
    unwrap_single_newlines,
)


def build_embedding_text(record: EmbeddingInputRecord) -> str:
    """
    Build the final text payload sent to the embedding provider.

    Parameters
    ----------
    record : EmbeddingInputRecord
        Embedding input record produced by the chunk loader.

    Returns
    -------
    str
        Clean embedding text ready for vector generation.

    Design rules
    ------------
    - preserve the semantically useful chunk text already selected upstream
    - repair conservative PDF wrapping artifacts when safe
    - remove avoidable spacing noise
    - flatten incidental single-line wrapping while preserving paragraphs
    """

    return _normalize_embedding_body_text(record.text)


def _normalize_embedding_body_text(text: str) -> str:
    """
    Apply the conservative cleanup used for the main embedding body text.

    Parameters
    ----------
    text : str
        Raw chunk text selected for embedding.

    Returns
    -------
    str
        Cleaned body text ready for vector generation.
    """

    candidate_text = normalize_block_whitespace(text)
    if not candidate_text:
        return ""

    candidate_text = join_hyphenated_linebreaks(candidate_text)
    candidate_text = normalize_block_whitespace(candidate_text)
    candidate_text = unwrap_single_newlines(candidate_text)

    return normalize_block_whitespace(candidate_text)
