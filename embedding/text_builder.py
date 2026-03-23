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
    - prefer the semantically useful chunk text already selected upstream
    - repair conservative PDF wrapping artifacts when safe
    - remove avoidable spacing noise
    - flatten incidental single-line wrapping while preserving paragraphs
    - avoid injecting additional structural prefixes at this stage
    """

    candidate_text = normalize_block_whitespace(record.text)
    if not candidate_text:
        return ""

    candidate_text = join_hyphenated_linebreaks(candidate_text)
    candidate_text = normalize_block_whitespace(candidate_text)
    candidate_text = unwrap_single_newlines(candidate_text)

    return normalize_block_whitespace(candidate_text)
