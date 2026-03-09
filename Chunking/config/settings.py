from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# -------------------------------------------------------------------------
# Project root resolution
# -------------------------------------------------------------------------
#
# This file lives in:
# Chunking/config/settings.py
#
# Therefore:
# parents[0] = config
# parents[1] = Chunking
# parents[2] = project root
#
# Keeping this dynamic makes the pipeline more portable and avoids hardcoded
# absolute paths that would break across developers, CI, or operating systems.
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class PipelineSettings:
    """
    Central runtime configuration for the chunking pipeline.

    Design goals:
    - keep configuration explicit and easy to inspect
    - avoid magic numbers spread across multiple modules
    - support fast tuning after real output inspection
    - provide settings that remain generic across legal/regulatory PDFs

    Important:
    chunk sizes are currently character-based, not token-based.
    This keeps the implementation lightweight and deterministic.
    """

    # ---------------------------------------------------------------------
    # Input / output folders
    # ---------------------------------------------------------------------
    #
    # These are resolved relative to the project root so the pipeline works
    # correctly on Windows, Linux, and macOS without path rewrites.
    # ---------------------------------------------------------------------
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    output_dir: Path = PROJECT_ROOT / "data" / "chunks"

    # ---------------------------------------------------------------------
    # Chunk sizing configuration
    # ---------------------------------------------------------------------
    #
    # Why these values:
    # - target_chunk_chars:
    #   preferred chunk size for semantically coherent retrieval units
    #
    # - hard_max_chunk_chars:
    #   safety ceiling used when a structural unit becomes too large
    #
    # - min_chunk_chars:
    #   avoids generating very small chunks with weak standalone meaning
    #
    # - overlap_chars:
    #   reserved for strategies that may later support overlap-based fallback
    #
    # These values are intentionally conservative for legal documents:
    # large enough to preserve context, but small enough to remain retrievable.
    # ---------------------------------------------------------------------
    target_chunk_chars: int = 1800
    hard_max_chunk_chars: int = 2600
    min_chunk_chars: int = 350
    overlap_chars: int = 180

    # ---------------------------------------------------------------------
    # Structural parsing / chunking behavior
    # ---------------------------------------------------------------------
    #
    # These switches help the pipeline remain explicit and tunable without
    # requiring code changes in parser or strategy modules.
    # ---------------------------------------------------------------------

    # Minimum number of repeated page occurrences required for a short line
    # to be considered likely header/footer noise.
    repeated_line_min_occurrences: int = 2

    # Fraction of document pages in which a short line must appear before it
    # is considered repeated layout noise.
    #
    # Example:
    # 0.5 means "appears in at least half the pages".
    repeated_line_min_page_ratio: float = 0.5

    # Maximum length of a line considered eligible for repeated-noise
    # detection. This protects real long content from accidental removal.
    repeated_line_max_chars: int = 120

    # Minimum number of consecutive TOC-like lines required before a block is
    # removed as a probable index/table-of-contents block.
    toc_block_min_lines: int = 4

    # Maximum number of title lines consumed after a chapter header.
    max_chapter_title_lines: int = 2

    # Maximum number of title lines consumed after an annex header.
    max_annex_title_lines: int = 3

    # Maximum number of title lines consumed after an article header.
    max_article_title_lines: int = 2

    # ---------------------------------------------------------------------
    # Export options
    # ---------------------------------------------------------------------
    #
    # DOCX inspection files are extremely useful for validating:
    # - chunk boundaries
    # - metadata quality
    # - parser decisions
    # - normalization side effects
    # ---------------------------------------------------------------------
    export_docx: bool = True
    export_json: bool = True
    export_intermediate_text: bool = True

    # ---------------------------------------------------------------------
    # Noise markers
    # ---------------------------------------------------------------------
    #
    # Common institutional phrases that often behave like layout noise
    # (headers / footers / cover furniture).
    #
    # These markers are not automatically removed by themselves, but they are
    # useful signals for heuristics and future tuning.
    # ---------------------------------------------------------------------
    likely_noise_markers: List[str] = field(
        default_factory=lambda: [
            "POLITÉCNICO DO PORTO",
            "P.PORTO",
            "REGULAMENTO",
            "DESPACHO",
        ]
    )

    # ---------------------------------------------------------------------
    # Supported file types
    # ---------------------------------------------------------------------
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf"]
    )