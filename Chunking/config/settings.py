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

    Design goals
    ------------
    - keep configuration explicit and easy to inspect
    - avoid magic numbers spread across multiple modules
    - support fast tuning after real output inspection
    - remain generic across legal / regulatory PDF corpora
    - keep the current implementation lightweight and pragmatic

    Important note
    --------------
    Chunk sizes are currently character-based, not token-based.

    Why this is still acceptable for now
    ------------------------------------
    - it keeps the implementation deterministic
    - it avoids adding tokenizer dependencies too early
    - it is sufficient for the current iteration of the project

    Future evolution
    ----------------
    If the team later wants tighter LLM-oriented control, these settings can
    evolve toward token-based sizing without changing the overall pipeline
    architecture.
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
    # Why these values
    # ----------------
    # - target_chunk_chars:
    #   preferred chunk size for semantically coherent retrieval units
    #
    # - hard_max_chunk_chars:
    #   hard safety ceiling used when a structural unit becomes too large
    #
    # - min_chunk_chars:
    #   avoids generating very small chunks with weak standalone meaning
    #
    # - overlap_chars:
    #   reserved for future overlap-based fallback strategies
    #
    # These defaults are intentionally conservative for legal documents:
    # large enough to preserve context, but small enough to remain retrievable.
    # ---------------------------------------------------------------------
    target_chunk_chars: int = 1800
    hard_max_chunk_chars: int = 2600
    min_chunk_chars: int = 350
    overlap_chars: int = 180

    # ---------------------------------------------------------------------
    # Repeated-line detection behavior
    # ---------------------------------------------------------------------
    #
    # These settings support page-furniture removal such as repeated headers
    # and footers. The current normalizer focuses repeated-line detection on
    # page margins to reduce accidental removal of valid body text.
    # ---------------------------------------------------------------------

    # Minimum number of page occurrences required before a short line can be
    # considered repeated layout furniture.
    repeated_line_min_occurrences: int = 2

    # Minimum page-ratio threshold for repeated layout furniture.
    #
    # Example:
    # 0.5 means "appears in at least half the pages".
    repeated_line_min_page_ratio: float = 0.5

    # Maximum line length eligible for repeated-line detection.
    # This helps protect real long content from accidental removal.
    repeated_line_max_chars: int = 120

    # Number of top / bottom non-empty lines inspected per page when looking
    # for repeated page furniture.
    repeated_line_margin_window: int = 4

    # ---------------------------------------------------------------------
    # Table-of-contents / front-matter cleanup behavior
    # ---------------------------------------------------------------------
    #
    # These settings help the normalizer remain explicit and tunable without
    # requiring code changes in cleanup logic.
    # ---------------------------------------------------------------------

    # Minimum number of consecutive TOC-like lines before a candidate block is
    # considered a probable index / table-of-contents block.
    toc_block_min_lines: int = 4

    # Maximum number of early pages where aggressive TOC cleanup is allowed.
    # This protects later legitimate structural content from being removed.
    max_toc_scan_pages: int = 5

    # ---------------------------------------------------------------------
    # Title-consumption behavior
    # ---------------------------------------------------------------------
    #
    # These values are used when the parser consumes title lines immediately
    # following structural headers.
    # ---------------------------------------------------------------------
    max_chapter_title_lines: int = 2
    max_annex_title_lines: int = 3
    max_article_title_lines: int = 2
    max_section_container_title_lines: int = 2

    # ---------------------------------------------------------------------
    # Strategy execution behavior
    # ---------------------------------------------------------------------
    #
    # These flags are useful for future tuning and explicit pipeline behavior.
    # Even when not fully used everywhere yet, keeping them here makes the
    # intended execution model clearer.
    # ---------------------------------------------------------------------

    # Allow the CLI to run all strategies in a single execution.
    allow_all_strategies: bool = True

    # ---------------------------------------------------------------------
    # Export options
    # ---------------------------------------------------------------------
    #
    # Inspection files are extremely useful for validating:
    # - chunk boundaries
    # - metadata quality
    # - parser decisions
    # - normalization side effects
    # - strategy comparison
    # ---------------------------------------------------------------------
    export_docx: bool = True
    export_json: bool = True
    export_intermediate_text: bool = True

    # Export an additional lightweight JSON quality summary for each
    # document/strategy run.
    export_quality_summary: bool = True

    # ---------------------------------------------------------------------
    # Chunk enrichment behavior
    # ---------------------------------------------------------------------
    #
    # These settings reflect the newer chunk model that supports both:
    # - visible text
    # - text enriched for embeddings
    # ---------------------------------------------------------------------

    # Include structure-enriched text_for_embedding in the final chunk model.
    include_text_for_embedding: bool = True

    # Link neighboring chunks using prev_chunk_id / next_chunk_id.
    enable_chunk_neighbor_links: bool = True

    # ---------------------------------------------------------------------
    # Noise markers
    # ---------------------------------------------------------------------
    #
    # Common institutional phrases that often behave like layout noise
    # (headers / footers / cover furniture).
    #
    # Important:
    # these markers should be treated as weak signals only, not as automatic
    # deletion rules by themselves.
    # ---------------------------------------------------------------------
    likely_noise_markers: List[str] = field(
        default_factory=lambda: [
            "POLITÉCNICO DO PORTO",
            "P.PORTO",
            "REGULAMENTO",
            "DESPACHO",
            "DIÁRIO DA REPÚBLICA",
            "ÍNDICE",
        ]
    )

    # ---------------------------------------------------------------------
    # Supported file types
    # ---------------------------------------------------------------------
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf"]
    )