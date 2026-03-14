from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ============================================================================
# Project root resolution
# ============================================================================
#
# This file lives in:
#     Chunking/config/settings.py
#
# Therefore:
#     parents[0] = config
#     parents[1] = Chunking
#     parents[2] = project root
#
# Keeping this dynamic makes the pipeline portable and avoids hardcoded
# absolute paths that would break across developers, CI environments, or
# operating systems.
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class PipelineSettings:
    """
    Central runtime configuration for the PDF structure extraction pipeline.

    Current project focus
    ---------------------
    The pipeline is no longer only a chunking pipeline.

    The primary goal at this stage is:
        PDF -> structured extraction -> normalization -> parsing
        -> canonical master-dictionary-style JSON

    Chunking is still supported, but it is now a downstream optional stage.

    Design goals
    ------------
    - keep configuration explicit and easy to inspect
    - avoid magic numbers spread across multiple modules
    - support fast tuning after real output inspection
    - remain generic across legal / regulatory PDF corpora
    - keep the implementation lightweight and pragmatic

    Important note
    --------------
    Some chunk-related settings are still present because chunking remains
    available as an optional later phase.
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
    # Extraction stage behavior
    # ---------------------------------------------------------------------
    #
    # These settings control the first major stage of the pipeline:
    # - native PDF extraction
    # - extraction quality analysis
    # - OCR fallback decision
    # ---------------------------------------------------------------------

    # Enable extraction-quality analysis immediately after native PDF extraction.
    enable_extraction_quality_analysis: bool = True

    # When True, the pipeline may switch to OCR fallback if the extracted text
    # appears severely corrupted.
    enable_ocr_fallback: bool = True

    # OCR rendering resolution.
    #
    # 300 DPI is a practical default for OCR:
    # - usually good enough for legal PDFs
    # - not too expensive
    # - widely used in document OCR workflows
    ocr_dpi: int = 300

    # Default Tesseract language code used by OCR fallback.
    #
    # For Portuguese legal documents, "por" is the correct default.
    ocr_language: str = "por"

    # Document-level suspicious-page threshold above which OCR fallback becomes
    # justified.
    #
    # Example:
    # 0.40 means "40% or more pages look suspicious".
    suspicious_page_ratio_threshold: float = 0.40

    # ---------------------------------------------------------------------
    # Repeated-line detection behavior
    # ---------------------------------------------------------------------
    #
    # These settings support page-furniture removal such as repeated headers
    # and footers. The normalizer focuses repeated-line detection on page
    # margins to reduce accidental removal of valid body text.
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
    repeated_line_max_chars: int = 140

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
    # considered a probable TOC/index block.
    toc_block_min_lines: int = 4

    # Maximum number of early pages where aggressive TOC cleanup is allowed.
    # This protects later legitimate structural content from being removed.
    max_toc_scan_pages: int = 5

    # ---------------------------------------------------------------------
    # Parser title-consumption behavior
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
    # Canonical structure export behavior
    # ---------------------------------------------------------------------
    #
    # These settings relate to the current project objective:
    # exporting a canonical master-dictionary-style JSON tree.
    # ---------------------------------------------------------------------

    # Export the generic internal structure tree for debugging.
    export_debug_structure_json: bool = True

    # Export the canonical master-dictionary-style JSON.
    export_master_json: bool = True

    # Preserve generic parser metadata in the canonical export when useful.
    include_filtered_node_metadata_in_master_json: bool = True

    # ---------------------------------------------------------------------
    # Chunk sizing configuration
    # ---------------------------------------------------------------------
    #
    # These settings remain relevant only for the optional chunking stage.
    #
    # Important note
    # --------------
    # Chunk sizes are currently character-based, not token-based.
    #
    # Why this is still acceptable for now
    # ------------------------------------
    # - keeps the implementation deterministic
    # - avoids tokenizer dependencies too early
    # - is sufficient for the current stage of experimentation
    # ---------------------------------------------------------------------
    target_chunk_chars: int = 1800
    hard_max_chunk_chars: int = 2600
    min_chunk_chars: int = 350
    overlap_chars: int = 180

    # ---------------------------------------------------------------------
    # Strategy execution behavior
    # ---------------------------------------------------------------------
    #
    # These flags control the optional chunking phase.
    # ---------------------------------------------------------------------

    # Allow the CLI to run all strategies in a single execution.
    allow_all_strategies: bool = True

    # Enable the hybrid strategy.
    enable_hybrid_strategy: bool = True

    # ---------------------------------------------------------------------
    # Export options
    # ---------------------------------------------------------------------
    #
    # Inspection files are useful for validating:
    # - extraction behavior
    # - normalization side effects
    # - parser decisions
    # - canonical JSON shape
    # - optional chunking quality
    # ---------------------------------------------------------------------
    export_docx: bool = True
    export_json: bool = True
    export_intermediate_text: bool = True
    export_quality_summary: bool = True
    export_extraction_quality_report: bool = True

    # ---------------------------------------------------------------------
    # Chunk enrichment behavior
    # ---------------------------------------------------------------------
    #
    # These settings apply only to the optional chunking stage.
    # ---------------------------------------------------------------------

    # Include structure-enriched text_for_embedding in final chunk payloads.
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