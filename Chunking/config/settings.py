from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


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
APPSETTINGS_PATH = PROJECT_ROOT / "config" / "appsettings.json"


def _load_appsettings() -> Dict[str, Any]:
    """
    Load the central application settings from the project configuration file.

    The loader is intentionally tolerant:
    - missing files fall back to defaults defined in PipelineSettings
    - invalid JSON falls back to defaults instead of breaking the pipeline
    - non-dictionary payloads are ignored
    """

    if not APPSETTINGS_PATH.exists():
        return {}

    try:
        with APPSETTINGS_PATH.open("r", encoding="utf-8") as settings_file:
            loaded_settings = json.load(settings_file)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(loaded_settings, dict):
        return {}

    return loaded_settings


def _get_nested_value(
    data: Dict[str, Any],
    path: List[str],
    default_value: Any,
) -> Any:
    """
    Read a nested configuration value from a dictionary.

    Parameters
    ----------
    data : Dict[str, Any]
        Root configuration dictionary.
    path : List[str]
        Ordered key path to the desired nested value.
    default_value : Any
        Fallback value returned when the path is missing or invalid.
    """

    current_value: Any = data

    for key in path:
        if not isinstance(current_value, dict) or key not in current_value:
            return default_value
        current_value = current_value[key]

    return current_value


def _resolve_project_path(value: Any, default_path: Path) -> Path:
    """
    Resolve a configuration path relative to the project root when needed.

    Parameters
    ----------
    value : Any
        Raw configuration value expected to represent a filesystem path.
    default_path : Path
        Fallback path used when the configuration value is missing or invalid.
    """

    if not isinstance(value, str) or not value.strip():
        return default_path

    candidate_path = Path(value)
    if candidate_path.is_absolute():
        return candidate_path

    return PROJECT_ROOT / candidate_path


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

    # Document-level replacement-character density above which native
    # extraction is treated as too corrupted to keep.
    document_ocr_replacement_ratio_threshold: float = 0.30

    # Document-level empty-page density above which native extraction is
    # treated as too incomplete to keep.
    document_ocr_empty_ratio_threshold: float = 0.30

    # Minimum suspicious-page density that can trigger document-level OCR when
    # the average page quality is already weak.
    document_ocr_low_quality_suspicious_ratio_threshold: float = 0.25

    # Average page-quality ceiling paired with suspicious-page density for
    # document-level OCR fallback.
    document_ocr_low_quality_average_score_threshold: float = 15.0

    # Minimum suspicious-page density that can trigger document-level OCR when
    # replacement-like corruption also appears repeatedly.
    document_ocr_replacement_mix_suspicious_ratio_threshold: float = 0.20

    # Replacement-character density paired with suspicious-page density for
    # document-level OCR fallback.
    document_ocr_replacement_mix_replacement_ratio_threshold: float = 0.15

    # Minimum legal-marker coverage below which a weak-quality document is
    # treated as semantically unsafe to keep native.
    document_ocr_low_legal_marker_coverage_threshold: float = 0.10

    # Average page-quality ceiling paired with weak legal-marker coverage for
    # document-level OCR fallback.
    document_ocr_low_legal_marker_average_score_threshold: float = 10.0

    # Enable per-page native-versus-OCR selection after OCR comparison is
    # triggered for a suspicious document.
    enable_hybrid_page_selection: bool = True

    # Minimum score and badness gaps required before OCR can win a page by the
    # standard multi-signal comparison path.
    hybrid_ocr_page_min_score_gap: float = 8.0
    hybrid_ocr_page_min_badness_gap: float = 0.75
    hybrid_ocr_page_min_reason_count: int = 2

    # Strong-signal thresholds used when OCR clearly dominates the native page.
    hybrid_ocr_strong_signal_min_score_gap: float = 18.0
    hybrid_ocr_strong_signal_min_badness_gap: float = 1.5

    # Conservative thresholds used when both page candidates are weak and the
    # pipeline must select the less harmful option.
    hybrid_ocr_less_harmful_min_score_gap: float = 12.0
    hybrid_ocr_less_harmful_min_badness_gap: float = 0.50

    # Minimum blended page score below which local degradation can mark a page
    # as unreliable when supported by strong corruption signals.
    local_unreliable_page_min_quality_score: float = 10.0

    # Severe blended page score below which a page is always treated as
    # locally unreliable regardless of individual reason combinations.
    local_unreliable_page_hard_floor_score: float = 0.0

    # Minimum blended page score that directly triggers native-versus-OCR
    # comparison for a locally suspicious page.
    local_ocr_trigger_page_quality_score: float = 12.0

    # Minimum suspicious-symbol density that justifies explicit OCR
    # comparison for a single page.
    local_ocr_trigger_suspicious_symbol_ratio: float = 0.025

    # Minimum content-quality thresholds used to trigger page-level OCR
    # comparison when lexical or prose integrity is locally weak.
    local_ocr_trigger_min_lexical_completeness: float = 0.55
    local_ocr_trigger_min_line_readability: float = 0.62
    local_ocr_trigger_min_prose_likeness: float = 0.50

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
    target_chunk_chars: int = 900
    hard_max_chunk_chars: int = 1024
    min_chunk_chars: int = 350
    overlap_chars: int = 80

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

    # Include optional structure-enriched meta_text in final chunk payloads.
    include_meta_text: bool = True

    # Link neighboring chunks using prev_chunk_id / next_chunk_id.
    enable_chunk_neighbor_links: bool = True

    # Visible-length cap below which split chunks are treated as potentially
    # undersized for standalone retrieval quality.
    validator_problematic_split_chunk_max_chars: int = 180

    # Minimum word count expected before a split chunk can be treated as
    # semantically self-sufficient by the validator.
    validator_low_autonomy_min_word_count: int = 8

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

    # ---------------------------------------------------------------------
    # Central pipeline settings from config/appsettings.json
    # ---------------------------------------------------------------------
    chunking_strategy: str = "article_smart"
    embedding_enabled: bool = False
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_input_root: Path = PROJECT_ROOT / "data" / "chunks"
    embedding_output_root: Path = PROJECT_ROOT / "data" / "embeddings"
    embedding_input_text_field: str = "text"
    embedding_batch_size: int = 100
    embedding_visualization_enabled: bool = False
    embedding_visualization_spotlight_enabled: bool = False

    def __post_init__(self) -> None:
        """
        Merge central appsettings values into the runtime settings object.

        Explicit constructor values remain authoritative because dataclass
        defaults are only overridden when the current field still matches its
        default value.
        """

        appsettings = _load_appsettings()
        chunking_settings = _get_nested_value(appsettings, ["chunking"], {})
        extraction_settings = _get_nested_value(appsettings, ["extraction"], {})
        embedding_settings = _get_nested_value(appsettings, ["embedding"], {})
        chunking_validation_settings = _get_nested_value(
            chunking_settings,
            ["validation"],
            {},
        )
        visualization_settings = _get_nested_value(
            embedding_settings,
            ["visualization"],
            {},
        )

        self.chunking_strategy = self._resolve_string_setting(
            current_value=self.chunking_strategy,
            default_value="article_smart",
            configured_value=_get_nested_value(
                chunking_settings,
                ["strategy"],
                "article_smart",
            ),
        )
        self.embedding_enabled = self._resolve_bool_setting(
            current_value=self.embedding_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                embedding_settings,
                ["enabled"],
                False,
            ),
        )
        self.embedding_provider = self._resolve_string_setting(
            current_value=self.embedding_provider,
            default_value="openai",
            configured_value=_get_nested_value(
                embedding_settings,
                ["provider"],
                "openai",
            ),
        )
        self.embedding_model = self._resolve_string_setting(
            current_value=self.embedding_model,
            default_value="text-embedding-3-large",
            configured_value=_get_nested_value(
                embedding_settings,
                ["model"],
                "text-embedding-3-large",
            ),
        )
        self.embedding_input_root = self._resolve_path_setting(
            current_value=self.embedding_input_root,
            default_value=PROJECT_ROOT / "data" / "chunks",
            configured_value=_get_nested_value(
                embedding_settings,
                ["input_root"],
                "data/chunks",
            ),
        )
        self.embedding_output_root = self._resolve_path_setting(
            current_value=self.embedding_output_root,
            default_value=PROJECT_ROOT / "data" / "embeddings",
            configured_value=_get_nested_value(
                embedding_settings,
                ["output_root"],
                "data/embeddings",
            ),
        )
        self.embedding_input_text_field = self._resolve_string_setting(
            current_value=self.embedding_input_text_field,
            default_value="text",
            configured_value=_get_nested_value(
                embedding_settings,
                ["input_text_field"],
                "text",
            ),
        )
        self.embedding_batch_size = self._resolve_int_setting(
            current_value=self.embedding_batch_size,
            default_value=100,
            configured_value=_get_nested_value(
                embedding_settings,
                ["batch_size"],
                100,
            ),
        )
        self.embedding_visualization_enabled = self._resolve_bool_setting(
            current_value=self.embedding_visualization_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                visualization_settings,
                ["enabled"],
                False,
            ),
        )
        self.embedding_visualization_spotlight_enabled = self._resolve_bool_setting(
            current_value=self.embedding_visualization_spotlight_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                visualization_settings,
                ["spotlight_enabled"],
                False,
            ),
        )
        self.enable_ocr_fallback = self._resolve_bool_setting(
            current_value=self.enable_ocr_fallback,
            default_value=True,
            configured_value=_get_nested_value(
                extraction_settings,
                ["enable_ocr_fallback"],
                True,
            ),
        )
        self.suspicious_page_ratio_threshold = self._resolve_float_setting(
            current_value=self.suspicious_page_ratio_threshold,
            default_value=0.40,
            configured_value=_get_nested_value(
                extraction_settings,
                ["suspicious_page_ratio_threshold"],
                0.40,
            ),
        )
        self.document_ocr_replacement_ratio_threshold = self._resolve_float_setting(
            current_value=self.document_ocr_replacement_ratio_threshold,
            default_value=0.30,
            configured_value=_get_nested_value(
                extraction_settings,
                ["document_ocr_replacement_ratio_threshold"],
                0.30,
            ),
        )
        self.document_ocr_empty_ratio_threshold = self._resolve_float_setting(
            current_value=self.document_ocr_empty_ratio_threshold,
            default_value=0.30,
            configured_value=_get_nested_value(
                extraction_settings,
                ["document_ocr_empty_ratio_threshold"],
                0.30,
            ),
        )
        self.document_ocr_low_quality_suspicious_ratio_threshold = (
            self._resolve_float_setting(
                current_value=self.document_ocr_low_quality_suspicious_ratio_threshold,
                default_value=0.25,
                configured_value=_get_nested_value(
                    extraction_settings,
                    ["document_ocr_low_quality_suspicious_ratio_threshold"],
                    0.25,
                ),
            )
        )
        self.document_ocr_low_quality_average_score_threshold = (
            self._resolve_float_setting(
                current_value=self.document_ocr_low_quality_average_score_threshold,
                default_value=15.0,
                configured_value=_get_nested_value(
                    extraction_settings,
                    ["document_ocr_low_quality_average_score_threshold"],
                    15.0,
                ),
            )
        )
        self.document_ocr_replacement_mix_suspicious_ratio_threshold = (
            self._resolve_float_setting(
                current_value=self.document_ocr_replacement_mix_suspicious_ratio_threshold,
                default_value=0.20,
                configured_value=_get_nested_value(
                    extraction_settings,
                    ["document_ocr_replacement_mix_suspicious_ratio_threshold"],
                    0.20,
                ),
            )
        )
        self.document_ocr_replacement_mix_replacement_ratio_threshold = (
            self._resolve_float_setting(
                current_value=self.document_ocr_replacement_mix_replacement_ratio_threshold,
                default_value=0.15,
                configured_value=_get_nested_value(
                    extraction_settings,
                    ["document_ocr_replacement_mix_replacement_ratio_threshold"],
                    0.15,
                ),
            )
        )
        self.document_ocr_low_legal_marker_coverage_threshold = (
            self._resolve_float_setting(
                current_value=self.document_ocr_low_legal_marker_coverage_threshold,
                default_value=0.10,
                configured_value=_get_nested_value(
                    extraction_settings,
                    ["document_ocr_low_legal_marker_coverage_threshold"],
                    0.10,
                ),
            )
        )
        self.document_ocr_low_legal_marker_average_score_threshold = (
            self._resolve_float_setting(
                current_value=self.document_ocr_low_legal_marker_average_score_threshold,
                default_value=10.0,
                configured_value=_get_nested_value(
                    extraction_settings,
                    ["document_ocr_low_legal_marker_average_score_threshold"],
                    10.0,
                ),
            )
        )
        self.enable_hybrid_page_selection = self._resolve_bool_setting(
            current_value=self.enable_hybrid_page_selection,
            default_value=True,
            configured_value=_get_nested_value(
                extraction_settings,
                ["enable_hybrid_page_selection"],
                True,
            ),
        )
        self.hybrid_ocr_page_min_score_gap = self._resolve_float_setting(
            current_value=self.hybrid_ocr_page_min_score_gap,
            default_value=8.0,
            configured_value=_get_nested_value(
                extraction_settings,
                ["hybrid_ocr_page_min_score_gap"],
                8.0,
            ),
        )
        self.hybrid_ocr_page_min_badness_gap = self._resolve_float_setting(
            current_value=self.hybrid_ocr_page_min_badness_gap,
            default_value=0.75,
            configured_value=_get_nested_value(
                extraction_settings,
                ["hybrid_ocr_page_min_badness_gap"],
                0.75,
            ),
        )
        self.hybrid_ocr_page_min_reason_count = self._resolve_int_setting(
            current_value=self.hybrid_ocr_page_min_reason_count,
            default_value=2,
            configured_value=_get_nested_value(
                extraction_settings,
                ["hybrid_ocr_page_min_reason_count"],
                2,
            ),
        )
        self.hybrid_ocr_strong_signal_min_score_gap = self._resolve_float_setting(
            current_value=self.hybrid_ocr_strong_signal_min_score_gap,
            default_value=18.0,
            configured_value=_get_nested_value(
                extraction_settings,
                ["hybrid_ocr_strong_signal_min_score_gap"],
                18.0,
            ),
        )
        self.hybrid_ocr_strong_signal_min_badness_gap = self._resolve_float_setting(
            current_value=self.hybrid_ocr_strong_signal_min_badness_gap,
            default_value=1.5,
            configured_value=_get_nested_value(
                extraction_settings,
                ["hybrid_ocr_strong_signal_min_badness_gap"],
                1.5,
            ),
        )
        self.hybrid_ocr_less_harmful_min_score_gap = self._resolve_float_setting(
            current_value=self.hybrid_ocr_less_harmful_min_score_gap,
            default_value=12.0,
            configured_value=_get_nested_value(
                extraction_settings,
                ["hybrid_ocr_less_harmful_min_score_gap"],
                12.0,
            ),
        )
        self.hybrid_ocr_less_harmful_min_badness_gap = self._resolve_float_setting(
            current_value=self.hybrid_ocr_less_harmful_min_badness_gap,
            default_value=0.50,
            configured_value=_get_nested_value(
                extraction_settings,
                ["hybrid_ocr_less_harmful_min_badness_gap"],
                0.50,
            ),
        )
        self.validator_problematic_split_chunk_max_chars = self._resolve_int_setting(
            current_value=self.validator_problematic_split_chunk_max_chars,
            default_value=180,
            configured_value=_get_nested_value(
                chunking_validation_settings,
                ["problematic_split_chunk_max_chars"],
                180,
            ),
        )
        self.validator_low_autonomy_min_word_count = self._resolve_int_setting(
            current_value=self.validator_low_autonomy_min_word_count,
            default_value=8,
            configured_value=_get_nested_value(
                chunking_validation_settings,
                ["low_autonomy_min_word_count"],
                8,
            ),
        )
        self.local_unreliable_page_min_quality_score = self._resolve_float_setting(
            current_value=self.local_unreliable_page_min_quality_score,
            default_value=10.0,
            configured_value=_get_nested_value(
                extraction_settings,
                ["local_unreliable_page_min_quality_score"],
                10.0,
            ),
        )
        self.local_unreliable_page_hard_floor_score = self._resolve_float_setting(
            current_value=self.local_unreliable_page_hard_floor_score,
            default_value=0.0,
            configured_value=_get_nested_value(
                extraction_settings,
                ["local_unreliable_page_hard_floor_score"],
                0.0,
            ),
        )
        self.local_ocr_trigger_page_quality_score = self._resolve_float_setting(
            current_value=self.local_ocr_trigger_page_quality_score,
            default_value=12.0,
            configured_value=_get_nested_value(
                extraction_settings,
                ["local_ocr_trigger_page_quality_score"],
                12.0,
            ),
        )
        self.local_ocr_trigger_suspicious_symbol_ratio = self._resolve_float_setting(
            current_value=self.local_ocr_trigger_suspicious_symbol_ratio,
            default_value=0.025,
            configured_value=_get_nested_value(
                extraction_settings,
                ["local_ocr_trigger_suspicious_symbol_ratio"],
                0.025,
            ),
        )
        self.local_ocr_trigger_min_lexical_completeness = self._resolve_float_setting(
            current_value=self.local_ocr_trigger_min_lexical_completeness,
            default_value=0.55,
            configured_value=_get_nested_value(
                extraction_settings,
                ["local_ocr_trigger_min_lexical_completeness"],
                0.55,
            ),
        )
        self.local_ocr_trigger_min_line_readability = self._resolve_float_setting(
            current_value=self.local_ocr_trigger_min_line_readability,
            default_value=0.62,
            configured_value=_get_nested_value(
                extraction_settings,
                ["local_ocr_trigger_min_line_readability"],
                0.62,
            ),
        )
        self.local_ocr_trigger_min_prose_likeness = self._resolve_float_setting(
            current_value=self.local_ocr_trigger_min_prose_likeness,
            default_value=0.50,
            configured_value=_get_nested_value(
                extraction_settings,
                ["local_ocr_trigger_min_prose_likeness"],
                0.50,
            ),
        )

    def _resolve_string_setting(
        self,
        current_value: str,
        default_value: str,
        configured_value: Any,
    ) -> str:
        """
        Resolve a string setting while preserving explicit constructor values.
        """

        if current_value != default_value:
            return current_value

        if isinstance(configured_value, str) and configured_value.strip():
            return configured_value

        return default_value

    def _resolve_bool_setting(
        self,
        current_value: bool,
        default_value: bool,
        configured_value: Any,
    ) -> bool:
        """
        Resolve a boolean setting while preserving explicit constructor values.
        """

        if current_value != default_value:
            return current_value

        if isinstance(configured_value, bool):
            return configured_value

        return default_value

    def _resolve_int_setting(
        self,
        current_value: int,
        default_value: int,
        configured_value: Any,
    ) -> int:
        """
        Resolve an integer setting while preserving explicit constructor values.
        """

        if current_value != default_value:
            return current_value

        if isinstance(configured_value, int) and not isinstance(configured_value, bool):
            return configured_value

        return default_value

    def _resolve_float_setting(
        self,
        current_value: float,
        default_value: float,
        configured_value: Any,
    ) -> float:
        """
        Resolve a float setting while preserving explicit constructor values.
        """

        if current_value != default_value:
            return current_value

        if isinstance(configured_value, (int, float)) and not isinstance(
            configured_value,
            bool,
        ):
            return float(configured_value)

        return default_value

    def _resolve_path_setting(
        self,
        current_value: Path,
        default_value: Path,
        configured_value: Any,
    ) -> Path:
        """
        Resolve a path setting while preserving explicit constructor values.
        """

        if current_value != default_value:
            return current_value

        return _resolve_project_path(
            value=configured_value,
            default_path=default_value,
        )
