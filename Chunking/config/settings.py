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
    target_chunk_chars: int = 480
    hard_max_chunk_chars: int = 650
    min_chunk_chars: int = 120
    overlap_chars: int = 120

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
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_input_root: Path = PROJECT_ROOT / "data" / "chunks"
    embedding_output_root: Path = PROJECT_ROOT / "data" / "embeddings"
    embedding_input_text_field: str = "text"
    embedding_batch_size: int = 100
    embedding_visualization_enabled: bool = False
    embedding_visualization_spotlight_enabled: bool = False
    embedding_visualization_benchmark_overlay_enabled: bool = False
    embedding_visualization_benchmark_overlay_output_path: Path = (
        PROJECT_ROOT / "data" / "embeddings" / "benchmark_overlay.jsonl"
    )
    embedding_comparison_enabled: bool = False
    embedding_comparison_candidate_models: List[str] = field(
        default_factory=lambda: [
            "all-MiniLM-L6-v2",
            "Qwen/Qwen3-VL-Embedding-8B",
        ]
    )
    embedding_comparison_output_root: Path = (
        PROJECT_ROOT / "data" / "embedding_comparison"
    )
    chromadb_mode: str = "cloud"
    chromadb_persist_directory: Path = PROJECT_ROOT / "data" / "chromadb"
    chromadb_collection_name: str = "rag_embeddings"
    chromadb_cloud_tenant: str = ""
    chromadb_cloud_database: str = ""
    chromadb_cloud_host: str = "api.trychroma.com"
    chromadb_cloud_port: int = 443
    chromadb_cloud_api_key_env_var: str = "CHROMA_API_KEY"

    # ---------------------------------------------------------------------
    # Retrieval runtime configuration
    # ---------------------------------------------------------------------
    retrieval_enabled: bool = True
    retrieval_top_k: int = 6
    retrieval_candidate_pool_size: int = 6
    retrieval_query_normalization_enabled: bool = False
    retrieval_query_normalization_strip_formatting_instructions: bool = True
    retrieval_query_normalization_extract_formatting_directives: bool = True
    retrieval_score_filtering_enabled: bool = False
    retrieval_min_similarity_score: float = 0.0
    retrieval_context_max_chunks: int = 4
    retrieval_context_max_characters: int = 6000
    retrieval_context_include_article_number: bool = False
    retrieval_context_include_article_title: bool = False
    retrieval_context_include_parent_structure: bool = False
    retrieval_context_primary_anchor_max_count: int = 1
    retrieval_context_single_intent_compaction_enabled: bool = True
    retrieval_context_single_intent_max_supporting_chunks: int = 1
    retrieval_routing_enabled: bool = False
    retrieval_routing_article_scoping_enabled: bool = True
    retrieval_routing_document_scoping_enabled: bool = True
    retrieval_routing_comparative_retrieval_enabled: bool = True
    retrieval_routing_weak_evidence_retry_enabled: bool = True
    retrieval_routing_document_inference_enabled: bool = True
    retrieval_routing_document_inference_min_score: float = 3.0
    retrieval_routing_document_inference_min_margin: float = 2.0
    retrieval_routing_scoped_candidate_pool_size: int = 8
    retrieval_routing_broad_candidate_pool_size: int = 6
    retrieval_second_pass_retry_enabled: bool = True
    retrieval_second_pass_retry_candidate_pool_size: int = 8
    retrieval_second_pass_dominant_document_min_share: float = 0.60
    retrieval_evidence_strong_min_score: float = 0.75
    retrieval_evidence_weak_min_score: float = 0.45
    retrieval_evidence_ambiguity_score_margin: float = 0.05
    retrieval_evidence_conflict_score_margin: float = 0.10
    retrieval_evidence_non_blocking_competitor_score_margin: float = 0.08
    retrieval_evidence_blocking_conflict_score_margin: float = 0.15
    retrieval_response_policy_cautious_answer_enabled: bool = True
    retrieval_response_policy_clarification_enabled: bool = True
    retrieval_grounding_derived_claims_enabled: bool = True
    retrieval_grounding_derived_numeric_claim_tolerance: int = 0
    retrieval_grounding_enumeration_coverage_threshold: float = 0.80

    # ---------------------------------------------------------------------
    # Response-generation runtime configuration
    # ---------------------------------------------------------------------
    response_generation_enabled: bool = True
    response_generation_provider: str = "openai"
    response_generation_model: str = "gpt-4o"
    response_generation_grounded_fallback_enabled: bool = True
    response_generation_openai_api_key_env_var: str = "OPENAI_API_KEY"
    response_generation_external_gpt4o_endpoint_url: str = ""
    response_generation_external_gpt4o_auth_env_var: str = "EXTERNAL_GPT4O_API_KEY"
    response_generation_external_gpt4o_question_field_name: str = "question"
    response_generation_external_gpt4o_context_field_name: str = "context"
    response_generation_external_gpt4o_instructions_field_name: str = "instructions"
    response_generation_external_gpt4o_metadata_field_name: str = "metadata"
    response_generation_external_gpt4o_channel_id: str = ""
    response_generation_external_gpt4o_thread_id: str = ""
    response_generation_external_gpt4o_user_info: str = "{}"
    response_generation_external_gpt4o_user_id: str = ""
    response_generation_external_gpt4o_user_name: str = ""
    response_generation_external_gpt4o_auth_header_name: str = "Authorization"
    response_generation_external_gpt4o_auth_header_prefix: str = "Bearer"
    response_generation_external_gpt4o_timeout_seconds: float = 30.0
    response_generation_external_gpt4o_max_retries: int = 2

    # ---------------------------------------------------------------------
    # Deterministic guardrail runtime configuration
    # ---------------------------------------------------------------------
    guardrails_enabled: bool = True
    guardrails_portuguese_coverage_enabled: bool = False
    guardrails_portuguese_jailbreak_pattern_checks_enabled: bool = False
    guardrails_pre_request_offensive_language_checks_enabled: bool = True
    guardrails_pre_request_sexual_content_checks_enabled: bool = True
    guardrails_pre_request_discriminatory_content_checks_enabled: bool = True
    guardrails_pre_request_criminal_or_dangerous_content_checks_enabled: bool = True
    guardrails_pre_request_sensitive_data_checks_enabled: bool = True
    guardrails_pre_request_dangerous_command_checks_enabled: bool = True
    guardrails_pre_request_jailbreak_pattern_checks_enabled: bool = False
    guardrails_post_response_unsafe_output_checks_enabled: bool = True
    guardrails_post_response_grounded_response_checks_enabled: bool = True
    guardrails_post_response_unsupported_answer_checks_enabled: bool = True
    guardrails_post_response_sensitive_data_checks_enabled: bool = False

    # ---------------------------------------------------------------------
    # Retrieval metrics runtime configuration
    # ---------------------------------------------------------------------
    metrics_enabled: bool = True
    metrics_track_deflection_rate: bool = True
    metrics_track_false_positive_rate: bool = True
    metrics_track_jailbreak_resistance: bool = True
    metrics_track_stage_latency: bool = True
    metrics_retrieval_quality_enabled: bool = False
    metrics_track_candidate_pool_size: bool = False
    metrics_track_selected_context_size: bool = False
    metrics_track_context_truncation: bool = False
    metrics_track_structural_richness: bool = False

    # ---------------------------------------------------------------------
    # Benchmark and evaluation configuration
    # ---------------------------------------------------------------------
    benchmark_enabled: bool = False
    benchmark_questions_path: Path = PROJECT_ROOT / "benchmark" / "questions.jsonl"
    benchmark_guardrails_path: Path = PROJECT_ROOT / "benchmark" / "guardrails.jsonl"
    benchmark_output_root: Path = PROJECT_ROOT / "data" / "benchmark_runs"

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
        retrieval_settings = _get_nested_value(appsettings, ["retrieval"], {})
        retrieval_score_filtering_settings = _get_nested_value(
            retrieval_settings,
            ["score_filtering"],
            {},
        )
        retrieval_query_normalization_settings = _get_nested_value(
            retrieval_settings,
            ["query_normalization"],
            {},
        )
        retrieval_context_settings = _get_nested_value(
            retrieval_settings,
            ["context"],
            {},
        )
        retrieval_routing_settings = _get_nested_value(
            retrieval_settings,
            ["routing"],
            {},
        )
        retrieval_evidence_quality_settings = _get_nested_value(
            retrieval_settings,
            ["evidence_quality"],
            {},
        )
        retrieval_second_pass_retry_settings = _get_nested_value(
            retrieval_settings,
            ["second_pass_retry"],
            {},
        )
        retrieval_response_policy_settings = _get_nested_value(
            retrieval_settings,
            ["response_policy"],
            {},
        )
        retrieval_grounding_settings = _get_nested_value(
            retrieval_settings,
            ["grounding"],
            {},
        )
        response_generation_settings = _get_nested_value(
            appsettings,
            ["response_generation"],
            {},
        )
        response_generation_external_gpt4o_settings = _get_nested_value(
            response_generation_settings,
            ["external_gpt4o"],
            {},
        )
        response_generation_openai_settings = _get_nested_value(
            response_generation_settings,
            ["openai"],
            {},
        )
        guardrails_settings = _get_nested_value(appsettings, ["guardrails"], {})
        guardrails_pre_request_settings = _get_nested_value(
            guardrails_settings,
            ["pre_request"],
            {},
        )
        guardrails_portuguese_coverage_settings = _get_nested_value(
            guardrails_settings,
            ["portuguese_coverage"],
            {},
        )
        guardrails_post_response_settings = _get_nested_value(
            guardrails_settings,
            ["post_response"],
            {},
        )
        metrics_settings = _get_nested_value(appsettings, ["metrics"], {})
        metrics_retrieval_quality_settings = _get_nested_value(
            metrics_settings,
            ["retrieval_quality"],
            {},
        )
        chromadb_settings = _get_nested_value(embedding_settings, ["chromadb"], {})
        chromadb_cloud_settings = _get_nested_value(
            chromadb_settings,
            ["cloud"],
            {},
        )
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
        benchmark_overlay_settings = _get_nested_value(
            visualization_settings,
            ["benchmark_overlay"],
            {},
        )
        embedding_comparison_settings = _get_nested_value(
            embedding_settings,
            ["comparison"],
            {},
        )
        benchmark_settings = _get_nested_value(appsettings, ["benchmark"], {})
        benchmark_datasets_settings = _get_nested_value(
            benchmark_settings,
            ["datasets"],
            {},
        )
        benchmark_outputs_settings = _get_nested_value(
            benchmark_settings,
            ["outputs"],
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
            default_value="sentence_transformers",
            configured_value=_get_nested_value(
                embedding_settings,
                ["provider"],
                "sentence_transformers",
            ),
        )
        self.embedding_model = self._resolve_string_setting(
            current_value=self.embedding_model,
            default_value="all-MiniLM-L6-v2",
            configured_value=_get_nested_value(
                embedding_settings,
                ["model"],
                "all-MiniLM-L6-v2",
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
        self.embedding_visualization_benchmark_overlay_enabled = (
            self._resolve_bool_setting(
                current_value=self.embedding_visualization_benchmark_overlay_enabled,
                default_value=False,
                configured_value=_get_nested_value(
                    benchmark_overlay_settings,
                    ["enabled"],
                    False,
                ),
            )
        )
        self.embedding_visualization_benchmark_overlay_output_path = (
            self._resolve_path_setting(
                current_value=self.embedding_visualization_benchmark_overlay_output_path,
                default_value=(
                    PROJECT_ROOT / "data" / "embeddings" / "benchmark_overlay.jsonl"
                ),
                configured_value=_get_nested_value(
                    benchmark_overlay_settings,
                    ["output_path"],
                    "data/embeddings/benchmark_overlay.jsonl",
                ),
            )
        )
        self.embedding_comparison_enabled = self._resolve_bool_setting(
            current_value=self.embedding_comparison_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                embedding_comparison_settings,
                ["enabled"],
                False,
            ),
        )
        self.embedding_comparison_candidate_models = (
            self._resolve_string_list_setting(
                current_value=self.embedding_comparison_candidate_models,
                default_value=[
                    "all-MiniLM-L6-v2",
                    "Qwen/Qwen3-VL-Embedding-8B",
                ],
                configured_value=_get_nested_value(
                    embedding_comparison_settings,
                    ["candidate_models"],
                    [
                        "all-MiniLM-L6-v2",
                        "Qwen/Qwen3-VL-Embedding-8B",
                    ],
                ),
            )
        )
        self.embedding_comparison_output_root = self._resolve_path_setting(
            current_value=self.embedding_comparison_output_root,
            default_value=PROJECT_ROOT / "data" / "embedding_comparison",
            configured_value=_get_nested_value(
                embedding_comparison_settings,
                ["output_root"],
                "data/embedding_comparison",
            ),
        )
        self.chromadb_mode = self._resolve_string_setting(
            current_value=self.chromadb_mode,
            default_value="cloud",
            configured_value=_get_nested_value(
                chromadb_settings,
                ["mode"],
                "cloud",
            ),
        )
        self.chromadb_persist_directory = self._resolve_path_setting(
            current_value=self.chromadb_persist_directory,
            default_value=PROJECT_ROOT / "data" / "chromadb",
            configured_value=_get_nested_value(
                chromadb_settings,
                ["persist_directory"],
                "data/chromadb",
            ),
        )
        self.chromadb_collection_name = self._resolve_string_setting(
            current_value=self.chromadb_collection_name,
            default_value="rag_embeddings",
            configured_value=_get_nested_value(
                chromadb_settings,
                ["collection_name"],
                "rag_embeddings",
            ),
        )
        self.chromadb_cloud_tenant = self._resolve_string_setting(
            current_value=self.chromadb_cloud_tenant,
            default_value="",
            configured_value=_get_nested_value(
                chromadb_cloud_settings,
                ["tenant"],
                "",
            ),
        )
        self.chromadb_cloud_database = self._resolve_string_setting(
            current_value=self.chromadb_cloud_database,
            default_value="",
            configured_value=_get_nested_value(
                chromadb_cloud_settings,
                ["database"],
                "",
            ),
        )
        self.chromadb_cloud_host = self._resolve_string_setting(
            current_value=self.chromadb_cloud_host,
            default_value="api.trychroma.com",
            configured_value=_get_nested_value(
                chromadb_cloud_settings,
                ["host"],
                "api.trychroma.com",
            ),
        )
        self.chromadb_cloud_port = self._resolve_int_setting(
            current_value=self.chromadb_cloud_port,
            default_value=443,
            configured_value=_get_nested_value(
                chromadb_cloud_settings,
                ["port"],
                443,
            ),
        )
        self.chromadb_cloud_api_key_env_var = self._resolve_string_setting(
            current_value=self.chromadb_cloud_api_key_env_var,
            default_value="CHROMA_API_KEY",
            configured_value=_get_nested_value(
                chromadb_cloud_settings,
                ["api_key_env_var"],
                "CHROMA_API_KEY",
            ),
        )
        self.retrieval_enabled = self._resolve_bool_setting(
            current_value=self.retrieval_enabled,
            default_value=True,
            configured_value=_get_nested_value(
                retrieval_settings,
                ["enabled"],
                True,
            ),
        )
        self.retrieval_top_k = self._resolve_int_setting(
            current_value=self.retrieval_top_k,
            default_value=6,
            configured_value=_get_nested_value(
                retrieval_settings,
                ["top_k"],
                6,
            ),
        )
        self.retrieval_candidate_pool_size = self._resolve_int_setting(
            current_value=self.retrieval_candidate_pool_size,
            default_value=6,
            configured_value=_get_nested_value(
                retrieval_settings,
                ["candidate_pool_size"],
                6,
            ),
        )
        self.retrieval_query_normalization_enabled = self._resolve_bool_setting(
            current_value=self.retrieval_query_normalization_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                retrieval_query_normalization_settings,
                ["enabled"],
                False,
            ),
        )
        self.retrieval_query_normalization_strip_formatting_instructions = (
            self._resolve_bool_setting(
                current_value=(
                    self.retrieval_query_normalization_strip_formatting_instructions
                ),
                default_value=True,
                configured_value=_get_nested_value(
                    retrieval_query_normalization_settings,
                    ["strip_formatting_instructions"],
                    True,
                ),
            )
        )
        self.retrieval_query_normalization_extract_formatting_directives = (
            self._resolve_bool_setting(
                current_value=(
                    self.retrieval_query_normalization_extract_formatting_directives
                ),
                default_value=True,
                configured_value=_get_nested_value(
                    retrieval_query_normalization_settings,
                    ["extract_formatting_directives"],
                    True,
                ),
            )
        )
        self.retrieval_score_filtering_enabled = self._resolve_bool_setting(
            current_value=self.retrieval_score_filtering_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                retrieval_score_filtering_settings,
                ["enabled"],
                False,
            ),
        )
        self.retrieval_min_similarity_score = self._resolve_float_setting(
            current_value=self.retrieval_min_similarity_score,
            default_value=0.0,
            configured_value=_get_nested_value(
                retrieval_score_filtering_settings,
                ["min_similarity_score"],
                0.0,
            ),
        )
        self.retrieval_context_max_chunks = self._resolve_int_setting(
            current_value=self.retrieval_context_max_chunks,
            default_value=2,
            configured_value=_get_nested_value(
                retrieval_context_settings,
                ["max_chunks"],
                2,
            ),
        )
        self.retrieval_context_max_characters = self._resolve_int_setting(
            current_value=self.retrieval_context_max_characters,
            default_value=6000,
            configured_value=_get_nested_value(
                retrieval_context_settings,
                ["max_characters"],
                6000,
            ),
        )
        self.retrieval_context_include_article_number = self._resolve_bool_setting(
            current_value=self.retrieval_context_include_article_number,
            default_value=False,
            configured_value=_get_nested_value(
                retrieval_context_settings,
                ["include_article_number"],
                False,
            ),
        )
        self.retrieval_context_include_article_title = self._resolve_bool_setting(
            current_value=self.retrieval_context_include_article_title,
            default_value=False,
            configured_value=_get_nested_value(
                retrieval_context_settings,
                ["include_article_title"],
                False,
            ),
        )
        self.retrieval_context_include_parent_structure = self._resolve_bool_setting(
            current_value=self.retrieval_context_include_parent_structure,
            default_value=False,
            configured_value=_get_nested_value(
                retrieval_context_settings,
                ["include_parent_structure"],
                False,
            ),
        )
        self.retrieval_context_primary_anchor_max_count = self._resolve_int_setting(
            current_value=self.retrieval_context_primary_anchor_max_count,
            default_value=1,
            configured_value=_get_nested_value(
                retrieval_context_settings,
                ["primary_anchor_max_count"],
                1,
            ),
        )
        self.retrieval_context_single_intent_compaction_enabled = (
            self._resolve_bool_setting(
                current_value=self.retrieval_context_single_intent_compaction_enabled,
                default_value=True,
                configured_value=_get_nested_value(
                    retrieval_context_settings,
                    ["single_intent_compaction_enabled"],
                    True,
                ),
            )
        )
        self.retrieval_context_single_intent_max_supporting_chunks = (
            self._resolve_int_setting(
                current_value=self.retrieval_context_single_intent_max_supporting_chunks,
                default_value=1,
                configured_value=_get_nested_value(
                    retrieval_context_settings,
                    ["single_intent_max_supporting_chunks"],
                    1,
                ),
            )
        )
        self.retrieval_routing_enabled = self._resolve_bool_setting(
            current_value=self.retrieval_routing_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                retrieval_routing_settings,
                ["enabled"],
                False,
            ),
        )
        self.retrieval_routing_article_scoping_enabled = self._resolve_bool_setting(
            current_value=self.retrieval_routing_article_scoping_enabled,
            default_value=True,
            configured_value=_get_nested_value(
                retrieval_routing_settings,
                ["article_scoping_enabled"],
                True,
            ),
        )
        self.retrieval_routing_document_scoping_enabled = self._resolve_bool_setting(
            current_value=self.retrieval_routing_document_scoping_enabled,
            default_value=True,
            configured_value=_get_nested_value(
                retrieval_routing_settings,
                ["document_scoping_enabled"],
                True,
            ),
        )
        self.retrieval_routing_comparative_retrieval_enabled = (
            self._resolve_bool_setting(
                current_value=self.retrieval_routing_comparative_retrieval_enabled,
                default_value=True,
                configured_value=_get_nested_value(
                    retrieval_routing_settings,
                    ["comparative_retrieval_enabled"],
                    True,
                ),
            )
        )
        self.retrieval_routing_weak_evidence_retry_enabled = (
            self._resolve_bool_setting(
                current_value=self.retrieval_routing_weak_evidence_retry_enabled,
                default_value=True,
                configured_value=_get_nested_value(
                    retrieval_routing_settings,
                    ["weak_evidence_retry_enabled"],
                    True,
                ),
            )
        )
        self.retrieval_routing_document_inference_enabled = (
            self._resolve_bool_setting(
                current_value=self.retrieval_routing_document_inference_enabled,
                default_value=True,
                configured_value=_get_nested_value(
                    retrieval_routing_settings,
                    ["document_inference_enabled"],
                    True,
                ),
            )
        )
        self.retrieval_routing_document_inference_min_score = (
            self._resolve_float_setting(
                current_value=self.retrieval_routing_document_inference_min_score,
                default_value=3.0,
                configured_value=_get_nested_value(
                    retrieval_routing_settings,
                    ["document_inference_min_score"],
                    3.0,
                ),
            )
        )
        self.retrieval_routing_document_inference_min_margin = (
            self._resolve_float_setting(
                current_value=self.retrieval_routing_document_inference_min_margin,
                default_value=2.0,
                configured_value=_get_nested_value(
                    retrieval_routing_settings,
                    ["document_inference_min_margin"],
                    2.0,
                ),
            )
        )
        self.retrieval_routing_scoped_candidate_pool_size = (
            self._resolve_int_setting(
                current_value=self.retrieval_routing_scoped_candidate_pool_size,
                default_value=8,
                configured_value=_get_nested_value(
                    retrieval_routing_settings,
                    ["scoped_candidate_pool_size"],
                    8,
                ),
            )
        )
        self.retrieval_routing_broad_candidate_pool_size = self._resolve_int_setting(
            current_value=self.retrieval_routing_broad_candidate_pool_size,
            default_value=6,
            configured_value=_get_nested_value(
                retrieval_routing_settings,
                ["broad_candidate_pool_size"],
                6,
            ),
        )
        self.retrieval_second_pass_retry_enabled = self._resolve_bool_setting(
            current_value=self.retrieval_second_pass_retry_enabled,
            default_value=True,
            configured_value=_get_nested_value(
                retrieval_second_pass_retry_settings,
                ["enabled"],
                True,
            ),
        )
        self.retrieval_second_pass_retry_candidate_pool_size = (
            self._resolve_int_setting(
                current_value=self.retrieval_second_pass_retry_candidate_pool_size,
                default_value=8,
                configured_value=_get_nested_value(
                    retrieval_second_pass_retry_settings,
                    ["candidate_pool_size"],
                    8,
                ),
            )
        )
        self.retrieval_second_pass_dominant_document_min_share = (
            self._resolve_float_setting(
                current_value=self.retrieval_second_pass_dominant_document_min_share,
                default_value=0.60,
                configured_value=_get_nested_value(
                    retrieval_second_pass_retry_settings,
                    ["dominant_document_min_share"],
                    0.60,
                ),
            )
        )
        self.retrieval_evidence_strong_min_score = self._resolve_float_setting(
            current_value=self.retrieval_evidence_strong_min_score,
            default_value=0.75,
            configured_value=_get_nested_value(
                retrieval_evidence_quality_settings,
                ["strong_min_score"],
                0.75,
            ),
        )
        self.retrieval_evidence_weak_min_score = self._resolve_float_setting(
            current_value=self.retrieval_evidence_weak_min_score,
            default_value=0.45,
            configured_value=_get_nested_value(
                retrieval_evidence_quality_settings,
                ["weak_min_score"],
                0.45,
            ),
        )
        self.retrieval_evidence_ambiguity_score_margin = (
            self._resolve_float_setting(
                current_value=self.retrieval_evidence_ambiguity_score_margin,
                default_value=0.05,
                configured_value=_get_nested_value(
                    retrieval_evidence_quality_settings,
                    ["ambiguity_score_margin"],
                    0.05,
                ),
            )
        )
        self.retrieval_evidence_conflict_score_margin = self._resolve_float_setting(
            current_value=self.retrieval_evidence_conflict_score_margin,
            default_value=0.10,
            configured_value=_get_nested_value(
                retrieval_evidence_quality_settings,
                ["conflict_score_margin"],
                0.10,
            ),
        )
        self.retrieval_evidence_non_blocking_competitor_score_margin = (
            self._resolve_float_setting(
                current_value=(
                    self.retrieval_evidence_non_blocking_competitor_score_margin
                ),
                default_value=0.08,
                configured_value=_get_nested_value(
                    retrieval_evidence_quality_settings,
                    ["non_blocking_competitor_score_margin"],
                    0.08,
                ),
            )
        )
        self.retrieval_evidence_blocking_conflict_score_margin = (
            self._resolve_float_setting(
                current_value=self.retrieval_evidence_blocking_conflict_score_margin,
                default_value=0.15,
                configured_value=_get_nested_value(
                    retrieval_evidence_quality_settings,
                    ["blocking_conflict_score_margin"],
                    0.15,
                ),
            )
        )
        self.retrieval_response_policy_cautious_answer_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.retrieval_response_policy_cautious_answer_enabled
                ),
                default_value=True,
                configured_value=_get_nested_value(
                    retrieval_response_policy_settings,
                    ["cautious_answer_enabled"],
                    True,
                ),
            )
        )
        self.retrieval_response_policy_clarification_enabled = (
            self._resolve_bool_setting(
                current_value=self.retrieval_response_policy_clarification_enabled,
                default_value=True,
                configured_value=_get_nested_value(
                    retrieval_response_policy_settings,
                    ["clarification_enabled"],
                    True,
                ),
            )
        )
        self.retrieval_grounding_derived_claims_enabled = self._resolve_bool_setting(
            current_value=self.retrieval_grounding_derived_claims_enabled,
            default_value=True,
            configured_value=_get_nested_value(
                retrieval_grounding_settings,
                ["derived_claims_enabled"],
                True,
            ),
        )
        self.retrieval_grounding_derived_numeric_claim_tolerance = (
            self._resolve_int_setting(
                current_value=self.retrieval_grounding_derived_numeric_claim_tolerance,
                default_value=0,
                configured_value=_get_nested_value(
                    retrieval_grounding_settings,
                    ["derived_numeric_claim_tolerance"],
                    0,
                ),
            )
        )
        self.retrieval_grounding_enumeration_coverage_threshold = (
            self._resolve_float_setting(
                current_value=self.retrieval_grounding_enumeration_coverage_threshold,
                default_value=0.80,
                configured_value=_get_nested_value(
                    retrieval_grounding_settings,
                    ["enumeration_coverage_threshold"],
                    0.80,
                ),
            )
        )
        self.response_generation_enabled = self._resolve_bool_setting(
            current_value=self.response_generation_enabled,
            default_value=True,
            configured_value=_get_nested_value(
                response_generation_settings,
                ["enabled"],
                True,
            ),
        )
        self.response_generation_provider = self._resolve_string_setting(
            current_value=self.response_generation_provider,
            default_value="openai",
            configured_value=_get_nested_value(
                response_generation_settings,
                ["provider"],
                "openai",
            ),
        )
        self.response_generation_model = self._resolve_string_setting(
            current_value=self.response_generation_model,
            default_value="gpt-4o",
            configured_value=_get_nested_value(
                response_generation_settings,
                ["model"],
                "gpt-4o",
            ),
        )
        self.response_generation_grounded_fallback_enabled = (
            self._resolve_bool_setting(
                current_value=self.response_generation_grounded_fallback_enabled,
                default_value=True,
                configured_value=_get_nested_value(
                    response_generation_settings,
                    ["grounded_fallback_enabled"],
                    True,
                ),
            )
        )
        self.response_generation_openai_api_key_env_var = (
            self._resolve_string_setting(
                current_value=self.response_generation_openai_api_key_env_var,
                default_value="OPENAI_API_KEY",
                configured_value=_get_nested_value(
                    response_generation_openai_settings,
                    ["api_key_env_var"],
                    "OPENAI_API_KEY",
                ),
            )
        )
        self.response_generation_external_gpt4o_endpoint_url = (
            self._resolve_string_setting(
                current_value=self.response_generation_external_gpt4o_endpoint_url,
                default_value="",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["endpoint_url"],
                    "",
                ),
            )
        )
        self.response_generation_external_gpt4o_auth_env_var = (
            self._resolve_string_setting(
                current_value=self.response_generation_external_gpt4o_auth_env_var,
                default_value="EXTERNAL_GPT4O_API_KEY",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["auth_env_var"],
                    "EXTERNAL_GPT4O_API_KEY",
                ),
            )
        )
        self.response_generation_external_gpt4o_question_field_name = (
            self._resolve_string_setting(
                current_value=(
                    self.response_generation_external_gpt4o_question_field_name
                ),
                default_value="question",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["question_field_name"],
                    "question",
                ),
            )
        )
        self.response_generation_external_gpt4o_context_field_name = (
            self._resolve_string_setting(
                current_value=self.response_generation_external_gpt4o_context_field_name,
                default_value="context",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["context_field_name"],
                    "context",
                ),
            )
        )
        self.response_generation_external_gpt4o_instructions_field_name = (
            self._resolve_string_setting(
                current_value=(
                    self.response_generation_external_gpt4o_instructions_field_name
                ),
                default_value="instructions",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["instructions_field_name"],
                    "instructions",
                ),
            )
        )
        self.response_generation_external_gpt4o_metadata_field_name = (
            self._resolve_string_setting(
                current_value=self.response_generation_external_gpt4o_metadata_field_name,
                default_value="metadata",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["metadata_field_name"],
                    "metadata",
                ),
            )
        )
        self.response_generation_external_gpt4o_channel_id = (
            self._resolve_optional_string_setting(
                current_value=self.response_generation_external_gpt4o_channel_id,
                default_value="",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["channel_id"],
                    "",
                ),
            )
        )
        self.response_generation_external_gpt4o_thread_id = (
            self._resolve_optional_string_setting(
                current_value=self.response_generation_external_gpt4o_thread_id,
                default_value="",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["thread_id"],
                    "",
                ),
            )
        )
        self.response_generation_external_gpt4o_user_info = (
            self._resolve_optional_string_setting(
                current_value=self.response_generation_external_gpt4o_user_info,
                default_value="{}",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["user_info"],
                    "{}",
                ),
            )
        )
        self.response_generation_external_gpt4o_user_id = (
            self._resolve_optional_string_setting(
                current_value=self.response_generation_external_gpt4o_user_id,
                default_value="",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["user_id"],
                    "",
                ),
            )
        )
        self.response_generation_external_gpt4o_user_name = (
            self._resolve_optional_string_setting(
                current_value=self.response_generation_external_gpt4o_user_name,
                default_value="",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["user_name"],
                    "",
                ),
            )
        )
        self.response_generation_external_gpt4o_auth_header_name = (
            self._resolve_string_setting(
                current_value=self.response_generation_external_gpt4o_auth_header_name,
                default_value="Authorization",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["auth_header_name"],
                    "Authorization",
                ),
            )
        )
        self.response_generation_external_gpt4o_auth_header_prefix = (
            self._resolve_optional_string_setting(
                current_value=self.response_generation_external_gpt4o_auth_header_prefix,
                default_value="Bearer",
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["auth_header_prefix"],
                    "Bearer",
                ),
            )
        )
        self.response_generation_external_gpt4o_timeout_seconds = (
            self._resolve_float_setting(
                current_value=self.response_generation_external_gpt4o_timeout_seconds,
                default_value=30.0,
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["timeout_seconds"],
                    30.0,
                ),
            )
        )
        self.response_generation_external_gpt4o_max_retries = (
            self._resolve_int_setting(
                current_value=self.response_generation_external_gpt4o_max_retries,
                default_value=2,
                configured_value=_get_nested_value(
                    response_generation_external_gpt4o_settings,
                    ["max_retries"],
                    2,
                ),
            )
        )
        self.guardrails_enabled = self._resolve_bool_setting(
            current_value=self.guardrails_enabled,
            default_value=True,
            configured_value=_get_nested_value(
                guardrails_settings,
                ["enabled"],
                True,
            ),
        )
        self.guardrails_portuguese_coverage_enabled = self._resolve_bool_setting(
            current_value=self.guardrails_portuguese_coverage_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                guardrails_portuguese_coverage_settings,
                ["enabled"],
                False,
            ),
        )
        self.guardrails_portuguese_jailbreak_pattern_checks_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.guardrails_portuguese_jailbreak_pattern_checks_enabled
                ),
                default_value=False,
                configured_value=_get_nested_value(
                    guardrails_portuguese_coverage_settings,
                    ["jailbreak_pattern_checks_enabled"],
                    False,
                ),
            )
        )
        self.guardrails_pre_request_offensive_language_checks_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.guardrails_pre_request_offensive_language_checks_enabled
                ),
                default_value=True,
                configured_value=_get_nested_value(
                    guardrails_pre_request_settings,
                    ["offensive_language_checks_enabled"],
                    True,
                ),
            )
        )
        self.guardrails_pre_request_sexual_content_checks_enabled = (
            self._resolve_bool_setting(
                current_value=self.guardrails_pre_request_sexual_content_checks_enabled,
                default_value=True,
                configured_value=_get_nested_value(
                    guardrails_pre_request_settings,
                    ["sexual_content_checks_enabled"],
                    True,
                ),
            )
        )
        self.guardrails_pre_request_discriminatory_content_checks_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.guardrails_pre_request_discriminatory_content_checks_enabled
                ),
                default_value=True,
                configured_value=_get_nested_value(
                    guardrails_pre_request_settings,
                    ["discriminatory_content_checks_enabled"],
                    True,
                ),
            )
        )
        self.guardrails_pre_request_criminal_or_dangerous_content_checks_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.guardrails_pre_request_criminal_or_dangerous_content_checks_enabled
                ),
                default_value=True,
                configured_value=_get_nested_value(
                    guardrails_pre_request_settings,
                    ["criminal_or_dangerous_content_checks_enabled"],
                    True,
                ),
            )
        )
        self.guardrails_pre_request_sensitive_data_checks_enabled = (
            self._resolve_bool_setting(
                current_value=self.guardrails_pre_request_sensitive_data_checks_enabled,
                default_value=True,
                configured_value=_get_nested_value(
                    guardrails_pre_request_settings,
                    ["sensitive_data_checks_enabled"],
                    True,
                ),
            )
        )
        self.guardrails_pre_request_dangerous_command_checks_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.guardrails_pre_request_dangerous_command_checks_enabled
                ),
                default_value=True,
                configured_value=_get_nested_value(
                    guardrails_pre_request_settings,
                    ["dangerous_command_checks_enabled"],
                    True,
                ),
            )
        )
        self.guardrails_pre_request_jailbreak_pattern_checks_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.guardrails_pre_request_jailbreak_pattern_checks_enabled
                ),
                default_value=False,
                configured_value=_get_nested_value(
                    guardrails_pre_request_settings,
                    ["jailbreak_pattern_checks_enabled"],
                    False,
                ),
            )
        )
        self.guardrails_post_response_unsafe_output_checks_enabled = (
            self._resolve_bool_setting(
                current_value=self.guardrails_post_response_unsafe_output_checks_enabled,
                default_value=True,
                configured_value=_get_nested_value(
                    guardrails_post_response_settings,
                    ["unsafe_output_checks_enabled"],
                    True,
                ),
            )
        )
        self.guardrails_post_response_grounded_response_checks_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.guardrails_post_response_grounded_response_checks_enabled
                ),
                default_value=True,
                configured_value=_get_nested_value(
                    guardrails_post_response_settings,
                    ["grounded_response_checks_enabled"],
                    True,
                ),
            )
        )
        self.guardrails_post_response_unsupported_answer_checks_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.guardrails_post_response_unsupported_answer_checks_enabled
                ),
                default_value=True,
                configured_value=_get_nested_value(
                    guardrails_post_response_settings,
                    ["unsupported_answer_checks_enabled"],
                    True,
                ),
            )
        )
        self.guardrails_post_response_sensitive_data_checks_enabled = (
            self._resolve_bool_setting(
                current_value=(
                    self.guardrails_post_response_sensitive_data_checks_enabled
                ),
                default_value=False,
                configured_value=_get_nested_value(
                    guardrails_post_response_settings,
                    ["sensitive_data_checks_enabled"],
                    False,
                ),
            )
        )
        self.metrics_enabled = self._resolve_bool_setting(
            current_value=self.metrics_enabled,
            default_value=True,
            configured_value=_get_nested_value(
                metrics_settings,
                ["enabled"],
                True,
            ),
        )
        self.metrics_retrieval_quality_enabled = self._resolve_bool_setting(
            current_value=self.metrics_retrieval_quality_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                metrics_retrieval_quality_settings,
                ["enabled"],
                False,
            ),
        )
        self.metrics_track_candidate_pool_size = self._resolve_bool_setting(
            current_value=self.metrics_track_candidate_pool_size,
            default_value=False,
            configured_value=_get_nested_value(
                metrics_retrieval_quality_settings,
                ["track_candidate_pool_size"],
                False,
            ),
        )
        self.metrics_track_selected_context_size = self._resolve_bool_setting(
            current_value=self.metrics_track_selected_context_size,
            default_value=False,
            configured_value=_get_nested_value(
                metrics_retrieval_quality_settings,
                ["track_selected_context_size"],
                False,
            ),
        )
        self.metrics_track_context_truncation = self._resolve_bool_setting(
            current_value=self.metrics_track_context_truncation,
            default_value=False,
            configured_value=_get_nested_value(
                metrics_retrieval_quality_settings,
                ["track_context_truncation"],
                False,
            ),
        )
        self.metrics_track_structural_richness = self._resolve_bool_setting(
            current_value=self.metrics_track_structural_richness,
            default_value=False,
            configured_value=_get_nested_value(
                metrics_retrieval_quality_settings,
                ["track_structural_richness"],
                False,
            ),
        )
        self.metrics_track_deflection_rate = self._resolve_bool_setting(
            current_value=self.metrics_track_deflection_rate,
            default_value=True,
            configured_value=_get_nested_value(
                metrics_settings,
                ["track_deflection_rate"],
                True,
            ),
        )
        self.metrics_track_false_positive_rate = self._resolve_bool_setting(
            current_value=self.metrics_track_false_positive_rate,
            default_value=True,
            configured_value=_get_nested_value(
                metrics_settings,
                ["track_false_positive_rate"],
                True,
            ),
        )
        self.metrics_track_jailbreak_resistance = self._resolve_bool_setting(
            current_value=self.metrics_track_jailbreak_resistance,
            default_value=True,
            configured_value=_get_nested_value(
                metrics_settings,
                ["track_jailbreak_resistance"],
                True,
            ),
        )
        self.metrics_track_stage_latency = self._resolve_bool_setting(
            current_value=self.metrics_track_stage_latency,
            default_value=True,
            configured_value=_get_nested_value(
                metrics_settings,
                ["track_stage_latency"],
                True,
            ),
        )
        self.benchmark_enabled = self._resolve_bool_setting(
            current_value=self.benchmark_enabled,
            default_value=False,
            configured_value=_get_nested_value(
                benchmark_settings,
                ["enabled"],
                False,
            ),
        )
        self.benchmark_questions_path = self._resolve_path_setting(
            current_value=self.benchmark_questions_path,
            default_value=PROJECT_ROOT / "benchmark" / "questions.jsonl",
            configured_value=_get_nested_value(
                benchmark_datasets_settings,
                ["questions_path"],
                "benchmark/questions.jsonl",
            ),
        )
        self.benchmark_guardrails_path = self._resolve_path_setting(
            current_value=self.benchmark_guardrails_path,
            default_value=PROJECT_ROOT / "benchmark" / "guardrails.jsonl",
            configured_value=_get_nested_value(
                benchmark_datasets_settings,
                ["guardrails_path"],
                "benchmark/guardrails.jsonl",
            ),
        )
        self.benchmark_output_root = self._resolve_path_setting(
            current_value=self.benchmark_output_root,
            default_value=PROJECT_ROOT / "data" / "benchmark_runs",
            configured_value=_get_nested_value(
                benchmark_outputs_settings,
                ["output_root"],
                "data/benchmark_runs",
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

    def _resolve_optional_string_setting(
        self,
        current_value: str,
        default_value: str,
        configured_value: Any,
    ) -> str:
        """
        Resolve a string setting where an explicit empty value is meaningful.
        """

        if current_value != default_value:
            return current_value

        if isinstance(configured_value, str):
            return configured_value.strip()

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

    def _resolve_string_list_setting(
        self,
        current_value: List[str],
        default_value: List[str],
        configured_value: Any,
    ) -> List[str]:
        """
        Resolve a string-list setting while preserving explicit constructor values.
        """

        if current_value != default_value:
            return current_value

        if not isinstance(configured_value, list):
            return list(default_value)

        resolved_values = [
            item.strip()
            for item in configured_value
            if isinstance(item, str) and item.strip()
        ]
        if resolved_values:
            return resolved_values

        return list(default_value)

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
