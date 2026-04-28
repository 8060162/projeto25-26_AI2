from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from Chunking.chunking.models import (
    Chunk,
    DocumentMetadata,
    ExtractedDocument,
    StructuralNode,
)
from Chunking.chunking.strategy_article_smart import ArticleSmartChunkingStrategy
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.chunking.strategy_hybrid import HybridChunkingStrategy
from Chunking.chunking.strategy_structure_first import StructureFirstChunkingStrategy
from Chunking.cleaning.normalizer import NormalizedDocument, TextNormalizer
from Chunking.config.settings import PipelineSettings
from Chunking.export.docx_exporter import DocxInspectionExporter
from Chunking.export.json_exporter import JsonExporter
from Chunking.extraction.extraction_quality import ExtractionQualityAnalyzer
from Chunking.extraction.ocr_fallback import OcrFallbackReader
from Chunking.extraction.pdf_reader import PdfReader
from Chunking.parsing.structure_parser import StructureParser
from Chunking.quality.chunk_quality_validator import ChunkQualityValidator
from Chunking.utils.text import build_canonical_document_id


def run_pipeline() -> None:
    """
    Main pipeline entry point used by main.py.

    High-level pipeline flow
    ------------------------
    1. Read all supported PDF files from the input folder
    2. Extract structured PDF content using native extraction
    3. Analyze extraction quality
    4. Assemble a page-level hybrid document when OCR comparison is justified
    5. Normalize extracted content conservatively
    6. Parse the normalized document into a structural tree
    7. Export:
       - normalized text/debug artifacts
       - generic structural tree
       - canonical master-dictionary-style JSON
    8. Optionally run chunking strategies and export chunk inspection artifacts

    Architectural note
    ------------------
    The project is currently transitioning from a "text-first" pipeline to a
    "structure-first" pipeline.

    Therefore the most important output at this stage is no longer only:
        normalized text -> chunks

    but rather:
        PDF -> structured extraction -> normalized document -> parsed tree
        -> canonical JSON representation

    Chunking remains available, but it should be treated as a downstream stage
    built on top of the structural representation.

    Design goals
    ------------
    - remain explicit and easy to run locally
    - support both native extraction and OCR fallback
    - preserve useful debugging artifacts
    - produce a canonical JSON structure aligned with the master dictionary goal
    - remain backward-compatible enough while the codebase is being migrated
    """
    default_settings = PipelineSettings()

    parser = argparse.ArgumentParser(
        description=(
            "Extract regulatory PDFs into a canonical structured JSON tree "
            "and optionally run chunking strategies."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(default_settings.raw_dir),
        help="Input folder with PDF documents.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_settings.output_dir),
        help="Output folder for all pipeline artifacts.",
    )
    args = parser.parse_args()

    settings = PipelineSettings(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
    )
    strategy_names = _resolve_strategy_names(settings)

    # ------------------------------------------------------------------
    # Pipeline components
    # ------------------------------------------------------------------
    native_reader = PdfReader()
    ocr_reader = OcrFallbackReader()
    extraction_quality_analyzer = ExtractionQualityAnalyzer(settings)
    normalizer = TextNormalizer()
    parser_engine = StructureParser()
    json_exporter = JsonExporter()
    docx_exporter = DocxInspectionExporter()

    print(f"[INFO] Raw directory: {settings.raw_dir}")
    print(f"[INFO] Output directory: {settings.output_dir}")

    if not settings.raw_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: '{settings.raw_dir}'. "
            f"Please place your source PDF files there."
        )

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        [
            path
            for path in settings.raw_dir.iterdir()
            if path.is_file() and path.suffix.lower() in settings.supported_extensions
        ]
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in '{settings.raw_dir}'. Please place your source files there."
        )

    _validate_unique_document_ids(pdf_files)

    for pdf_path in pdf_files:
        print(f"[INFO] Processing: {pdf_path.name}")

        document_metadata = _build_document_metadata(pdf_path)

        # --------------------------------------------------------------
        # 1) Native structured extraction
        #
        # We now prefer the richer `extract_document()` API instead of
        # extracting only `PageText`. This preserves page-level structure,
        # extraction mode, quality, and future parser-friendly information.
        # --------------------------------------------------------------
        native_extracted_document = native_reader.extract_document(pdf_path)

        # --------------------------------------------------------------
        # 2) Analyze native extraction quality
        #
        # Why this matters:
        # if native extraction is badly corrupted, downstream normalization
        # and parsing may still run but produce structurally meaningless output.
        # Detecting this early allows the pipeline to switch to OCR.
        # --------------------------------------------------------------
        native_extraction_quality = extraction_quality_analyzer.analyze_document(
            native_extracted_document
        )

        active_extracted_document = native_extracted_document
        active_extraction_quality = native_extraction_quality
        extraction_mode_used = "native"
        ocr_comparison_trigger = _build_ocr_comparison_trigger(
            native_extraction_quality=native_extraction_quality,
            enable_hybrid_page_selection=settings.enable_hybrid_page_selection,
        )
        extraction_decision = _build_native_only_extraction_decision(
            native_extracted_document=native_extracted_document,
            native_extraction_quality=native_extraction_quality,
            ocr_comparison_trigger=ocr_comparison_trigger,
        )

        # --------------------------------------------------------------
        # 3) OCR-assisted hybrid page assembly when native extraction looks
        #    corrupted
        #
        # OCR is slower and structurally poorer than native PDF extraction,
        # so we only compare against it when the analyzer considers native
        # text suspicious enough to justify fallback. Once OCR is available,
        # the active document is assembled page by page instead of switching
        # the entire file to OCR blindly.
        # --------------------------------------------------------------
        if settings.enable_ocr_fallback and bool(
            ocr_comparison_trigger.get("should_run_ocr_comparison", False)
        ):
            trigger_pages = list(
                ocr_comparison_trigger.get("pages_requiring_ocr_comparison", [])
            )
            trigger_reason = (
                "document corruption"
                if ocr_comparison_trigger.get("document_likely_corrupted", False)
                else "local page comparison"
            )
            print(
                "[WARN] Triggering OCR comparison for "
                f"{pdf_path.name} due to {trigger_reason}. "
                f"Candidate pages: {trigger_pages or 'all'}"
            )

            ocr_extracted_document = ocr_reader.extract_document(pdf_path)
            ocr_extraction_quality = extraction_quality_analyzer.analyze_document(
                ocr_extracted_document
            )

            if settings.enable_hybrid_page_selection:
                (
                    active_extracted_document,
                    extraction_mode_used,
                    extraction_decision,
                ) = _build_hybrid_extraction_result(
                    source_path=str(pdf_path),
                    native_extracted_document=native_extracted_document,
                    native_extraction_quality=native_extraction_quality,
                    ocr_extracted_document=ocr_extracted_document,
                    ocr_extraction_quality=ocr_extraction_quality,
                    extraction_quality_analyzer=extraction_quality_analyzer,
                    ocr_comparison_trigger=ocr_comparison_trigger,
                )
            else:
                active_extracted_document = ocr_extracted_document
                extraction_mode_used = "ocr"
                extraction_decision = _build_ocr_only_extraction_decision(
                    ocr_extracted_document=ocr_extracted_document,
                    native_extraction_quality=native_extraction_quality,
                    ocr_extraction_quality=ocr_extraction_quality,
                    ocr_comparison_trigger=ocr_comparison_trigger,
                )
            active_extraction_quality = extraction_quality_analyzer.analyze_document(
                active_extracted_document
            )

        print(f"[INFO] Extraction mode used: {extraction_mode_used}")

        # --------------------------------------------------------------
        # 4) Normalize extracted content conservatively
        #
        # The normalizer now accepts the richer extracted document directly.
        # It cleans obvious layout noise while preserving parser-useful
        # structural line boundaries.
        # --------------------------------------------------------------
        normalized = normalizer.normalize(active_extracted_document)

        # --------------------------------------------------------------
        # 5) Parse normalized content into a structural tree
        #
        # The parser now accepts the normalized document directly instead of
        # requiring a manually flattened page tuple list.
        # --------------------------------------------------------------
        structure_root = parser_engine.parse(normalized)

        # --------------------------------------------------------------
        # 6) Create base output folder for this document
        #
        # We now separate:
        # - structure-level outputs
        # - chunking strategy outputs
        # so the canonical structure export exists even when chunking is not run.
        # --------------------------------------------------------------
        document_output_dir = settings.output_dir / document_metadata.doc_id
        document_output_dir.mkdir(parents=True, exist_ok=True)

        structure_output_dir = document_output_dir / "structure"
        structure_output_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------
        # 7) Export structure-stage artifacts
        # --------------------------------------------------------------
        if settings.export_intermediate_text:
            _write_structure_stage_outputs(
                structure_output_dir=structure_output_dir,
                normalized=normalized,
                extraction_quality=active_extraction_quality,
                extraction_mode_used=extraction_mode_used,
                extraction_decision=extraction_decision,
            )

        if settings.export_json:
            # Generic tree export for debugging/internal inspection.
            json_exporter.write_structure(
                structure_root,
                structure_output_dir / "03_structure_debug.json",
            )

            # Canonical master-dictionary-style export.
            json_exporter.write_master_dictionary(
                document_metadata=document_metadata,
                root=structure_root,
                output_path=structure_output_dir / "04_master_dictionary.json",
            )

            if settings.export_quality_summary:
                (structure_output_dir / "04b_structure_quality_summary.json").write_text(
                    _dict_to_pretty_json(
                        _build_structure_quality_summary(
                            document_metadata=document_metadata,
                            extraction_quality=active_extraction_quality,
                            normalized=normalized,
                            structure_root=structure_root,
                            extraction_mode_used=extraction_mode_used,
                            extraction_decision=extraction_decision,
                        )
                    ),
                    encoding="utf-8",
                )

        # --------------------------------------------------------------
        # 8) Optional chunking stage
        #
        # Chunking is no longer mandatory for the pipeline to be useful.
        # The canonical JSON tree is already a valid end product for the
        # current phase of the project.
        # --------------------------------------------------------------
        if strategy_names:
            for strategy_name in strategy_names:
                strategy = _build_strategy(strategy_name, settings)

                print(f"[INFO] Running strategy: {strategy.name}")

                chunks = strategy.build_chunks(document_metadata, structure_root)

                strategy_output_dir = document_output_dir / strategy.name
                strategy_output_dir.mkdir(parents=True, exist_ok=True)

                if settings.export_json:
                    json_exporter.write_chunks(
                        chunks,
                        strategy_output_dir / "05_chunks.json",
                    )

                    if settings.export_quality_summary:
                        (strategy_output_dir / "05b_chunk_quality_summary.json").write_text(
                            _dict_to_pretty_json(
                                _build_chunk_quality_summary(
                                    document_metadata=document_metadata,
                                    extraction_quality=active_extraction_quality,
                                    normalized=normalized,
                                    structure_root=structure_root,
                                    chunks=chunks,
                                    strategy_name=strategy.name,
                                    extraction_mode_used=extraction_mode_used,
                                    extraction_decision=extraction_decision,
                                    settings=settings,
                                )
                            ),
                            encoding="utf-8",
                        )

                if settings.export_docx:
                    docx_exporter.write_chunks_docx(
                        document_metadata=document_metadata,
                        strategy_name=strategy.name,
                        chunks=chunks,
                        output_path=strategy_output_dir / "06_chunk_inspection.docx",
                    )

        print(f"[INFO] Done: {pdf_path.name} -> {document_output_dir}")

    print("[INFO] Pipeline execution completed successfully.")


def _resolve_strategy_names(settings: PipelineSettings) -> List[str]:
    """
    Resolve the configured chunking strategy into one or more strategy names.

    Why this helper exists
    ----------------------
    The pipeline now reads strategy selection from settings so chunking
    behavior stays aligned with the central application configuration.

    The configured value may still express:
    - a single strategy
    - all strategies
    - or no chunking at all

    This is useful because the current project phase is primarily focused on
    extracting PDFs into a canonical JSON tree. Chunking should therefore be
    optional rather than mandatory.

    Parameters
    ----------
    settings : PipelineSettings
        Shared runtime configuration.

    Returns
    -------
    List[str]
        Strategy names to execute in order.
    """
    allowed_strategy_names = {
        "article_smart",
        "structure_first",
        "hybrid",
        "all",
        "none",
    }
    strategy_argument = settings.chunking_strategy.strip().lower()

    if strategy_argument not in allowed_strategy_names:
        raise ValueError(
            "Invalid chunking strategy configured in appsettings.json: "
            f"'{settings.chunking_strategy}'."
        )

    if strategy_argument == "all":
        return ["article_smart", "structure_first", "hybrid"]

    if strategy_argument == "none":
        return []

    return [strategy_argument]


def _build_document_metadata(pdf_path: Path) -> DocumentMetadata:
    """
    Build document metadata for one input PDF.

    Why this helper exists
    ----------------------
    It keeps document metadata construction centralized and easy to extend.

    Parameters
    ----------
    pdf_path : Path
        Source PDF path.

    Returns
    -------
    DocumentMetadata
        Base metadata object for the document.
    """
    return DocumentMetadata(
        doc_id=build_canonical_document_id(pdf_path.stem),
        file_name=pdf_path.name,
        title=pdf_path.stem,
        source_path=str(pdf_path),
        metadata={
            "source_extension": pdf_path.suffix.lower(),
        },
    )


def _validate_unique_document_ids(pdf_files: List[Path]) -> None:
    """
    Validate that all input PDFs resolve to distinct document identifiers.

    Parameters
    ----------
    pdf_files : List[Path]
        Source PDF paths selected for the current pipeline run.

    Raises
    ------
    ValueError
        Raised when two different files would write to the same document output
        folder and produce overlapping chunk identifiers.
    """

    document_id_sources: Dict[str, List[str]] = {}

    for pdf_path in pdf_files:
        document_id = build_canonical_document_id(pdf_path.stem)
        document_id_sources.setdefault(document_id, []).append(pdf_path.name)

    duplicated_document_ids = {
        document_id: file_names
        for document_id, file_names in document_id_sources.items()
        if len(file_names) > 1
    }

    if not duplicated_document_ids:
        return

    duplicate_descriptions = [
        f"{document_id}: {', '.join(file_names)}"
        for document_id, file_names in sorted(duplicated_document_ids.items())
    ]
    raise ValueError(
        "Multiple input PDFs resolve to the same document id. Rename or remove "
        "duplicates before running the pipeline: "
        + "; ".join(duplicate_descriptions)
    )


def _write_structure_stage_outputs(
    structure_output_dir: Path,
    normalized: NormalizedDocument,
    extraction_quality: Dict[str, object],
    extraction_mode_used: str,
    extraction_decision: Dict[str, object],
) -> None:
    """
    Write structure-stage intermediate artifacts.

    Exported files
    --------------
    - 01_normalized_text.txt
    - 02_dropped_lines_report.json
    - 02b_extraction_quality.json

    Why these files matter
    ----------------------
    They are extremely useful for understanding where quality is being lost:
    - extraction issues
    - OCR fallback behavior
    - normalization side effects
    - front matter contamination
    - repeated-line removal behavior

    Parameters
    ----------
    structure_output_dir : Path
        Structure-stage output folder.

    normalized : NormalizedDocument
        Normalized document result.

    extraction_quality : Dict[str, object]
        Extraction quality report for the active extraction mode.

    extraction_mode_used : str
        "native", "ocr", or "hybrid".

    extraction_decision : Dict[str, object]
        Hybrid-selection diagnostics for the active extraction result.
    """
    (structure_output_dir / "01_normalized_text.txt").write_text(
        normalized.full_text,
        encoding="utf-8",
    )

    (structure_output_dir / "02_dropped_lines_report.json").write_text(
        _dict_to_pretty_json(normalized.dropped_lines_report),
        encoding="utf-8",
    )

    (structure_output_dir / "02b_extraction_quality.json").write_text(
        _dict_to_pretty_json(
            {
                "extraction_mode_used": extraction_mode_used,
                "extraction_summary": _build_extraction_summary(
                    extraction_mode_used=extraction_mode_used,
                    extraction_decision=extraction_decision,
                ),
                "extraction_decision": extraction_decision,
                "quality": extraction_quality,
            }
        ),
        encoding="utf-8",
    )


def _build_strategy(name: str, settings: PipelineSettings) -> BaseChunkingStrategy:
    """
    Return the requested chunking strategy implementation.

    Parameters
    ----------
    name : str
        Strategy name.

    settings : PipelineSettings
        Shared runtime configuration.

    Returns
    -------
    BaseChunkingStrategy
        Instantiated strategy object.
    """
    strategies: Dict[str, BaseChunkingStrategy] = {
        "article_smart": ArticleSmartChunkingStrategy(settings),
        "structure_first": StructureFirstChunkingStrategy(settings),
        "hybrid": HybridChunkingStrategy(settings),
    }
    return strategies[name]


def _build_structure_quality_summary(
    document_metadata: DocumentMetadata,
    extraction_quality: Dict[str, object],
    normalized: NormalizedDocument,
    structure_root: StructuralNode,
    extraction_mode_used: str,
    extraction_decision: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Build a lightweight structure-stage quality summary.

    Why this helper exists
    ----------------------
    The project is currently more interested in:
    - extraction quality
    - normalization quality
    - parsing quality
    than in chunking quality alone.

    Therefore structure-stage diagnostics deserve their own summary artifact.

    Parameters
    ----------
    document_metadata : DocumentMetadata
        Source document metadata.

    extraction_quality : Dict[str, object]
        Extraction quality report.

    normalized : NormalizedDocument
        Normalized document output.

    structure_root : StructuralNode
        Parsed structure tree.

    extraction_mode_used : str
        Active extraction mode ("native", "ocr", or "hybrid").

    extraction_decision : Dict[str, object]
        Hybrid-selection diagnostics for the active extraction result.

    Returns
    -------
    Dict[str, object]
        Structure-stage summary dictionary suitable for JSON export.
    """
    effective_extraction_decision = extraction_decision or _build_fallback_extraction_decision(
        extraction_mode_used=extraction_mode_used,
        page_count=len(normalized.pages),
    )
    structural_integrity_summary = _build_structural_integrity_summary(structure_root)
    diagnostic_summary = _build_structure_stage_diagnostic_summary(
        extraction_mode_used=extraction_mode_used,
        extraction_decision=effective_extraction_decision,
        structural_integrity_summary=structural_integrity_summary,
    )

    return {
        "document_id": document_metadata.doc_id,
        "document_title": document_metadata.title,
        "extraction_mode_used": extraction_mode_used,
        "extraction_summary": _build_extraction_summary(
            extraction_mode_used=extraction_mode_used,
            extraction_decision=effective_extraction_decision,
        ),
        "extraction_decision": effective_extraction_decision,
        "extraction_quality": extraction_quality,
        "normalized_page_count": len(normalized.pages),
        "non_empty_normalized_pages": sum(
            1 for page in normalized.pages if page.text.strip()
        ),
        "dropped_lines_report": normalized.dropped_lines_report,
        "normalized_page_reports": normalized.page_reports,
        "structural_integrity_summary": structural_integrity_summary,
        "has_structural_integrity_warnings": structural_integrity_summary[
            "has_structural_integrity_warnings"
        ],
        "diagnostic_summary": diagnostic_summary,
        "structure_counts": {
            "front_matter": _count_node_type(structure_root, "FRONT_MATTER"),
            "preamble": _count_node_type(structure_root, "PREAMBLE"),
            "annex": _count_node_type(structure_root, "ANNEX"),
            "chapter": _count_node_type(structure_root, "CHAPTER"),
            "section_container": _count_node_type(structure_root, "SECTION_CONTAINER"),
            "article": _count_node_type(structure_root, "ARTICLE"),
            "section": _count_node_type(structure_root, "SECTION"),
            "lettered_item": _count_node_type(structure_root, "LETTERED_ITEM"),
        },
    }


def _build_chunk_quality_summary(
    document_metadata: DocumentMetadata,
    extraction_quality: Dict[str, object],
    normalized: NormalizedDocument,
    structure_root: StructuralNode,
    chunks: List[Chunk],
    strategy_name: str,
    extraction_mode_used: str,
    settings: PipelineSettings,
    extraction_decision: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Build a lightweight chunk-stage quality summary for one document run.

    Why this helper exists
    ----------------------
    Manual inspection remains important, but a compact machine-readable summary
    helps quickly compare outputs across documents and strategies.

    Parameters
    ----------
    document_metadata : DocumentMetadata
        Source document metadata.

    extraction_quality : Dict[str, object]
        Extraction quality report.

    normalized : NormalizedDocument
        Normalized document output.

    structure_root : StructuralNode
        Parsed structure tree.

    chunks : List[Chunk]
        Final chunk list.

    strategy_name : str
        Strategy used for this run.

    extraction_mode_used : str
        "native", "ocr", or "hybrid".

    extraction_decision : Dict[str, object]
        Hybrid-selection diagnostics for the active extraction result.

    settings : PipelineSettings
        Shared runtime configuration used by the chunk validator.

    Returns
    -------
    Dict[str, object]
        Chunk-stage summary dictionary suitable for JSON export.
    """
    effective_extraction_decision = extraction_decision or _build_fallback_extraction_decision(
        extraction_mode_used=extraction_mode_used,
        page_count=len(normalized.pages),
    )
    chunk_count = len(chunks)
    char_counts = [getattr(chunk, "char_count", len(chunk.text)) for chunk in chunks]
    chunk_reasons = CounterLike()
    validator = ChunkQualityValidator(settings)
    validation_report = validator.validate_chunks(chunks)
    blocking_overview = _build_blocking_failure_overview(validation_report)
    next_phase_decision = _build_next_phase_decision(
        strategy_name=strategy_name,
        validation_report=validation_report,
    )
    structural_integrity_summary = _build_structural_integrity_summary(structure_root)
    validator_summary = _build_validator_summary(validation_report)
    diagnostic_summary = _build_chunk_stage_diagnostic_summary(
        extraction_mode_used=extraction_mode_used,
        extraction_decision=effective_extraction_decision,
        structural_integrity_summary=structural_integrity_summary,
        validator_summary=validator_summary,
        next_phase_decision=next_phase_decision,
    )

    for chunk in chunks:
        chunk_reasons.increment(getattr(chunk, "chunk_reason", "") or "unspecified")

    return {
        "document_id": document_metadata.doc_id,
        "document_title": document_metadata.title,
        "strategy": strategy_name,
        "extraction_mode_used": extraction_mode_used,
        "extraction_summary": _build_extraction_summary(
            extraction_mode_used=extraction_mode_used,
            extraction_decision=effective_extraction_decision,
        ),
        "extraction_decision": effective_extraction_decision,
        "extraction_quality": extraction_quality,
        "normalized_page_count": len(normalized.pages),
        "non_empty_normalized_pages": sum(
            1 for page in normalized.pages if page.text.strip()
        ),
        "dropped_lines_report": normalized.dropped_lines_report,
        "structural_integrity_summary": structural_integrity_summary,
        "has_structural_integrity_warnings": structural_integrity_summary[
            "has_structural_integrity_warnings"
        ],
        "structure_counts": {
            "front_matter": _count_node_type(structure_root, "FRONT_MATTER"),
            "preamble": _count_node_type(structure_root, "PREAMBLE"),
            "annex": _count_node_type(structure_root, "ANNEX"),
            "chapter": _count_node_type(structure_root, "CHAPTER"),
            "section_container": _count_node_type(structure_root, "SECTION_CONTAINER"),
            "article": _count_node_type(structure_root, "ARTICLE"),
            "section": _count_node_type(structure_root, "SECTION"),
            "lettered_item": _count_node_type(structure_root, "LETTERED_ITEM"),
        },
        "chunk_count": chunk_count,
        "chunk_reason_counts": chunk_reasons.to_dict(),
        "chunk_size_stats": {
            "min_chars": min(char_counts) if char_counts else 0,
            "max_chars": max(char_counts) if char_counts else 0,
            "avg_chars": (sum(char_counts) / len(char_counts)) if char_counts else 0,
        },
        "overall_status": validation_report["overall_status"],
        "has_blocking_failures": blocking_overview["has_blocking_failures"],
        "valid_chunk_count": blocking_overview["valid_chunk_count"],
        "invalid_chunk_count": blocking_overview["invalid_chunk_count"],
        "acceptable_for_next_phase": next_phase_decision["is_acceptable"],
        "blocking_failure_count": blocking_overview["blocking_failure_count"],
        "blocking_failure_types": blocking_overview["blocking_failure_types"],
        "blocking_failures": blocking_overview["blocking_failures"],
        "failed_chunk_ids": blocking_overview["failed_chunk_ids"],
        "acceptance_basis": "validator_blocking_failures",
        "acceptance_reason": next_phase_decision["reason"],
        "final_decision": next_phase_decision["decision"],
        "next_phase_decision": next_phase_decision,
        "validator_summary": validator_summary,
        "diagnostic_summary": diagnostic_summary,
        "chunk_neighbor_links_complete": all(
            (index == 0 or getattr(chunks[index], "prev_chunk_id", None) is not None)
            and (
                index == len(chunks) - 1
                or getattr(chunks[index], "next_chunk_id", None) is not None
            )
            for index in range(len(chunks))
        ),
    }


def _build_validator_summary(validation_report: Dict[str, object]) -> Dict[str, object]:
    """
    Build a compact validator summary for chunk-quality reporting.

    Parameters
    ----------
    validation_report : Dict[str, object]
        Aggregate validator output for one chunk sequence.

    Returns
    -------
    Dict[str, object]
        Reduced validator payload focused on automation-friendly fields.
    """
    issue_type_counts = dict(validation_report.get("issue_type_counts", {}))
    issue_examples = dict(validation_report.get("issue_examples", {}))
    blocking_overview = _build_blocking_failure_overview(validation_report)

    return {
        "chunk_count": blocking_overview["chunk_count"],
        "valid_chunk_count": blocking_overview["valid_chunk_count"],
        "invalid_chunk_count": blocking_overview["invalid_chunk_count"],
        "overall_status": blocking_overview["overall_status"],
        "has_blocking_failures": blocking_overview["has_blocking_failures"],
        "blocking_failure_count": blocking_overview["blocking_failure_count"],
        "blocking_failure_types": blocking_overview["blocking_failure_types"],
        "blocking_failures": blocking_overview["blocking_failures"],
        "failure_type_counts": issue_type_counts,
        "failure_examples": issue_examples,
        "failure_messages": _build_failure_message_map(validation_report),
        "failed_chunk_ids": blocking_overview["failed_chunk_ids"],
    }


def _build_failure_message_map(
    validation_report: Dict[str, object],
) -> Dict[str, List[str]]:
    """
    Collect stable validator messages for each blocking failure code.

    Parameters
    ----------
    validation_report : Dict[str, object]
        Aggregate validator output for one chunk sequence.

    Returns
    -------
    Dict[str, List[str]]
        Failure messages grouped by issue code and limited for compact export.
    """
    failure_messages: Dict[str, List[str]] = {}

    for chunk_report in validation_report.get("chunk_reports", []):
        for issue in chunk_report.get("issues", []):
            issue_code = str(issue.get("code", "")).strip()
            issue_message = str(issue.get("message", "")).strip()
            if not issue_code or not issue_message:
                continue

            messages = failure_messages.setdefault(issue_code, [])
            if issue_message not in messages and len(messages) < 3:
                messages.append(issue_message)

    return failure_messages


def _build_fallback_extraction_decision(
    extraction_mode_used: str,
    page_count: int,
) -> Dict[str, object]:
    """
    Build a minimal extraction decision payload for backward-compatible callers.

    Parameters
    ----------
    extraction_mode_used : str
        Active document-level extraction mode.

    page_count : int
        Number of normalized pages available to downstream stages.

    Returns
    -------
    Dict[str, object]
        Stable extraction decision payload that matches the exported schema.
    """
    is_ocr = extraction_mode_used == "ocr"
    return {
        "comparison_performed": False,
        "document_composition": _classify_document_composition(extraction_mode_used),
        "page_count": page_count,
        "compared_page_count": 0,
        "native_page_count": 0 if is_ocr else page_count,
        "ocr_page_count": page_count if is_ocr else 0,
        "page_decisions": [],
    }


def _build_extraction_summary(
    extraction_mode_used: str,
    extraction_decision: Dict[str, object],
) -> Dict[str, object]:
    """
    Build a compact extraction summary suitable for exported diagnostics.

    Parameters
    ----------
    extraction_mode_used : str
        Active document-level extraction mode.

    extraction_decision : Dict[str, object]
        Detailed extraction decision payload.

    Returns
    -------
    Dict[str, object]
        Compact extraction summary with document composition and page choices.
    """
    page_count = int(extraction_decision.get("page_count", 0))
    page_decisions = list(extraction_decision.get("page_decisions", []))
    page_source_selection = _build_page_source_selection(
        extraction_mode_used=extraction_mode_used,
        extraction_decision=extraction_decision,
    )

    return {
        "document_mode": extraction_mode_used,
        "document_composition": extraction_decision.get(
            "document_composition",
            _classify_document_composition(extraction_mode_used),
        ),
        "comparison_performed": bool(
            extraction_decision.get("comparison_performed", False)
        ),
        "page_count": page_count,
        "compared_page_count": int(extraction_decision.get("compared_page_count", 0)),
        "native_page_count": int(extraction_decision.get("native_page_count", 0)),
        "ocr_page_count": int(extraction_decision.get("ocr_page_count", 0)),
        "page_modes_selected": page_source_selection,
        "selected_page_modes_present": bool(page_source_selection)
        and len(page_source_selection) == page_count,
        "hybrid_pages_selected": any(
            page_selection.get("selected_source") == "ocr"
            for page_selection in page_source_selection
        )
        and any(
            page_selection.get("selected_source") == "native"
            for page_selection in page_source_selection
        ),
        "page_selection_reason_codes": {
            str(page_decision.get("page_number")): list(page_decision.get("reason_codes", []))
            for page_decision in page_decisions
        },
        "ocr_comparison_trigger": _build_ocr_comparison_trigger_summary(
            extraction_decision
        ),
    }


def _build_ocr_comparison_trigger_summary(
    extraction_decision: Dict[str, object],
) -> Dict[str, object]:
    """
    Reduce OCR trigger diagnostics to the fields needed for exported summaries.

    Parameters
    ----------
    extraction_decision : Dict[str, object]
        Detailed extraction decision payload.

    Returns
    -------
    Dict[str, object]
        Stable OCR trigger summary for compact diagnostic exports.
    """
    trigger = dict(extraction_decision.get("ocr_comparison_trigger", {}))

    return {
        "document_likely_corrupted": bool(
            trigger.get("document_likely_corrupted", False)
        ),
        "has_local_pages_requiring_ocr_comparison": bool(
            trigger.get("has_local_pages_requiring_ocr_comparison", False)
        ),
        "pages_requiring_ocr_comparison": [
            int(page_number)
            for page_number in trigger.get("pages_requiring_ocr_comparison", [])
        ],
        "local_page_comparison_supported": bool(
            trigger.get("local_page_comparison_supported", False)
        ),
        "should_run_ocr_comparison": bool(
            trigger.get("should_run_ocr_comparison", False)
        ),
    }


def _build_page_source_selection(
    extraction_mode_used: str,
    extraction_decision: Dict[str, object],
) -> List[Dict[str, object]]:
    """
    Convert extraction decisions into an explicit page-by-page selection list.

    Parameters
    ----------
    extraction_mode_used : str
        Active document-level extraction mode.

    extraction_decision : Dict[str, object]
        Detailed extraction decision payload.

    Returns
    -------
    List[Dict[str, object]]
        Per-page extraction source selection details.
    """
    page_decisions = list(extraction_decision.get("page_decisions", []))
    if page_decisions:
        return [
            {
                "page_number": int(page_decision.get("page_number", 0)),
                "selected_source": str(page_decision.get("selected_source", "native")),
                "selected_mode": str(
                    page_decision.get(
                        "selected_mode",
                        page_decision.get("selected_source", "native"),
                    )
                ),
            }
            for page_decision in page_decisions
        ]

    selected_source = "ocr" if extraction_mode_used == "ocr" else "native"
    return [
        {
            "page_number": page_number,
            "selected_source": selected_source,
            "selected_mode": selected_source,
        }
        for page_number in range(1, int(extraction_decision.get("page_count", 0)) + 1)
    ]


def _build_structure_stage_diagnostic_summary(
    extraction_mode_used: str,
    extraction_decision: Dict[str, object],
    structural_integrity_summary: Dict[str, object],
) -> Dict[str, object]:
    """
    Build a compact structure-stage diagnostic payload for exported summaries.

    Parameters
    ----------
    extraction_mode_used : str
        Active document-level extraction mode.

    extraction_decision : Dict[str, object]
        Detailed extraction decision payload.

    structural_integrity_summary : Dict[str, object]
        Parser integrity summary for the current structure tree.

    Returns
    -------
    Dict[str, object]
        Compact diagnostic payload for structure-stage auditing.
    """
    return {
        "extraction": _build_extraction_diagnostic_summary(
            extraction_mode_used=extraction_mode_used,
            extraction_decision=extraction_decision,
        ),
        "structure": _build_structure_diagnostic_summary(
            structural_integrity_summary=structural_integrity_summary
        ),
    }


def _build_chunk_stage_diagnostic_summary(
    extraction_mode_used: str,
    extraction_decision: Dict[str, object],
    structural_integrity_summary: Dict[str, object],
    validator_summary: Dict[str, object],
    next_phase_decision: Dict[str, object],
) -> Dict[str, object]:
    """
    Build a compact chunk-stage diagnostic payload for exported summaries.

    Parameters
    ----------
    extraction_mode_used : str
        Active document-level extraction mode.

    extraction_decision : Dict[str, object]
        Detailed extraction decision payload.

    structural_integrity_summary : Dict[str, object]
        Parser integrity summary for the current structure tree.

    validator_summary : Dict[str, object]
        Reduced validator payload for the current chunk sequence.

    next_phase_decision : Dict[str, object]
        Final accept/reject decision for downstream consumption.

    Returns
    -------
    Dict[str, object]
        Compact diagnostic payload explaining the final chunk-stage decision.
    """
    return {
        "extraction": _build_extraction_diagnostic_summary(
            extraction_mode_used=extraction_mode_used,
            extraction_decision=extraction_decision,
        ),
        "structure": _build_structure_diagnostic_summary(
            structural_integrity_summary=structural_integrity_summary
        ),
        "validation": _build_validation_diagnostic_summary(validator_summary),
        "final_decision": {
            "decision": next_phase_decision["decision"],
            "is_acceptable": next_phase_decision["is_acceptable"],
            "reason": next_phase_decision["reason"],
            "decision_drivers": _build_decision_drivers(
                extraction_mode_used=extraction_mode_used,
                extraction_decision=extraction_decision,
                structural_integrity_summary=structural_integrity_summary,
                validator_summary=validator_summary,
                next_phase_decision=next_phase_decision,
            ),
        },
    }


def _build_extraction_diagnostic_summary(
    extraction_mode_used: str,
    extraction_decision: Dict[str, object],
) -> Dict[str, object]:
    """
    Highlight the extraction choices that matter for debugging downstream runs.

    Parameters
    ----------
    extraction_mode_used : str
        Active document-level extraction mode.

    extraction_decision : Dict[str, object]
        Detailed extraction decision payload.

    Returns
    -------
    Dict[str, object]
        Extraction-side highlights focused on comparison and page selection.
    """
    page_decisions = list(extraction_decision.get("page_decisions", []))
    pages_switched_to_ocr: List[int] = []
    pages_kept_native_after_comparison: List[int] = []
    pages_kept_native_without_comparison: List[int] = []
    ocr_comparison_trigger = _build_ocr_comparison_trigger_summary(extraction_decision)

    for page_decision in page_decisions:
        page_number = int(page_decision.get("page_number", 0))
        selected_source = str(page_decision.get("selected_source", "native"))
        compared_with_ocr = _did_page_run_ocr_comparison(
            page_decision=page_decision,
            extraction_decision=extraction_decision,
        )

        if selected_source == "ocr":
            pages_switched_to_ocr.append(page_number)
            continue

        if compared_with_ocr:
            pages_kept_native_after_comparison.append(page_number)
            continue

        pages_kept_native_without_comparison.append(page_number)

    return {
        "document_mode": extraction_mode_used,
        "document_composition": extraction_decision.get(
            "document_composition",
            _classify_document_composition(extraction_mode_used),
        ),
        "comparison_performed": bool(
            extraction_decision.get("comparison_performed", False)
        ),
        "ocr_comparison_trigger": ocr_comparison_trigger,
        "pages_switched_to_ocr": pages_switched_to_ocr,
        "pages_kept_native_after_comparison": pages_kept_native_after_comparison,
        "pages_kept_native_without_comparison": (
            pages_kept_native_without_comparison
        ),
        "page_decision_trace": [
            {
                "page_number": int(page_decision.get("page_number", 0)),
                "compared_with_ocr": _did_page_run_ocr_comparison(
                    page_decision=page_decision,
                    extraction_decision=extraction_decision,
                ),
                "selected_source": str(page_decision.get("selected_source", "native")),
                "selected_mode": str(
                    page_decision.get(
                        "selected_mode",
                        page_decision.get("selected_source", "native"),
                    )
                ),
                "decision": str(page_decision.get("decision", "")),
                "score_gap": page_decision.get("score_gap"),
                "reason_codes": list(page_decision.get("reason_codes", [])),
            }
            for page_decision in page_decisions
        ],
    }


def _did_page_run_ocr_comparison(
    page_decision: Dict[str, object],
    extraction_decision: Dict[str, object],
) -> bool:
    """
    Determine whether one page actually entered OCR-vs-native comparison.

    Parameters
    ----------
    page_decision : Dict[str, object]
        Exported decision payload for one page.

    extraction_decision : Dict[str, object]
        Document-level extraction decision payload.

    Returns
    -------
    bool
        True when the page was compared against OCR before final selection.
    """
    explicit_flag = page_decision.get("compared_with_ocr")
    if explicit_flag is not None:
        return bool(explicit_flag)

    if not bool(extraction_decision.get("comparison_performed", False)):
        return False

    return True


def _build_structure_diagnostic_summary(
    structural_integrity_summary: Dict[str, object],
) -> Dict[str, object]:
    """
    Highlight parser integrity outcomes that affect downstream trust.

    Parameters
    ----------
    structural_integrity_summary : Dict[str, object]
        Parser integrity summary for the current structure tree.

    Returns
    -------
    Dict[str, object]
        Reduced structure-side diagnostics suitable for summary exports.
    """
    return {
        "has_structural_integrity_warnings": bool(
            structural_integrity_summary.get("has_structural_integrity_warnings", False)
        ),
        "structurally_incomplete_article_count": int(
            structural_integrity_summary.get(
                "structurally_incomplete_article_count",
                0,
            )
        ),
        "articles_with_truncation_signals": int(
            structural_integrity_summary.get("articles_with_truncation_signals", 0)
        ),
        "articles_with_integrity_warnings": int(
            structural_integrity_summary.get("articles_with_integrity_warnings", 0)
        ),
        "integrity_warning_counts": dict(
            structural_integrity_summary.get("integrity_warning_counts", {})
        ),
        "truncation_signal_counts": dict(
            structural_integrity_summary.get("truncation_signal_counts", {})
        ),
        "example_articles": list(
            structural_integrity_summary.get("example_articles", [])
        )[:3],
    }


def _build_validation_diagnostic_summary(
    validator_summary: Dict[str, object],
) -> Dict[str, object]:
    """
    Highlight validator outcomes without duplicating validator decision logic.

    Parameters
    ----------
    validator_summary : Dict[str, object]
        Reduced validator payload for the current chunk sequence.

    Returns
    -------
    Dict[str, object]
        Validation highlights focused on blocking failures and examples.
    """
    blocking_failures = list(validator_summary.get("blocking_failures", []))
    failure_messages = dict(validator_summary.get("failure_messages", {}))

    return {
        "overall_status": str(validator_summary.get("overall_status", "fail")),
        "has_blocking_failures": bool(
            validator_summary.get("has_blocking_failures", False)
        ),
        "blocking_failure_count": int(
            validator_summary.get("blocking_failure_count", 0)
        ),
        "invalid_chunk_count": int(validator_summary.get("invalid_chunk_count", 0)),
        "blocking_failures": [
            {
                "code": str(failure.get("code", "")),
                "count": int(failure.get("count", 0)),
                "example_chunk_ids": list(failure.get("example_chunk_ids", [])),
                "messages": list(
                    failure_messages.get(str(failure.get("code", "")), [])
                ),
            }
            for failure in blocking_failures
        ],
    }


def _build_decision_drivers(
    extraction_mode_used: str,
    extraction_decision: Dict[str, object],
    structural_integrity_summary: Dict[str, object],
    validator_summary: Dict[str, object],
    next_phase_decision: Dict[str, object],
) -> List[str]:
    """
    Build short sentences explaining the major factors behind the final decision.

    Parameters
    ----------
    extraction_mode_used : str
        Active document-level extraction mode.

    extraction_decision : Dict[str, object]
        Detailed extraction decision payload.

    structural_integrity_summary : Dict[str, object]
        Parser integrity summary for the current structure tree.

    validator_summary : Dict[str, object]
        Reduced validator payload for the current chunk sequence.

    next_phase_decision : Dict[str, object]
        Final accept/reject decision for downstream consumption.

    Returns
    -------
    List[str]
        Ordered high-level statements that explain the exported decision.
    """
    drivers: List[str] = []
    compared_page_count = int(extraction_decision.get("compared_page_count", 0))
    ocr_page_count = int(extraction_decision.get("ocr_page_count", 0))

    if extraction_mode_used == "hybrid":
        drivers.append(
            "Hybrid extraction kept native text where it remained stronger and "
            f"switched {ocr_page_count} page(s) to OCR."
        )
    elif extraction_mode_used == "ocr" and compared_page_count > 0:
        drivers.append(
            "OCR won after extraction comparison, so all compared pages were "
            "exported from OCR."
        )
    elif compared_page_count > 0:
        drivers.append(
            "Extraction comparison ran, but native extraction remained the best "
            "available document-level result."
        )
    else:
        drivers.append(
            "No OCR comparison was needed because native extraction stayed within "
            "the accepted quality profile."
        )

    if structural_integrity_summary.get("has_structural_integrity_warnings", False):
        drivers.append(
            "Parser integrity warnings remained present in "
            f"{int(structural_integrity_summary.get('structurally_incomplete_article_count', 0))} "
            "article(s)."
        )

    if validator_summary.get("has_blocking_failures", False):
        failure_codes = list(validator_summary.get("blocking_failure_types", []))
        failure_label = ", ".join(failure_codes[:3])
        if len(failure_codes) > 3:
            failure_label = f"{failure_label}, ..."
        drivers.append(
            "Validator blocked downstream acceptance due to "
            f"{failure_label or 'blocking chunk-quality failures'}."
        )
    else:
        drivers.append(
            "Validator reported no blocking chunk-quality failures for downstream "
            "embedding consumption."
        )

    drivers.append(
        f"Final decision: {str(next_phase_decision.get('decision', 'reject'))}."
    )
    return drivers


def _build_structural_integrity_summary(
    structure_root: StructuralNode,
) -> Dict[str, object]:
    """
    Build one compact view of parser integrity warnings across ARTICLE nodes.

    Parameters
    ----------
    structure_root : StructuralNode
        Parsed document tree.

    Returns
    -------
    Dict[str, object]
        Article-level integrity summary for exported diagnostics.
    """
    article_nodes = _collect_nodes_by_type(structure_root, "ARTICLE")
    truncation_signal_counter = CounterLike()
    integrity_warning_counter = CounterLike()
    article_examples: List[Dict[str, object]] = []
    articles_with_truncation_signals = 0
    articles_with_integrity_warnings = 0
    structurally_incomplete_article_count = 0

    for article in article_nodes:
        metadata = article.metadata or {}
        truncation_signals = [
            str(signal).strip()
            for signal in metadata.get("truncation_signals", [])
            if str(signal).strip()
        ]
        integrity_warnings = [
            str(signal).strip()
            for signal in metadata.get("integrity_warnings", [])
            if str(signal).strip()
        ]
        is_structurally_incomplete = bool(
            metadata.get("is_structurally_incomplete", False)
        )

        if truncation_signals:
            articles_with_truncation_signals += 1
            for signal in truncation_signals:
                truncation_signal_counter.increment(signal)

        if integrity_warnings:
            articles_with_integrity_warnings += 1
            for warning in integrity_warnings:
                integrity_warning_counter.increment(warning)

        if is_structurally_incomplete:
            structurally_incomplete_article_count += 1
            if len(article_examples) < 5:
                article_examples.append(
                    {
                        "article_label": article.label,
                        "article_title": article.title,
                        "page_start": article.page_start,
                        "page_end": article.page_end,
                        "truncation_signals": truncation_signals,
                        "integrity_warnings": integrity_warnings,
                    }
                )

    return {
        "article_count": len(article_nodes),
        "structurally_incomplete_article_count": structurally_incomplete_article_count,
        "articles_with_truncation_signals": articles_with_truncation_signals,
        "articles_with_integrity_warnings": articles_with_integrity_warnings,
        "has_structural_integrity_warnings": structurally_incomplete_article_count > 0,
        "truncation_signal_counts": truncation_signal_counter.to_dict(),
        "integrity_warning_counts": integrity_warning_counter.to_dict(),
        "example_articles": article_examples,
    }


def _build_native_only_extraction_decision(
    native_extracted_document: ExtractedDocument,
    native_extraction_quality: Dict[str, object],
    ocr_comparison_trigger: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Build a stable extraction-decision payload when OCR comparison is skipped.

    Parameters
    ----------
    native_extracted_document : ExtractedDocument
        Native extraction result selected without OCR comparison.

    native_extraction_quality : Dict[str, object]
        Analyzer output for the native extraction.

    ocr_comparison_trigger : Optional[Dict[str, object]]
        Diagnostic payload describing whether OCR comparison was justified.

    Returns
    -------
    Dict[str, object]
        Explainable extraction decision payload.
    """
    page_count = len(native_extracted_document.pages)
    return {
        "comparison_performed": False,
        "document_composition": "all_native",
        "page_count": page_count,
        "compared_page_count": 0,
        "native_page_count": page_count,
        "ocr_page_count": 0,
        "native_extraction_quality": native_extraction_quality,
        "ocr_comparison_trigger": ocr_comparison_trigger or {},
        "page_decisions": [],
    }


def _build_ocr_comparison_trigger(
    native_extraction_quality: Dict[str, object],
    enable_hybrid_page_selection: bool,
) -> Dict[str, object]:
    """
    Summarize whether OCR comparison should run for the native extraction.

    Parameters
    ----------
    native_extraction_quality : Dict[str, object]
        Analyzer output for the native extraction candidate.

    enable_hybrid_page_selection : bool
        Whether the pipeline can safely compare and select pages individually.

    Returns
    -------
    Dict[str, object]
        Explainable trigger payload used by pipeline orchestration and exports.
    """
    pages_requiring_ocr_comparison = [
        int(page_number)
        for page_number in native_extraction_quality.get(
            "pages_requiring_ocr_comparison",
            [],
        )
    ]
    document_likely_corrupted = bool(
        native_extraction_quality.get("document_likely_corrupted", False)
    )
    has_local_pages_requiring_ocr_comparison = bool(
        native_extraction_quality.get(
            "has_local_pages_requiring_ocr_comparison",
            False,
        )
    ) or bool(pages_requiring_ocr_comparison)
    local_page_comparison_supported = (
        enable_hybrid_page_selection and has_local_pages_requiring_ocr_comparison
    )

    return {
        "document_likely_corrupted": document_likely_corrupted,
        "has_local_pages_requiring_ocr_comparison": (
            has_local_pages_requiring_ocr_comparison
        ),
        "pages_requiring_ocr_comparison": pages_requiring_ocr_comparison,
        "local_page_comparison_supported": local_page_comparison_supported,
        "should_run_ocr_comparison": (
            document_likely_corrupted or local_page_comparison_supported
        ),
    }


def _build_hybrid_extraction_result(
    source_path: str,
    native_extracted_document: ExtractedDocument,
    native_extraction_quality: Dict[str, object],
    ocr_extracted_document: ExtractedDocument,
    ocr_extraction_quality: Dict[str, object],
    extraction_quality_analyzer: ExtractionQualityAnalyzer,
    ocr_comparison_trigger: Optional[Dict[str, object]] = None,
) -> tuple[ExtractedDocument, str, Dict[str, object]]:
    """
    Assemble the best available document page by page from native and OCR.

    Parameters
    ----------
    source_path : str
        Source PDF path used for the assembled document.

    native_extracted_document : ExtractedDocument
        Native extraction candidate.

    native_extraction_quality : Dict[str, object]
        Analyzer output for the native candidate.

    ocr_extracted_document : ExtractedDocument
        OCR extraction candidate.

    ocr_extraction_quality : Dict[str, object]
        Analyzer output for the OCR candidate.

    extraction_quality_analyzer : ExtractionQualityAnalyzer
        Shared analyzer reused for page-level comparison.

    ocr_comparison_trigger : Optional[Dict[str, object]]
        Diagnostic payload describing why OCR comparison was run.

    Returns
    -------
    tuple[ExtractedDocument, str, Dict[str, object]]
        Assembled document, document-level extraction mode, and decision
        diagnostics describing which source won per page.
    """
    native_pages = native_extracted_document.pages
    ocr_pages = ocr_extracted_document.pages

    if len(native_pages) != len(ocr_pages):
        raise ValueError(
            "Native and OCR extraction produced different page counts. "
            "Hybrid page assembly requires one OCR candidate per native page."
        )

    selected_pages = []
    page_decisions: List[Dict[str, Any]] = []
    comparison_trigger = ocr_comparison_trigger or {}
    compare_all_pages = bool(
        comparison_trigger.get("document_likely_corrupted", False)
    )
    pages_requiring_ocr_comparison = {
        int(page_number)
        for page_number in comparison_trigger.get("pages_requiring_ocr_comparison", [])
    }
    compared_page_count = 0

    for native_page, ocr_page in zip(native_pages, ocr_pages):
        page_requires_comparison = compare_all_pages or (
            native_page.page_number in pages_requiring_ocr_comparison
        )

        if page_requires_comparison:
            comparison = extraction_quality_analyzer.compare_page_versions(
                native_page=native_page,
                ocr_page=ocr_page,
            )
            selected_page = (
                ocr_page if comparison["preferred_source"] == "ocr" else native_page
            )
            selected_pages.append(selected_page)
            page_decisions.append(
                {
                    "page_number": comparison["page_number"],
                    "compared_with_ocr": True,
                    "selected_source": comparison["preferred_source"],
                    "selected_mode": comparison["preferred_mode"],
                    "decision": comparison["decision"],
                    "score_gap": comparison["score_gap"],
                    "reason_codes": comparison["reason_codes"],
                }
            )
            compared_page_count += 1
            continue

        selected_pages.append(native_page)
        page_decisions.append(
            {
                "page_number": native_page.page_number,
                "compared_with_ocr": False,
                "selected_source": "native",
                "selected_mode": native_page.selected_mode,
                "decision": "keep_native_without_ocr_comparison",
                "score_gap": None,
                "reason_codes": ["page_not_flagged_for_ocr_comparison"],
            }
        )

    active_extracted_document = ExtractedDocument(
        source_path=source_path,
        page_count=len(selected_pages),
        pages=selected_pages,
    )
    extraction_mode_used = _classify_document_extraction_mode(active_extracted_document)

    native_page_count = sum(
        1 for page in active_extracted_document.pages if page.selected_mode != "ocr"
    )
    ocr_page_count = len(active_extracted_document.pages) - native_page_count

    extraction_decision = {
        "comparison_performed": True,
        "document_composition": _classify_document_composition(
            extraction_mode_used=extraction_mode_used
        ),
        "page_count": len(active_extracted_document.pages),
        "compared_page_count": compared_page_count,
        "native_page_count": native_page_count,
        "ocr_page_count": ocr_page_count,
        "native_extraction_quality": native_extraction_quality,
        "ocr_extraction_quality": ocr_extraction_quality,
        "ocr_comparison_trigger": comparison_trigger,
        "page_decisions": page_decisions,
    }

    return active_extracted_document, extraction_mode_used, extraction_decision


def _build_ocr_only_extraction_decision(
    ocr_extracted_document: ExtractedDocument,
    native_extraction_quality: Dict[str, object],
    ocr_extraction_quality: Dict[str, object],
    ocr_comparison_trigger: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Build a stable extraction-decision payload when hybrid page selection is disabled.

    Parameters
    ----------
    ocr_extracted_document : ExtractedDocument
        OCR extraction result selected as the active document.

    native_extraction_quality : Dict[str, object]
        Analyzer output for the native extraction.

    ocr_extraction_quality : Dict[str, object]
        Analyzer output for the OCR extraction.

    ocr_comparison_trigger : Optional[Dict[str, object]]
        Diagnostic payload describing why OCR comparison was run.

    Returns
    -------
    Dict[str, object]
        Explainable extraction decision payload.
    """
    page_count = len(ocr_extracted_document.pages)
    return {
        "comparison_performed": False,
        "document_composition": "all_ocr",
        "page_count": page_count,
        "compared_page_count": 0,
        "native_page_count": 0,
        "ocr_page_count": page_count,
        "native_extraction_quality": native_extraction_quality,
        "ocr_extraction_quality": ocr_extraction_quality,
        "ocr_comparison_trigger": ocr_comparison_trigger or {},
        "page_decisions": [],
    }


def _classify_document_extraction_mode(extracted_document: ExtractedDocument) -> str:
    """
    Classify the assembled document mode from the selected page sources.

    Parameters
    ----------
    extracted_document : ExtractedDocument
        Active extracted document.

    Returns
    -------
    str
        "native", "ocr", or "hybrid".
    """
    if not extracted_document.pages:
        return "native"

    ocr_page_count = sum(
        1 for page in extracted_document.pages if page.selected_mode == "ocr"
    )
    if ocr_page_count == 0:
        return "native"
    if ocr_page_count == len(extracted_document.pages):
        return "ocr"
    return "hybrid"


def _classify_document_composition(extraction_mode_used: str) -> str:
    """
    Convert the active extraction mode into a stable composition label.

    Parameters
    ----------
    extraction_mode_used : str
        Document-level mode selected for downstream processing.

    Returns
    -------
    str
        "all_native", "all_ocr", or "hybrid".
    """
    if extraction_mode_used == "ocr":
        return "all_ocr"
    if extraction_mode_used == "hybrid":
        return "hybrid"
    return "all_native"


def _build_next_phase_decision(
    strategy_name: str,
    validation_report: Dict[str, object],
) -> Dict[str, object]:
    """
    Convert validator output into a clear accept/reject pipeline signal.

    Parameters
    ----------
    strategy_name : str
        Strategy evaluated for the current summary.

    validation_report : Dict[str, object]
        Aggregate validator output for one chunk sequence.

    Returns
    -------
    Dict[str, object]
        Deterministic acceptance decision for downstream consumption.
    """
    blocking_overview = _build_blocking_failure_overview(validation_report)
    is_acceptable = not blocking_overview["has_blocking_failures"]

    if is_acceptable:
        decision_reason = (
            "Validator reported no blocking chunk-quality failures, so the chunk "
            "sequence is acceptable for downstream embedding consumption."
        )
    else:
        blocking_failure_types = blocking_overview["blocking_failure_types"]
        failure_label = ", ".join(blocking_failure_types[:3])
        if len(blocking_failure_types) > 3:
            failure_label = f"{failure_label}, ..."
        decision_reason = (
            "Validator rejected the chunk sequence for downstream embedding "
            f"consumption: {blocking_overview['invalid_chunk_count']} invalid "
            f"chunk(s) across {blocking_overview['blocking_failure_count']} "
            f"blocking issue(s) ({failure_label})."
        )

    return {
        "target_phase": "embedding_consumption",
        "strategy": strategy_name,
        "decision": "accept" if is_acceptable else "reject",
        "is_acceptable": is_acceptable,
        "valid_chunk_count": blocking_overview["valid_chunk_count"],
        "invalid_chunk_count": blocking_overview["invalid_chunk_count"],
        "blocking_failure_count": blocking_overview["blocking_failure_count"],
        "blocking_failure_types": blocking_overview["blocking_failure_types"],
        "failed_chunk_ids": blocking_overview["failed_chunk_ids"],
        "reason": decision_reason,
    }


def _build_blocking_failure_overview(
    validation_report: Dict[str, object],
) -> Dict[str, object]:
    """
    Build one canonical blocking-failure view from validator output.

    Parameters
    ----------
    validation_report : Dict[str, object]
        Aggregate validator output for one chunk sequence.

    Returns
    -------
    Dict[str, object]
        Reduced validator facts reused by summary and decision helpers.
    """
    issue_type_counts = dict(validation_report.get("issue_type_counts", {}))
    issue_examples = dict(validation_report.get("issue_examples", {}))
    invalid_chunk_count = int(validation_report.get("invalid_chunk_count", 0))

    return {
        "chunk_count": int(validation_report.get("chunk_count", 0)),
        "valid_chunk_count": int(validation_report.get("valid_chunk_count", 0)),
        "invalid_chunk_count": invalid_chunk_count,
        "overall_status": str(validation_report.get("overall_status", "fail")),
        "has_blocking_failures": invalid_chunk_count > 0,
        "blocking_failure_count": sum(issue_type_counts.values()),
        "blocking_failure_types": _get_sorted_blocking_failure_types(validation_report),
        "blocking_failures": [
            {
                "code": issue_code,
                "count": issue_type_counts[issue_code],
                "example_chunk_ids": list(issue_examples.get(issue_code, [])),
            }
            for issue_code in _get_sorted_blocking_failure_types(validation_report)
        ],
        "failed_chunk_ids": [
            report.get("chunk_id")
            for report in validation_report.get("chunk_reports", [])
            if not report.get("is_valid") and report.get("chunk_id")
        ][:10],
    }


def _get_sorted_blocking_failure_types(
    validation_report: Dict[str, object],
) -> List[str]:
    """
    Order blocking failure codes by frequency for stable summary export.

    Parameters
    ----------
    validation_report : Dict[str, object]
        Aggregate validator output for one chunk sequence.

    Returns
    -------
    List[str]
        Blocking validator issue codes ordered by descending frequency.
    """
    issue_type_counts = dict(validation_report.get("issue_type_counts", {}))

    return [
        issue_code
        for issue_code, _count in sorted(
            issue_type_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _count_node_type(node: StructuralNode, node_type: str) -> int:
    """
    Recursively count nodes of a given type inside the parsed structure tree.

    Parameters
    ----------
    node : StructuralNode
        Current node.

    node_type : str
        Node type to count.

    Returns
    -------
    int
        Total count of matching nodes.
    """
    count = 1 if node.node_type == node_type else 0

    for child in node.children:
        count += _count_node_type(child, node_type)

    return count


def _collect_nodes_by_type(node: StructuralNode, node_type: str) -> List[StructuralNode]:
    """
    Collect all nodes of a given type inside the parsed structure tree.

    Parameters
    ----------
    node : StructuralNode
        Current root node for traversal.

    node_type : str
        Node type to collect.

    Returns
    -------
    List[StructuralNode]
        Matching nodes in traversal order.
    """
    matching_nodes: List[StructuralNode] = []

    if node.node_type == node_type:
        matching_nodes.append(node)

    for child in node.children:
        matching_nodes.extend(_collect_nodes_by_type(child, node_type))

    return matching_nodes


def _dict_to_pretty_json(data: dict) -> str:
    """
    Serialize dictionaries using readable UTF-8 JSON formatting.

    Parameters
    ----------
    data : dict
        Dictionary to serialize.

    Returns
    -------
    str
        Pretty JSON string.
    """
    return json.dumps(data, ensure_ascii=False, indent=2)


class CounterLike:
    """
    Small deterministic counter helper for lightweight JSON summaries.

    Why this helper exists
    ----------------------
    The standard library Counter would also work, but this helper keeps the
    export payload intentionally simple and explicit.
    """

    def __init__(self) -> None:
        """
        Initialize an empty counter.
        """
        self._data: Dict[str, int] = {}

    def increment(self, key: str) -> None:
        """
        Increment one counter key.

        Parameters
        ----------
        key : str
            Counter key to increment.
        """
        self._data[key] = self._data.get(key, 0) + 1

    def to_dict(self) -> Dict[str, int]:
        """
        Return the internal counter data as a plain dictionary.

        Returns
        -------
        Dict[str, int]
            Counter contents.
        """
        return dict(self._data)
