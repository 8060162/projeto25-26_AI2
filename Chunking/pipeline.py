from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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
from Chunking.utils.text import slugify_file_stem


def run_pipeline() -> None:
    """
    Main pipeline entry point used by main.py.

    High-level pipeline flow
    ------------------------
    1. Read all supported PDF files from the input folder
    2. Extract structured PDF content using native extraction
    3. Analyze extraction quality
    4. Trigger OCR fallback when native extraction looks too corrupted
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
    extraction_quality_analyzer = ExtractionQualityAnalyzer()
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

        # --------------------------------------------------------------
        # 3) OCR fallback when native extraction looks corrupted
        #
        # OCR is slower and structurally poorer than native PDF extraction,
        # so we only use it when the analyzer considers native text
        # suspicious enough to justify fallback.
        # --------------------------------------------------------------
        if native_extraction_quality.get("document_likely_corrupted", False):
            print(
                "[WARN] Native extraction appears corrupted. "
                f"Triggering OCR fallback for: {pdf_path.name}"
            )

            ocr_extracted_document = ocr_reader.extract_document(pdf_path)
            ocr_extraction_quality = extraction_quality_analyzer.analyze_document(
                ocr_extracted_document
            )

            # ----------------------------------------------------------
            # Decision policy:
            # if native extraction is flagged as corrupted, OCR becomes the
            # active input. This is intentionally decisive because once native
            # text is badly corrupted, parsing the native content is usually
            # much more harmful than using OCR.
            #
            # A more sophisticated "compare both and choose best" policy can
            # be added later if needed.
            # ----------------------------------------------------------
            active_extracted_document = ocr_extracted_document
            active_extraction_quality = ocr_extraction_quality
            extraction_mode_used = "ocr"

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
        doc_id=slugify_file_stem(pdf_path.stem),
        file_name=pdf_path.name,
        title=pdf_path.stem,
        source_path=str(pdf_path),
        metadata={
            "source_extension": pdf_path.suffix.lower(),
        },
    )


def _write_structure_stage_outputs(
    structure_output_dir: Path,
    normalized: NormalizedDocument,
    extraction_quality: Dict[str, object],
    extraction_mode_used: str,
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
        "native" or "ocr".
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
        Active extraction mode ("native" or "ocr").

    Returns
    -------
    Dict[str, object]
        Structure-stage summary dictionary suitable for JSON export.
    """
    return {
        "document_id": document_metadata.doc_id,
        "document_title": document_metadata.title,
        "extraction_mode_used": extraction_mode_used,
        "extraction_quality": extraction_quality,
        "normalized_page_count": len(normalized.pages),
        "non_empty_normalized_pages": sum(
            1 for page in normalized.pages if page.text.strip()
        ),
        "dropped_lines_report": normalized.dropped_lines_report,
        "normalized_page_reports": normalized.page_reports,
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
        "native" or "ocr".

    Returns
    -------
    Dict[str, object]
        Chunk-stage summary dictionary suitable for JSON export.
    """
    chunk_count = len(chunks)
    char_counts = [getattr(chunk, "char_count", len(chunk.text)) for chunk in chunks]
    chunk_reasons = CounterLike()

    for chunk in chunks:
        chunk_reasons.increment(getattr(chunk, "chunk_reason", "") or "unspecified")

    return {
        "document_id": document_metadata.doc_id,
        "document_title": document_metadata.title,
        "strategy": strategy_name,
        "extraction_mode_used": extraction_mode_used,
        "extraction_quality": extraction_quality,
        "normalized_page_count": len(normalized.pages),
        "non_empty_normalized_pages": sum(
            1 for page in normalized.pages if page.text.strip()
        ),
        "dropped_lines_report": normalized.dropped_lines_report,
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
        "chunk_neighbor_links_complete": all(
            (index == 0 or getattr(chunks[index], "prev_chunk_id", None) is not None)
            and (
                index == len(chunks) - 1
                or getattr(chunks[index], "next_chunk_id", None) is not None
            )
            for index in range(len(chunks))
        ),
    }


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
