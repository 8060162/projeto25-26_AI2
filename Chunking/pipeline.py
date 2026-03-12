from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.chunking.strategy_article_smart import ArticleSmartChunkingStrategy
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.chunking.strategy_hybrid import HybridChunkingStrategy
from Chunking.chunking.strategy_structure_first import StructureFirstChunkingStrategy
from Chunking.cleaning.normalizer import NormalizedDocument, TextNormalizer
from Chunking.config.settings import PipelineSettings
from Chunking.export.docx_exporter import DocxInspectionExporter
from Chunking.export.json_exporter import JsonExporter
from Chunking.extraction.extraction_quality import ExtractionQualityAnalyzer
from Chunking.extraction.pdf_reader import PdfReader
from Chunking.parsing.structure_parser import StructureParser
from Chunking.utils.text import slugify_file_stem


def run_pipeline() -> None:
    """
    Entry point used by main.py.

    High-level pipeline flow
    ------------------------
    1. Read all supported PDF files from the input folder
    2. Extract page-level text
    3. Analyze extraction quality
    4. Normalize noisy PDF text conservatively
    5. Parse document structure
    6. Generate chunks with one or more strategies
    7. Export intermediate artifacts and inspection outputs

    Design goals
    ------------
    - remain easy to run locally
    - remain explicit and readable
    - produce enough artifacts for debugging and quality inspection
    - avoid hardcoded absolute paths
    - support both single-strategy and all-strategy execution
    - detect extraction problems early before parsing/chunking

    Important note
    --------------
    This pipeline still remains intentionally lightweight.
    It is not a full orchestration framework; it is a practical execution
    layer for the current chunking project.

    Future evolution
    ----------------
    This version introduces extraction-quality analysis but does not yet
    perform OCR fallback automatically. That is the next step in the
    extraction pipeline roadmap.
    """
    default_settings = PipelineSettings()

    parser = argparse.ArgumentParser(
        description="Chunk regulatory PDFs into clean semantically useful chunks."
    )
    parser.add_argument(
        "--strategy",
        choices=["article_smart", "structure_first", "hybrid", "all"],
        default="article_smart",
        help="Chunking strategy to use. Use 'all' to run every available strategy.",
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
        help="Output folder for all chunking artifacts.",
    )
    args = parser.parse_args()

    settings = PipelineSettings(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
    )

    reader = PdfReader()
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

    strategy_names = _resolve_strategy_names(args.strategy)

    for pdf_path in pdf_files:
        print(f"[INFO] Processing: {pdf_path.name}")

        document_metadata = _build_document_metadata(pdf_path)

        # ---------------------------------------------------------
        # 1) Extract raw text page by page
        # ---------------------------------------------------------
        pages = reader.extract_pages(pdf_path)

        # ---------------------------------------------------------
        # 2) Analyze extraction quality before normalization
        #
        # Why this matters:
        # if the text layer is corrupted, downstream stages may still run but
        # produce structurally meaningless output. Detecting this early gives
        # us diagnostic visibility and prepares the pipeline for OCR fallback
        # in a future step.
        # ---------------------------------------------------------
        extraction_quality = extraction_quality_analyzer.analyze_document(pages)

        if extraction_quality.get("document_likely_corrupted", False):
            print(
                "[WARN] Extracted text appears suspicious or partially corrupted: "
                f"{pdf_path.name}"
            )

        # ---------------------------------------------------------
        # 3) Normalize noisy PDF text
        # ---------------------------------------------------------
        normalized = normalizer.normalize(pages)

        # ---------------------------------------------------------
        # 4) Build page tuples expected by the structure parser
        # ---------------------------------------------------------
        page_tuples = [(page.page_number, page.text) for page in normalized.pages]

        # ---------------------------------------------------------
        # 5) Parse structure (front matter, chapters, articles, etc.)
        # ---------------------------------------------------------
        structure_root = parser_engine.parse(page_tuples)

        # ---------------------------------------------------------
        # 6) Run the selected strategy or all strategies
        # ---------------------------------------------------------
        for strategy_name in strategy_names:
            strategy = _build_strategy(strategy_name, settings)

            print(f"[INFO] Running strategy: {strategy.name}")

            chunks = strategy.build_chunks(document_metadata, structure_root)

            # -----------------------------------------------------
            # 7) Create output folder for this document + strategy
            # -----------------------------------------------------
            document_output_dir = (
                settings.output_dir / document_metadata.doc_id / strategy.name
            )
            document_output_dir.mkdir(parents=True, exist_ok=True)

            # -----------------------------------------------------
            # 8) Export intermediate normalized text and dropped lines
            # -----------------------------------------------------
            if settings.export_intermediate_text:
                _write_intermediate_outputs(
                    document_output_dir=document_output_dir,
                    normalized=normalized,
                )

            # -----------------------------------------------------
            # 9) Export structure and chunks to JSON
            # -----------------------------------------------------
            if settings.export_json:
                json_exporter.write_structure(
                    structure_root,
                    document_output_dir / "03_structure.json",
                )
                json_exporter.write_chunks(
                    chunks,
                    document_output_dir / "04_chunks.json",
                )

                # Export a lightweight quality / diagnostics summary to help
                # compare runs and quickly spot suspicious results.
                if settings.export_quality_summary:
                    (document_output_dir / "04b_quality_summary.json").write_text(
                        _dict_to_pretty_json(
                            _build_quality_summary(
                                document_metadata=document_metadata,
                                extraction_quality=extraction_quality,
                                normalized=normalized,
                                structure_root=structure_root,
                                chunks=chunks,
                                strategy_name=strategy.name,
                            )
                        ),
                        encoding="utf-8",
                    )

            # -----------------------------------------------------
            # 10) Export DOCX inspection file for human validation
            # -----------------------------------------------------
            if settings.export_docx:
                docx_exporter.write_chunks_docx(
                    document_metadata=document_metadata,
                    strategy_name=strategy.name,
                    chunks=chunks,
                    output_path=document_output_dir / "05_chunk_inspection.docx",
                )

            print(f"[INFO] Done: {pdf_path.name} -> {document_output_dir}")

    print("[INFO] Pipeline execution completed successfully.")


def _resolve_strategy_names(strategy_argument: str) -> List[str]:
    """
    Resolve the CLI strategy argument into one or more concrete strategy names.

    Why this helper exists
    ----------------------
    The pipeline supports:
    - a single strategy
    - or all strategies in one run

    This is useful when comparing chunking quality across multiple strategies
    for the same input documents.

    Parameters
    ----------
    strategy_argument : str
        CLI argument value.

    Returns
    -------
    List[str]
        Strategy names to execute in order.
    """
    if strategy_argument == "all":
        return ["article_smart", "structure_first", "hybrid"]

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


def _write_intermediate_outputs(
    document_output_dir: Path,
    normalized: NormalizedDocument,
) -> None:
    """
    Write intermediate normalization artifacts.

    Exported files
    --------------
    - 01_normalized_text.txt
    - 02_dropped_lines_report.json

    Why these files matter
    ----------------------
    They are extremely useful for understanding where quality is being lost:
    - extraction issues
    - cleanup side effects
    - front matter contamination
    - repeated-line removal behavior

    Parameters
    ----------
    document_output_dir : Path
        Strategy-specific output folder.
    normalized : NormalizedDocument
        Normalized document result.
    """
    (document_output_dir / "01_normalized_text.txt").write_text(
        normalized.full_text,
        encoding="utf-8",
    )
    (document_output_dir / "02_dropped_lines_report.json").write_text(
        _dict_to_pretty_json(normalized.dropped_lines_report),
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


def _build_quality_summary(
    document_metadata: DocumentMetadata,
    extraction_quality: Dict[str, object],
    normalized: NormalizedDocument,
    structure_root: StructuralNode,
    chunks: List[Chunk],
    strategy_name: str,
) -> Dict[str, object]:
    """
    Build a lightweight quality / diagnostics summary for one document run.

    Why this helper exists
    ----------------------
    Manual inspection is still important, but a compact machine-readable
    summary helps quickly compare outputs across documents and strategies.

    This summary now also includes extraction-quality diagnostics so the team
    can distinguish between:
    - extraction failures
    - normalization issues
    - parsing issues
    - chunking issues

    Parameters
    ----------
    document_metadata : DocumentMetadata
        Source document metadata.
    extraction_quality : Dict[str, object]
        Extraction quality report produced immediately after text extraction.
    normalized : NormalizedDocument
        Normalized document output.
    structure_root : StructuralNode
        Parsed structure tree.
    chunks : List[Chunk]
        Final chunk list.
    strategy_name : str
        Strategy used for this run.

    Returns
    -------
    Dict[str, object]
        Summary dictionary suitable for JSON export.
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