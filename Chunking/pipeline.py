from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from Chunking.chunking.models import DocumentMetadata
from Chunking.chunking.strategy_article_smart import ArticleSmartChunkingStrategy
from Chunking.chunking.strategy_base import BaseChunkingStrategy
from Chunking.chunking.strategy_hybrid import HybridChunkingStrategy
from Chunking.chunking.strategy_structure_first import StructureFirstChunkingStrategy
from Chunking.cleaning.normalizer import TextNormalizer
from Chunking.config.settings import PipelineSettings
from Chunking.export.docx_exporter import DocxInspectionExporter
from Chunking.export.json_exporter import JsonExporter
from Chunking.extraction.pdf_reader import PdfReader
from Chunking.parsing.structure_parser import StructureParser
from Chunking.utils.text import slugify_file_stem


def run_pipeline() -> None:
    """
    Entry point used by main.py.

    The pipeline is intentionally simple:
    1. read all PDFs from data/raw
    2. extract text page by page
    3. normalize and clean noise
    4. parse structure
    5. generate chunks with the selected strategy
    6. export JSON and DOCX inspection artifacts into data/chunks

    Important:
    We intentionally resolve default folders through PipelineSettings
    so the code works correctly on Windows/Linux/macOS without relying
    on hardcoded absolute paths such as "/data/raw".
    """
    default_settings = PipelineSettings()

    parser = argparse.ArgumentParser(
        description="Chunk regulatory PDFs into clean semantically useful chunks."
    )
    parser.add_argument(
        "--strategy",
        choices=["article_smart", "structure_first", "hybrid"],
        default="article_smart",
        help="Chunking strategy to use.",
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

    strategy = _build_strategy(args.strategy, settings)

    reader = PdfReader()
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

        document_metadata = DocumentMetadata(
            doc_id=slugify_file_stem(pdf_path.stem),
            file_name=pdf_path.name,
            title=pdf_path.stem,
            source_path=str(pdf_path),
        )

        # ---------------------------------------------------------
        # 1) Extract raw text page by page
        # ---------------------------------------------------------
        pages = reader.extract_pages(pdf_path)

        # ---------------------------------------------------------
        # 2) Normalize noisy PDF text
        # ---------------------------------------------------------
        normalized = normalizer.normalize(pages)

        # ---------------------------------------------------------
        # 3) Build page tuples expected by the structure parser
        # ---------------------------------------------------------
        page_tuples = [(page.page_number, page.text) for page in normalized.pages]

        # ---------------------------------------------------------
        # 4) Parse structure (chapters, articles, sections, etc.)
        # ---------------------------------------------------------
        structure_root = parser_engine.parse(page_tuples)

        # ---------------------------------------------------------
        # 5) Generate chunks using the selected strategy
        # ---------------------------------------------------------
        chunks = strategy.build_chunks(document_metadata, structure_root)

        # ---------------------------------------------------------
        # 6) Create output folder for this document + strategy
        # ---------------------------------------------------------
        document_output_dir = settings.output_dir / document_metadata.doc_id / strategy.name
        document_output_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # 7) Export intermediate normalized text and dropped lines
        # ---------------------------------------------------------
        if settings.export_intermediate_text:
            (document_output_dir / "01_normalized_text.txt").write_text(
                normalized.full_text,
                encoding="utf-8",
            )
            (document_output_dir / "02_dropped_lines_report.json").write_text(
                _dict_to_pretty_json(normalized.dropped_lines_report),
                encoding="utf-8",
            )

        # ---------------------------------------------------------
        # 8) Export structure and chunks to JSON
        # ---------------------------------------------------------
        if settings.export_json:
            json_exporter.write_structure(
                structure_root,
                document_output_dir / "03_structure.json",
            )
            json_exporter.write_chunks(
                chunks,
                document_output_dir / "04_chunks.json",
            )

        # ---------------------------------------------------------
        # 9) Export DOCX inspection file for human validation
        # ---------------------------------------------------------
        if settings.export_docx:
            docx_exporter.write_chunks_docx(
                document_metadata=document_metadata,
                strategy_name=strategy.name,
                chunks=chunks,
                output_path=document_output_dir / "05_chunk_inspection.docx",
            )

        print(f"[INFO] Done: {pdf_path.name} -> {document_output_dir}")

    print("[INFO] Pipeline execution completed successfully.")


def _build_strategy(name: str, settings: PipelineSettings) -> BaseChunkingStrategy:
    """
    Return the requested chunking strategy implementation.
    """
    strategies: Dict[str, BaseChunkingStrategy] = {
        "article_smart": ArticleSmartChunkingStrategy(settings),
        "structure_first": StructureFirstChunkingStrategy(settings),
        "hybrid": HybridChunkingStrategy(settings),
    }
    return strategies[name]


def _dict_to_pretty_json(data: dict) -> str:
    """
    Serialize dictionaries using readable UTF-8 JSON formatting.
    """
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)