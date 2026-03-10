from __future__ import annotations

import re
from pathlib import Path
from typing import List

from docx import Document

from Chunking.chunking.models import Chunk, DocumentMetadata


# -------------------------------------------------------------------------
# XML 1.0 valid character sanitizer
#
# python-docx writes XML under the hood. Some PDF extractions may contain
# invisible control characters that are valid in Python strings but invalid
# in XML / DOCX documents.
#
# This sanitizer removes those characters before writing text to Word.
# -------------------------------------------------------------------------
INVALID_XML_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F]"
)


def sanitize_for_docx(text: str) -> str:
    """
    Remove characters that are not valid in XML 1.0.

    Why this is necessary
    ---------------------
    - PDF extraction sometimes introduces hidden control characters
    - python-docx serializes content as XML
    - XML rejects NULL bytes and several control characters

    We intentionally preserve normal whitespace such as:
    - newline
    - carriage return
    - tab

    Parameters
    ----------
    text : str
        Input text that may contain invalid XML characters.

    Returns
    -------
    str
        Sanitized text safe to write into a DOCX document.
    """
    if not text:
        return ""

    return INVALID_XML_CHARS_RE.sub("", text)


class DocxInspectionExporter:
    """
    Export chunk inspection results into a DOCX file.

    Why this exporter matters
    -------------------------
    Human inspection remains extremely important for validating:
    - chunk boundaries
    - text cleanliness
    - metadata richness
    - structural traceability
    - semantic coherence

    This exporter is intentionally inspection-oriented rather than
    presentation-oriented.
    """

    def write_chunks_docx(
        self,
        document_metadata: DocumentMetadata,
        strategy_name: str,
        chunks: List[Chunk],
        output_path: Path,
    ) -> None:
        """
        Write a chunk inspection DOCX file.

        Report structure
        ----------------
        The report includes:
        - document-level summary
        - one section per chunk
        - explicit structural fields
        - metadata
        - visible chunk text
        - optional embedding text when different from visible text

        Parameters
        ----------
        document_metadata : DocumentMetadata
            High-level source document metadata.
        strategy_name : str
            Name of the chunking strategy used.
        chunks : List[Chunk]
            Final chunk list in document order.
        output_path : Path
            Destination DOCX path.
        """
        document = Document()

        document.add_heading("Chunk Inspection Report", level=1)
        document.add_paragraph(f"Document: {sanitize_for_docx(document_metadata.title)}")
        document.add_paragraph(f"File: {sanitize_for_docx(document_metadata.file_name)}")
        document.add_paragraph(f"Strategy: {sanitize_for_docx(strategy_name)}")
        document.add_paragraph(f"Total chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            self._write_chunk_section(
                document=document,
                chunk=chunk,
                index=index,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        document.save(output_path)

    def _write_chunk_section(
        self,
        document: Document,
        chunk: Chunk,
        index: int,
    ) -> None:
        """
        Write one chunk section into the inspection DOCX.

        Why this helper exists
        ----------------------
        Splitting chunk rendering into a dedicated helper keeps the main export
        method easier to read and makes future formatting changes simpler.

        Parameters
        ----------
        document : Document
            Target DOCX document object.
        chunk : Chunk
            Chunk to render.
        index : int
            Human-friendly chunk number in the report.
        """
        document.add_heading(f"Chunk {index}", level=2)

        document.add_paragraph(f"Chunk ID: {sanitize_for_docx(chunk.chunk_id)}")
        document.add_paragraph(f"Strategy: {sanitize_for_docx(chunk.strategy)}")
        document.add_paragraph(f"Pages: {chunk.page_start} -> {chunk.page_end}")
        document.add_paragraph(f"Source node type: {sanitize_for_docx(chunk.source_node_type)}")
        document.add_paragraph(f"Source node label: {sanitize_for_docx(chunk.source_node_label)}")
        document.add_paragraph(f"Chunk reason: {sanitize_for_docx(chunk.chunk_reason)}")
        document.add_paragraph(f"Character count: {chunk.char_count}")
        document.add_paragraph(f"Previous chunk: {sanitize_for_docx(chunk.prev_chunk_id or '')}")
        document.add_paragraph(f"Next chunk: {sanitize_for_docx(chunk.next_chunk_id or '')}")

        if chunk.hierarchy_path:
            document.add_paragraph("Hierarchy path:")
            document.add_paragraph(
                sanitize_for_docx(" > ".join(chunk.hierarchy_path))
            )

        if chunk.metadata:
            document.add_paragraph("Metadata:")
            for key, value in chunk.metadata.items():
                safe_key = sanitize_for_docx(str(key))
                safe_value = sanitize_for_docx(str(value))
                document.add_paragraph(f"- {safe_key}: {safe_value}")

        document.add_paragraph("Visible text:")
        document.add_paragraph(sanitize_for_docx(chunk.text))

        # Only show embedding text when it meaningfully differs from the visible
        # chunk text. This keeps the report readable while still making embedding
        # enrichment inspectable.
        if chunk.text_for_embedding and chunk.text_for_embedding != chunk.text:
            document.add_paragraph("Embedding text:")
            document.add_paragraph(sanitize_for_docx(chunk.text_for_embedding))