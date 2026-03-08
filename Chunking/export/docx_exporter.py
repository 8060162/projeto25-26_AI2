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

    Why this is necessary:
    - PDF extraction sometimes introduces hidden control characters
    - python-docx serializes content as XML
    - XML rejects NULL bytes and several control characters

    We keep normal whitespace such as:
    - newline
    - carriage return
    - tab
    """
    if not text:
        return ""
    return INVALID_XML_CHARS_RE.sub("", text)


class DocxInspectionExporter:
    """
    Export chunk inspection results into a DOCX file.

    This inspection file is meant for human validation of:
    - chunk boundaries
    - text cleanliness
    - metadata richness
    - semantic coherence
    """

    def write_chunks_docx(
        self,
        document_metadata: DocumentMetadata,
        strategy_name: str,
        chunks: List[Chunk],
        output_path: Path,
    ) -> None:
        document = Document()

        document.add_heading("Chunk Inspection Report", level=1)
        document.add_paragraph(f"Document: {sanitize_for_docx(document_metadata.title)}")
        document.add_paragraph(f"File: {sanitize_for_docx(document_metadata.file_name)}")
        document.add_paragraph(f"Strategy: {sanitize_for_docx(strategy_name)}")
        document.add_paragraph(f"Total chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            document.add_heading(f"Chunk {index}", level=2)

            document.add_paragraph(f"Chunk ID: {sanitize_for_docx(chunk.chunk_id)}")
            document.add_paragraph(f"Pages: {chunk.page_start} -> {chunk.page_end}")

            # Write metadata in a readable format
            if chunk.metadata:
                document.add_paragraph("Metadata:")
                for key, value in chunk.metadata.items():
                    safe_key = sanitize_for_docx(str(key))
                    safe_value = sanitize_for_docx(str(value))
                    document.add_paragraph(f"- {safe_key}: {safe_value}")

            document.add_paragraph("Text:")
            document.add_paragraph(sanitize_for_docx(chunk.text))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        document.save(output_path)