from __future__ import annotations

from pathlib import Path
from typing import List

from Chunking.chunking.models import PageText

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyMuPDF is required. Install package 'pymupdf'."
    ) from exc


class PdfReader:
    """
    Robust PDF text extractor used as the first stage of the chunking pipeline.

    Design goals
    ------------
    This component is intentionally focused on **reliable text extraction**
    rather than semantic interpretation.

    The responsibilities of this class are:

    1. Extract text page-by-page
    2. Preserve reading order as much as possible
    3. Avoid flattening layout too early
    4. Provide clean page-level text to the normalization stage

    Why page-level extraction?
    --------------------------
    Page boundaries are extremely useful later in the pipeline for:

    - page_start / page_end metadata
    - debugging chunk boundaries
    - identifying repeated headers / footers
    - traceability back to the source document

    Extraction strategy
    -------------------
    Instead of using the basic:

        page.get_text("text")

    we use:

        page.get_text("dict")

    This provides a **block-based representation** of the page.

    Advantages:
    - better preservation of reading order
    - avoids merging columns
    - allows later structural improvements
    - reduces layout artifacts

    The extractor then reconstructs the page text from:

        blocks -> lines -> spans

    in a controlled way.
    """

    def extract_pages(self, pdf_path: Path) -> List[PageText]:
        """
        Extract text from a PDF document page by page.

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file to be processed.

        Returns
        -------
        List[PageText]
            A list of PageText objects containing:
            - page_number
            - reconstructed page text

        Important implementation detail
        -------------------------------
        This method attempts block-based extraction first.
        If anything unexpected occurs (rare but possible depending on PDF
        structure), it falls back to PyMuPDF's simple text extraction.

        This ensures the pipeline remains robust across many PDF variants.
        """

        pages: List[PageText] = []

        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):

                try:
                    # Preferred extraction method
                    text = self._extract_page_via_blocks(page)

                except Exception:
                    # Fallback extraction
                    # This ensures we never lose text even if block parsing fails.
                    text = page.get_text("text")

                pages.append(
                    PageText(
                        page_number=page_index,
                        text=text or "",
                    )
                )

        return pages

    def _extract_page_via_blocks(self, page: fitz.Page) -> str:
        """
        Extract page text using block-based layout reconstruction.

        Why block extraction?
        ---------------------
        Many institutional PDFs contain:

        - multiple columns
        - floating headers
        - footers
        - numbered lists
        - text boxes

        Using PyMuPDF's "dict" extraction mode allows us to access the page
        layout structure and rebuild the text in a more controlled way.

        Structure returned by PyMuPDF:

            page
             └── blocks
                 └── lines
                     └── spans

        Each span contains a piece of text.

        The reconstruction logic concatenates spans into lines and lines into
        blocks, preserving line breaks.

        Parameters
        ----------
        page : fitz.Page
            PyMuPDF page object.

        Returns
        -------
        str
            Reconstructed text for the page.
        """

        page_dict = page.get_text("dict")

        blocks = page_dict.get("blocks", [])
        reconstructed_lines: List[str] = []

        for block in blocks:
            # Ignore non-text blocks such as images
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])

                # Rebuild the line from its spans
                line_text = "".join(span.get("text", "") for span in spans)

                if line_text.strip():
                    reconstructed_lines.append(line_text.strip())

        # Join reconstructed lines into page text
        return "\n".join(reconstructed_lines)