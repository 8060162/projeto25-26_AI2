from __future__ import annotations

import io
from pathlib import Path
from typing import List

from Chunking.chunking.models import PageText

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyMuPDF is required. Install package 'pymupdf'."
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Pillow is required for OCR fallback. Install package 'pillow'."
    ) from exc

try:
    import pytesseract
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "pytesseract is required for OCR fallback. Install package 'pytesseract'."
    ) from exc


class OcrFallbackReader:
    """
    OCR-based fallback extractor for problematic PDFs.

    Why this component exists
    -------------------------
    Some PDFs visually render correctly but contain unusable or corrupted
    text layers. In those cases, normal text extraction may return garbled
    output such as symbol-heavy noise instead of readable Portuguese text.

    Example failure mode:
        "*+,-*-.-/-1/2*-34+*4/-5/-1/6-/"

    For those documents, the pipeline needs an alternate path:
    - render each page as an image
    - run OCR over the rendered page
    - return page-level text in the same format expected by the rest of the
      pipeline

    Design goals
    ------------
    - integrate cleanly with the existing PageText model
    - remain page-based for metadata traceability
    - be explicit and easy to debug
    - be used only as a fallback, not as the primary extraction path

    Important operational note
    --------------------------
    This module requires:
    - Pillow
    - pytesseract
    - a working Tesseract installation available in the system PATH

    Language note
    -------------
    By default this reader uses Portuguese OCR via "por".
    """

    def __init__(
        self,
        dpi: int = 300,
        tesseract_lang: str = "por",
    ) -> None:
        """
        Initialize the OCR fallback reader.

        Parameters
        ----------
        dpi : int, default=300
            Rendering DPI used when converting PDF pages into images.

            Why 300 DPI?
            ------------
            It is a practical balance between:
            - OCR readability
            - processing time
            - memory usage

            Lower values may reduce OCR quality.
            Much higher values may increase processing cost without major gains.

        tesseract_lang : str, default="por"
            Tesseract OCR language code.

            For Portuguese legal documents, "por" is the appropriate default.
        """
        self.dpi = dpi
        self.tesseract_lang = tesseract_lang

    def extract_pages(self, pdf_path: Path) -> List[PageText]:
        """
        Extract text from a PDF document using OCR page by page.

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file that should be processed via OCR.

        Returns
        -------
        List[PageText]
            A list of PageText objects where each page text comes from OCR.

        Why the output shape matches PdfReader
        -------------------------------------
        The rest of the pipeline expects page-level text in the form of
        PageText objects. Keeping that contract identical allows the OCR
        fallback to drop into the pipeline without changing normalization,
        parsing, or chunking interfaces.
        """
        pages: List[PageText] = []

        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                image = self._render_page_to_image(page)
                text = self._ocr_image(image)

                pages.append(
                    PageText(
                        page_number=page_index,
                        text=text or "",
                    )
                )

        return pages

    def _render_page_to_image(self, page: fitz.Page) -> Image.Image:
        """
        Render one PDF page into a raster image suitable for OCR.

        Why this helper exists
        ----------------------
        OCR engines operate on images, not PDF vector content.
        Therefore each page must first be rendered to a bitmap image.

        Rendering process
        -----------------
        - scale the page according to the configured DPI
        - render with PyMuPDF pixmap
        - convert the pixmap bytes into a Pillow image

        Parameters
        ----------
        page : fitz.Page
            PyMuPDF page object.

        Returns
        -------
        Image.Image
            Pillow image ready for OCR.
        """
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_bytes = pixmap.tobytes("png")

        return Image.open(io.BytesIO(image_bytes))

    def _ocr_image(self, image: Image.Image) -> str:
        """
        Run OCR over one rendered page image.

        Why this helper exists
        ----------------------
        OCR invocation is isolated here so that:
        - preprocessing can be extended later
        - OCR configuration remains centralized
        - debugging stays simpler

        Current OCR configuration
        -------------------------
        - language: Portuguese by default
        - page segmentation mode: left as Tesseract default for now

        Parameters
        ----------
        image : Image.Image
            Pillow image for one page.

        Returns
        -------
        str
            OCR-extracted text.
        """
        prepared_image = self._prepare_image_for_ocr(image)

        text = pytesseract.image_to_string(
            prepared_image,
            lang=self.tesseract_lang,
        )

        return text.strip()

    def _prepare_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Apply lightweight image preparation before OCR.

        Why this helper exists
        ----------------------
        Even simple preprocessing can improve OCR quality by reducing noise
        and increasing contrast consistency.

        Current preparation steps
        -------------------------
        - convert to grayscale

        Why keep it simple for now
        --------------------------
        Over-aggressive preprocessing can damage characters and make OCR worse.
        This first version intentionally stays conservative.

        Parameters
        ----------
        image : Image.Image
            Original rendered page image.

        Returns
        -------
        Image.Image
            Prepared image for OCR.
        """
        return image.convert("L")