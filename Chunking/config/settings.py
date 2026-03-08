from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------
# Resolve project root dynamically
#
# This file lives in:
# Chunking/config/settings.py
#
# Therefore:
# parents[0] = config
# parents[1] = Chunking
# parents[2] = project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class PipelineSettings:
    """
    Central runtime configuration.

    This class intentionally keeps the knobs explicit and easy to tune.
    You will likely adjust these values after the first inspection of the
    generated chunk DOCX files.
    """

    # ---------------------------------------------------------
    # Input / Output folders
    #
    # These are now resolved relative to the project root so the
    # pipeline works correctly on Windows, Linux, and macOS.
    # ---------------------------------------------------------
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    output_dir: Path = PROJECT_ROOT / "data" / "chunks"

    # ---------------------------------------------------------
    # Chunk sizing configuration
    #
    # Character-based splitting for now. This keeps the system
    # lightweight without introducing tokenizer dependencies.
    #
    # Later this can be upgraded to token-aware splitting.
    # ---------------------------------------------------------
    target_chunk_chars: int = 1800
    hard_max_chunk_chars: int = 2600
    min_chunk_chars: int = 350
    overlap_chars: int = 180

    # ---------------------------------------------------------
    # Export options
    #
    # DOCX inspection files are extremely useful for validating
    # chunk quality manually.
    # ---------------------------------------------------------
    export_docx: bool = True
    export_json: bool = True
    export_intermediate_text: bool = True

    # ---------------------------------------------------------
    # Common repeated institutional phrases that often behave
    # like layout noise (headers / footers).
    #
    # These are not always removed automatically, but are used
    # as signals during normalization.
    # ---------------------------------------------------------
    likely_noise_markers: List[str] = field(
        default_factory=lambda: [
            "POLITÉCNICO DO PORTO",
            "P.PORTO",
            "REGULAMENTO",
            "DESPACHO",
        ]
    )

    # ---------------------------------------------------------
    # Supported file types
    # ---------------------------------------------------------
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf"]
    )