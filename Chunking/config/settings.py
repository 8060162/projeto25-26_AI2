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
    You will likely adjust these values after inspection of the generated
    chunk outputs and intermediate debug files.
    """

    # ---------------------------------------------------------
    # Input / Output folders
    # ---------------------------------------------------------
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    output_dir: Path = PROJECT_ROOT / "data" / "chunks"

    # ---------------------------------------------------------
    # Chunk sizing configuration
    #
    # Character-based splitting for now. This keeps the system
    # lightweight without introducing tokenizer dependencies.
    # ---------------------------------------------------------
    target_chunk_chars: int = 1800
    hard_max_chunk_chars: int = 2600
    min_chunk_chars: int = 350

    # Reserved for future overlap-aware chunking strategies.
    overlap_chars: int = 180

    # ---------------------------------------------------------
    # Export options
    # ---------------------------------------------------------
    export_docx: bool = True
    export_json: bool = True
    export_intermediate_text: bool = True

    # ---------------------------------------------------------
    # Normalization / parsing heuristics
    # ---------------------------------------------------------
    repeated_line_max_chars: int = 120
    toc_block_min_lines: int = 4
    title_max_chars: int = 120
    title_max_words: int = 14

    # ---------------------------------------------------------
    # Repeated institutional lines that may help identify
    # headers / footers when combined with repetition logic.
    #
    # These are only hints, not direct drop rules.
    # ---------------------------------------------------------
    repeated_header_footer_hints: List[str] = field(
        default_factory=lambda: [
            "INSTITUTO POLITÉCNICO DO PORTO",
            "POLITÉCNICO DO PORTO",
            "DIÁRIO DA REPÚBLICA",
        ]
    )

    # ---------------------------------------------------------
    # Supported file types
    # ---------------------------------------------------------
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf"]
    )