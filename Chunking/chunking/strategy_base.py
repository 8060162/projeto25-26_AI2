from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.config.settings import PipelineSettings


class BaseChunkingStrategy(ABC):
    """
    Common interface for all chunking strategies.

    Why this base class exists
    --------------------------
    All chunking strategies in the project should follow the same public
    contract:

        parsed structural tree -> list of Chunk objects

    This keeps the pipeline simple and interchangeable, allowing the caller to
    switch strategies without changing the orchestration logic.

    Architectural role
    ------------------
    The pipeline is now structured roughly as:

        PDF
        -> extraction
        -> normalization
        -> structure parsing
        -> chunking strategy
        -> chunk export / downstream retrieval

    At this stage, chunking strategies should:
    - consume the parsed structural tree
    - produce clean visible chunk text
    - attach rich structural metadata
    - remain deterministic and inspectable

    Design principles
    -----------------
    - strategies should be configuration-driven through PipelineSettings
    - strategies should not mutate the structural tree in surprising ways
    - strategies should return chunks in stable document order
    - strategies should remain easy to compare during quality evaluation
    """

    name: str = "base"

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the strategy with shared pipeline settings.

        Parameters
        ----------
        settings : PipelineSettings
            Shared runtime configuration used across the pipeline.
        """
        self.settings = settings

    @abstractmethod
    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        """
        Build chunks from a parsed structural tree.

        Parameters
        ----------
        document_metadata : DocumentMetadata
            Source document metadata.

        root : StructuralNode
            Root DOCUMENT node of the parsed structural tree.

        Returns
        -------
        List[Chunk]
            Final chunk list in document order.
        """
        raise NotImplementedError