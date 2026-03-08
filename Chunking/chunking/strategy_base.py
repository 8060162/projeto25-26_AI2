from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from Chunking.chunking.models import Chunk, DocumentMetadata, StructuralNode
from Chunking.config.settings import PipelineSettings


class BaseChunkingStrategy(ABC):
    """Common interface for all chunking strategies."""

    name: str = "base"

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    @abstractmethod
    def build_chunks(
        self,
        document_metadata: DocumentMetadata,
        root: StructuralNode,
    ) -> List[Chunk]:
        raise NotImplementedError