from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from Chunking.config.settings import PipelineSettings
from retrieval.models import RetrievalContext, RetrievedChunkResult


@dataclass(frozen=True, slots=True)
class _PreparedChunk:
    """
    Hold one retrieved chunk together with its original input position.

    Parameters
    ----------
    chunk : RetrievedChunkResult
        Normalized retrieved chunk preserved for the final context payload.

    original_index : int
        Stable zero-based input order used for deterministic tie-breaking.
    """

    chunk: RetrievedChunkResult
    original_index: int


class RetrievalContextBuilder:
    """
    Build compact grounded context from already retrieved chunk results.

    Design goals
    ------------
    - keep context selection separate from vector-store access
    - preserve deterministic ordering and deduplication behavior
    - enforce shared retrieval and context budgets from settings
    - emit explicit metadata describing how the context was assembled
    """

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """
        Initialize the builder with shared runtime settings.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared project settings. Default settings are loaded when omitted.
        """

        self.settings = settings or PipelineSettings()

    def build_context(
        self,
        retrieved_chunks: Iterable[RetrievedChunkResult],
        *,
        top_k: Optional[int] = None,
        max_chunks: Optional[int] = None,
        max_characters: Optional[int] = None,
    ) -> RetrievalContext:
        """
        Select, deduplicate, and pack retrieved chunks into grounded context.

        Parameters
        ----------
        retrieved_chunks : Iterable[RetrievedChunkResult]
            Raw retrieved results produced by the storage layer.

        top_k : Optional[int]
            Optional override for the maximum number of ranked chunks considered.

        max_chunks : Optional[int]
            Optional override for the maximum number of chunks included in the
            final context payload.

        max_characters : Optional[int]
            Optional override for the final packed context budget.

        Returns
        -------
        RetrievalContext
            Compact grounded context ready for answer generation.
        """

        prepared_chunks = self._prepare_chunks(retrieved_chunks)
        ordered_chunks = self._sort_chunks(prepared_chunks)
        deduplicated_chunks, duplicate_count = self._deduplicate_chunks(ordered_chunks)
        filtered_chunks, score_filtered_count, missing_score_count = (
            self._filter_chunks_by_similarity(deduplicated_chunks)
        )

        effective_top_k = self._resolve_positive_limit(
            value=top_k,
            default_value=self.settings.retrieval_top_k,
            label="top_k",
        )
        effective_max_chunks = self._resolve_positive_limit(
            value=max_chunks,
            default_value=self.settings.retrieval_context_max_chunks,
            label="max_chunks",
        )
        effective_max_characters = self._resolve_positive_limit(
            value=max_characters,
            default_value=self.settings.retrieval_context_max_characters,
            label="max_characters",
        )

        candidate_limit = min(effective_top_k, effective_max_chunks)
        limited_chunks = filtered_chunks[:candidate_limit]
        omitted_by_rank_limit = max(0, len(filtered_chunks) - len(limited_chunks))

        (
            selected_chunks,
            context_text,
            truncated,
            budget_omitted_count,
        ) = self._pack_chunks_within_budget(
            chunks=limited_chunks,
            max_characters=effective_max_characters,
        )

        return RetrievalContext(
            chunks=[prepared_chunk.chunk for prepared_chunk in selected_chunks],
            context_text=context_text,
            chunk_count=len(selected_chunks),
            character_count=len(context_text),
            truncated=truncated,
            metadata={
                "total_input_chunks": len(prepared_chunks),
                "duplicate_count": duplicate_count,
                "score_filtered_count": score_filtered_count,
                "missing_similarity_score_count": missing_score_count,
                "omitted_by_rank_limit_count": omitted_by_rank_limit,
                "omitted_by_budget_count": budget_omitted_count,
                "effective_top_k": effective_top_k,
                "effective_max_chunks": effective_max_chunks,
                "effective_max_characters": effective_max_characters,
                "selected_chunk_ids": [
                    prepared_chunk.chunk.chunk_id for prepared_chunk in selected_chunks
                ],
                "selected_record_ids": [
                    prepared_chunk.chunk.record_id for prepared_chunk in selected_chunks
                ],
            },
        )

    def _prepare_chunks(
        self,
        retrieved_chunks: Iterable[RetrievedChunkResult],
    ) -> List[_PreparedChunk]:
        """
        Normalize the input iterable into prepared chunk records.

        Parameters
        ----------
        retrieved_chunks : Iterable[RetrievedChunkResult]
            Raw retrieved results supplied by the caller.

        Returns
        -------
        List[_PreparedChunk]
            Prepared chunk records with stable original positions.
        """

        prepared_chunks: List[_PreparedChunk] = []

        for original_index, chunk in enumerate(retrieved_chunks):
            if not isinstance(chunk, RetrievedChunkResult):
                continue
            if not chunk.text:
                continue
            prepared_chunks.append(
                _PreparedChunk(
                    chunk=chunk,
                    original_index=original_index,
                )
            )

        return prepared_chunks

    def _sort_chunks(
        self,
        prepared_chunks: Sequence[_PreparedChunk],
    ) -> List[_PreparedChunk]:
        """
        Order chunks deterministically before selection and packing.

        Parameters
        ----------
        prepared_chunks : Sequence[_PreparedChunk]
            Candidate chunk records to order.

        Returns
        -------
        List[_PreparedChunk]
            Ordered chunk records ready for deduplication.
        """

        return sorted(
            prepared_chunks,
            key=self._build_sort_key,
        )

    def _build_sort_key(
        self,
        prepared_chunk: _PreparedChunk,
    ) -> Tuple[int, float, float, int]:
        """
        Build the deterministic ordering key for one chunk.

        Parameters
        ----------
        prepared_chunk : _PreparedChunk
            Candidate chunk with stable input position.

        Returns
        -------
        Tuple[int, float, float, int]
            Composite key preferring explicit rank, then higher similarity,
            then shorter distance, then original order.
        """

        chunk = prepared_chunk.chunk

        rank_sort_value = chunk.rank if chunk.rank is not None else 10**9
        similarity_sort_value = (
            -chunk.similarity_score
            if chunk.similarity_score is not None
            else 0.0
        )
        distance_sort_value = (
            chunk.distance if chunk.distance is not None else float("inf")
        )

        return (
            rank_sort_value,
            similarity_sort_value,
            distance_sort_value,
            prepared_chunk.original_index,
        )

    def _deduplicate_chunks(
        self,
        prepared_chunks: Sequence[_PreparedChunk],
    ) -> Tuple[List[_PreparedChunk], int]:
        """
        Remove duplicate retrieval records while preserving first occurrence.

        Parameters
        ----------
        prepared_chunks : Sequence[_PreparedChunk]
            Ordered chunk records to deduplicate.

        Returns
        -------
        Tuple[List[_PreparedChunk], int]
            Deduplicated chunk list and duplicate count.
        """

        deduplicated_chunks: List[_PreparedChunk] = []
        seen_keys: set[Tuple[str, str, str]] = set()
        duplicate_count = 0

        for prepared_chunk in prepared_chunks:
            deduplication_key = self._build_deduplication_key(prepared_chunk.chunk)
            if deduplication_key in seen_keys:
                duplicate_count += 1
                continue

            seen_keys.add(deduplication_key)
            deduplicated_chunks.append(prepared_chunk)

        return deduplicated_chunks, duplicate_count

    def _build_deduplication_key(
        self,
        chunk: RetrievedChunkResult,
    ) -> Tuple[str, str, str]:
        """
        Build one stable deduplication key for a retrieved chunk.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Candidate chunk to deduplicate.

        Returns
        -------
        Tuple[str, str, str]
            Stable identifier tuple preserving document and text fallback.
        """

        primary_identifier = chunk.record_id or chunk.chunk_id
        secondary_identifier = chunk.chunk_id or chunk.doc_id
        text_fingerprint = " ".join(chunk.text.split())
        return (
            primary_identifier,
            secondary_identifier,
            text_fingerprint,
        )

    def _filter_chunks_by_similarity(
        self,
        prepared_chunks: Sequence[_PreparedChunk],
    ) -> Tuple[List[_PreparedChunk], int, int]:
        """
        Apply the configured similarity filter when it is enabled.

        Parameters
        ----------
        prepared_chunks : Sequence[_PreparedChunk]
            Deduplicated candidate chunks.

        Returns
        -------
        Tuple[List[_PreparedChunk], int, int]
            Filtered chunk list, removed chunk count, and missing-score count.
        """

        if not self.settings.retrieval_score_filtering_enabled:
            return list(prepared_chunks), 0, 0

        filtered_chunks: List[_PreparedChunk] = []
        score_filtered_count = 0
        missing_score_count = 0

        for prepared_chunk in prepared_chunks:
            similarity_score = prepared_chunk.chunk.similarity_score
            if similarity_score is None:
                missing_score_count += 1
                filtered_chunks.append(prepared_chunk)
                continue

            if similarity_score < self.settings.retrieval_min_similarity_score:
                score_filtered_count += 1
                continue

            filtered_chunks.append(prepared_chunk)

        return filtered_chunks, score_filtered_count, missing_score_count

    def _resolve_positive_limit(
        self,
        *,
        value: Optional[int],
        default_value: int,
        label: str,
    ) -> int:
        """
        Resolve one positive integer limit used by context selection.

        Parameters
        ----------
        value : Optional[int]
            Optional caller-supplied override.

        default_value : int
            Shared runtime default applied when the override is omitted.

        label : str
            Human-readable setting name used in validation errors.

        Returns
        -------
        int
            Positive integer limit used by the builder.
        """

        resolved_value = default_value if value is None else int(value)
        if resolved_value <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        return resolved_value

    def _pack_chunks_within_budget(
        self,
        *,
        chunks: Sequence[_PreparedChunk],
        max_characters: int,
    ) -> Tuple[List[_PreparedChunk], str, bool, int]:
        """
        Pack ordered chunks into one bounded context string.

        Parameters
        ----------
        chunks : Sequence[_PreparedChunk]
            Ordered chunk candidates that already passed selection filters.

        max_characters : int
            Maximum character budget available for the packed context.

        Returns
        -------
        Tuple[List[_PreparedChunk], str, bool, int]
            Selected chunks, final context text, truncation flag, and count of
            chunks omitted because of the character budget.
        """

        selected_chunks: List[_PreparedChunk] = []
        context_parts: List[str] = []
        character_count = 0
        truncated = False
        budget_omitted_count = 0

        for source_index, prepared_chunk in enumerate(chunks, start=1):
            candidate_block = self._format_context_block(
                chunk=prepared_chunk.chunk,
                source_index=source_index,
            )
            separator = "\n\n" if context_parts else ""
            candidate_length = len(separator) + len(candidate_block)

            if character_count + candidate_length <= max_characters:
                if separator:
                    context_parts.append(separator)
                context_parts.append(candidate_block)
                character_count += candidate_length
                selected_chunks.append(prepared_chunk)
                continue

            remaining_characters = max_characters - character_count - len(separator)
            if remaining_characters > 0 and not context_parts:
                truncated_block = candidate_block[:remaining_characters].rstrip()
                if truncated_block:
                    context_parts.append(truncated_block)
                    selected_chunks.append(prepared_chunk)
                    character_count = len(truncated_block)
                    truncated = True
                    budget_omitted_count = len(chunks) - 1
                    break

            truncated = True
            budget_omitted_count = len(chunks) - len(selected_chunks)
            break

        return selected_chunks, "".join(context_parts), truncated, budget_omitted_count

    def _format_context_block(
        self,
        *,
        chunk: RetrievedChunkResult,
        source_index: int,
    ) -> str:
        """
        Format one retrieved chunk into the final grounding payload.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Selected chunk to serialize.

        source_index : int
            One-based source number used in the packed context.

        Returns
        -------
        str
            Compact context block preserving the key grounding metadata.
        """

        header_fields = [
            f"Source {source_index}",
            f"doc_id={chunk.doc_id or 'unknown'}",
            f"chunk_id={chunk.chunk_id or chunk.record_id or 'unknown'}",
        ]

        if chunk.source_file:
            header_fields.append(f"source_file={chunk.source_file}")

        section_title = self._resolve_section_title(chunk)
        if section_title:
            header_fields.append(f"section_title={section_title}")

        page_range = self._resolve_page_range(chunk)
        if page_range:
            header_fields.append(f"pages={page_range}")

        return f"[{' | '.join(header_fields)}]\n{chunk.text}"

    def _resolve_section_title(
        self,
        chunk: RetrievedChunkResult,
    ) -> str:
        """
        Resolve one human-readable section title from chunk metadata.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Selected chunk whose metadata may contain section labels.

        Returns
        -------
        str
            First available section title, otherwise an empty string.
        """

        candidate_values = (
            chunk.chunk_metadata.get("section_title"),
            chunk.chunk_metadata.get("section"),
            chunk.document_metadata.get("document_title"),
            chunk.metadata.get("section_title"),
        )

        for value in candidate_values:
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def _resolve_page_range(
        self,
        chunk: RetrievedChunkResult,
    ) -> str:
        """
        Resolve one compact page label from chunk metadata when available.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Selected chunk whose metadata may contain page information.

        Returns
        -------
        str
            Compact page label, otherwise an empty string.
        """

        page_start = self._read_page_value(chunk, "page_start")
        page_end = self._read_page_value(chunk, "page_end")

        if page_start is None and page_end is None:
            return ""
        if page_start is None:
            return str(page_end)
        if page_end is None or page_end == page_start:
            return str(page_start)
        return f"{page_start}-{page_end}"

    def _read_page_value(
        self,
        chunk: RetrievedChunkResult,
        field_name: str,
    ) -> Optional[int]:
        """
        Read one optional page value from the available metadata scopes.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Selected chunk whose metadata may contain page values.

        field_name : str
            Metadata field to resolve.

        Returns
        -------
        Optional[int]
            Parsed integer page value, otherwise `None`.
        """

        candidate_values = (
            chunk.chunk_metadata.get(field_name),
            chunk.metadata.get(field_name),
        )

        for value in candidate_values:
            if isinstance(value, bool) or value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue

        return None
