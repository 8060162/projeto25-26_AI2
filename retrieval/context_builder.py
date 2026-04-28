from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from Chunking.config.settings import PipelineSettings
from retrieval.models import (
    EvidenceQualityClassification,
    RetrievalContext,
    RetrievalRouteDecision,
    RetrievedChunkResult,
)

_ARTICLE_REFERENCE_PATTERN = re.compile(
    r"\bart(?:igo)?\.?\s*(\d+[a-z]?)\b",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_LEGAL_INTENT_PHRASE_MIN_SIZE = 2
_LEGAL_INTENT_PHRASE_MAX_SIZE = 4
_DOMINANT_ANCHOR_MIN_SCORE = 30.0
_DOMINANT_ANCHOR_MIN_MARGIN = 18.0
_GENERAL_SCOPE_TOKENS = frozenset(
    {
        "base",
        "default",
        "geral",
        "general",
        "normal",
        "ordinario",
        "padrao",
        "standard",
    }
)
_SPECIAL_SCOPE_TOKENS = frozenset(
    {
        "atraso",
        "consequencia",
        "consequencias",
        "divida",
        "dividas",
        "especial",
        "especifica",
        "especifico",
        "excecional",
        "excepcional",
        "incumprimento",
        "mora",
        "regularizacao",
        "regularizar",
        "specific",
        "special",
    }
)
_CONSEQUENCE_SCOPE_TOKENS = frozenset(
    {
        "atraso",
        "consequencia",
        "consequencias",
        "divida",
        "dividas",
        "incumprimento",
        "mora",
        "regularizacao",
        "regularizar",
    }
)
_LOW_SIGNAL_TOKENS = frozenset(
    {
        "about",
        "ainda",
        "aos",
        "apos",
        "apply",
        "applicable",
        "artigo",
        "article",
        "base",
        "como",
        "com",
        "context",
        "contexto",
        "does",
        "dos",
        "das",
        "estao",
        "esta",
        "este",
        "have",
        "isto",
        "mais",
        "menos",
        "num",
        "numa",
        "para",
        "pela",
        "pelo",
        "per",
        "por",
        "qual",
        "quais",
        "question",
        "regulation",
        "regulamento",
        "sobre",
        "that",
        "the",
        "uma",
    }
)


def _normalize_match_text(value: str) -> str:
    """
    Normalize text into an ASCII-like comparison form for lexical matching.

    Parameters
    ----------
    value : str
        Raw text fragment used in structural matching.

    Returns
    -------
    str
        Lowercased accent-free text with collapsed separator spacing.
    """

    normalized_value = unicodedata.normalize("NFKD", value)
    ascii_value = normalized_value.encode("ascii", "ignore").decode("ascii")
    return " ".join(_TOKEN_PATTERN.findall(ascii_value.lower()))


def _tokenize_match_text(value: str) -> FrozenSet[str]:
    """
    Tokenize one text fragment into stable high-signal lexical tokens.

    Parameters
    ----------
    value : str
        Raw text fragment used in lexical overlap checks.

    Returns
    -------
    FrozenSet[str]
        Distinct relevant tokens kept for structural matching.
    """

    tokens: set[str] = set()

    for token in _normalize_match_text(value).split():
        if len(token) < 3 or token in _LOW_SIGNAL_TOKENS:
            continue

        tokens.add(token)
        normalized_token = _normalize_lexical_token(token)
        if normalized_token and normalized_token not in _LOW_SIGNAL_TOKENS:
            tokens.add(normalized_token)

    return frozenset(tokens)


def _ordered_match_tokens(value: str) -> List[str]:
    """
    Tokenize one fragment into ordered high-signal lexical tokens.

    Parameters
    ----------
    value : str
        Raw text fragment used in phrase-level legal-intent matching.

    Returns
    -------
    List[str]
        Ordered normalized tokens with low-signal terms removed.
    """

    ordered_tokens: List[str] = []

    for token in _normalize_match_text(value).split():
        if len(token) < 3 or token in _LOW_SIGNAL_TOKENS:
            continue

        normalized_token = _normalize_lexical_token(token)
        if normalized_token and normalized_token not in _LOW_SIGNAL_TOKENS:
            ordered_tokens.append(normalized_token)

    return ordered_tokens


def _extract_legal_intent_phrases(value: str) -> FrozenSet[str]:
    """
    Extract compact phrase-level legal-intent cues from one text fragment.

    Parameters
    ----------
    value : str
        Raw query or legal text used for deterministic intent matching.

    Returns
    -------
    FrozenSet[str]
        Distinct normalized n-grams representing local normative intent.
    """

    ordered_tokens = _ordered_match_tokens(value)
    intent_phrases: set[str] = set()

    for phrase_size in range(
        _LEGAL_INTENT_PHRASE_MIN_SIZE,
        _LEGAL_INTENT_PHRASE_MAX_SIZE + 1,
    ):
        if len(ordered_tokens) < phrase_size:
            break

        for start_index in range(0, len(ordered_tokens) - phrase_size + 1):
            intent_phrases.add(
                " ".join(ordered_tokens[start_index : start_index + phrase_size])
            )

    return frozenset(intent_phrases)


def _normalize_lexical_token(token: str) -> str:
    """
    Normalize simple Portuguese plural forms for lexical cue matching.

    Parameters
    ----------
    token : str
        Accent-free lowercase token extracted from query or chunk text.

    Returns
    -------
    str
        Conservative singular-like token variant.
    """

    if len(token) <= 4:
        return token

    if token.endswith("coes") and len(token) > 6:
        return f"{token[:-4]}cao"
    if token.endswith("ais") and len(token) > 5:
        return f"{token[:-3]}al"
    if token.endswith("es") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]

    return token


@dataclass(frozen=True, slots=True)
class _QueryStructuralCues:
    """
    Hold lightweight deterministic structural cues extracted from the query.

    Parameters
    ----------
    normalized_query_text : str
        Normalized semantic query text used for lexical phrase matching.

    article_numbers : FrozenSet[str]
        Explicit article-number references extracted from the query.

    document_titles : FrozenSet[str]
        Explicit document-title targets extracted from metadata or routing.

    article_titles : FrozenSet[str]
        Explicit article-title targets extracted from metadata or routing.

    lexical_tokens : FrozenSet[str]
        High-signal query tokens used for overlap-based structural alignment.

    legal_intent_phrases : FrozenSet[str]
        Phrase-level query cues used to distinguish close normative intents.

    comparative : bool
        Whether routing identified the question as comparative across legal
        anchors.
    """

    normalized_query_text: str
    article_numbers: FrozenSet[str]
    document_titles: FrozenSet[str]
    article_titles: FrozenSet[str]
    lexical_tokens: FrozenSet[str]
    legal_intent_phrases: FrozenSet[str]
    comparative: bool = False


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


@dataclass(frozen=True, slots=True)
class _PrimaryAnchorSelection:
    """
    Describe the selected legal anchor that should govern answer generation.

    Parameters
    ----------
    anchor : str
        Compact legal-anchor label resolved from selected context.

    primary_chunk_ids : Tuple[str, ...]
        Selected chunk identifiers treated as the primary evidence anchor.

    supporting_chunk_ids : Tuple[str, ...]
        Selected same-anchor chunk identifiers retained as supporting evidence.

    score : float
        Deterministic legal-intent alignment score for the selected anchor.
    """

    anchor: str = ""
    primary_chunk_ids: Tuple[str, ...] = ()
    supporting_chunk_ids: Tuple[str, ...] = ()
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class _CompetitorAssessment:
    """
    Describe one omitted legal competitor relative to the primary anchor.

    Parameters
    ----------
    chunk_id : str
        Omitted competitor chunk identifier.

    anchor : str
        Legal anchor represented by the competitor.

    category : str
        Graded competition category: supportive, alternative_scope, or blocking.

    score : float
        Deterministic legal-intent alignment score for the competitor.

    relationship : str
        Refined legal relationship relative to the selected primary anchor.
    """

    chunk_id: str
    anchor: str
    category: str
    score: float
    relationship: str


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
        query_text: str = "",
        query_metadata: Optional[Mapping[str, Any]] = None,
        route_decision: Optional[RetrievalRouteDecision] = None,
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

        query_text : str
            Normalized semantic query text used for structural prioritization.

        query_metadata : Optional[Mapping[str, Any]]
            Optional deterministic query hints already extracted upstream.

        route_decision : Optional[RetrievalRouteDecision]
            Optional retrieval route emitted by the deterministic router.

        Returns
        -------
        RetrievalContext
            Compact grounded context ready for answer generation.
        """

        prepared_chunks = self._prepare_chunks(retrieved_chunks)
        query_structural_cues = self._extract_query_structural_cues(
            query_text=query_text,
            query_metadata=query_metadata,
            route_decision=route_decision,
        )
        ordered_chunks = self._sort_chunks(prepared_chunks, query_structural_cues)
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

        effective_candidate_pool_size = self._resolve_candidate_pool_size(
            top_k=effective_top_k,
            max_chunks=effective_max_chunks,
            route_decision=route_decision,
        )
        candidate_chunks = filtered_chunks[:effective_candidate_pool_size]
        packable_chunks = self._focus_candidate_chunks_on_dominant_anchor(
            candidate_chunks,
            query_structural_cues,
        )
        omitted_by_rank_limit = max(0, len(filtered_chunks) - len(candidate_chunks))

        (
            selected_chunks,
            context_text,
            truncated,
            budget_omitted_count,
            max_chunks_omitted_count,
        ) = self._pack_chunks_within_budget(
            chunks=packable_chunks,
            max_chunks=effective_max_chunks,
            max_characters=effective_max_characters,
            query_structural_cues=query_structural_cues,
        )
        primary_anchor_selection = self._resolve_primary_anchor_selection(
            selected_chunks=selected_chunks,
            query_structural_cues=query_structural_cues,
        )
        evidence_quality = self._classify_evidence_quality(
            candidate_chunks=candidate_chunks,
            selected_chunks=selected_chunks,
            query_structural_cues=query_structural_cues,
            primary_anchor_selection=primary_anchor_selection,
        )
        evidence_metadata = evidence_quality.metadata

        return RetrievalContext(
            chunks=[prepared_chunk.chunk for prepared_chunk in selected_chunks],
            context_text=context_text,
            chunk_count=len(selected_chunks),
            character_count=len(context_text),
            truncated=truncated,
            metadata={
                "total_input_chunks": len(prepared_chunks),
                "candidate_chunk_count": len(candidate_chunks),
                "duplicate_count": duplicate_count,
                "score_filtered_count": score_filtered_count,
                "missing_similarity_score_count": missing_score_count,
                "omitted_by_rank_limit_count": omitted_by_rank_limit,
                "omitted_by_budget_count": budget_omitted_count,
                "effective_top_k": effective_top_k,
                "effective_candidate_pool_size": effective_candidate_pool_size,
                "effective_max_chunks": effective_max_chunks,
                "effective_max_characters": effective_max_characters,
                "omitted_by_max_chunks_count": max_chunks_omitted_count,
                "selected_chunk_ids": [
                    prepared_chunk.chunk.chunk_id for prepared_chunk in selected_chunks
                ],
                "selected_record_ids": [
                    prepared_chunk.chunk.record_id for prepared_chunk in selected_chunks
                ],
                "primary_anchor": primary_anchor_selection.anchor,
                "primary_anchor_role": (
                    "main_regime" if primary_anchor_selection.anchor else ""
                ),
                "primary_anchor_chunk_ids": list(
                    primary_anchor_selection.primary_chunk_ids
                ),
                "supporting_anchor_chunk_ids": list(
                    primary_anchor_selection.supporting_chunk_ids
                ),
                "primary_anchor_score": primary_anchor_selection.score,
                "selection_query_article_numbers": sorted(
                    query_structural_cues.article_numbers
                ),
                "selection_query_document_titles": sorted(
                    query_structural_cues.document_titles
                ),
                "selection_query_article_titles": sorted(
                    query_structural_cues.article_titles
                ),
                "close_competitor_chunk_ids": (
                    evidence_quality.close_competitor_chunk_ids
                ),
                "conflicting_chunk_ids": evidence_quality.conflicting_chunk_ids,
                "supportive_competitor_chunk_ids": evidence_metadata.get(
                    "supportive_competitor_chunk_ids",
                    [],
                ),
                "alternative_scope_competitor_chunk_ids": evidence_metadata.get(
                    "alternative_scope_competitor_chunk_ids",
                    [],
                ),
                "blocking_conflict_chunk_ids": evidence_metadata.get(
                    "blocking_conflict_chunk_ids",
                    [],
                ),
                "consequence_competitor_chunk_ids": evidence_metadata.get(
                    "consequence_competitor_chunk_ids",
                    [],
                ),
                "neighboring_non_governing_chunk_ids": evidence_metadata.get(
                    "neighboring_non_governing_chunk_ids",
                    [],
                ),
            },
            route_decision=route_decision,
            evidence_quality=evidence_quality,
        )

    def _extract_query_structural_cues(
        self,
        *,
        query_text: str,
        query_metadata: Optional[Mapping[str, Any]],
        route_decision: Optional[RetrievalRouteDecision],
    ) -> _QueryStructuralCues:
        """
        Extract deterministic structural cues from the semantic query payload.

        Parameters
        ----------
        query_text : str
            Normalized semantic query text supplied by the caller.

        query_metadata : Optional[Mapping[str, Any]]
            Optional upstream query hints that may contain structural cues.

        route_decision : Optional[RetrievalRouteDecision]
            Optional deterministic route carrying legal retrieval targets.

        Returns
        -------
        _QueryStructuralCues
            Lightweight cues reused during final chunk prioritization.
        """

        normalized_query_text = _normalize_match_text(query_text)
        normalized_metadata = (
            dict(query_metadata) if isinstance(query_metadata, Mapping) else {}
        )

        structural_fragments = [query_text]
        structural_fragments.extend(
            self._read_string_list_metadata(normalized_metadata, "article_titles")
        )
        structural_fragments.extend(
            self._read_string_list_metadata(normalized_metadata, "document_titles")
        )

        route_article_numbers: List[str] = []
        route_article_titles: List[str] = []
        route_document_titles: List[str] = []
        comparative = False
        if isinstance(route_decision, RetrievalRouteDecision):
            route_article_numbers = route_decision.target_article_numbers
            route_article_titles = route_decision.target_article_titles
            route_document_titles = route_decision.target_document_titles
            comparative = route_decision.comparative

        structural_fragments.extend(route_article_titles)
        structural_fragments.extend(route_document_titles)

        for metadata_key in ("article_title", "document_title", "section_title"):
            metadata_value = normalized_metadata.get(metadata_key)
            if isinstance(metadata_value, str) and metadata_value.strip():
                structural_fragments.append(metadata_value.strip())

        article_numbers = set(_ARTICLE_REFERENCE_PATTERN.findall(normalized_query_text))
        article_numbers.update(
            self._read_string_list_metadata(normalized_metadata, "article_numbers")
        )
        article_numbers.update(route_article_numbers)

        article_number_value = normalized_metadata.get("article_number")
        if isinstance(article_number_value, str) and article_number_value.strip():
            article_numbers.add(article_number_value.strip().lower())

        document_titles = self._resolve_target_texts(
            normalized_metadata,
            single_field_name="document_title",
            list_field_name="document_titles",
            route_values=route_document_titles,
        )
        article_titles = self._resolve_target_texts(
            normalized_metadata,
            single_field_name="article_title",
            list_field_name="article_titles",
            route_values=route_article_titles,
        )

        lexical_tokens = frozenset().union(
            *[_tokenize_match_text(fragment) for fragment in structural_fragments if fragment]
        )
        legal_intent_phrases = _extract_legal_intent_phrases(query_text)

        return _QueryStructuralCues(
            normalized_query_text=normalized_query_text,
            article_numbers=frozenset(
                article_number.strip().lower()
                for article_number in article_numbers
                if article_number and article_number.strip()
            ),
            document_titles=frozenset(
                _normalize_match_text(document_title)
                for document_title in document_titles
                if document_title
            ),
            article_titles=frozenset(
                _normalize_match_text(article_title)
                for article_title in article_titles
                if article_title
            ),
            lexical_tokens=lexical_tokens,
            legal_intent_phrases=legal_intent_phrases,
            comparative=comparative,
        )

    def _resolve_target_texts(
        self,
        metadata: Mapping[str, Any],
        *,
        single_field_name: str,
        list_field_name: str,
        route_values: Sequence[str],
    ) -> List[str]:
        """
        Resolve explicit target titles from query metadata and routing.

        Parameters
        ----------
        metadata : Mapping[str, Any]
            Deterministic query metadata supplied by the caller.

        single_field_name : str
            Metadata field carrying a single target title.

        list_field_name : str
            Metadata field carrying multiple target titles.

        route_values : Sequence[str]
            Target titles emitted by the retrieval router.

        Returns
        -------
        List[str]
            Ordered non-empty target title values.
        """

        target_texts = self._read_string_list_metadata(metadata, list_field_name)
        single_value = metadata.get(single_field_name)
        if isinstance(single_value, str) and single_value.strip():
            target_texts.append(single_value.strip())
        target_texts.extend(value for value in route_values if value)
        return target_texts

    def _read_string_list_metadata(
        self,
        metadata: Mapping[str, Any],
        field_name: str,
    ) -> List[str]:
        """
        Read one optional string-list field from query metadata.

        Parameters
        ----------
        metadata : Mapping[str, Any]
            Deterministic query metadata supplied by the caller.

        field_name : str
            Metadata field to normalize into a string list.

        Returns
        -------
        List[str]
            Clean ordered string values extracted from the metadata field.
        """

        raw_value = metadata.get(field_name)
        if not isinstance(raw_value, list):
            return []

        normalized_values: List[str] = []

        for item in raw_value:
            if isinstance(item, str) and item.strip():
                normalized_values.append(item.strip())

        return normalized_values

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
        query_structural_cues: _QueryStructuralCues,
    ) -> List[_PreparedChunk]:
        """
        Order chunks deterministically before selection and packing.

        Parameters
        ----------
        prepared_chunks : Sequence[_PreparedChunk]
            Candidate chunk records to order.

        query_structural_cues : _QueryStructuralCues
            Deterministic semantic cues reused to prioritize structural matches.

        Returns
        -------
        List[_PreparedChunk]
            Ordered chunk records ready for deduplication.
        """

        return sorted(
            prepared_chunks,
            key=lambda prepared_chunk: self._build_sort_key(
                prepared_chunk,
                query_structural_cues,
            ),
        )

    def _build_sort_key(
        self,
        prepared_chunk: _PreparedChunk,
        query_structural_cues: _QueryStructuralCues,
    ) -> Tuple[int, int, int, int, float, float, int]:
        """
        Build the deterministic ordering key for one chunk.

        Parameters
        ----------
        prepared_chunk : _PreparedChunk
            Candidate chunk with stable input position.

        query_structural_cues : _QueryStructuralCues
            Deterministic semantic cues reused to prioritize structural matches.

        Returns
        -------
        Tuple[int, int, int, int, float, float, int]
            Composite key preferring explicit legal targets, content-intent
            alignment, structural alignment, explicit rank, higher similarity,
            shorter distance, and original order.
        """

        chunk = prepared_chunk.chunk
        explicit_target_alignment_score = self._score_explicit_target_alignment(
            chunk=chunk,
            query_structural_cues=query_structural_cues,
        )
        structural_alignment_score = self._score_structural_alignment(
            chunk=chunk,
            query_structural_cues=query_structural_cues,
        )
        legal_content_alignment_score = self._score_legal_content_alignment(
            chunk=chunk,
            query_structural_cues=query_structural_cues,
        )

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
            -explicit_target_alignment_score,
            -legal_content_alignment_score,
            -structural_alignment_score,
            rank_sort_value,
            similarity_sort_value,
            distance_sort_value,
            prepared_chunk.original_index,
        )

    def _score_explicit_target_alignment(
        self,
        *,
        chunk: RetrievedChunkResult,
        query_structural_cues: _QueryStructuralCues,
    ) -> int:
        """
        Score direct article or title targets extracted from query routing.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Retrieved chunk candidate being ranked for final context packing.

        query_structural_cues : _QueryStructuralCues
            Deterministic structural cues extracted from the semantic query.

        Returns
        -------
        int
            Explicit-target score used before inferred content and structure
            alignment.
        """

        context_metadata = chunk.context_metadata
        score = 0

        if (
            context_metadata.article_number
            and context_metadata.article_number.lower()
            in query_structural_cues.article_numbers
        ):
            score += 100

        if self._matches_target_title(
            context_metadata.document_title,
            query_structural_cues.document_titles,
        ):
            score += 90

        if self._matches_target_title(
            context_metadata.article_title,
            query_structural_cues.article_titles,
        ):
            score += 80

        return score

    def _score_structural_alignment(
        self,
        *,
        chunk: RetrievedChunkResult,
        query_structural_cues: _QueryStructuralCues,
    ) -> int:
        """
        Score one chunk according to deterministic structural query alignment.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Retrieved chunk candidate being ranked for final context packing.

        query_structural_cues : _QueryStructuralCues
            Deterministic structural cues extracted from the semantic query.

        Returns
        -------
        int
            Integer alignment score used only for deterministic ordering.
        """

        context_metadata = chunk.context_metadata
        score = 0

        if (
            context_metadata.article_number
            and context_metadata.article_number.lower()
            in query_structural_cues.article_numbers
        ):
            score += 100

        score += 18 * self._count_query_token_overlap(
            context_metadata.article_title,
            query_structural_cues,
        )
        score += 28 * self._count_legal_intent_phrase_overlap(
            context_metadata.article_title,
            query_structural_cues,
        )
        score += 14 * self._count_query_token_overlap(
            self._resolve_section_title(chunk),
            query_structural_cues,
        )
        score += 20 * self._count_legal_intent_phrase_overlap(
            self._resolve_section_title(chunk),
            query_structural_cues,
        )
        score += 10 * self._count_query_token_overlap(
            context_metadata.document_title,
            query_structural_cues,
        )
        score += 6 * self._count_query_token_overlap(
            " ".join(context_metadata.parent_structure),
            query_structural_cues,
        )
        score += 12 * self._count_legal_intent_phrase_overlap(
            " ".join(context_metadata.parent_structure),
            query_structural_cues,
        )

        score += 40 * self._count_phrase_matches(
            context_metadata.article_title,
            query_structural_cues,
        )
        score += 30 * self._count_phrase_matches(
            context_metadata.document_title,
            query_structural_cues,
        )

        normalized_document_title = _normalize_match_text(
            context_metadata.document_title
        )
        normalized_article_title = _normalize_match_text(context_metadata.article_title)

        if (
            normalized_document_title
            and normalized_document_title in query_structural_cues.document_titles
        ):
            score += 90

        if (
            normalized_article_title
            and normalized_article_title in query_structural_cues.article_titles
        ):
            score += 80

        return score

    def _score_legal_content_alignment(
        self,
        *,
        chunk: RetrievedChunkResult,
        query_structural_cues: _QueryStructuralCues,
    ) -> int:
        """
        Score deterministic legal-content overlap between the query and a chunk.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Retrieved chunk candidate being ranked for final context packing.

        query_structural_cues : _QueryStructuralCues
            Deterministic structural cues extracted from the semantic query.

        Returns
        -------
        int
            Integer content score used only after stronger structural signals.
        """

        if not query_structural_cues.lexical_tokens:
            return 0

        chunk_tokens = _tokenize_match_text(chunk.text)
        body_token_overlap = len(chunk_tokens & query_structural_cues.lexical_tokens)
        body_token_occurrences = self._count_query_token_occurrences(
            chunk.text,
            query_structural_cues,
        )
        body_phrase_overlap = self._count_legal_intent_phrase_overlap(
            chunk.text,
            query_structural_cues,
        )
        structural_signal_text = self._build_legal_content_signal_text(chunk)
        structural_token_overlap = self._count_query_token_overlap(
            structural_signal_text,
            query_structural_cues,
        )
        structural_phrase_overlap = self._count_legal_intent_phrase_overlap(
            structural_signal_text,
            query_structural_cues,
        )
        article_reference_score = 0

        for article_number in query_structural_cues.article_numbers:
            if re.search(
                rf"\bart(?:igo)?\.?\s*{re.escape(article_number)}\b",
                _normalize_match_text(chunk.text),
                re.IGNORECASE,
            ):
                article_reference_score += 12

        return (
            (4 * body_token_overlap)
            + (5 * body_token_occurrences)
            + (10 * body_phrase_overlap)
            + (6 * structural_token_overlap)
            + (18 * structural_phrase_overlap)
            + article_reference_score
        )

    def _count_query_token_occurrences(
        self,
        legal_text: str,
        query_structural_cues: _QueryStructuralCues,
    ) -> int:
        """
        Count repeated query-token occurrences in one legal text fragment.

        Parameters
        ----------
        legal_text : str
            Legal chunk text used for deterministic content scoring.

        query_structural_cues : _QueryStructuralCues
            Query cues containing high-signal lexical tokens.

        Returns
        -------
        int
            Bounded occurrence count that preserves repeated operative terms
            without allowing long chunks to dominate solely by length.
        """

        if not legal_text or not query_structural_cues.lexical_tokens:
            return 0

        occurrence_count = 0
        normalized_tokens = _ordered_match_tokens(legal_text)
        for query_token in query_structural_cues.lexical_tokens:
            occurrence_count += min(3, normalized_tokens.count(query_token))

        return occurrence_count

    def _build_legal_content_signal_text(
        self,
        chunk: RetrievedChunkResult,
    ) -> str:
        """
        Build legal-anchor text used for content-intent scoring.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Retrieved chunk whose structural metadata contributes legal intent.

        Returns
        -------
        str
            Compact metadata text containing title and hierarchy signals.
        """

        context_metadata = chunk.context_metadata
        signal_fragments = [
            context_metadata.article_title,
            self._resolve_section_title(chunk),
            " ".join(context_metadata.parent_structure),
        ]
        return " ".join(fragment for fragment in signal_fragments if fragment)

    def _count_query_token_overlap(
        self,
        structural_text: str,
        query_structural_cues: _QueryStructuralCues,
    ) -> int:
        """
        Count lexical overlap between one structural field and the query cues.

        Parameters
        ----------
        structural_text : str
            Structural metadata field attached to one retrieved chunk.

        query_structural_cues : _QueryStructuralCues
            Deterministic structural cues extracted from the semantic query.

        Returns
        -------
        int
            Number of shared high-signal tokens.
        """

        if not structural_text or not query_structural_cues.lexical_tokens:
            return 0

        return len(
            _tokenize_match_text(structural_text)
            & query_structural_cues.lexical_tokens
        )

    def _count_phrase_matches(
        self,
        structural_text: str,
        query_structural_cues: _QueryStructuralCues,
    ) -> int:
        """
        Count direct phrase matches for one structural field inside the query.

        Parameters
        ----------
        structural_text : str
            Structural metadata field attached to one retrieved chunk.

        query_structural_cues : _QueryStructuralCues
            Deterministic structural cues extracted from the semantic query.

        Returns
        -------
        int
            One when the normalized structural field appears in the query,
            otherwise zero.
        """

        normalized_structural_text = _normalize_match_text(structural_text)
        if (
            not normalized_structural_text
            or not query_structural_cues.normalized_query_text
        ):
            return 0

        if normalized_structural_text in query_structural_cues.normalized_query_text:
            return 1

        return 0

    def _count_legal_intent_phrase_overlap(
        self,
        legal_text: str,
        query_structural_cues: _QueryStructuralCues,
    ) -> int:
        """
        Count phrase-level overlap between legal text and query intent cues.

        Parameters
        ----------
        legal_text : str
            Legal chunk text or structural metadata field.

        query_structural_cues : _QueryStructuralCues
            Query cues containing normalized legal-intent phrases.

        Returns
        -------
        int
            Number of shared phrase-level normative intent cues.
        """

        if not legal_text or not query_structural_cues.legal_intent_phrases:
            return 0

        return len(
            _extract_legal_intent_phrases(legal_text)
            & query_structural_cues.legal_intent_phrases
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

    def _resolve_candidate_pool_size(
        self,
        *,
        top_k: int,
        max_chunks: int,
        route_decision: Optional[RetrievalRouteDecision],
    ) -> int:
        """
        Resolve the ranked candidate pool considered before final packing.

        Parameters
        ----------
        top_k : int
            Maximum number of ranked retrieval results available to consider.

        max_chunks : int
            Maximum number of chunks allowed in the final packed context.

        route_decision : Optional[RetrievalRouteDecision]
            Optional router decision carrying a route-specific candidate pool.

        Returns
        -------
        int
            Ranked candidate pool size kept broader than the final output size
            whenever the configured retrieval settings allow it.
        """

        configured_candidate_pool_size = self._resolve_route_candidate_pool_size(
            route_decision
        )
        if configured_candidate_pool_size is None:
            configured_candidate_pool_size = self._resolve_positive_limit(
                value=self.settings.retrieval_candidate_pool_size,
                default_value=self.settings.retrieval_candidate_pool_size,
                label="candidate_pool_size",
            )

        return min(top_k, max(max_chunks, configured_candidate_pool_size))

    def _focus_candidate_chunks_on_dominant_anchor(
        self,
        candidate_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
    ) -> List[_PreparedChunk]:
        """
        Keep the final context focused on one dominant legal anchor when safe.

        Parameters
        ----------
        candidate_chunks : Sequence[_PreparedChunk]
            Ranked legal candidates available after route-aware ordering.

        query_structural_cues : _QueryStructuralCues
            Query cues used to decide whether the question asks for one legal
            answer or a comparative set of anchors.

        Returns
        -------
        List[_PreparedChunk]
            Original candidates, or a same-anchor subset when one legal anchor
            is strong enough to answer a single-intent question.
        """

        if (
            len(candidate_chunks) < 2
            or query_structural_cues.comparative
            or query_structural_cues.article_numbers
        ):
            return list(candidate_chunks)

        dominant_anchor = self._resolve_dominant_legal_anchor(
            candidate_chunks,
            query_structural_cues,
        )
        if not dominant_anchor:
            return list(candidate_chunks)

        focused_chunks = [
            prepared_chunk
            for prepared_chunk in candidate_chunks
            if self._build_legal_anchor(prepared_chunk.chunk) == dominant_anchor
        ]

        if not focused_chunks:
            return list(candidate_chunks)

        if not self.settings.retrieval_context_single_intent_compaction_enabled:
            return focused_chunks

        primary_anchor_limit = max(
            1,
            int(self.settings.retrieval_context_primary_anchor_max_count),
        )
        supporting_anchor_limit = max(
            0,
            int(self.settings.retrieval_context_single_intent_max_supporting_chunks),
        )
        focused_chunk_limit = primary_anchor_limit + supporting_anchor_limit

        return focused_chunks[:focused_chunk_limit]

    def _resolve_dominant_legal_anchor(
        self,
        candidate_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
    ) -> str:
        """
        Resolve the strongest legal anchor for a single-intent legal question.

        Parameters
        ----------
        candidate_chunks : Sequence[_PreparedChunk]
            Candidate chunks already ordered for context selection.

        query_structural_cues : _QueryStructuralCues
            Query cues used for legal-intent scoring.

        Returns
        -------
        str
            Dominant legal anchor label, or an empty string when the evidence
            should remain multi-anchor and be classified as ambiguous.
        """

        anchor_scores: dict[str, float] = {}

        for prepared_chunk in candidate_chunks:
            anchor = self._build_legal_anchor(prepared_chunk.chunk)
            if not anchor:
                continue

            chunk_score = self._score_legal_anchor_dominance(
                prepared_chunk.chunk,
                query_structural_cues,
            )
            anchor_scores[anchor] = max(anchor_scores.get(anchor, 0.0), chunk_score)

        if len(anchor_scores) < 2:
            return ""

        ordered_anchor_scores = sorted(
            anchor_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_anchor, top_score = ordered_anchor_scores[0]
        second_score = ordered_anchor_scores[1][1]

        if top_score < _DOMINANT_ANCHOR_MIN_SCORE:
            return ""

        if top_score - second_score >= _DOMINANT_ANCHOR_MIN_MARGIN:
            return top_anchor

        top_chunk = next(
            (
                prepared_chunk.chunk
                for prepared_chunk in candidate_chunks
                if self._build_legal_anchor(prepared_chunk.chunk) == top_anchor
            ),
            None,
        )
        if top_chunk and self._has_default_rule_alignment(
            top_chunk,
            query_structural_cues,
        ):
            return top_anchor

        return ""

    def _score_legal_anchor_dominance(
        self,
        chunk: RetrievedChunkResult,
        query_structural_cues: _QueryStructuralCues,
    ) -> float:
        """
        Score whether one chunk's legal anchor should dominate final packing.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Candidate chunk being scored at legal-anchor level.

        query_structural_cues : _QueryStructuralCues
            Query cues used for legal-intent scoring.

        Returns
        -------
        float
            Combined deterministic score for single-anchor focus.
        """

        return float(
            self._score_explicit_target_alignment(
                chunk=chunk,
                query_structural_cues=query_structural_cues,
            )
            + self._score_structural_alignment(
                chunk=chunk,
                query_structural_cues=query_structural_cues,
            )
            + self._score_legal_content_alignment(
                chunk=chunk,
                query_structural_cues=query_structural_cues,
            )
            + self._score_default_rule_alignment(
                chunk,
                query_structural_cues,
            )
        )

    def _has_default_rule_alignment(
        self,
        chunk: RetrievedChunkResult,
        query_structural_cues: _QueryStructuralCues,
    ) -> bool:
        """
        Check whether a chunk represents the default legal rule requested.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Candidate chunk whose legal anchor is being assessed.

        query_structural_cues : _QueryStructuralCues
            Query cues used to detect special-case terms.

        Returns
        -------
        bool
            `True` when the query has no special-case scope and the chunk
            exposes a general/default legal-rule anchor.
        """

        return self._score_default_rule_alignment(chunk, query_structural_cues) > 0

    def _score_default_rule_alignment(
        self,
        chunk: RetrievedChunkResult,
        query_structural_cues: _QueryStructuralCues,
    ) -> int:
        """
        Score generic legal-rule alignment for same-document competition.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Candidate chunk being checked for default-rule alignment.

        query_structural_cues : _QueryStructuralCues
            Query cues used to avoid default-rule bias for special cases.

        Returns
        -------
        int
            Positive score when a generic query aligns with a general/default
            legal anchor and not with a special-case anchor.
        """

        if not query_structural_cues.lexical_tokens:
            return 0

        query_scope_tokens = query_structural_cues.lexical_tokens
        if query_scope_tokens & _SPECIAL_SCOPE_TOKENS:
            return 0

        anchor_text = self._build_legal_content_signal_text(chunk)
        anchor_tokens = _tokenize_match_text(anchor_text)
        if not anchor_tokens or not (anchor_tokens & _GENERAL_SCOPE_TOKENS):
            return 0

        topic_overlap = len(query_scope_tokens & (anchor_tokens - _GENERAL_SCOPE_TOKENS))
        if topic_overlap < 2:
            return 0

        return 75 + (10 * min(topic_overlap, 4))

    def _resolve_route_candidate_pool_size(
        self,
        route_decision: Optional[RetrievalRouteDecision],
    ) -> Optional[int]:
        """
        Read the route-specific candidate pool when the router provides one.

        Parameters
        ----------
        route_decision : Optional[RetrievalRouteDecision]
            Optional router decision attached to the context build request.

        Returns
        -------
        Optional[int]
            Positive route candidate pool size, otherwise `None`.
        """

        if not isinstance(route_decision, RetrievalRouteDecision):
            return None

        raw_candidate_pool_size = route_decision.metadata.get("candidate_pool_size")
        try:
            candidate_pool_size = int(raw_candidate_pool_size)
        except (TypeError, ValueError):
            return None

        if candidate_pool_size <= 0:
            return None

        return candidate_pool_size

    def _pack_chunks_within_budget(
        self,
        *,
        chunks: Sequence[_PreparedChunk],
        max_chunks: int,
        max_characters: int,
        query_structural_cues: _QueryStructuralCues,
    ) -> Tuple[List[_PreparedChunk], str, bool, int, int]:
        """
        Pack ordered chunks into one bounded context string.

        Parameters
        ----------
        chunks : Sequence[_PreparedChunk]
            Ordered chunk candidates that already passed selection filters.

        max_chunks : int
            Maximum number of chunks allowed in the final packed context.

        max_characters : int
            Maximum character budget available for the packed context.

        query_structural_cues : _QueryStructuralCues
            Deterministic query cues used to explain selected chunks.

        Returns
        -------
        Tuple[List[_PreparedChunk], str, bool, int, int]
            Selected chunks, final context text, truncation flag, and count of
            chunks omitted because of the character budget or max-chunk limit.
        """

        selected_chunks: List[_PreparedChunk] = []
        context_parts: List[str] = []
        character_count = 0
        truncated = False
        budget_omitted_count = 0
        max_chunks_omitted_count = 0

        for source_index, prepared_chunk in enumerate(chunks, start=1):
            if len(selected_chunks) >= max_chunks:
                max_chunks_omitted_count = len(chunks) - len(selected_chunks)
                break

            candidate_block = self._format_context_block(
                chunk=prepared_chunk.chunk,
                source_index=source_index,
                selection_reason=self._build_selection_reason(
                    prepared_chunk.chunk,
                    query_structural_cues,
                ),
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

        return (
            selected_chunks,
            "".join(context_parts),
            truncated,
            budget_omitted_count,
            max_chunks_omitted_count,
        )

    def _format_context_block(
        self,
        *,
        chunk: RetrievedChunkResult,
        source_index: int,
        selection_reason: str = "",
    ) -> str:
        """
        Format one retrieved chunk into the final grounding payload.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Selected chunk to serialize.

        source_index : int
            One-based source number used in the packed context.

        selection_reason : str
            Compact deterministic reason explaining why the chunk was kept.

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

        header_fields.extend(self._build_structural_header_fields(chunk))

        if selection_reason:
            header_fields.append(f"selection_reason={selection_reason}")

        page_range = self._resolve_page_range(chunk)
        if page_range:
            header_fields.append(f"pages={page_range}")

        return f"[{' | '.join(header_fields)}]\n{chunk.text}"

    def _build_structural_header_fields(
        self,
        chunk: RetrievedChunkResult,
    ) -> List[str]:
        """
        Build explicit structural header fields for one selected chunk.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Selected chunk whose metadata is serialized into the context.

        Returns
        -------
        List[str]
            Ordered explicit structural fields preserved in the final context.
        """

        structural_fields: List[str] = []
        context_metadata = chunk.context_metadata

        legal_anchor = self._build_legal_anchor(chunk)
        if legal_anchor:
            structural_fields.append(f"legal_anchor={legal_anchor}")

        if context_metadata.document_title:
            structural_fields.append(
                f"document_title={context_metadata.document_title}"
            )

        if (
            self.settings.retrieval_context_include_article_number
            and context_metadata.article_number
        ):
            structural_fields.append(
                f"article_number={context_metadata.article_number}"
            )

        if (
            self.settings.retrieval_context_include_article_title
            and context_metadata.article_title
        ):
            structural_fields.append(
                f"article_title={context_metadata.article_title}"
            )

        section_title = self._resolve_section_title(chunk)
        if section_title:
            structural_fields.append(f"section_title={section_title}")

        if (
            self.settings.retrieval_context_include_parent_structure
            and context_metadata.parent_structure
        ):
            structural_fields.append(
                "parent_structure="
                + " > ".join(context_metadata.parent_structure)
            )

        return structural_fields

    def _build_legal_anchor(
        self,
        chunk: RetrievedChunkResult,
    ) -> str:
        """
        Build one compact article-level anchor for the serialized context.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Selected chunk whose structural metadata will be summarized.

        Returns
        -------
        str
            Compact legal anchor exposing the clearest available structural id.
        """

        context_metadata = chunk.context_metadata
        anchor_segments: List[str] = []

        if context_metadata.document_title:
            anchor_segments.append(context_metadata.document_title)

        article_anchor = ""
        if context_metadata.article_number and context_metadata.article_title:
            article_anchor = (
                f"Article {context_metadata.article_number} - "
                f"{context_metadata.article_title}"
            )
        elif context_metadata.article_number:
            article_anchor = f"Article {context_metadata.article_number}"
        elif context_metadata.article_title:
            article_anchor = context_metadata.article_title
        else:
            article_anchor = self._resolve_section_title(chunk)

        if article_anchor:
            anchor_segments.append(article_anchor)

        return " > ".join(anchor_segments)

    def _build_selection_reason(
        self,
        chunk: RetrievedChunkResult,
        query_structural_cues: _QueryStructuralCues,
    ) -> str:
        """
        Build a compact deterministic explanation for context inclusion.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Selected chunk whose inclusion is being explained.

        query_structural_cues : _QueryStructuralCues
            Deterministic query cues used during ranking.

        Returns
        -------
        str
            Short semicolon-separated selection explanation.
        """

        reasons: List[str] = []
        context_metadata = chunk.context_metadata
        has_query_cues = any(
            [
                query_structural_cues.article_numbers,
                query_structural_cues.document_titles,
                query_structural_cues.article_titles,
                query_structural_cues.lexical_tokens,
                query_structural_cues.legal_intent_phrases,
            ]
        )

        if not has_query_cues:
            return ""

        if (
            context_metadata.article_number
            and context_metadata.article_number.lower()
            in query_structural_cues.article_numbers
        ):
            reasons.append("matched_article_number")

        if self._matches_target_title(
            context_metadata.document_title,
            query_structural_cues.document_titles,
        ):
            reasons.append("matched_document_title")

        if self._matches_target_title(
            context_metadata.article_title,
            query_structural_cues.article_titles,
        ):
            reasons.append("matched_article_title")

        if self._score_legal_content_alignment(
            chunk=chunk,
            query_structural_cues=query_structural_cues,
        ):
            reasons.append("matched_legal_terms")

        return ";".join(reasons[:3])

    def _matches_target_title(
        self,
        value: str,
        target_titles: FrozenSet[str],
    ) -> bool:
        """
        Check whether one normalized title matches explicit target titles.

        Parameters
        ----------
        value : str
            Candidate title attached to a retrieved chunk.

        target_titles : FrozenSet[str]
            Normalized title targets extracted from query metadata or routing.

        Returns
        -------
        bool
            `True` when the title is an explicit legal target.
        """

        normalized_value = _normalize_match_text(value)
        return bool(normalized_value and normalized_value in target_titles)

    def _resolve_primary_anchor_selection(
        self,
        *,
        selected_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
    ) -> _PrimaryAnchorSelection:
        """
        Resolve the governing legal anchor from selected context chunks.

        Parameters
        ----------
        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for final context packing.

        query_structural_cues : _QueryStructuralCues
            Query cues used to score legal-anchor alignment.

        Returns
        -------
        _PrimaryAnchorSelection
            Primary anchor metadata, or an empty selection when no legal anchor
            is available.
        """

        anchor_order: dict[str, int] = {}
        anchor_chunk_ids: dict[str, List[str]] = {}

        for selected_index, selected_chunk in enumerate(selected_chunks):
            anchor = self._build_legal_anchor(selected_chunk.chunk)
            if not anchor:
                continue

            anchor_order.setdefault(anchor, selected_index)
            anchor_chunk_ids.setdefault(anchor, []).append(
                selected_chunk.chunk.chunk_id
            )

        if not anchor_chunk_ids:
            return _PrimaryAnchorSelection()

        anchor_scores = self._build_selected_anchor_scores(
            selected_chunks,
            query_structural_cues,
        )
        ordered_anchors = sorted(
            anchor_chunk_ids,
            key=lambda anchor: (
                -anchor_scores.get(anchor, 0.0),
                anchor_order.get(anchor, 10**9),
                anchor,
            ),
        )
        primary_anchor = ordered_anchors[0]
        primary_anchor_limit = max(
            1,
            int(self.settings.retrieval_context_primary_anchor_max_count),
        )
        selected_anchor_chunk_ids = anchor_chunk_ids[primary_anchor]

        return _PrimaryAnchorSelection(
            anchor=primary_anchor,
            primary_chunk_ids=tuple(
                selected_anchor_chunk_ids[:primary_anchor_limit]
            ),
            supporting_chunk_ids=tuple(
                selected_anchor_chunk_ids[primary_anchor_limit:]
            ),
            score=float(anchor_scores.get(primary_anchor, 0.0)),
        )

    def _classify_evidence_quality(
        self,
        *,
        candidate_chunks: Sequence[_PreparedChunk],
        selected_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
        primary_anchor_selection: _PrimaryAnchorSelection,
    ) -> EvidenceQualityClassification:
        """
        Classify close legal competitors among selected context candidates.

        Parameters
        ----------
        candidate_chunks : Sequence[_PreparedChunk]
            Ranked chunks considered for final packing.

        selected_chunks : Sequence[_PreparedChunk]
            Chunks that survived final packing.

        query_structural_cues : _QueryStructuralCues
            Deterministic query cues used during ranking and explanation.

        primary_anchor_selection : _PrimaryAnchorSelection
            Selected primary legal anchor used to grade omitted competitors.

        Returns
        -------
        EvidenceQualityClassification
            Typed ambiguity and conflict signals for downstream routing.
        """

        selected_ids = {
            prepared_chunk.chunk.chunk_id for prepared_chunk in selected_chunks
        }
        competitor_chunks = [
            prepared_chunk
            for prepared_chunk in candidate_chunks
            if prepared_chunk.chunk.chunk_id not in selected_ids
            and self._is_close_legal_competitor(
                candidate=prepared_chunk.chunk,
                selected_chunks=selected_chunks,
                query_structural_cues=query_structural_cues,
            )
        ]
        competitor_chunks = sorted(
            competitor_chunks,
            key=self._build_competitor_sort_key,
        )
        competitor_assessments = [
            self._assess_legal_competitor(
                candidate=prepared_chunk.chunk,
                selected_chunks=selected_chunks,
                query_structural_cues=query_structural_cues,
                primary_anchor_selection=primary_anchor_selection,
            )
            for prepared_chunk in competitor_chunks
        ]

        close_competitor_ids = [
            prepared_chunk.chunk.chunk_id for prepared_chunk in competitor_chunks
        ]
        supportive_competitor_ids = [
            assessment.chunk_id
            for assessment in competitor_assessments
            if assessment.category == "supportive"
        ]
        alternative_scope_competitor_ids = [
            assessment.chunk_id
            for assessment in competitor_assessments
            if assessment.category == "alternative_scope"
        ]
        blocking_conflict_ids = [
            assessment.chunk_id
            for assessment in competitor_assessments
            if assessment.category == "blocking"
        ]
        conflicting_ids = [
            prepared_chunk.chunk.chunk_id
            for prepared_chunk in competitor_chunks
            if prepared_chunk.chunk.chunk_id in blocking_conflict_ids
        ]
        selected_anchor_conflict_ids = self._find_conflicting_selected_anchor_ids(
            selected_chunks=selected_chunks,
            query_structural_cues=query_structural_cues,
        )
        conflicting_ids.extend(
            chunk_id
            for chunk_id in selected_anchor_conflict_ids
            if chunk_id not in conflicting_ids
        )

        reasons: List[str] = []
        if close_competitor_ids:
            reasons.append("close_legal_competitors_detected")
        if supportive_competitor_ids:
            reasons.append("supportive_legal_competitors_detected")
        if alternative_scope_competitor_ids:
            reasons.append("alternative_scope_competitors_detected")
        if conflicting_ids:
            reasons.append("blocking_legal_competitors_detected")
        if selected_anchor_conflict_ids:
            reasons.append("ambiguous_selected_legal_anchors_detected")

        return EvidenceQualityClassification(
            strength="strong" if selected_chunks else "empty",
            ambiguity="ambiguous" if close_competitor_ids else "clear",
            conflict="conflicting" if conflicting_ids else "none",
            sufficient_for_answer=bool(selected_chunks) and not conflicting_ids,
            close_competitor_chunk_ids=close_competitor_ids,
            conflicting_chunk_ids=conflicting_ids,
            reasons=reasons,
            metadata={
                "selected_chunk_ids": list(selected_ids),
                "candidate_chunk_count": len(candidate_chunks),
                "primary_anchor": primary_anchor_selection.anchor,
                "primary_anchor_chunk_ids": list(
                    primary_anchor_selection.primary_chunk_ids
                ),
                "supporting_anchor_chunk_ids": list(
                    primary_anchor_selection.supporting_chunk_ids
                ),
                "primary_anchor_score": primary_anchor_selection.score,
                "supportive_competitor_chunk_ids": supportive_competitor_ids,
                "alternative_scope_competitor_chunk_ids": (
                    alternative_scope_competitor_ids
                ),
                "blocking_conflict_chunk_ids": blocking_conflict_ids,
                "competitor_assessments": [
                    {
                        "chunk_id": assessment.chunk_id,
                        "anchor": assessment.anchor,
                        "category": assessment.category,
                        "score": assessment.score,
                        "relationship": assessment.relationship,
                    }
                    for assessment in competitor_assessments
                ],
                "consequence_competitor_chunk_ids": [
                    assessment.chunk_id
                    for assessment in competitor_assessments
                    if assessment.relationship == "consequence_rule"
                ],
                "neighboring_non_governing_chunk_ids": [
                    assessment.chunk_id
                    for assessment in competitor_assessments
                    if assessment.relationship == "neighboring_non_governing_rule"
                ],
                "selected_anchor_scores": self._build_selected_anchor_score_metadata(
                    selected_chunks,
                    query_structural_cues,
                ),
            },
        )

    def _find_conflicting_selected_anchor_ids(
        self,
        *,
        selected_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
    ) -> List[str]:
        """
        Find selected chunks that keep unresolved same-document legal ambiguity.

        Parameters
        ----------
        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for final context.

        query_structural_cues : _QueryStructuralCues
            Query cues used to score legal-intent alignment.

        Returns
        -------
        List[str]
            Selected chunk ids from non-leading anchors when top anchors remain
            too close for normal generation.
        """

        if (
            len(selected_chunks) < 2
            or query_structural_cues.comparative
            or query_structural_cues.article_numbers
        ):
            return []

        anchor_scores = self._build_selected_anchor_scores(
            selected_chunks,
            query_structural_cues,
        )
        if len(anchor_scores) < 2:
            return []

        ordered_anchor_scores = sorted(
            anchor_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_anchor, top_score = ordered_anchor_scores[0]
        second_score = ordered_anchor_scores[1][1]
        minimum_margin = max(
            15.0,
            top_score * float(self.settings.retrieval_evidence_conflict_score_margin),
        )

        if top_score - second_score > minimum_margin:
            return []

        conflicting_ids: List[str] = []
        for selected_chunk in selected_chunks:
            selected_anchor = self._build_legal_anchor(selected_chunk.chunk)
            if selected_anchor and selected_anchor != top_anchor:
                conflicting_ids.append(selected_chunk.chunk.chunk_id)

        return conflicting_ids

    def _build_selected_anchor_scores(
        self,
        selected_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
    ) -> dict[str, float]:
        """
        Score each selected legal anchor against the query intent.

        Parameters
        ----------
        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for final context.

        query_structural_cues : _QueryStructuralCues
            Query cues used to score legal-intent alignment.

        Returns
        -------
        dict[str, float]
            Highest alignment score observed for each selected legal anchor.
        """

        anchor_scores: dict[str, float] = {}

        for selected_chunk in selected_chunks:
            anchor = self._build_legal_anchor(selected_chunk.chunk)
            if not anchor:
                continue

            chunk_score = float(
                self._score_explicit_target_alignment(
                    chunk=selected_chunk.chunk,
                    query_structural_cues=query_structural_cues,
                )
                + self._score_structural_alignment(
                    chunk=selected_chunk.chunk,
                    query_structural_cues=query_structural_cues,
                )
                + self._score_legal_content_alignment(
                    chunk=selected_chunk.chunk,
                    query_structural_cues=query_structural_cues,
                )
            )
            anchor_scores[anchor] = max(anchor_scores.get(anchor, 0.0), chunk_score)

        return anchor_scores

    def _build_selected_anchor_score_metadata(
        self,
        selected_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
    ) -> dict[str, float]:
        """
        Build deterministic selected-anchor score metadata.

        Parameters
        ----------
        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for final context.

        query_structural_cues : _QueryStructuralCues
            Query cues used to score legal-intent alignment.

        Returns
        -------
        dict[str, float]
            Selected legal-anchor scores sorted by anchor label.
        """

        return dict(
            sorted(
                self._build_selected_anchor_scores(
                    selected_chunks,
                    query_structural_cues,
                ).items()
            )
        )

    def _build_competitor_sort_key(
        self,
        prepared_chunk: _PreparedChunk,
    ) -> Tuple[int, str, str]:
        """
        Build a stable legal-anchor ordering key for omitted competitors.

        Parameters
        ----------
        prepared_chunk : _PreparedChunk
            Omitted competitor chunk being surfaced in evidence metadata.

        Returns
        -------
        Tuple[int, str, str]
            Numeric article ordering when available, then article label and
            chunk id for deterministic metadata output.
        """

        article_number = prepared_chunk.chunk.context_metadata.article_number.lower()
        numeric_match = re.match(r"^(\d+)", article_number)
        numeric_article = int(numeric_match.group(1)) if numeric_match else 10**9

        return (
            numeric_article,
            article_number,
            prepared_chunk.chunk.chunk_id,
        )

    def _assess_legal_competitor(
        self,
        *,
        candidate: RetrievedChunkResult,
        selected_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
        primary_anchor_selection: _PrimaryAnchorSelection,
    ) -> _CompetitorAssessment:
        """
        Grade one omitted legal competitor against the primary evidence anchor.

        Parameters
        ----------
        candidate : RetrievedChunkResult
            Omitted candidate chunk being classified.

        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for the final context.

        query_structural_cues : _QueryStructuralCues
            Query cues used to score legal-intent alignment.

        primary_anchor_selection : _PrimaryAnchorSelection
            Primary legal anchor selected from final context.

        Returns
        -------
        _CompetitorAssessment
            Deterministic competitor category and score.
        """

        candidate_anchor = self._build_legal_anchor(candidate)
        candidate_score = float(
            self._score_structural_alignment(
                chunk=candidate,
                query_structural_cues=query_structural_cues,
            )
            + self._score_legal_content_alignment(
                chunk=candidate,
                query_structural_cues=query_structural_cues,
            )
        )

        if self._shares_selected_article_anchor(candidate, selected_chunks):
            return _CompetitorAssessment(
                chunk_id=candidate.chunk_id,
                anchor=candidate_anchor,
                category="supportive",
                score=candidate_score,
                relationship="supporting_same_anchor",
            )

        if self._selected_context_matches_explicit_article(
            selected_chunks,
            query_structural_cues,
        ):
            return _CompetitorAssessment(
                chunk_id=candidate.chunk_id,
                anchor=candidate_anchor,
                category="alternative_scope",
                score=candidate_score,
                relationship=self._resolve_competitor_relationship(
                    candidate,
                    selected_chunks,
                ),
            )

        if (
            not candidate_anchor
            or candidate_anchor == primary_anchor_selection.anchor
        ):
            return _CompetitorAssessment(
                chunk_id=candidate.chunk_id,
                anchor=candidate_anchor,
                category="supportive",
                score=candidate_score,
                relationship="supporting_same_anchor",
            )

        if not primary_anchor_selection.anchor:
            return _CompetitorAssessment(
                chunk_id=candidate.chunk_id,
                anchor=candidate_anchor,
                category="alternative_scope",
                score=candidate_score,
                relationship="alternative_regime",
            )

        if not self._has_same_document_anchor_with_any_selected(
            candidate,
            selected_chunks,
        ):
            return _CompetitorAssessment(
                chunk_id=candidate.chunk_id,
                anchor=candidate_anchor,
                category="alternative_scope",
                score=candidate_score,
                relationship="alternative_regime",
            )

        relationship = self._resolve_competitor_relationship(
            candidate,
            selected_chunks,
        )

        if query_structural_cues.lexical_tokens & _SPECIAL_SCOPE_TOKENS:
            return _CompetitorAssessment(
                chunk_id=candidate.chunk_id,
                anchor=candidate_anchor,
                category="blocking",
                score=candidate_score,
                relationship=relationship,
            )

        if self._is_blocking_legal_competitor(
            candidate_score=candidate_score,
            primary_anchor_score=primary_anchor_selection.score,
        ):
            return _CompetitorAssessment(
                chunk_id=candidate.chunk_id,
                anchor=candidate_anchor,
                category="blocking",
                score=candidate_score,
                relationship=relationship,
            )

        return _CompetitorAssessment(
            chunk_id=candidate.chunk_id,
            anchor=candidate_anchor,
            category="alternative_scope",
            score=candidate_score,
            relationship=relationship,
        )

    def _resolve_competitor_relationship(
        self,
        candidate: RetrievedChunkResult,
        selected_chunks: Sequence[_PreparedChunk],
    ) -> str:
        """
        Resolve a refined legal relationship for one omitted competitor.

        Parameters
        ----------
        candidate : RetrievedChunkResult
            Omitted chunk being described relative to the selected evidence.

        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for the final context.

        Returns
        -------
        str
            Stable relationship label used in evidence metadata and diagnostics.
        """

        if self._shares_selected_article_anchor(candidate, selected_chunks):
            return "supporting_same_anchor"

        if not self._has_same_document_anchor_with_any_selected(
            candidate,
            selected_chunks,
        ):
            return "alternative_regime"

        if self._is_consequence_rule(candidate):
            return "consequence_rule"

        if self._is_neighboring_non_governing_rule(candidate, selected_chunks):
            return "neighboring_non_governing_rule"

        return "alternative_regime"

    def _is_blocking_legal_competitor(
        self,
        *,
        candidate_score: float,
        primary_anchor_score: float,
    ) -> bool:
        """
        Decide whether a distinct same-document competitor should block answers.

        Parameters
        ----------
        candidate_score : float
            Legal-intent score for the omitted competitor.

        primary_anchor_score : float
            Legal-intent score for the selected primary anchor.

        Returns
        -------
        bool
            `True` when the competitor is close enough to be a blocking
            conflict under the configured evidence margin.
        """

        if primary_anchor_score <= 0:
            return candidate_score > 0

        blocking_margin = max(
            1.0,
            primary_anchor_score
            * float(self.settings.retrieval_evidence_blocking_conflict_score_margin),
        )
        non_blocking_margin = max(
            1.0,
            primary_anchor_score
            * float(
                self.settings.retrieval_evidence_non_blocking_competitor_score_margin
            ),
        )
        effective_margin = max(blocking_margin, non_blocking_margin)

        return candidate_score >= primary_anchor_score - effective_margin

    def _is_close_legal_competitor(
        self,
        *,
        candidate: RetrievedChunkResult,
        selected_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
    ) -> bool:
        """
        Determine whether one omitted chunk is a close legal competitor.

        Parameters
        ----------
        candidate : RetrievedChunkResult
            Omitted candidate being evaluated.

        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for the final context.

        query_structural_cues : _QueryStructuralCues
            Deterministic query cues used during ranking.

        Returns
        -------
        bool
            `True` when the omitted chunk remains relevant enough to surface.
        """

        if not selected_chunks:
            return False

        candidate_score = self._score_structural_alignment(
            chunk=candidate,
            query_structural_cues=query_structural_cues,
        ) + self._score_legal_content_alignment(
            chunk=candidate,
            query_structural_cues=query_structural_cues,
        )

        if candidate_score > 0:
            return True

        return any(
            self._has_overlapping_legal_anchor(candidate, selected_chunk.chunk)
            for selected_chunk in selected_chunks
        )

    def _is_consequence_rule(
        self,
        candidate: RetrievedChunkResult,
    ) -> bool:
        """
        Detect whether one competitor looks like a consequence-oriented rule.

        Parameters
        ----------
        candidate : RetrievedChunkResult
            Omitted chunk being classified.

        Returns
        -------
        bool
            `True` when the chunk signals enforcement, arrears, or consequence
            semantics instead of a governing base rule.
        """

        signal_tokens = _tokenize_match_text(
            f"{self._build_legal_content_signal_text(candidate)} {candidate.text}"
        )
        return bool(signal_tokens & _CONSEQUENCE_SCOPE_TOKENS)

    def _is_neighboring_non_governing_rule(
        self,
        candidate: RetrievedChunkResult,
        selected_chunks: Sequence[_PreparedChunk],
    ) -> bool:
        """
        Detect a nearby same-document rule that is not the governing anchor.

        Parameters
        ----------
        candidate : RetrievedChunkResult
            Omitted chunk being classified.

        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for the final context.

        Returns
        -------
        bool
            `True` when the competitor is structurally near the selected
            article-level anchor without sharing the same article.
        """

        candidate_article_number = self._extract_numeric_article_number(candidate)
        if candidate_article_number is None:
            return False

        for selected_chunk in selected_chunks:
            if not self._has_same_document_anchor(candidate, selected_chunk.chunk):
                continue

            selected_article_number = self._extract_numeric_article_number(
                selected_chunk.chunk
            )
            if selected_article_number is None:
                continue

            if abs(candidate_article_number - selected_article_number) <= 1:
                return True

        return False

    def _has_overlapping_legal_anchor(
        self,
        candidate: RetrievedChunkResult,
        selected_chunk: RetrievedChunkResult,
    ) -> bool:
        """
        Check whether two chunks share a document or article-level legal anchor.

        Parameters
        ----------
        candidate : RetrievedChunkResult
            Omitted candidate chunk.

        selected_chunk : RetrievedChunkResult
            Selected context chunk.

        Returns
        -------
        bool
            `True` when the chunks are legal neighbors in the same context.
        """

        candidate_metadata = candidate.context_metadata
        selected_metadata = selected_chunk.context_metadata

        if (
            candidate_metadata.document_title
            and selected_metadata.document_title
            and candidate_metadata.document_title == selected_metadata.document_title
        ):
            return True

        if (
            candidate_metadata.article_number
            and selected_metadata.article_number
            and candidate_metadata.article_number == selected_metadata.article_number
        ):
            return True

        return candidate.doc_id and candidate.doc_id == selected_chunk.doc_id

    def _is_legally_different_competitor(
        self,
        *,
        candidate: RetrievedChunkResult,
        selected_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
    ) -> bool:
        """
        Detect whether a competitor points to a distinct legal anchor.

        Parameters
        ----------
        candidate : RetrievedChunkResult
            Omitted candidate chunk.

        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for the final context.

        query_structural_cues : _QueryStructuralCues
            Deterministic query cues used to recognize explicit article targets.

        Returns
        -------
        bool
            `True` when the competitor may represent conflicting legal context.
        """

        if self._shares_selected_article_anchor(candidate, selected_chunks):
            return False

        if self._selected_context_matches_explicit_article(
            selected_chunks,
            query_structural_cues,
        ):
            return False

        candidate_score = self._score_structural_alignment(
            chunk=candidate,
            query_structural_cues=query_structural_cues,
        ) + self._score_legal_content_alignment(
            chunk=candidate,
            query_structural_cues=query_structural_cues,
        )
        selected_scores = [
            self._score_structural_alignment(
                chunk=selected_chunk.chunk,
                query_structural_cues=query_structural_cues,
            )
            + self._score_legal_content_alignment(
                chunk=selected_chunk.chunk,
                query_structural_cues=query_structural_cues,
            )
            for selected_chunk in selected_chunks
        ]
        if selected_scores and max(selected_scores) > candidate_score:
            return False

        candidate_anchor = self._build_legal_anchor(candidate)
        if not candidate_anchor:
            return False

        selected_anchors = self._build_selected_legal_anchors(selected_chunks)
        if len(selected_anchors) != 1:
            return False

        return candidate_anchor not in selected_anchors

    def _shares_selected_article_anchor(
        self,
        candidate: RetrievedChunkResult,
        selected_chunks: Sequence[_PreparedChunk],
    ) -> bool:
        """
        Check whether an omitted competitor belongs to a selected article.

        Parameters
        ----------
        candidate : RetrievedChunkResult
            Omitted candidate chunk being assessed.

        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for the final context.

        Returns
        -------
        bool
            `True` when candidate and selected context share document and article.
        """

        candidate_metadata = candidate.context_metadata
        candidate_article_number = candidate_metadata.article_number.lower()
        if not candidate_article_number:
            return False

        for selected_chunk in selected_chunks:
            selected_metadata = selected_chunk.chunk.context_metadata
            selected_article_number = selected_metadata.article_number.lower()
            if candidate_article_number != selected_article_number:
                continue
            if self._has_same_document_anchor(candidate, selected_chunk.chunk):
                return True

        return False

    def _selected_context_matches_explicit_article(
        self,
        selected_chunks: Sequence[_PreparedChunk],
        query_structural_cues: _QueryStructuralCues,
    ) -> bool:
        """
        Check whether selected context satisfies an explicit article request.

        Parameters
        ----------
        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for the final context.

        query_structural_cues : _QueryStructuralCues
            Query cues containing optional requested article numbers.

        Returns
        -------
        bool
            `True` when at least one selected chunk matches the requested article.
        """

        if not query_structural_cues.article_numbers:
            return False

        return any(
            selected_chunk.chunk.context_metadata.article_number.lower()
            in query_structural_cues.article_numbers
            for selected_chunk in selected_chunks
        )

    def _has_same_document_anchor(
        self,
        first_chunk: RetrievedChunkResult,
        second_chunk: RetrievedChunkResult,
    ) -> bool:
        """
        Check whether two chunks belong to the same legal document.

        Parameters
        ----------
        first_chunk : RetrievedChunkResult
            First chunk being compared.

        second_chunk : RetrievedChunkResult
            Second chunk being compared.

        Returns
        -------
        bool
            `True` when document identifiers or titles match.
        """

        first_metadata = first_chunk.context_metadata
        second_metadata = second_chunk.context_metadata

        if first_chunk.doc_id and first_chunk.doc_id == second_chunk.doc_id:
            return True

        return bool(
            first_metadata.document_title
            and first_metadata.document_title == second_metadata.document_title
        )

    def _has_same_document_anchor_with_any_selected(
        self,
        candidate: RetrievedChunkResult,
        selected_chunks: Sequence[_PreparedChunk],
    ) -> bool:
        """
        Check whether one candidate belongs to any selected legal document.

        Parameters
        ----------
        candidate : RetrievedChunkResult
            Omitted candidate chunk being assessed.

        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for the final context.

        Returns
        -------
        bool
            `True` when the candidate shares a document anchor with selected
            evidence.
        """

        return any(
            self._has_same_document_anchor(candidate, selected_chunk.chunk)
            for selected_chunk in selected_chunks
        )

    def _build_selected_legal_anchors(
        self,
        selected_chunks: Sequence[_PreparedChunk],
    ) -> FrozenSet[str]:
        """
        Build the distinct legal anchors represented in selected context.

        Parameters
        ----------
        selected_chunks : Sequence[_PreparedChunk]
            Chunks selected for the final context.

        Returns
        -------
        FrozenSet[str]
            Distinct non-empty anchors already available to answer generation.
        """

        anchors = {
            legal_anchor
            for selected_chunk in selected_chunks
            if (legal_anchor := self._build_legal_anchor(selected_chunk.chunk))
        }
        return frozenset(anchors)

    def _extract_numeric_article_number(
        self,
        chunk: RetrievedChunkResult,
    ) -> Optional[int]:
        """
        Extract the leading numeric article number from one chunk when present.

        Parameters
        ----------
        chunk : RetrievedChunkResult
            Chunk whose structural metadata may contain an article number.

        Returns
        -------
        Optional[int]
            Leading numeric article component, or `None` when unavailable.
        """

        article_number = chunk.context_metadata.article_number.lower()
        numeric_match = re.match(r"^(\d+)", article_number)
        if not numeric_match:
            return None

        return int(numeric_match.group(1))

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
