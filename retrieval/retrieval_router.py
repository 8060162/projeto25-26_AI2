from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Pattern, Tuple

from Chunking.config.settings import PipelineSettings
from retrieval.models import RetrievalRouteDecision, UserQuestionInput


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_LEGAL_SIGNAL_STOPWORDS = frozenset(
    {
        "a",
        "as",
        "by",
        "com",
        "da",
        "das",
        "de",
        "del",
        "do",
        "dos",
        "e",
        "em",
        "for",
        "in",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "of",
        "os",
        "para",
        "por",
        "the",
        "to",
    }
)
_LEGAL_INTENT_SIGNAL_TERMS: Dict[str, Tuple[str, ...]] = {
    "payment_plan": (
        "plano pagamento",
        "pagamento prestacoes",
        "prestacoes pagamento",
        "payment plan",
    ),
    "general_payment_plan": (
        "plano geral pagamento",
        "regime geral pagamento",
        "general payment plan",
    ),
    "specific_payment_plan": (
        "plano especifico pagamento",
        "plano especial pagamento",
        "regime especifico pagamento",
        "specific payment plan",
    ),
    "regularization_plan": (
        "plano regularizacao",
        "regularizacao divida",
        "regularizacao pagamento",
        "regularizacao propinas",
    ),
    "international_student": (
        "estudante internacional",
        "estudantes internacionais",
        "aluno internacional",
        "international student",
    ),
    "installment_schedule": (
        "prestacoes",
        "pagamento faseado",
        "calendario pagamento",
        "installments",
        "payment schedule",
    ),
    "deadline_question": (
        "prazo",
        "prazos",
        "data limite",
        "deadline",
    ),
    "document_requirement_question": (
        "documentos necessarios",
        "documentacao necessaria",
        "documentos exigidos",
        "required documents",
    ),
}
_GENERIC_LEGAL_INTENT_SIGNALS = frozenset(
    {
        "deadline_question",
        "document_requirement_question",
    }
)
_LOW_SIGNAL_DOCUMENT_TOKENS = frozenset(
    {
        "alteracao",
        "aprovado",
        "artigo",
        "curso",
        "cursos",
        "das",
        "de",
        "despacho",
        "do",
        "dos",
        "ensino",
        "estudante",
        "estudantes",
        "instituto",
        "inscricao",
        "inscricoes",
        "matricula",
        "matriculas",
        "norma",
        "para",
        "pdf",
        "politecnico",
        "porto",
        "pporto",
        "prazo",
        "prazos",
        "regime",
        "regimes",
        "regulamento",
        "signed",
        "superior",
    }
)
_ARTICLE_REFERENCE_PATTERN = re.compile(
    r"\bart(?:igo)?\.?\s*(\d+[a-z]?)\b",
    re.IGNORECASE,
)
_COMPARATIVE_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"\b(?:comparar|compara|compare|comparacao|comparação)\b"),
    re.compile(
        r"\b(?:diferenca|diferença|diferencas|diferenças|difference|differences)\b"
    ),
    re.compile(r"\b(?:entre|between)\b.+\b(?:e|and|vs|versus)\b"),
    re.compile(r"\b(?:ambos|ambas|both)\b"),
)


@dataclass(frozen=True, slots=True)
class _DocumentSignalProfile:
    """
    Hold reusable document-level routing signals discovered from chunk metadata.

    Parameters
    ----------
    doc_id : str
        Stable document identifier used by storage metadata filters.
    document_title : str
        Human-readable document title from chunk metadata.
    title_tokens : FrozenSet[str]
        High-signal tokens from document titles, source names, and aliases.
    structural_tokens : FrozenSet[str]
        High-signal tokens from structural chunk metadata such as article labels.
    legal_intent_signals : FrozenSet[str]
        Generic legal-intent signals discovered from document chunks.
    """

    doc_id: str
    document_title: str
    title_tokens: FrozenSet[str]
    structural_tokens: FrozenSet[str]
    legal_intent_signals: FrozenSet[str]


def _normalize_match_text(value: str) -> str:
    """
    Normalize one text value into an accent-free comparison form.

    Parameters
    ----------
    value : str
        Raw text value.

    Returns
    -------
    str
        Lowercased alphanumeric comparison text.
    """

    normalized_value = unicodedata.normalize("NFKD", value)
    ascii_value = normalized_value.encode("ascii", "ignore").decode("ascii")
    return " ".join(_TOKEN_PATTERN.findall(ascii_value.lower()))


def _normalize_legal_signal_text(value: str) -> str:
    """
    Normalize text for legal-intent phrase matching while ignoring fillers.

    Parameters
    ----------
    value : str
        Raw text value.

    Returns
    -------
    str
        Accent-free normalized text with low-information connector tokens
        removed so intent phrases remain stable across natural phrasing.
    """

    normalized_tokens = [
        token
        for token in _normalize_match_text(value).split()
        if token not in _LEGAL_SIGNAL_STOPWORDS
    ]
    return " ".join(normalized_tokens)


def _normalize_lexical_token(token: str) -> str:
    """
    Normalize simple Portuguese plural forms for document-signal matching.

    Parameters
    ----------
    token : str
        Accent-free lowercase token extracted from query or metadata text.

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


def _tokenize_document_signal_text(value: str) -> FrozenSet[str]:
    """
    Tokenize one text fragment into high-signal document-routing tokens.

    Parameters
    ----------
    value : str
        Raw metadata or query text.

    Returns
    -------
    FrozenSet[str]
        Distinct lexical tokens suitable for generic document matching.
    """

    tokens: set[str] = set()

    for token in _normalize_match_text(value).split():
        if len(token) < 4 or token in _LOW_SIGNAL_DOCUMENT_TOKENS:
            continue

        tokens.add(token)
        normalized_token = _normalize_lexical_token(token)
        if normalized_token and normalized_token not in _LOW_SIGNAL_DOCUMENT_TOKENS:
            tokens.add(normalized_token)

    return frozenset(tokens)


def _normalize_string(value: Any) -> str:
    """
    Normalize one optional value into a stripped string.

    Parameters
    ----------
    value : Any
        Candidate string value.

    Returns
    -------
    str
        Stripped string when present, otherwise an empty string.
    """

    if isinstance(value, str):
        return value.strip()
    return ""


def _normalize_string_list(value: Any) -> List[str]:
    """
    Normalize one optional scalar or list value into ordered strings.

    Parameters
    ----------
    value : Any
        Candidate scalar or list value.

    Returns
    -------
    List[str]
        Ordered non-empty strings with duplicates removed.
    """

    if isinstance(value, str):
        candidate_values = [value]
    elif isinstance(value, list):
        candidate_values = value
    else:
        return []

    normalized_values: List[str] = []
    seen_values = set()

    for candidate_value in candidate_values:
        normalized_value = _normalize_string(candidate_value)
        comparison_value = _normalize_match_text(normalized_value)

        if normalized_value and comparison_value not in seen_values:
            normalized_values.append(normalized_value)
            seen_values.add(comparison_value)

    return normalized_values


def _extract_legal_intent_signals(value: str) -> FrozenSet[str]:
    """
    Extract generic legal-intent signals from one normalized text fragment.

    Parameters
    ----------
    value : str
        Raw query, metadata, or chunk text.

    Returns
    -------
    FrozenSet[str]
        Stable legal-intent signal names detected in the text.
    """

    normalized_value = _normalize_match_text(value)
    normalized_signal_value = _normalize_legal_signal_text(value)
    if not normalized_value and not normalized_signal_value:
        return frozenset()

    detected_signals = {
        intent_name
        for intent_name, intent_terms in _LEGAL_INTENT_SIGNAL_TERMS.items()
        if any(
            intent_term in normalized_value
            or _normalize_legal_signal_text(intent_term) in normalized_signal_value
            for intent_term in intent_terms
        )
    }
    if detected_signals & {
        "general_payment_plan",
        "specific_payment_plan",
        "regularization_plan",
        "installment_schedule",
    }:
        detected_signals.add("payment_plan")

    return frozenset(detected_signals)


def _metadata_values(
    metadata: Dict[str, Any],
    plural_key: str,
    singular_key: str,
) -> List[str]:
    """
    Read one metadata value family from plural and singular query metadata keys.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Query metadata emitted by normalization or upstream callers.
    plural_key : str
        Metadata key expected to contain a list.
    singular_key : str
        Metadata key expected to contain one string.

    Returns
    -------
    List[str]
        Ordered unique metadata values.
    """

    plural_values = _normalize_string_list(metadata.get(plural_key))
    plural_comparison_values = {
        _normalize_match_text(existing_value) for existing_value in plural_values
    }

    return plural_values + [
        value
        for value in _normalize_string_list(metadata.get(singular_key))
        if _normalize_match_text(value) not in plural_comparison_values
    ]


def _extend_unique(values: List[str], additions: Iterable[str]) -> List[str]:
    """
    Append values while preserving order and avoiding duplicates.

    Parameters
    ----------
    values : List[str]
        Existing ordered values.
    additions : Iterable[str]
        Candidate values to append.

    Returns
    -------
    List[str]
        Updated ordered values.
    """

    seen_values = {_normalize_match_text(value) for value in values}

    for addition in additions:
        normalized_addition = _normalize_string(addition)
        comparison_addition = _normalize_match_text(normalized_addition)

        if normalized_addition and comparison_addition not in seen_values:
            values.append(normalized_addition)
            seen_values.add(comparison_addition)

    return values


class RetrievalRouter:
    """
    Decide deterministic retrieval behavior from a normalized user question.

    The router owns pre-retrieval routing only. It does not access vector
    storage, build context, generate answers, or evaluate grounding.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the retrieval router.

        Parameters
        ----------
        settings : PipelineSettings
            Central pipeline settings containing retrieval-routing controls.
        """

        self.settings = settings
        self._document_alias_index: Dict[str, List[str]] | None = None
        self._document_signal_profiles: List[_DocumentSignalProfile] | None = None

    def route(self, question: UserQuestionInput) -> RetrievalRouteDecision:
        """
        Build one retrieval route decision for a normalized question.

        Parameters
        ----------
        question : UserQuestionInput
            User question carrying normalized text and query metadata.

        Returns
        -------
        RetrievalRouteDecision
            Deterministic retrieval profile, scope, targets, and reasons.
        """

        query_text = question.normalized_query_text or question.question_text
        query_metadata = dict(question.query_metadata)
        route_features = self._extract_route_features(query_text, query_metadata)

        if not self.settings.retrieval_routing_enabled:
            return self._build_disabled_decision(route_features)

        if (
            route_features["comparative"]
            and self.settings.retrieval_routing_comparative_retrieval_enabled
        ):
            return self._build_comparative_decision(route_features)

        if route_features["document_titles"] and route_features["article_numbers"]:
            return self._build_article_document_decision(route_features)

        if route_features["document_titles"]:
            return self._build_document_decision(route_features)

        if route_features["article_numbers"]:
            return self._build_article_decision(route_features)

        return self._build_default_decision(route_features)

    def _extract_route_features(
        self,
        query_text: str,
        query_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract normalized legal routing features from query text and metadata.

        Parameters
        ----------
        query_text : str
            Normalized semantic retrieval query.
        query_metadata : Dict[str, Any]
            Metadata emitted by query normalization or upstream callers.

        Returns
        -------
        Dict[str, Any]
            Deterministic routing features used by route builders.
        """

        article_numbers = _metadata_values(
            query_metadata,
            "article_numbers",
            "article_number",
        )
        article_numbers = _extend_unique(
            article_numbers,
            _ARTICLE_REFERENCE_PATTERN.findall(query_text),
        )
        document_titles = _metadata_values(
            query_metadata,
            "document_titles",
            "document_title",
        )
        article_titles = _metadata_values(
            query_metadata,
            "article_titles",
            "article_title",
        )
        doc_ids = _metadata_values(query_metadata, "doc_ids", "doc_id")
        doc_ids = _extend_unique(
            doc_ids,
            self._resolve_document_doc_ids(document_titles),
        )
        comparative = self._is_comparative_query(query_text, query_metadata)
        legal_intents = self._extract_query_legal_intents(
            query_text=query_text,
            query_metadata=query_metadata,
        )
        inferred_document = self._infer_document_target(
            query_text=query_text,
            query_legal_intents=legal_intents,
            explicit_document_titles=document_titles,
            comparative=comparative,
        )
        inference_evidence = self._build_inference_evidence(
            query_text=query_text,
            query_legal_intents=legal_intents,
            inferred_document=inferred_document,
        )

        if inferred_document:
            document_titles = _extend_unique(
                document_titles,
                [inferred_document.document_title],
            )
            doc_ids = _extend_unique(doc_ids, [inferred_document.doc_id])

        return {
            "article_numbers": article_numbers,
            "article_titles": article_titles,
            "comparative": comparative,
            "doc_ids": doc_ids,
            "document_inferred": inferred_document is not None,
            "document_titles": document_titles,
            "inference_evidence": inference_evidence,
            "inferred_document": inferred_document,
            "legal_intents": legal_intents,
            "query_metadata": query_metadata,
        }

    def _extract_query_legal_intents(
        self,
        *,
        query_text: str,
        query_metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Read legal-intent routing signals from metadata and query text.

        Parameters
        ----------
        query_text : str
            Normalized semantic retrieval query.
        query_metadata : Dict[str, Any]
            Metadata emitted by query normalization or upstream callers.

        Returns
        -------
        List[str]
            Ordered legal-intent signal names usable by routing.
        """

        legal_intents = _metadata_values(
            query_metadata,
            "legal_intents",
            "legal_intent",
        )
        legal_intents = _extend_unique(
            legal_intents,
            _normalize_string_list(query_metadata.get("legal_intent_signals")),
        )
        legal_intents = _extend_unique(
            legal_intents,
            sorted(_extract_legal_intent_signals(query_text)),
        )

        return legal_intents

    def _build_inference_evidence(
        self,
        *,
        query_text: str,
        query_legal_intents: List[str],
        inferred_document: _DocumentSignalProfile | None,
    ) -> Dict[str, int]:
        """
        Summarize why one document inference was selected.

        Parameters
        ----------
        query_text : str
            Normalized semantic retrieval query.
        query_legal_intents : List[str]
            Generic legal-intent signals extracted for routing.
        inferred_document : _DocumentSignalProfile | None
            Selected inferred document profile when inference succeeded.

        Returns
        -------
        Dict[str, int]
            Counts of matched title, structural, and legal-intent evidence.
        """

        if inferred_document is None:
            return {}

        query_tokens = _tokenize_document_signal_text(query_text)
        query_intent_signals = frozenset(query_legal_intents)

        return {
            "title_token_overlap_count": len(
                query_tokens & inferred_document.title_tokens
            ),
            "structural_token_overlap_count": len(
                query_tokens
                & (inferred_document.structural_tokens - inferred_document.title_tokens)
            ),
            "legal_intent_overlap_count": len(
                query_intent_signals & inferred_document.legal_intent_signals
            ),
        }

    def _infer_document_target(
        self,
        *,
        query_text: str,
        query_legal_intents: List[str],
        explicit_document_titles: List[str],
        comparative: bool,
    ) -> _DocumentSignalProfile | None:
        """
        Infer one target document from query-to-document signal overlap.

        Parameters
        ----------
        query_text : str
            Normalized semantic retrieval query.
        query_legal_intents : List[str]
            Generic legal-intent signals extracted by query normalization.
        explicit_document_titles : List[str]
            Explicit document targets already extracted by the normalizer.
        comparative : bool
            Whether the question should preserve multi-document retrieval.

        Returns
        -------
        _DocumentSignalProfile | None
            Confident single document target, otherwise `None`.
        """

        if not self.settings.retrieval_routing_document_inference_enabled:
            return None
        if explicit_document_titles or comparative:
            return None

        query_tokens = _tokenize_document_signal_text(query_text)
        query_intent_signals = frozenset(query_legal_intents)
        if not query_tokens and not query_intent_signals:
            return None

        scored_profiles = sorted(
            [
                (
                    self._score_document_profile(
                        query_tokens=query_tokens,
                        query_legal_intents=query_intent_signals,
                        profile=profile,
                    ),
                    profile,
                )
                for profile in self._load_document_signal_profiles()
            ],
            key=lambda item: (-item[0], item[1].doc_id),
        )
        if not scored_profiles:
            return None

        top_score, top_profile = scored_profiles[0]
        second_score = scored_profiles[1][0] if len(scored_profiles) > 1 else 0.0
        min_score = float(self.settings.retrieval_routing_document_inference_min_score)
        min_margin = float(
            self.settings.retrieval_routing_document_inference_min_margin
        )

        if top_score < min_score:
            return None
        if top_score - second_score < min_margin:
            return None

        return top_profile

    def _score_document_profile(
        self,
        *,
        query_tokens: FrozenSet[str],
        query_legal_intents: FrozenSet[str],
        profile: _DocumentSignalProfile,
    ) -> float:
        """
        Score query overlap against one document signal profile.

        Parameters
        ----------
        query_tokens : FrozenSet[str]
            High-signal normalized query tokens.
        query_legal_intents : FrozenSet[str]
            Generic legal-intent signals extracted from the query.
        profile : _DocumentSignalProfile
            Document profile discovered from active chunk metadata.

        Returns
        -------
        float
            Deterministic score where title overlap is stronger than structural
            metadata overlap.
        """

        title_overlap_count = len(query_tokens & profile.title_tokens)
        structural_overlap_count = len(
            query_tokens & (profile.structural_tokens - profile.title_tokens)
        )
        intent_overlap_count = len(query_legal_intents & profile.legal_intent_signals)
        generic_intent_overlap_count = len(
            query_legal_intents
            & profile.legal_intent_signals
            & _GENERIC_LEGAL_INTENT_SIGNALS
        )
        specific_intent_overlap_count = (
            intent_overlap_count - generic_intent_overlap_count
        )

        return (
            (3.0 * title_overlap_count)
            + float(structural_overlap_count)
            + (4.0 * specific_intent_overlap_count)
            + (1.5 * generic_intent_overlap_count)
        )

    def _resolve_document_doc_ids(self, document_titles: List[str]) -> List[str]:
        """
        Resolve explicit document-title targets to known local document ids.

        Parameters
        ----------
        document_titles : List[str]
            Document titles or aliases extracted from the question.

        Returns
        -------
        List[str]
            Matching document identifiers discovered from active chunk outputs.
        """

        if not document_titles:
            return []

        alias_index = self._load_document_alias_index()
        resolved_doc_ids: List[str] = []

        for document_title in document_titles:
            title_key = _normalize_match_text(document_title)
            title_tokens = set(title_key.split())
            if not title_key or not title_tokens:
                continue

            for alias_key, doc_ids in alias_index.items():
                alias_tokens = set(alias_key.split())
                if title_key == alias_key or title_tokens.issubset(alias_tokens):
                    _extend_unique(resolved_doc_ids, doc_ids)

        return resolved_doc_ids

    def _load_document_signal_profiles(self) -> List[_DocumentSignalProfile]:
        """
        Build document-level routing profiles from active chunk outputs.

        Returns
        -------
        List[_DocumentSignalProfile]
            Profiles carrying generic title and structure signals per document.
        """

        if self._document_signal_profiles is not None:
            return self._document_signal_profiles

        profile_payloads: Dict[str, Dict[str, Any]] = {}
        input_root = Path(self.settings.embedding_input_root)
        strategy_name = self.settings.chunking_strategy.strip().lower()

        if not input_root.exists() or not strategy_name:
            self._document_signal_profiles = []
            return []

        for chunk_file_path in sorted(input_root.rglob("05_chunks.json")):
            if chunk_file_path.parent.name != strategy_name:
                continue
            self._collect_document_profile_payloads(
                chunk_file_path=chunk_file_path,
                profile_payloads=profile_payloads,
            )

        self._document_signal_profiles = [
            _DocumentSignalProfile(
                doc_id=doc_id,
                document_title=payload["document_title"],
                title_tokens=frozenset(payload["title_tokens"]),
                structural_tokens=frozenset(payload["structural_tokens"]),
                legal_intent_signals=frozenset(payload["legal_intent_signals"]),
            )
            for doc_id, payload in sorted(profile_payloads.items())
            if (
                payload["title_tokens"]
                or payload["structural_tokens"]
                or payload["legal_intent_signals"]
            )
        ]
        return self._document_signal_profiles

    def _collect_document_profile_payloads(
        self,
        *,
        chunk_file_path: Path,
        profile_payloads: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Add document signal payloads from one active chunk file.

        Parameters
        ----------
        chunk_file_path : Path
            Active chunk output file to inspect.
        profile_payloads : Dict[str, Dict[str, Any]]
            Mutable document payloads updated in place.
        """

        try:
            with chunk_file_path.open("r", encoding="utf-8") as chunk_file:
                payload = json.load(chunk_file)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(payload, list):
            return

        for chunk_item in payload:
            if isinstance(chunk_item, dict):
                self._collect_chunk_document_signals(chunk_item, profile_payloads)

    def _collect_chunk_document_signals(
        self,
        chunk_item: Dict[str, Any],
        profile_payloads: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Add generic document-routing signals from one chunk payload.

        Parameters
        ----------
        chunk_item : Dict[str, Any]
            Chunk record loaded from an active chunk output.
        profile_payloads : Dict[str, Dict[str, Any]]
            Mutable document payloads updated in place.
        """

        doc_id = _normalize_string(chunk_item.get("doc_id"))
        metadata = chunk_item.get("metadata")
        if not doc_id or not isinstance(metadata, dict):
            return

        document_title = _normalize_string(metadata.get("document_title"))
        source_file_name = _normalize_string(metadata.get("source_file_name"))
        profile_payload = profile_payloads.setdefault(
            doc_id,
            {
                "document_title": document_title or doc_id,
                "title_tokens": set(),
                "structural_tokens": set(),
                "legal_intent_signals": set(),
            },
        )
        if document_title and profile_payload["document_title"] == doc_id:
            profile_payload["document_title"] = document_title

        for title_fragment in (doc_id, document_title, source_file_name):
            profile_payload["title_tokens"].update(
                _tokenize_document_signal_text(title_fragment)
            )

        structural_fragments = [
            metadata.get("article_number"),
            metadata.get("article_title"),
            metadata.get("section_title"),
            chunk_item.get("source_node_label"),
            chunk_item.get("meta_text"),
        ]
        hierarchy_path = chunk_item.get("hierarchy_path")
        if isinstance(hierarchy_path, list):
            structural_fragments.extend(hierarchy_path)

        for structural_fragment in structural_fragments:
            if isinstance(structural_fragment, str):
                profile_payload["structural_tokens"].update(
                    _tokenize_document_signal_text(structural_fragment)
                )

        intent_fragments = [
            document_title,
            source_file_name,
            chunk_item.get("text"),
            chunk_item.get("meta_text"),
        ]
        if isinstance(metadata, dict):
            intent_fragments.extend(
                [
                    metadata.get("article_title"),
                    metadata.get("section_title"),
                    metadata.get("document_title"),
                ]
            )

        for intent_fragment in intent_fragments:
            if isinstance(intent_fragment, str):
                profile_payload["legal_intent_signals"].update(
                    _extract_legal_intent_signals(intent_fragment)
                )

    def _load_document_alias_index(self) -> Dict[str, List[str]]:
        """
        Build a lightweight document-title alias index from chunk outputs.

        Returns
        -------
        Dict[str, List[str]]
            Mapping from normalized document labels to known document ids.
        """

        if self._document_alias_index is not None:
            return self._document_alias_index

        alias_index: Dict[str, List[str]] = {}
        input_root = Path(self.settings.embedding_input_root)
        strategy_name = self.settings.chunking_strategy.strip().lower()

        if not input_root.exists() or not strategy_name:
            self._document_alias_index = alias_index
            return alias_index

        for chunk_file_path in sorted(input_root.rglob("05_chunks.json")):
            if chunk_file_path.parent.name != strategy_name:
                continue
            self._index_chunk_file_documents(
                chunk_file_path=chunk_file_path,
                alias_index=alias_index,
            )

        self._document_alias_index = alias_index
        return alias_index

    def _index_chunk_file_documents(
        self,
        *,
        chunk_file_path: Path,
        alias_index: Dict[str, List[str]],
    ) -> None:
        """
        Add document aliases from one chunk file to the router index.

        Parameters
        ----------
        chunk_file_path : Path
            Active chunk output file to inspect.

        alias_index : Dict[str, List[str]]
            Mutable alias index updated in place.
        """

        try:
            with chunk_file_path.open("r", encoding="utf-8") as chunk_file:
                payload = json.load(chunk_file)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(payload, list):
            return

        for chunk_item in payload:
            if not isinstance(chunk_item, dict):
                continue

            doc_id = _normalize_string(chunk_item.get("doc_id"))
            metadata = chunk_item.get("metadata")
            document_title = ""
            source_file_name = ""
            if isinstance(metadata, dict):
                document_title = _normalize_string(metadata.get("document_title"))
                source_file_name = _normalize_string(metadata.get("source_file_name"))

            if not doc_id:
                continue

            for alias in (doc_id, document_title, source_file_name):
                alias_key = _normalize_match_text(alias)
                if not alias_key:
                    continue
                alias_index.setdefault(alias_key, [])
                _extend_unique(alias_index[alias_key], [doc_id])

    def _is_comparative_query(
        self,
        query_text: str,
        query_metadata: Dict[str, Any],
    ) -> bool:
        """
        Determine whether a question should keep multi-document retrieval.

        Parameters
        ----------
        query_text : str
            Normalized semantic retrieval query.
        query_metadata : Dict[str, Any]
            Query metadata supplied by the normalizer or caller.

        Returns
        -------
        bool
            `True` when comparative retrieval behavior is indicated.
        """

        if bool(
            query_metadata.get("comparative")
            or query_metadata.get("is_comparative")
        ):
            return True

        normalized_query_text = _normalize_match_text(query_text)
        return any(
            pattern.search(normalized_query_text)
            for pattern in _COMPARATIVE_PATTERNS
        )

    def _build_disabled_decision(
        self,
        route_features: Dict[str, Any],
    ) -> RetrievalRouteDecision:
        """
        Build the default route used when explicit routing is disabled.

        Parameters
        ----------
        route_features : Dict[str, Any]
            Extracted routing features.

        Returns
        -------
        RetrievalRouteDecision
            Broad standard route with routing-disabled metadata.
        """

        return RetrievalRouteDecision(
            route_name="routing_disabled",
            retrieval_profile="standard",
            retrieval_scope="broad",
            target_doc_ids=route_features["doc_ids"],
            target_document_titles=route_features["document_titles"],
            target_article_numbers=route_features["article_numbers"],
            target_article_titles=route_features["article_titles"],
            comparative=route_features["comparative"],
            allow_second_pass=False,
            reasons=["routing_disabled"],
            metadata={
                "candidate_pool_size": self.settings.retrieval_candidate_pool_size,
                "routing_enabled": False,
            },
        )

    def _build_comparative_decision(
        self,
        route_features: Dict[str, Any],
    ) -> RetrievalRouteDecision:
        """
        Build a broad multi-document route for comparative questions.

        Parameters
        ----------
        route_features : Dict[str, Any]
            Extracted routing features.

        Returns
        -------
        RetrievalRouteDecision
            Broad comparative route.
        """

        return RetrievalRouteDecision(
            route_name="comparative_broad",
            retrieval_profile="comparative",
            retrieval_scope="broad",
            target_doc_ids=route_features["doc_ids"],
            target_document_titles=route_features["document_titles"],
            target_article_numbers=route_features["article_numbers"],
            target_article_titles=route_features["article_titles"],
            comparative=True,
            allow_second_pass=self.settings.retrieval_routing_weak_evidence_retry_enabled,
            reasons=["comparative_query_detected", "multi_document_retrieval_preserved"],
            metadata={
                "candidate_pool_size": self.settings.retrieval_routing_broad_candidate_pool_size,
                "routing_enabled": True,
            },
        )

    def _build_article_document_decision(
        self,
        route_features: Dict[str, Any],
    ) -> RetrievalRouteDecision:
        """
        Build a route for questions with explicit document and article targets.

        Parameters
        ----------
        route_features : Dict[str, Any]
            Extracted routing features.

        Returns
        -------
        RetrievalRouteDecision
            Scoped or broad article-document route based on settings.
        """

        if self._should_preserve_broad_for_inferred_document(route_features):
            return self._build_inferred_document_decision(
                route_features=route_features,
                additional_reasons=["article_target_detected"],
            )

        document_scoping_enabled = (
            self.settings.retrieval_routing_document_scoping_enabled
        )
        article_scoping_enabled = self.settings.retrieval_routing_article_scoping_enabled
        scope = (
            "scoped"
            if document_scoping_enabled or article_scoping_enabled
            else "broad"
        )
        profile = (
            "article_document_scoped"
            if scope == "scoped"
            else "article_document_broad"
        )
        reasons = [
            self._document_target_reason(route_features),
            "article_target_detected",
        ]

        if scope == "scoped":
            reasons.append("scoped_retrieval_selected")

        return self._build_targeted_decision(
            route_name=profile,
            retrieval_profile=profile,
            retrieval_scope=scope,
            reasons=reasons,
            route_features=route_features,
        )

    def _build_document_decision(
        self,
        route_features: Dict[str, Any],
    ) -> RetrievalRouteDecision:
        """
        Build a route for questions with explicit document targets.

        Parameters
        ----------
        route_features : Dict[str, Any]
            Extracted routing features.

        Returns
        -------
        RetrievalRouteDecision
            Document-scoped route when document scoping is enabled.
        """

        if self._should_preserve_broad_for_inferred_document(route_features):
            return self._build_inferred_document_decision(route_features=route_features)

        if self.settings.retrieval_routing_document_scoping_enabled:
            return self._build_targeted_decision(
                route_name="document_scoped",
                retrieval_profile="document_scoped",
                retrieval_scope="scoped",
                reasons=[
                    self._document_target_reason(route_features),
                    "scoped_retrieval_selected",
                ],
                route_features=route_features,
            )

        return self._build_default_decision(
            route_features,
            additional_reasons=[
                self._document_target_reason(route_features),
                "document_scoping_disabled",
            ],
        )

    def _document_target_reason(self, route_features: Dict[str, Any]) -> str:
        """
        Resolve the reason code for explicit or inferred document targets.

        Parameters
        ----------
        route_features : Dict[str, Any]
            Extracted routing features.

        Returns
        -------
        str
            Stable reason code describing how the document target was found.
        """

        if route_features.get("document_inferred"):
            return "dynamic_document_target_inferred"

        return "document_target_detected"

    def _should_preserve_broad_for_inferred_document(
        self,
        route_features: Dict[str, Any],
    ) -> bool:
        """
        Decide whether an inferred document should be carried as a retry target.

        Parameters
        ----------
        route_features : Dict[str, Any]
            Extracted routing features.

        Returns
        -------
        bool
            `True` when the inference is driven by legal-intent evidence rather
            than direct document-title evidence.
        """

        if not route_features.get("document_inferred"):
            return False

        inference_evidence = route_features.get("inference_evidence")
        if not isinstance(inference_evidence, dict):
            return False

        legal_intent_overlap_count = int(
            inference_evidence.get("legal_intent_overlap_count", 0)
        )
        title_token_overlap_count = int(
            inference_evidence.get("title_token_overlap_count", 0)
        )

        return legal_intent_overlap_count > 0 and title_token_overlap_count == 0

    def _build_inferred_document_decision(
        self,
        route_features: Dict[str, Any],
        additional_reasons: List[str] | None = None,
    ) -> RetrievalRouteDecision:
        """
        Build a broad route that carries one inferred document retry candidate.

        Parameters
        ----------
        route_features : Dict[str, Any]
            Extracted routing features with an inferred document target.
        additional_reasons : List[str] | None
            Optional extra route reasons, such as article bias.

        Returns
        -------
        RetrievalRouteDecision
            Evidence-aware route that preserves broad retrieval while exposing
            a document-focused candidate for second-pass recovery.
        """

        reasons = [
            "dynamic_document_target_inferred",
            "retry_candidate_document_scope_selected",
        ]
        if additional_reasons:
            reasons.extend(additional_reasons)

        candidate_pool_size = max(
            int(self.settings.retrieval_candidate_pool_size),
            int(self.settings.retrieval_routing_broad_candidate_pool_size),
            int(self.settings.retrieval_second_pass_retry_candidate_pool_size),
        )

        return RetrievalRouteDecision(
            route_name="retry_candidate_document_scoped",
            retrieval_profile="retry_candidate_document_scoped",
            retrieval_scope="retry_candidate_document_scoped",
            target_doc_ids=route_features["doc_ids"],
            target_document_titles=route_features["document_titles"],
            target_article_numbers=route_features["article_numbers"],
            target_article_titles=route_features["article_titles"],
            comparative=False,
            allow_second_pass=self.settings.retrieval_routing_weak_evidence_retry_enabled,
            reasons=reasons,
            metadata={
                "candidate_pool_size": candidate_pool_size,
                "document_inferred": True,
                "inference_evidence": route_features["inference_evidence"],
                "inferred_target_doc_ids": route_features["doc_ids"],
                "inferred_target_document_titles": route_features["document_titles"],
                "legal_intents": route_features["legal_intents"],
                "routing_enabled": True,
            },
        )

    def _build_article_decision(
        self,
        route_features: Dict[str, Any],
    ) -> RetrievalRouteDecision:
        """
        Build a route for questions with explicit article targets.

        Parameters
        ----------
        route_features : Dict[str, Any]
            Extracted routing features.

        Returns
        -------
        RetrievalRouteDecision
            Article-biased route that keeps broad search when no document target exists.
        """

        if self.settings.retrieval_routing_article_scoping_enabled:
            return self._build_targeted_decision(
                route_name="article_biased",
                retrieval_profile="article_biased",
                retrieval_scope="broad",
                reasons=["article_target_detected", "article_bias_selected"],
                route_features=route_features,
            )

        return self._build_default_decision(
            route_features,
            additional_reasons=["article_target_detected", "article_scoping_disabled"],
        )

    def _build_default_decision(
        self,
        route_features: Dict[str, Any],
        additional_reasons: List[str] | None = None,
    ) -> RetrievalRouteDecision:
        """
        Build the standard broad retrieval route.

        Parameters
        ----------
        route_features : Dict[str, Any]
            Extracted routing features.
        additional_reasons : List[str] | None
            Optional extra reason codes.

        Returns
        -------
        RetrievalRouteDecision
            Standard broad retrieval decision.
        """

        legal_intents = route_features["legal_intents"]
        expanded_broad = bool(legal_intents)
        reasons = ["no_explicit_retrieval_target"]
        if additional_reasons:
            reasons = list(additional_reasons)
            reasons.append("standard_broad_retrieval_selected")
        elif expanded_broad:
            reasons.append("legal_intent_broad_expansion_selected")

        candidate_pool_size = max(
            int(self.settings.retrieval_candidate_pool_size),
            int(self.settings.retrieval_routing_broad_candidate_pool_size),
        )
        if expanded_broad:
            candidate_pool_size = max(
                candidate_pool_size,
                int(self.settings.retrieval_second_pass_retry_candidate_pool_size),
            )

        return RetrievalRouteDecision(
            route_name="broad_expanded" if expanded_broad else "standard_broad",
            retrieval_profile="broad_expanded" if expanded_broad else "standard",
            retrieval_scope="broad_expanded" if expanded_broad else "broad",
            target_doc_ids=route_features["doc_ids"],
            target_document_titles=route_features["document_titles"],
            target_article_numbers=route_features["article_numbers"],
            target_article_titles=route_features["article_titles"],
            comparative=route_features["comparative"],
            allow_second_pass=self.settings.retrieval_routing_weak_evidence_retry_enabled,
            reasons=reasons,
            metadata={
                "candidate_pool_size": candidate_pool_size,
                "inference_evidence": route_features["inference_evidence"],
                "document_inference_attempted": (
                    self.settings.retrieval_routing_document_inference_enabled
                ),
                "legal_intents": legal_intents,
                "routing_enabled": True,
            },
        )

    def _build_targeted_decision(
        self,
        route_name: str,
        retrieval_profile: str,
        retrieval_scope: str,
        reasons: List[str],
        route_features: Dict[str, Any],
    ) -> RetrievalRouteDecision:
        """
        Build one targeted retrieval route from extracted features.

        Parameters
        ----------
        route_name : str
            Stable route identifier.
        retrieval_profile : str
            Retrieval behavior profile.
        retrieval_scope : str
            Retrieval scope name.
        reasons : List[str]
            Explainable route reason codes.
        route_features : Dict[str, Any]
            Extracted routing features.

        Returns
        -------
        RetrievalRouteDecision
            Target-aware route decision.
        """

        candidate_pool_size = (
            self.settings.retrieval_routing_scoped_candidate_pool_size
            if retrieval_scope == "scoped"
            else self.settings.retrieval_routing_broad_candidate_pool_size
        )
        candidate_pool_size = max(
            int(candidate_pool_size),
            int(self.settings.retrieval_candidate_pool_size),
        )

        return RetrievalRouteDecision(
            route_name=route_name,
            retrieval_profile=retrieval_profile,
            retrieval_scope=retrieval_scope,
            target_doc_ids=route_features["doc_ids"],
            target_document_titles=route_features["document_titles"],
            target_article_numbers=route_features["article_numbers"],
            target_article_titles=route_features["article_titles"],
            comparative=False,
            allow_second_pass=self.settings.retrieval_routing_weak_evidence_retry_enabled,
            reasons=reasons,
            metadata={
                "candidate_pool_size": candidate_pool_size,
                "document_inferred": route_features["document_inferred"],
                "inference_evidence": route_features["inference_evidence"],
                "legal_intents": route_features["legal_intents"],
                "routing_enabled": True,
            },
        )
