from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Match, Optional, Pattern, Tuple

from Chunking.config.settings import PipelineSettings
from retrieval.models import UserQuestionInput


_WHITESPACE_PATTERN = re.compile(r"\s+")
_SEPARATOR_PATTERN = re.compile(r"\s*[,;:]\s*")
_CLAUSE_SEPARATOR_PATTERN = re.compile(r"\s*[,;:]\s*")
_LEADING_CONNECTOR_PATTERN = re.compile(
    r"^(?:e|and|then|entao|então)\b[\s,;:.-]*",
    re.IGNORECASE,
)
_TRAILING_CONNECTOR_PATTERN = re.compile(
    r"[\s,;:.-]*\b(?:e|and|com|with)\b[\s,;:.-]*$",
    re.IGNORECASE,
)
_ARTICLE_REFERENCE_PATTERN = re.compile(
    r"\bart(?:igo)?\.?\s*(\d+[a-z]?)\b",
    re.IGNORECASE,
)
_LEGAL_DOCUMENT_REFERENCE_PATTERN = re.compile(
    r"\b(?:regulamento|despacho|estatuto|norma|normativo)\s+"
    r"(?:interno\s+)?(?:de|do|da|dos|das|sobre|relativo\s+a)\s+"
    r"[a-z0-9][^?!.;,]{2,80}",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_LANGUAGE_DIRECTIVE_CLAUSE_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(
        r"\b(?:responde|responda|respond|answer|reply|write|escreve|escreva|"
        r"redige|redija|devolve|devolva|fornece|forneca|forneça|apresenta|"
        r"apresente|quero|pretendo|preciso|gostava)\b.*\b(?:pt pt|pt-pt|"
        r"portugues europeu|portuguese|portugues|português|english|en us|en-us)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:resposta|output|texto)\b.*\b(?:em|in)\s+(?:pt pt|pt-pt|"
        r"portugues europeu|portuguese|portugues|português|english|en us|en-us)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:quero|pretendo|preciso|gostava)\s+(?:a\s+|uma\s+)?"
        r"(?:resposta|output|texto)\b$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:quero|pretendo|preciso|gostava)(?:\s+(?:a|uma))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:resposta|output|texto)\b$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:em|in)\s+(?:pt pt|pt-pt|portugues europeu|portuguese|"
        r"portugues|português|english|en us|en-us)\b",
        re.IGNORECASE,
    ),
)
_CITATION_DIRECTIVE_CLAUSE_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(
        r"\b(?:indica|indique|cita|cite|menciona|mencione|identifica|"
        r"identifique|refere|refira|inclui|inclua|aponta|aponte|mostra|mostre)\b"
        r".*\b(?:regulamento|regulation|despacho|artigo|article|fonte|source|"
        r"referencia|referência|base legal|alinea|alínea)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:com|incluindo|with)\b.*\b(?:referencia|referência|citacao|"
        r"citação|citacoes|citações|base legal)\b.*\b(?:regulamento|article|"
        r"artigo|despacho|fonte|source)\b",
        re.IGNORECASE,
    ),
)
_GROUNDING_DIRECTIVE_CLAUSE_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(
        r"\b(?:com base|baseado|baseada|baseando|usando|utilizando|considerando|"
        r"com apoio|com recurso)\b.*\b(?:regulamentos recuperados|contexto recuperado|"
        r"contexto fornecido|fontes fornecidas|documentos fornecidos|retrieved "
        r"regulations|retrieved context|provided context|provided sources)\b",
        re.IGNORECASE,
    ),
)
_FORMATTING_DIRECTIVE_CLAUSE_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(
        r"\b(?:em|in|formato|format)\b.*\b(?:bullet points|pontos|lista|list|"
        r"tabela|table|markdown|json)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:resumo|resumida|resumido|sumario|sumário|curta|curto|breve|"
        r"passo a passo)\b",
        re.IGNORECASE,
    ),
)
_LEGAL_INTENT_RULES: Tuple[Tuple[str, Tuple[Pattern[str], ...]], ...] = (
    (
        "payment_plan",
        (
            re.compile(r"\bplano\s+(?:de\s+)?pagamento\b"),
            re.compile(r"\bpagamento\s+(?:em\s+)?prestaco(?:es|e?s)\b"),
            re.compile(r"\bprestaco(?:es|e?s)\s+(?:de\s+)?pagamento\b"),
            re.compile(r"\b(?:fee|tuition)\s+payment\s+plan\b"),
        ),
    ),
    (
        "general_payment_plan",
        (
            re.compile(r"\bplano\s+geral\s+(?:de\s+)?pagamento\b"),
            re.compile(r"\bregime\s+geral\s+(?:de\s+)?pagamento\b"),
            re.compile(r"\bgeneral\s+(?:fee|tuition)?\s*payment\s+plan\b"),
        ),
    ),
    (
        "specific_payment_plan",
        (
            re.compile(r"\bplano\s+(?:especifico|especial)\s+(?:de\s+)?pagamento\b"),
            re.compile(r"\bregime\s+(?:especifico|especial)\s+(?:de\s+)?pagamento\b"),
            re.compile(r"\bspecific\s+(?:fee|tuition)?\s*payment\s+plan\b"),
        ),
    ),
    (
        "regularization_plan",
        (
            re.compile(r"\bplano\s+(?:de\s+)?regularizacao\b"),
            re.compile(r"\bregularizacao\s+(?:de\s+)?(?:divida|dividas|pagamento|propina|propinas)\b"),
            re.compile(r"\b(?:debt|payment)\s+regulari[sz]ation\s+plan\b"),
        ),
    ),
    (
        "international_student",
        (
            re.compile(r"\bestudante(?:s)?\s+internacion(?:al|ais)\b"),
            re.compile(r"\baluno(?:s)?\s+internacion(?:al|ais)\b"),
            re.compile(r"\binternational\s+student(?:s)?\b"),
            re.compile(r"\bestudante(?:s)?\s+(?:estrangeiro(?:s)?|externo(?:s)?)\b"),
        ),
    ),
    (
        "installment_schedule",
        (
            re.compile(r"\bprestaco(?:es|e?s)\b"),
            re.compile(r"\bpagamento\s+faseado\b"),
            re.compile(r"\bcalendario\s+(?:de\s+)?pagamento\b"),
            re.compile(r"\binstallment(?:s)?\b"),
            re.compile(r"\bpayment\s+schedule\b"),
        ),
    ),
    (
        "deadline_question",
        (
            re.compile(r"\b(?:qual|quais|que|quando|ate\s+quando)\b.*\bprazo(?:s)?\b"),
            re.compile(r"\bprazo(?:s)?\s+(?:para|de|do|da)\b"),
            re.compile(r"\bdeadline(?:s)?\b"),
            re.compile(r"\b(?:data|datas)\s+limite\b"),
        ),
    ),
    (
        "document_requirement_question",
        (
            re.compile(r"\b(?:documento|documentos|documentacao)\s+(?:necessario|necessarios|exigido|exigidos|a\s+entregar|a\s+apresentar)\b"),
            re.compile(r"\b(?:que|quais)\s+(?:documento|documentos|documentacao)\b"),
            re.compile(r"\brequisito(?:s)?\s+(?:documental|documentais|de\s+documentacao)\b"),
            re.compile(r"\brequired\s+document(?:s)?\b"),
            re.compile(r"\bdocument\s+requirement(?:s)?\b"),
        ),
    ),
    (
        "matriculation_cancellation",
        (
            re.compile(
                r"\b(?:anul(?:acao|ar)|cancelamento|cancelar|desistencia|desistir)\s+"
                r"(?:da\s+|de\s+)?matricul(?:a|as)\b"
            ),
            re.compile(
                r"\bmatricul(?:a|as)\s+(?:anulada|anuladas|cancelada|canceladas)\b"
            ),
            re.compile(
                r"\b(?:cancel|cancellation|withdrawal)\s+of\s+matriculation\b"
            ),
        ),
    ),
    (
        "mandatory_attendance",
        (
            re.compile(r"\bassiduidade\s+obrigatoria\b"),
            re.compile(r"\bfrequencia\s+obrigatoria\b"),
            re.compile(r"\bpresenca\s+obrigatoria\b"),
            re.compile(r"\battendance\s+is\s+mandatory\b"),
            re.compile(r"\bmandatory\s+attendance\b"),
        ),
    ),
    (
        "liminary_rejection",
        (
            re.compile(r"\bindeferimento\s+liminar\b"),
            re.compile(r"\b(?:pedido|requerimento|candidatura)\s+liminarmente\s+indeferid[oa]\b"),
            re.compile(r"\b(?:liminary|summary)\s+rejection\b"),
        ),
    ),
    (
        "exclusion",
        (
            re.compile(r"\bexclus(?:ao|oes)\b"),
            re.compile(r"\bexclui(?:do|da|dos|das)\b"),
            re.compile(r"\b(?:expulsao|expulsar)\b"),
            re.compile(r"\bexclusion\b"),
        ),
    ),
    (
        "non_compliance_consequence",
        (
            re.compile(r"\bconsequenc(?:ia|ias)\s+(?:do|da|de)\s+incumprimento\b"),
            re.compile(r"\b(?:efeito|efeitos|consequenc(?:ia|ias)|penaliza(?:cao|coes)|sanc(?:ao|oes))\s+(?:do|da|de)\s+(?:nao\s+)?cumprimento\b"),
            re.compile(r"\bincumprimento\s+(?:implica|determina|leva\s+a|origina)\b"),
            re.compile(r"\bwhat\s+happens\s+if\s+(?:there\s+is\s+)?non[ -]?compliance\b"),
            re.compile(r"\bnon[ -]?compliance\s+consequence(?:s)?\b"),
        ),
    ),
)
_COMPARATIVE_INTENT_PATTERN = re.compile(
    r"\b(?:comparar|compara|comparacao|diferenca|diferencas|entre|versus|vs|"
    r"compare|comparison|difference|differences|between)\b"
)
_PAYMENT_INTENT_SIGNALS = frozenset(
    {
        "payment_plan",
        "general_payment_plan",
        "specific_payment_plan",
        "regularization_plan",
        "installment_schedule",
    }
)
_PAYMENT_PLAN_DERIVED_SIGNALS = frozenset(
    {
        "general_payment_plan",
        "specific_payment_plan",
        "regularization_plan",
        "installment_schedule",
    }
)
_QUALIFIER_INTENT_SIGNALS = frozenset({"international_student"})


def _normalize_whitespace(value: str) -> str:
    """
    Normalize arbitrary whitespace into one clean single-space string.

    Parameters
    ----------
    value : str
        Raw text fragment.

    Returns
    -------
    str
        Stripped text with collapsed whitespace.
    """

    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def _normalize_match_text(value: str) -> str:
    """
    Normalize one text fragment into an accent-free comparison form.

    Parameters
    ----------
    value : str
        Raw text fragment.

    Returns
    -------
    str
        Lowercased text reduced to alphanumeric comparison tokens.
    """

    normalized_value = unicodedata.normalize("NFKD", value)
    ascii_value = normalized_value.encode("ascii", "ignore").decode("ascii")
    return " ".join(_TOKEN_PATTERN.findall(ascii_value.lower()))


def _clean_directive_text(value: str) -> str:
    """
    Clean one extracted directive clause for stable downstream storage.

    Parameters
    ----------
    value : str
        Raw matched directive text.

    Returns
    -------
    str
        Cleaned directive text without redundant separators.
    """

    cleaned_value = _normalize_whitespace(value.strip(" ,;:."))
    cleaned_value = _LEADING_CONNECTOR_PATTERN.sub("", cleaned_value)
    cleaned_value = _TRAILING_CONNECTOR_PATTERN.sub("", cleaned_value)
    return cleaned_value.strip(" ,;:.")


def _clean_semantic_query_text(value: str) -> str:
    """
    Clean the semantic query after directive removal.

    Parameters
    ----------
    value : str
        Query candidate after directive spans have been stripped.

    Returns
    -------
    str
        Semantic query text kept readable for embedding.
    """

    cleaned_value = _SEPARATOR_PATTERN.sub(", ", value)
    cleaned_value = re.sub(r"\s+[?]", "?", cleaned_value)
    cleaned_value = re.sub(r"\s+[!]", "!", cleaned_value)
    cleaned_value = re.sub(r"^[,;:.\- ]+", "", cleaned_value)
    cleaned_value = re.sub(r"[,;:.\- ]+$", "", cleaned_value)
    cleaned_value = _LEADING_CONNECTOR_PATTERN.sub("", cleaned_value)
    cleaned_value = _TRAILING_CONNECTOR_PATTERN.sub("", cleaned_value)
    return _normalize_whitespace(cleaned_value)


def _extract_language_hint(directives: List[str]) -> str:
    """
    Extract one normalized output-language hint from directive clauses.

    Parameters
    ----------
    directives : List[str]
        Extracted formatting directives.

    Returns
    -------
    str
        Canonical language hint when detected, otherwise an empty string.
    """

    combined_directives = " ".join(directives).lower()

    if (
        "pt-pt" in combined_directives
        or "portuguese" in combined_directives
        or "portugues europeu" in combined_directives
    ):
        return "pt-pt"
    if "português europeu" in combined_directives:
        return "pt-pt"
    if "português" in combined_directives:
        return "pt"
    if "english" in combined_directives or "en-us" in combined_directives:
        return "en"

    return ""


def _contains_any_pattern(
    value: str,
    patterns: Tuple[Pattern[str], ...],
) -> bool:
    """
    Check whether one text fragment matches any compiled directive pattern.

    Parameters
    ----------
    value : str
        Text fragment already normalized for stable matching.

    patterns : Tuple[Pattern[str], ...]
        Compiled patterns evaluated against the fragment.

    Returns
    -------
    bool
        `True` when at least one pattern matches.
    """

    return any(pattern.search(value) for pattern in patterns)


def _deduplicate_preserving_order(values: List[str]) -> List[str]:
    """
    Remove duplicate strings while preserving first appearance order.

    Parameters
    ----------
    values : List[str]
        Candidate ordered values.

    Returns
    -------
    List[str]
        Ordered unique values.
    """

    unique_values: List[str] = []
    seen_values: set[str] = set()

    for value in values:
        if not value or value in seen_values:
            continue
        seen_values.add(value)
        unique_values.append(value)

    return unique_values


def _order_directives_by_appearance(
    question_text: str,
    directives: List[str],
) -> List[str]:
    """
    Order extracted directives by first appearance in the original question.

    Parameters
    ----------
    question_text : str
        Original user question used as the stable ordering source.

    directives : List[str]
        Extracted directives already deduplicated.

    Returns
    -------
    List[str]
        Directives ordered according to their position in the source text.
    """

    normalized_question = question_text.lower()

    return sorted(
        directives,
        key=lambda directive: (
            normalized_question.find(directive.lower())
            if normalized_question.find(directive.lower()) >= 0
            else len(normalized_question)
        ),
    )


@dataclass(frozen=True, slots=True)
class _DirectivePattern:
    """
    Hold one deterministic formatting-directive extraction rule.

    Parameters
    ----------
    pattern : Pattern[str]
        Compiled regular expression used to capture one directive span.

    metadata_key : str
        Optional metadata key set to `True` when the pattern matches.
    """

    pattern: Pattern[str]
    metadata_key: str = ""


class SemanticQueryNormalizer:
    """
    Build a deterministic semantic retrieval query from raw user wording.

    Design goals
    ------------
    - keep the original user question intact for later answer generation
    - strip retrieval-harmful formatting directives before embedding
    - extract lightweight explicit hints without model-based rewriting
    - remain generic for legal and regulatory questions in Portuguese or English
    """

    _DIRECTIVE_PATTERNS: Tuple[_DirectivePattern, ...] = (
        _DirectivePattern(
            pattern=re.compile(
                r"\b(?:responde|responda|answer|respond|reply|write)\b"
                r"[^,;:.?!]*(?:pt-pt|pt pt|português europeu|portugues europeu|"
                r"português|portugues|english|en-us|en us)\b[^,;:.?!]*",
                re.IGNORECASE,
            ),
            metadata_key="language_directive_detected",
        ),
        _DirectivePattern(
            pattern=re.compile(
                r"\b(?:indica|indique|cita|cite|mention|show|identify)\b"
                r"[^,;:.?!]*(?:regulamento|regulation|artigo|article)\b[^,;:.?!]*",
                re.IGNORECASE,
            ),
            metadata_key="citation_directive_detected",
        ),
        _DirectivePattern(
            pattern=re.compile(
                r"\b(?:com base|baseando(?:-te)?|using|based on)\b"
                r"[^,;:.?!]*(?:regulamentos recuperados|retrieved regulations|"
                r"retrieved context|contexto recuperado|contexto fornecido)\b"
                r"[^,;:.?!]*",
                re.IGNORECASE,
            ),
            metadata_key="grounding_directive_detected",
        ),
        _DirectivePattern(
            pattern=re.compile(
                r"\b(?:em|in)\s+(?:bullet points|pontos|lista|list)\b[^,;:.?!]*",
                re.IGNORECASE,
            ),
            metadata_key="formatting_directive_detected",
        ),
        _DirectivePattern(
            pattern=re.compile(
                r"\b(?:quero|pretendo|preciso|gostava)\b[^,;:.?!]*"
                r"\b(?:a\s+|uma\s+)?(?:resposta|output|texto)\b[^,;:.?!]*"
                r"\b(?:em|in)\s+(?:pt-pt|pt pt|português europeu|portugues europeu|"
                r"português|portugues|english|en-us|en us)\b[^,;:.?!]*",
                re.IGNORECASE,
            ),
            metadata_key="language_directive_detected",
        ),
        _DirectivePattern(
            pattern=re.compile(
                r"\b(?:resposta|output|texto)\b[^,;:.?!]*\b(?:em|in)\s+"
                r"(?:pt-pt|pt pt|português europeu|portugues europeu|"
                r"português|portugues|english|en-us|en us)\b[^,;:.?!]*",
                re.IGNORECASE,
            ),
            metadata_key="language_directive_detected",
        ),
        _DirectivePattern(
            pattern=re.compile(
                r"\b(?:em|in)\s+"
                r"(?:pt-pt|pt pt|português europeu|portugues europeu|"
                r"português|portugues|english|en-us|en us)\b",
                re.IGNORECASE,
            ),
            metadata_key="language_directive_detected",
        ),
        _DirectivePattern(
            pattern=re.compile(
                r"\b(?:com|incluindo|with)\b[^,;:.?!]*"
                r"(?:referência|referencia|citação|citacao|citacoes|citações|"
                r"base legal)\b[^,;:.?!]*(?:regulamento|regulation|artigo|article|"
                r"fonte|source|alínea|alinea)\b[^,;:.?!]*",
                re.IGNORECASE,
            ),
            metadata_key="citation_directive_detected",
        ),
        _DirectivePattern(
            pattern=re.compile(
                r"\b(?:usando|utilizando|considerando)\b[^,;:.?!]*"
                r"(?:regulamentos recuperados|retrieved regulations|retrieved context|"
                r"contexto recuperado|contexto fornecido|fontes fornecidas|"
                r"provided context|provided sources)\b[^,;:.?!]*",
                re.IGNORECASE,
            ),
            metadata_key="grounding_directive_detected",
        ),
    )

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """
        Initialize the normalizer with shared runtime settings.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared settings used to enable or disable normalization behavior.
        """

        self.settings = settings or PipelineSettings()

    def normalize(
        self,
        question: UserQuestionInput | str,
    ) -> UserQuestionInput:
        """
        Normalize one public question into a cleaner semantic retrieval query.

        Parameters
        ----------
        question : UserQuestionInput | str
            Raw user question contract or plain string.

        Returns
        -------
        UserQuestionInput
            Normalized question preserving the original wording while exposing
            semantic query text, formatting directives, and query metadata.
        """

        question_input = self._coerce_question_input(question)

        if not self.settings.retrieval_query_normalization_enabled:
            return UserQuestionInput(
                question_text=question_input.question_text,
                request_id=question_input.request_id,
                conversation_id=question_input.conversation_id,
                metadata=dict(question_input.metadata),
                normalized_query_text=question_input.question_text,
                formatting_instructions=list(question_input.formatting_instructions),
                query_metadata=dict(question_input.query_metadata),
            )

        semantic_query_text, formatting_directives, query_metadata = (
            self._normalize_query_text(question_input.question_text)
        )

        merged_query_metadata = dict(question_input.query_metadata)
        merged_query_metadata.update(query_metadata)

        merged_formatting_instructions = _deduplicate_preserving_order(
            list(question_input.formatting_instructions) + formatting_directives
        )

        return UserQuestionInput(
            question_text=question_input.question_text,
            request_id=question_input.request_id,
            conversation_id=question_input.conversation_id,
            metadata=dict(question_input.metadata),
            normalized_query_text=semantic_query_text,
            formatting_instructions=merged_formatting_instructions,
            query_metadata=merged_query_metadata,
        )

    def _coerce_question_input(
        self,
        question: UserQuestionInput | str,
    ) -> UserQuestionInput:
        """
        Normalize the public normalizer input into a question contract.

        Parameters
        ----------
        question : UserQuestionInput | str
            Raw normalizer input supplied by the caller.

        Returns
        -------
        UserQuestionInput
            Question contract with validated text content.
        """

        if isinstance(question, UserQuestionInput):
            question_input = question
        elif isinstance(question, str):
            question_input = UserQuestionInput(question_text=question)
        else:
            raise ValueError("Question must be a UserQuestionInput instance or string.")

        if not question_input.question_text:
            raise ValueError("Question text cannot be empty.")

        return question_input

    def _normalize_query_text(
        self,
        question_text: str,
    ) -> Tuple[str, List[str], Dict[str, object]]:
        """
        Extract semantic text and formatting directives from one question.

        Parameters
        ----------
        question_text : str
            Original user wording.

        Returns
        -------
        Tuple[str, List[str], Dict[str, object]]
            Semantic query text, formatting directives, and extracted hints.
        """

        working_text = _normalize_whitespace(question_text)
        extracted_directives: List[str] = []
        metadata_flags: Dict[str, object] = {}

        for directive_pattern in self._DIRECTIVE_PATTERNS:
            working_text, matched_directives = self._strip_pattern_matches(
                working_text,
                directive_pattern.pattern,
            )

            if matched_directives and directive_pattern.metadata_key:
                metadata_flags[directive_pattern.metadata_key] = True

            extracted_directives.extend(matched_directives)

        (
            working_text,
            clause_directives,
            clause_metadata_flags,
        ) = self._strip_directive_clauses(working_text)
        extracted_directives.extend(clause_directives)
        metadata_flags.update(clause_metadata_flags)

        extracted_directives = _order_directives_by_appearance(
            question_text,
            _deduplicate_preserving_order(extracted_directives),
        )
        semantic_query_text = working_text

        if self.settings.retrieval_query_normalization_strip_formatting_instructions:
            semantic_query_text = _clean_semantic_query_text(semantic_query_text)

        if not semantic_query_text:
            semantic_query_text = _normalize_whitespace(question_text)

        if not self.settings.retrieval_query_normalization_extract_formatting_directives:
            extracted_directives = []
            metadata_flags = {}

        requested_language = _extract_language_hint(extracted_directives)
        if requested_language:
            metadata_flags["requested_output_language"] = requested_language

        metadata_flags.update(
            self._extract_structural_query_metadata(semantic_query_text)
        )
        metadata_flags.update(
            self._extract_legal_intent_query_metadata(semantic_query_text)
        )
        metadata_flags["formatting_directive_count"] = len(extracted_directives)

        return semantic_query_text, extracted_directives, metadata_flags

    def _strip_directive_clauses(
        self,
        text: str,
    ) -> Tuple[str, List[str], Dict[str, object]]:
        """
        Remove directive-only clauses that survive narrow regex stripping.

        Parameters
        ----------
        text : str
            Current working question text.

        Returns
        -------
        Tuple[str, List[str], Dict[str, object]]
            Remaining semantic text, extracted directives, and metadata flags.
        """

        extracted_directives: List[str] = []
        metadata_flags: Dict[str, object] = {}
        remaining_clauses: List[str] = []

        for raw_clause in _CLAUSE_SEPARATOR_PATTERN.split(text):
            cleaned_clause = _clean_directive_text(raw_clause)
            if not cleaned_clause:
                continue

            directive_flags = self._classify_directive_clause(cleaned_clause)
            if directive_flags:
                extracted_directives.append(cleaned_clause)
                metadata_flags.update(directive_flags)
                continue

            remaining_clauses.append(cleaned_clause)

        return ", ".join(remaining_clauses), extracted_directives, metadata_flags

    def _classify_directive_clause(
        self,
        clause_text: str,
    ) -> Dict[str, object]:
        """
        Classify one clause as directive-only when deterministic signals exist.

        Parameters
        ----------
        clause_text : str
            Clause candidate preserved from the user question.

        Returns
        -------
        Dict[str, object]
            Metadata flags for the directive categories detected in the clause.
        """

        normalized_clause = _normalize_match_text(clause_text)
        metadata_flags: Dict[str, object] = {}

        if _contains_any_pattern(
            normalized_clause,
            _LANGUAGE_DIRECTIVE_CLAUSE_PATTERNS,
        ):
            metadata_flags["language_directive_detected"] = True
        if _contains_any_pattern(
            normalized_clause,
            _CITATION_DIRECTIVE_CLAUSE_PATTERNS,
        ):
            metadata_flags["citation_directive_detected"] = True
        if _contains_any_pattern(
            normalized_clause,
            _GROUNDING_DIRECTIVE_CLAUSE_PATTERNS,
        ):
            metadata_flags["grounding_directive_detected"] = True
        if _contains_any_pattern(
            normalized_clause,
            _FORMATTING_DIRECTIVE_CLAUSE_PATTERNS,
        ):
            metadata_flags["formatting_directive_detected"] = True

        return metadata_flags

    def _extract_structural_query_metadata(
        self,
        semantic_query_text: str,
    ) -> Dict[str, object]:
        """
        Extract structural legal cues that can improve later context selection.

        Parameters
        ----------
        semantic_query_text : str
            Clean semantic retrieval query after directive removal.

        Returns
        -------
        Dict[str, object]
            Deterministic structural query metadata.
        """

        metadata_flags: Dict[str, object] = {}
        article_numbers = _deduplicate_preserving_order(
            [
                article_number.lower()
                for article_number in _ARTICLE_REFERENCE_PATTERN.findall(
                    semantic_query_text
                )
                if article_number
            ]
        )
        document_titles = _deduplicate_preserving_order(
            [
                _normalize_whitespace(match.group(0).strip(" ,;:."))
                for match in _LEGAL_DOCUMENT_REFERENCE_PATTERN.finditer(
                    semantic_query_text
                )
            ]
        )

        if article_numbers:
            metadata_flags["article_numbers"] = article_numbers
            if len(article_numbers) == 1:
                metadata_flags["article_number"] = article_numbers[0]

        if document_titles:
            metadata_flags["document_titles"] = document_titles
            if len(document_titles) == 1:
                metadata_flags["document_title"] = document_titles[0]

        return metadata_flags

    def _extract_legal_intent_query_metadata(
        self,
        semantic_query_text: str,
    ) -> Dict[str, object]:
        """
        Extract generic operational legal-intent signals from a semantic query.

        Parameters
        ----------
        semantic_query_text : str
            Clean semantic retrieval query after directive removal.

        Returns
        -------
        Dict[str, object]
            Deterministic legal-intent metadata for routing and context selection.
        """

        normalized_query_text = _normalize_match_text(semantic_query_text)
        legal_intents: List[str] = []

        for intent_name, intent_patterns in _LEGAL_INTENT_RULES:
            if _contains_any_pattern(normalized_query_text, intent_patterns):
                legal_intents.append(intent_name)

        legal_intents = _deduplicate_preserving_order(legal_intents)
        if (
            "payment_plan" not in legal_intents
            and set(legal_intents) & _PAYMENT_PLAN_DERIVED_SIGNALS
        ):
            legal_intents.insert(0, "payment_plan")

        if not legal_intents:
            return {}

        metadata_flags: Dict[str, object] = {
            "legal_intent_signals": legal_intents,
            "legal_intents": legal_intents,
            "legal_intent_count": len(legal_intents),
        }
        for legal_intent in legal_intents:
            metadata_flags[legal_intent] = True

        if self._is_single_intent_question(normalized_query_text, legal_intents):
            metadata_flags["single_intent_question"] = True

        return metadata_flags

    def _is_single_intent_question(
        self,
        normalized_query_text: str,
        legal_intents: List[str],
    ) -> bool:
        """
        Determine whether extracted legal signals describe one operational issue.

        Parameters
        ----------
        normalized_query_text : str
            Accent-free normalized query text used for deterministic matching.

        legal_intents : List[str]
            Ordered legal-intent signals extracted from the query.

        Returns
        -------
        bool
            `True` when the query is focused enough for single-anchor treatment.
        """

        if not legal_intents:
            return False
        if _COMPARATIVE_INTENT_PATTERN.search(normalized_query_text):
            return False

        material_intents = [
            legal_intent
            for legal_intent in legal_intents
            if legal_intent not in _QUALIFIER_INTENT_SIGNALS
        ]
        has_payment_intent = any(
            legal_intent in _PAYMENT_INTENT_SIGNALS
            for legal_intent in material_intents
        )
        focus_count = 1 if has_payment_intent else 0

        for legal_intent in material_intents:
            if legal_intent not in _PAYMENT_INTENT_SIGNALS:
                focus_count += 1

        return focus_count == 1

    def _strip_pattern_matches(
        self,
        text: str,
        pattern: Pattern[str],
    ) -> Tuple[str, List[str]]:
        """
        Remove all matches of one directive pattern from the working text.

        Parameters
        ----------
        text : str
            Current query text candidate.

        pattern : Pattern[str]
            Directive pattern to remove.

        Returns
        -------
        Tuple[str, List[str]]
            Updated text plus extracted directive fragments.
        """

        extracted_directives: List[str] = []

        def _replacement(match: Match[str]) -> str:
            """
            Record one matched directive while removing it from the text.

            Parameters
            ----------
            match : Match[str]
                Regular-expression match for one directive span.

            Returns
            -------
            str
                One separator placeholder preserving surrounding readability.
            """

            cleaned_directive = _clean_directive_text(match.group(0))
            if cleaned_directive:
                extracted_directives.append(cleaned_directive)
            return " "

        updated_text = pattern.sub(_replacement, text)
        return _normalize_whitespace(updated_text), extracted_directives
