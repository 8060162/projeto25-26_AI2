from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Pattern, Sequence

from Chunking.config.settings import PipelineSettings
from retrieval.models import GuardrailDecision, RetrievalContext, UserQuestionInput


@dataclass(frozen=True, slots=True)
class _RuleSpec:
    """
    Describe one deterministic guardrail rule.

    Attributes
    ----------
    rule_name : str
        Stable identifier reported in the guardrail decision.

    category : str
        Logical safety category associated with the rule.

    reason : str
        Human-readable explanation emitted when the rule matches.

    pattern : Pattern[str]
        Compiled regular expression used for deterministic matching.
    """

    rule_name: str
    category: str
    reason: str
    pattern: Pattern[str]


def _compile_pattern(expression: str) -> Pattern[str]:
    """
    Compile one case-insensitive multiline regular expression safely.

    Parameters
    ----------
    expression : str
        Raw regular expression source.

    Returns
    -------
    Pattern[str]
        Compiled pattern used by deterministic guardrails.
    """

    return re.compile(expression, re.IGNORECASE | re.MULTILINE)


class DeterministicGuardrails:
    """
    Evaluate deterministic safety checks before and after answer generation.

    Design goals
    ------------
    - keep guardrail logic centralized in one module
    - make every decision explicit and explainable
    - remain fully deterministic and easy to test
    """

    _PRE_REQUEST_RULES: Dict[str, Sequence[_RuleSpec]] = {
        "offensive_language": (
            _RuleSpec(
                rule_name="offensive_language.profanity",
                category="offensive_language",
                reason="The request contains offensive or abusive language.",
                pattern=_compile_pattern(
                    r"\b(fuck|shit|bitch|asshole|bastard|idiot|moron|dumbass)\b"
                ),
            ),
        ),
        "sexual_content": (
            _RuleSpec(
                rule_name="sexual_content.explicit_terms",
                category="sexual_content",
                reason="The request contains explicit sexual content.",
                pattern=_compile_pattern(
                    r"\b(sex|sexual|porn|pornography|nude|nudes|fetish|xxx)\b"
                ),
            ),
        ),
        "discriminatory_content": (
            _RuleSpec(
                rule_name="discriminatory_content.slurs_or_bias",
                category="discriminatory_content",
                reason="The request contains discriminatory or hateful language.",
                pattern=_compile_pattern(
                    r"\b("
                    r"racial slur|ethnic cleansing|white power|black people are inferior|"
                    r"hate (women|men|gays|muslims|jews|immigrants)|"
                    r"inferior race|superior race"
                    r")\b"
                ),
            ),
        ),
        "criminal_or_dangerous_content": (
            _RuleSpec(
                rule_name="criminal_or_dangerous_content.illicit_harm",
                category="criminal_or_dangerous_content",
                reason="The request asks for criminal, violent, or dangerous guidance.",
                pattern=_compile_pattern(
                    r"\b("
                    r"build a bomb|make a bomb|how to kill|murder someone|"
                    r"poison (someone|a person)|bypass security|break into|"
                    r"steal credentials|phishing kit|malware|ransomware|explosive device"
                    r")\b"
                ),
            ),
        ),
        "sensitive_data": (
            _RuleSpec(
                rule_name="sensitive_data.secret_exfiltration",
                category="sensitive_data",
                reason="The request appears to seek sensitive data or secrets.",
                pattern=_compile_pattern(
                    r"\b("
                    r"password|passcode|api key|access token|secret key|private key|"
                    r"ssh key|credit card number|cvv|security code|social security number|"
                    r"bank account number|iban"
                    r")\b"
                ),
            ),
        ),
        "dangerous_command": (
            _RuleSpec(
                rule_name="dangerous_command.destructive_shell",
                category="dangerous_command",
                reason="The request includes a dangerous system command pattern.",
                pattern=_compile_pattern(
                    r"("
                    r"\brm\s+-rf\s+/\b|"
                    r"\bmkfs(\.\w+)?\b|"
                    r"\bdd\s+if=.*\bof=/dev/|"
                    r"\bshutdown\s+-h\s+now\b|"
                    r"\bcurl\b.+\|\s*(sh|bash)\b|"
                    r"\bchmod\s+-R\s+777\b"
                    r")"
                ),
            ),
        ),
    }

    _PRE_REQUEST_JAILBREAK_RULES: Sequence[_RuleSpec] = (
        _RuleSpec(
            rule_name="jailbreak_pattern.prompt_override",
            category="jailbreak_pattern",
            reason="The request contains a prompt-override or jailbreak attempt.",
            pattern=_compile_pattern(
                r"\b("
                r"ignore (all|any|the|previous) instructions|"
                r"disregard (all|previous) instructions|"
                r"reveal (the )?(system|hidden) prompt|"
                r"bypass (your|all) safety"
                r")\b"
            ),
        ),
    )

    _PORTUGUESE_PRE_REQUEST_RULES: Dict[str, Sequence[_RuleSpec]] = {
        "offensive_language": (
            _RuleSpec(
                rule_name="offensive_language.portuguese_profanity",
                category="offensive_language",
                reason="The request contains offensive or abusive language in Portuguese.",
                pattern=_compile_pattern(
                    r"\b("
                    r"cabr[aã]o|filh[oa] da puta|fdp|ot[aá]rio|imbecil|"
                    r"est[uú]pid[oa]|burro|caralho|merda|foda[- ]?se|"
                    r"vai[- ]?te foder|vai se foder"
                    r")\b"
                ),
            ),
        ),
        "sexual_content": (
            _RuleSpec(
                rule_name="sexual_content.portuguese_explicit_terms",
                category="sexual_content",
                reason="The request contains explicit sexual content in Portuguese.",
                pattern=_compile_pattern(
                    r"\b("
                    r"pornografia|porno|porn[oó]gr[aá]fico|"
                    r"conte[uú]do sexual expl[ií]cito|sexo expl[ií]cito|"
                    r"nudes?|material pornogr[aá]fico|"
                    r"conte[uú]do er[oó]tico|v[ií]deo [ií]ntimo|imagens? [ií]ntimas?|"
                    r"atos? sexuais expl[ií]citos?"
                    r")\b"
                ),
            ),
        ),
        "discriminatory_content": (
            _RuleSpec(
                rule_name="discriminatory_content.portuguese_slurs_or_bias",
                category="discriminatory_content",
                reason="The request contains discriminatory or hateful language in Portuguese.",
                pattern=_compile_pattern(
                    r"\b("
                    r"supremacia branca|poder branco|limpeza [ée]tnica|"
                    r"ra[çc]a inferior|ra[çc]a superior|"
                    r"rac[aã] pura|"
                    r"(pretos|negros|ciganos|judeus|mu[çc]ulmanos|imigrantes)"
                    r"\s+s[aã]o\s+(inferiores|uma praga|criminosos)|"
                    r"odeio (mulheres|homens|gays|homossexuais|mu[çc]ulmanos|judeus|imigrantes)|"
                    r"expulsar (imigrantes|mu[çc]ulmanos|judeus)"
                    r")\b"
                ),
            ),
        ),
        "criminal_or_dangerous_content": (
            _RuleSpec(
                rule_name="criminal_or_dangerous_content.portuguese_illicit_harm",
                category="criminal_or_dangerous_content",
                reason="The request asks for criminal, violent, or dangerous guidance in Portuguese.",
                pattern=_compile_pattern(
                    r"\b("
                    r"como (fazer|fabricar) (uma )?bomba|"
                    r"fabricar explosiv[oa]s?|"
                    r"como matar algu[ée]m|"
                    r"envenenar algu[ée]m|"
                    r"invadir (um )?(sistema|site|servidor|conta)|"
                    r"contornar a seguran[çc]a|"
                    r"contornar (a )?(autentica[çc][aã]o|verifica[çc][aã]o|mfa|2fa)|"
                    r"obter acesso n[aã]o autorizado|"
                    r"aceder sem permiss[aã]o|"
                    r"elevar privil[eé]gios|escalar privil[eé]gios|"
                    r"roubar credenciais|"
                    r"kit de phishing|"
                    r"malware|ransomware"
                    r")\b"
                ),
            ),
            _RuleSpec(
                rule_name="criminal_or_dangerous_content.portuguese_indirect_unauthorized_student_access",
                category="criminal_or_dangerous_content",
                reason="The request seeks indirect unauthorized access to another person's academic account or private data in Portuguese.",
                pattern=_compile_pattern(
                    r"("
                    r"\b("
                    r"entrar|aceder|abrir|usar|consultar|ver|espreitar|"
                    r"observar|explorar|navegar"
                    r")\b.{0,40}\b("
                    r"conta|perfil|sess[aã]o|portal|[aá]rea reservada|"
                    r"intranet|credenciais|login"
                    r")\b.{0,40}\b("
                    r"de outro aluno|de outra aluna|de outro estudante|"
                    r"de outra estudante|de outro utilizador|de outra pessoa|"
                    r"alhe[ií]a|que n[aã]o [ée] minha|que n[aã]o me pertence"
                    r")"
                    r"|"
                    r"\b("
                    r"usar|aproveitar|reutilizar|inserir|testar"
                    r")\b.{0,40}\b("
                    r"credenciais|login|palavra[- ]?passe|senha|conta"
                    r")\b.{0,40}\b("
                    r"de outro aluno|de outra aluna|de outro estudante|"
                    r"de outra estudante|de outra pessoa|alhe[ií]as"
                    r")"
                    r"|"
                    r"\b("
                    r"ver|consultar|aceder|obter|mostrar"
                    r")\b.{0,40}\b("
                    r"dados acad[eé]micos|dados privados|notas|propinas|"
                    r"hist[oó]rico escolar|avalia[çc][õo]es|inscri[çc][õo]es|"
                    r"documentos pessoais"
                    r")\b.{0,50}\b("
                    r"de outro aluno|de outra aluna|de outro estudante|"
                    r"de outra estudante|de outra pessoa"
                    r")\b.{0,50}\b("
                    r"sem autoriza[çc][aã]o|sem permiss[aã]o|atrav[eé]s da conta|"
                    r"pelo portal|usando credenciais|com o login"
                    r")"
                    r")"
                ),
            ),
        ),
        "sensitive_data": (
            _RuleSpec(
                rule_name="sensitive_data.portuguese_secret_exfiltration",
                category="sensitive_data",
                reason="The request appears to seek sensitive data or secrets in Portuguese.",
                pattern=_compile_pattern(
                    r"\b("
                    r"palavra[- ]?passe|senha|chave api|token de acesso|"
                    r"chave secreta|chave privada|credenciais|"
                    r"n[uú]mero do cart[aã]o|cart[aã]o de cr[eé]dito|"
                    r"c[oó]digo de seguran[çc]a|iban|n[uú]mero de conta"
                    r")\b"
                ),
            ),
            _RuleSpec(
                rule_name="sensitive_data.portuguese_personal_data_exfiltration",
                category="sensitive_data",
                reason="The request appears to seek personal or student data in Portuguese.",
                pattern=_compile_pattern(
                    r"\b("
                    r"revel[ae]|mostr[ae]|fornec[ea]|partilh[ae]|list[ae]|export[ae]|"
                    r"envi[ae]|exib[ae]|d[aá]|diz[- ]?me|obt[eé]m"
                    r")\b.{0,80}\b("
                    r"dados pessoais|dados dos alunos|dados de estudantes|"
                    r"lista de alunos|lista de estudantes|"
                    r"emails? dos alunos|emails? dos estudantes|"
                    r"moradas?|contactos? pessoais|nif|n[uú]mero de identifica[çc][aã]o"
                    r")\b"
                ),
            ),
        ),
        "dangerous_command": (
            _RuleSpec(
                rule_name="dangerous_command.portuguese_destructive_intent",
                category="dangerous_command",
                reason="The request describes destructive system or database actions in Portuguese.",
                pattern=_compile_pattern(
                    r"\b("
                    r"apaga|apagar|elimina|eliminar|remove|remover|"
                    r"destr[oó]i|destruir|limpa|limpar|drop"
                    r")\b.{0,60}\b("
                    r"base de dados|base de dados inteira|bd|database|"
                    r"tabelas?|registos?|logs?|backups?|ficheiros? do sistema"
                    r")\b"
                ),
            ),
        ),
    }

    _PORTUGUESE_PRE_REQUEST_JAILBREAK_RULES: Sequence[_RuleSpec] = (
        _RuleSpec(
            rule_name="jailbreak_pattern.portuguese_prompt_override",
            category="jailbreak_pattern",
            reason="The request contains a prompt-override or jailbreak attempt in Portuguese.",
            pattern=_compile_pattern(
                r"\b("
                r"ignora (as |todas as )?instru[çc][õo]es|"
                r"esquece as regras|"
                r"revela (o )?prompt (de sistema|oculto)|"
                r"contorna (as )?restri[çc][õo]es|"
                r"sem restri[çc][õo]es|"
                r"sem filtros|sem limita[çc][õo]es|"
                r"(faz de conta|finge|simula) .{0,40}(sem restri[çc][õo]es|sem filtros|"
                r"que podes ignorar as regras|que tens permiss[aã]o total)|"
                r"(isto [ée] (s[oó] )?(uma )?(simula[çc][aã]o|auditoria|roleplay)|"
                r"em modo de (simula[çc][aã]o|auditoria)).{0,40}"
                r"(ignora|contorna|sem restri[çc][õo]es|sem filtros)"
                r")\b"
            ),
        ),
    )

    _POST_RESPONSE_UNSAFE_RULES: Sequence[_RuleSpec] = (
        _RuleSpec(
            rule_name="unsafe_output.dangerous_command",
            category="unsafe_output",
            reason="The response contains a dangerous command pattern.",
            pattern=_compile_pattern(
                r"("
                r"\brm\s+-rf\s+/\b|"
                r"\bmkfs(\.\w+)?\b|"
                r"\bdd\s+if=.*\bof=/dev/|"
                r"\bcurl\b.+\|\s*(sh|bash)\b"
                r")"
            ),
        ),
    )

    _POST_RESPONSE_SENSITIVE_DATA_RULES: Sequence[_RuleSpec] = (
        _RuleSpec(
            rule_name="unsafe_output.sensitive_data",
            category="unsafe_output",
            reason="The response appears to expose sensitive data.",
            pattern=_compile_pattern(
                r"("
                r"sk-[a-z0-9]{12,}|"
                r"-----BEGIN (RSA |EC )?PRIVATE KEY-----|"
                r"\bpassword\s*[:=]\s*\S+|"
                r"\bsenha\s*[:=]\s*\S+|"
                r"\bpalavra[- ]?passe\s*[:=]\s*\S+|"
                r"\bapi[_ -]?key\s*[:=]\s*\S+|"
                r"\bchave api\s*[:=]\s*\S+|"
                r"\btoken de acesso\s*[:=]\s*\S+|"
                r"\bcvv\s*[:=]?\s*\d{3,4}\b"
                r")"
            ),
        ),
        _RuleSpec(
            rule_name="unsafe_output.illicit_instructions",
            category="unsafe_output",
            reason="The response contains unsafe criminal or harmful instructions.",
            pattern=_compile_pattern(
                r"\b("
                r"build a bomb|make a bomb|kill them|poison them|"
                r"bypass security|steal credentials|deploy malware|ransomware"
                r")\b"
            ),
        ),
    )

    _PORTUGUESE_POST_RESPONSE_UNSAFE_RULES: Sequence[_RuleSpec] = (
        _RuleSpec(
            rule_name="unsafe_output.portuguese_illicit_instructions",
            category="unsafe_output",
            reason="The response contains unsafe criminal or harmful instructions in Portuguese.",
            pattern=_compile_pattern(
                r"\b("
                r"como (fazer|fabricar) (uma )?bomba|"
                r"envenenar algu[ée]m|"
                r"roubar credenciais|"
                r"contornar a seguran[çc]a|"
                r"contornar (a )?(autentica[çc][aã]o|verifica[çc][aã]o|mfa|2fa)|"
                r"obter acesso n[aã]o autorizado|"
                r"instalar malware|"
                r"lan[çc]ar ransomware"
                r")\b"
            ),
        ),
        _RuleSpec(
            rule_name="unsafe_output.portuguese_destructive_actions",
            category="unsafe_output",
            reason="The response contains destructive system or database instructions in Portuguese.",
            pattern=_compile_pattern(
                r"\b("
                r"apaga|apagar|elimina|eliminar|remove|remover|"
                r"destr[oó]i|destruir|limpa|limpar|drop"
                r")\b.{0,60}\b("
                r"base de dados|base de dados inteira|bd|database|"
                r"tabelas?|registos?|logs?|backups?|ficheiros? do sistema"
                r")\b"
            ),
        ),
    )

    _POST_RESPONSE_UNSUPPORTED_RULES: Sequence[_RuleSpec] = (
        _RuleSpec(
            rule_name="unsupported_answer.self_referential_model",
            category="unsupported_answer",
            reason="The response uses unsupported model self-reference.",
            pattern=_compile_pattern(
                r"\b(as an ai|as a language model|my training data|i cannot browse)\b"
            ),
        ),
        _RuleSpec(
            rule_name="unsupported_answer.placeholder_content",
            category="unsupported_answer",
            reason="The response contains placeholder or unresolved content.",
            pattern=_compile_pattern(
                r"(\[citation needed\]|\btodo\b|\blorem ipsum\b|<insert .+?>)"
            ),
        ),
    )

    _PORTUGUESE_POST_RESPONSE_UNSUPPORTED_RULES: Sequence[_RuleSpec] = (
        _RuleSpec(
            rule_name="unsupported_answer.portuguese_self_referential_model",
            category="unsupported_answer",
            reason="The response uses unsupported model self-reference in Portuguese.",
            pattern=_compile_pattern(
                r"\b("
                r"como modelo de linguagem|"
                r"n[aã]o consigo navegar|"
                r"n[aã]o posso navegar|"
                r"n[aã]o tenho acesso (ao |aos |[àa] |[àa]s )?"
                r"(regulamentos?|documentos?|internet|web|conte[uú]do em tempo real)|"
                r"n[aã]o posso consultar a internet|"
                r"n[aã]o tenho acesso [àa] base de dados|"
                r"n[aã]o consigo ver (os )?documentos? anexos|"
                r"n[aã]o tenho contexto suficiente"
                r")\b"
            ),
        ),
    )

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """
        Initialize guardrails from the shared runtime settings.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared project settings. Default settings are loaded when omitted.
        """

        self.settings = settings or PipelineSettings()

    def evaluate_pre_request(
        self,
        question: UserQuestionInput | str,
    ) -> GuardrailDecision:
        """
        Evaluate deterministic guardrails before retrieval starts.

        Parameters
        ----------
        question : UserQuestionInput | str
            User question contract or plain question text.

        Returns
        -------
        GuardrailDecision
            Explainable decision describing whether the request is allowed.
        """

        if not self.settings.guardrails_enabled:
            return self._build_allow_decision(stage="pre_request")

        question_text = self._extract_question_text(question)
        enabled_rule_groups = self._get_enabled_pre_request_rule_groups()

        return self._evaluate_text_rules(
            text=question_text,
            stage="pre_request",
            action="block",
            rule_groups=enabled_rule_groups,
        )

    def evaluate_post_response(
        self,
        answer_text: str,
        context: Optional[RetrievalContext] = None,
        grounded: bool = False,
    ) -> GuardrailDecision:
        """
        Evaluate deterministic guardrails after answer generation.

        Parameters
        ----------
        answer_text : str
            Generated answer text produced by the answer adapter.

        context : Optional[RetrievalContext]
            Grounded context available to support the response.

        grounded : bool
            Explicit grounding flag emitted by the retrieval service.

        Returns
        -------
        GuardrailDecision
            Explainable decision describing whether the response is allowed.
        """

        if not self.settings.guardrails_enabled:
            return self._build_allow_decision(stage="post_response")

        normalized_answer = answer_text.strip()

        if self.settings.guardrails_post_response_unsafe_output_checks_enabled:
            unsafe_decision = self._evaluate_rule_specs(
                text=normalized_answer,
                stage="post_response",
                action="block",
                rule_specs=self._build_post_response_unsafe_rules(),
            )
            if not unsafe_decision.allowed:
                return unsafe_decision

        if self.settings.guardrails_post_response_grounded_response_checks_enabled:
            grounded_decision = self._evaluate_grounded_response(
                answer_text=normalized_answer,
                context=context,
                grounded=grounded,
            )
            if not grounded_decision.allowed:
                return grounded_decision

        if self.settings.guardrails_post_response_unsupported_answer_checks_enabled:
            unsupported_decision = self._evaluate_rule_specs(
                text=normalized_answer,
                stage="post_response",
                action="deflect",
                rule_specs=self._build_post_response_unsupported_rules(),
            )
            if not unsupported_decision.allowed:
                return unsupported_decision

        return self._build_allow_decision(stage="post_response")

    def _extract_question_text(self, question: UserQuestionInput | str) -> str:
        """
        Normalize the input question into plain text.

        Parameters
        ----------
        question : UserQuestionInput | str
            User question contract or plain string.

        Returns
        -------
        str
            Stripped question text used by pre-request guardrails.
        """

        if isinstance(question, UserQuestionInput):
            return question.question_text
        if isinstance(question, str):
            return question.strip()
        return ""

    def _get_enabled_pre_request_rule_groups(self) -> Dict[str, Sequence[_RuleSpec]]:
        """
        Resolve the enabled pre-request rule groups from runtime settings.

        Returns
        -------
        Dict[str, Sequence[_RuleSpec]]
            Mapping of enabled categories to their deterministic rules.
        """

        enabled_rule_groups: Dict[str, Sequence[_RuleSpec]] = {}

        if self.settings.guardrails_pre_request_offensive_language_checks_enabled:
            enabled_rule_groups["offensive_language"] = self._build_pre_request_rule_specs(
                category="offensive_language"
            )

        if self.settings.guardrails_pre_request_sexual_content_checks_enabled:
            enabled_rule_groups["sexual_content"] = self._build_pre_request_rule_specs(
                category="sexual_content"
            )

        if self.settings.guardrails_pre_request_discriminatory_content_checks_enabled:
            enabled_rule_groups["discriminatory_content"] = (
                self._build_pre_request_rule_specs(category="discriminatory_content")
            )

        if (
            self.settings.guardrails_pre_request_criminal_or_dangerous_content_checks_enabled
        ):
            enabled_rule_groups["criminal_or_dangerous_content"] = self._build_pre_request_rule_specs(
                category="criminal_or_dangerous_content"
            )

        if self.settings.guardrails_pre_request_sensitive_data_checks_enabled:
            enabled_rule_groups["sensitive_data"] = self._build_pre_request_rule_specs(
                category="sensitive_data"
            )

        if self.settings.guardrails_pre_request_dangerous_command_checks_enabled:
            enabled_rule_groups["dangerous_command"] = self._build_pre_request_rule_specs(
                category="dangerous_command"
            )

        if self.settings.guardrails_pre_request_jailbreak_pattern_checks_enabled:
            enabled_rule_groups["jailbreak_pattern"] = self._build_pre_request_jailbreak_rule_specs()

        return enabled_rule_groups

    def _build_pre_request_rule_specs(self, category: str) -> Sequence[_RuleSpec]:
        """
        Build one pre-request rule group with optional Portuguese coverage.

        Parameters
        ----------
        category : str
            Logical guardrail category requested by the evaluation flow.

        Returns
        -------
        Sequence[_RuleSpec]
            Ordered deterministic rules for the requested category.
        """

        rule_specs = list(self._PRE_REQUEST_RULES.get(category, ()))

        if self.settings.guardrails_portuguese_coverage_enabled:
            rule_specs.extend(self._PORTUGUESE_PRE_REQUEST_RULES.get(category, ()))

        return tuple(rule_specs)

    def _build_pre_request_jailbreak_rule_specs(self) -> Sequence[_RuleSpec]:
        """
        Build the enabled deterministic jailbreak rules for pre-request checks.

        Returns
        -------
        Sequence[_RuleSpec]
            Ordered jailbreak-pattern rules enabled for the current runtime.
        """

        rule_specs = list(self._PRE_REQUEST_JAILBREAK_RULES)

        if (
            self.settings.guardrails_portuguese_coverage_enabled
            and self.settings.guardrails_portuguese_jailbreak_pattern_checks_enabled
        ):
            rule_specs.extend(self._PORTUGUESE_PRE_REQUEST_JAILBREAK_RULES)

        return tuple(rule_specs)

    def _build_post_response_unsafe_rules(self) -> Sequence[_RuleSpec]:
        """
        Build the enabled unsafe-output rules for post-response checks.

        Returns
        -------
        Sequence[_RuleSpec]
            Ordered unsafe-output rules enabled for the current runtime.
        """

        rule_specs = list(self._POST_RESPONSE_UNSAFE_RULES)

        if self.settings.guardrails_post_response_sensitive_data_checks_enabled:
            rule_specs.extend(self._POST_RESPONSE_SENSITIVE_DATA_RULES)

        if self.settings.guardrails_portuguese_coverage_enabled:
            rule_specs.extend(self._PORTUGUESE_POST_RESPONSE_UNSAFE_RULES)

        return tuple(rule_specs)

    def _build_post_response_unsupported_rules(self) -> Sequence[_RuleSpec]:
        """
        Build the enabled unsupported-answer rules for post-response checks.

        Returns
        -------
        Sequence[_RuleSpec]
            Ordered unsupported-answer rules enabled for the current runtime.
        """

        rule_specs = list(self._POST_RESPONSE_UNSUPPORTED_RULES)

        if self.settings.guardrails_portuguese_coverage_enabled:
            rule_specs.extend(self._PORTUGUESE_POST_RESPONSE_UNSUPPORTED_RULES)

        return tuple(rule_specs)

    def _evaluate_text_rules(
        self,
        text: str,
        stage: str,
        action: str,
        rule_groups: Dict[str, Sequence[_RuleSpec]],
    ) -> GuardrailDecision:
        """
        Evaluate multiple deterministic rule groups in declaration order.

        Parameters
        ----------
        text : str
            Candidate text payload to inspect.

        stage : str
            Guardrail stage that produced the decision.

        action : str
            Action emitted when a rule matches.

        rule_groups : Dict[str, Sequence[_RuleSpec]]
            Enabled rule groups organized by logical category.

        Returns
        -------
        GuardrailDecision
            First blocking decision, otherwise an allow decision.
        """

        for rule_specs in rule_groups.values():
            decision = self._evaluate_rule_specs(
                text=text,
                stage=stage,
                action=action,
                rule_specs=rule_specs,
            )
            if not decision.allowed:
                return decision

        return self._build_allow_decision(stage=stage)

    def _evaluate_rule_specs(
        self,
        text: str,
        stage: str,
        action: str,
        rule_specs: Iterable[_RuleSpec],
    ) -> GuardrailDecision:
        """
        Evaluate one ordered collection of deterministic rules.

        Parameters
        ----------
        text : str
            Candidate text payload to inspect.

        stage : str
            Guardrail stage that produced the decision.

        action : str
            Action emitted when a rule matches.

        rule_specs : Iterable[_RuleSpec]
            Ordered deterministic rule specifications.

        Returns
        -------
        GuardrailDecision
            First matching decision, otherwise an allow decision.
        """

        for rule_spec in rule_specs:
            matched_rules = self._collect_matches(text=text, rule_spec=rule_spec)
            if not matched_rules:
                continue

            return GuardrailDecision(
                stage=stage,
                allowed=False,
                category=rule_spec.category,
                action=action,
                reason=rule_spec.reason,
                matched_rules=matched_rules,
                metadata={
                    "match_count": len(matched_rules),
                },
            )

        return self._build_allow_decision(stage=stage)

    def _evaluate_grounded_response(
        self,
        answer_text: str,
        context: Optional[RetrievalContext],
        grounded: bool,
    ) -> GuardrailDecision:
        """
        Deflect answers that do not have reliable grounded context.

        Parameters
        ----------
        answer_text : str
            Generated answer text.

        context : Optional[RetrievalContext]
            Grounded context assembled by the retrieval layer.

        grounded : bool
            Explicit grounding flag emitted by the service.

        Returns
        -------
        GuardrailDecision
            Allow decision when grounding is sufficient, otherwise deflection.
        """

        if not answer_text:
            return self._build_allow_decision(stage="post_response")

        if grounded:
            return self._build_allow_decision(stage="post_response")

        has_context = False
        context_chunk_count = 0

        if context is not None:
            context_chunk_count = context.chunk_count or len(context.chunks)
            has_context = bool(context.context_text.strip()) and context_chunk_count > 0

        if has_context:
            return self._build_allow_decision(stage="post_response")

        return GuardrailDecision(
            stage="post_response",
            allowed=False,
            category="grounded_response",
            action="deflect",
            reason="The response must be deflected because no reliable grounded context exists.",
            matched_rules=["grounded_response.missing_context"],
            metadata={
                "grounded": grounded,
                "context_chunk_count": context_chunk_count,
                "has_context_text": bool(
                    context is not None and context.context_text.strip()
                ),
            },
        )

    def _collect_matches(self, text: str, rule_spec: _RuleSpec) -> List[str]:
        """
        Collect stable match descriptors for one rule evaluation.

        Parameters
        ----------
        text : str
            Candidate text payload to inspect.

        rule_spec : _RuleSpec
            Deterministic rule specification being evaluated.

        Returns
        -------
        List[str]
            Stable matched-rule descriptors reported in the decision payload.
        """

        if not text:
            return []

        matches = rule_spec.pattern.findall(text)
        if not matches:
            return []

        return [rule_spec.rule_name]

    def _build_allow_decision(self, stage: str) -> GuardrailDecision:
        """
        Build one normalized allow decision for the requested stage.

        Parameters
        ----------
        stage : str
            Guardrail stage emitting the allow decision.

        Returns
        -------
        GuardrailDecision
            Normalized allow decision with no matched rules.
        """

        return GuardrailDecision(
            stage=stage,
            allowed=True,
            category="",
            action="allow",
            reason="",
            matched_rules=[],
            metadata={},
        )
