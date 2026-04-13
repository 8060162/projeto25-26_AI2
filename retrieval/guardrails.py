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
        _RuleSpec(
            rule_name="unsafe_output.sensitive_data",
            category="unsafe_output",
            reason="The response appears to expose sensitive data.",
            pattern=_compile_pattern(
                r"("
                r"sk-[a-z0-9]{12,}|"
                r"-----BEGIN (RSA |EC )?PRIVATE KEY-----|"
                r"\bpassword\s*[:=]\s*\S+|"
                r"\bapi[_ -]?key\s*[:=]\s*\S+|"
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
                rule_specs=self._POST_RESPONSE_UNSAFE_RULES,
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
                rule_specs=self._POST_RESPONSE_UNSUPPORTED_RULES,
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
            enabled_rule_groups["offensive_language"] = self._PRE_REQUEST_RULES[
                "offensive_language"
            ]

        if self.settings.guardrails_pre_request_sexual_content_checks_enabled:
            enabled_rule_groups["sexual_content"] = self._PRE_REQUEST_RULES[
                "sexual_content"
            ]

        if self.settings.guardrails_pre_request_discriminatory_content_checks_enabled:
            enabled_rule_groups["discriminatory_content"] = self._PRE_REQUEST_RULES[
                "discriminatory_content"
            ]

        if (
            self.settings.guardrails_pre_request_criminal_or_dangerous_content_checks_enabled
        ):
            enabled_rule_groups["criminal_or_dangerous_content"] = (
                self._PRE_REQUEST_RULES["criminal_or_dangerous_content"]
            )

        if self.settings.guardrails_pre_request_sensitive_data_checks_enabled:
            enabled_rule_groups["sensitive_data"] = self._PRE_REQUEST_RULES[
                "sensitive_data"
            ]

        if self.settings.guardrails_pre_request_dangerous_command_checks_enabled:
            enabled_rule_groups["dangerous_command"] = self._PRE_REQUEST_RULES[
                "dangerous_command"
            ]

        return enabled_rule_groups

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
