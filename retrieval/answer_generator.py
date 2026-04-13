from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from Chunking.config.settings import PipelineSettings
from retrieval.models import AnswerGenerationInput


class AnswerGenerationError(RuntimeError):
    """
    Raised when the answer-generation adapter cannot produce a response.
    """


@dataclass(slots=True)
class GeneratedAnswer:
    """
    Normalized answer payload returned by the generation adapter.

    Parameters
    ----------
    answer_text : str
        Final answer text returned to the retrieval service.

    grounded : bool
        Whether the adapter considers the answer grounded in the supplied
        retrieval context.

    metadata : Dict[str, Any]
        Minimal adapter metadata needed by the retrieval service.
    """

    answer_text: str
    grounded: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize the generated-answer payload after initialization.
        """

        self.answer_text = self.answer_text.strip()
        self.metadata = dict(self.metadata)


class AnswerGenerator(Protocol):
    """
    Structural contract implemented by grounded answer generators.
    """

    def generate_answer(
        self,
        generation_input: AnswerGenerationInput,
    ) -> GeneratedAnswer:
        """
        Generate one grounded answer from the provided retrieval payload.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized generation payload containing question, context, and
            optional instructions.

        Returns
        -------
        GeneratedAnswer
            Generated answer text plus minimal adapter metadata.
        """


GeneratorBuilder = Callable[[PipelineSettings], AnswerGenerator]


def create_answer_generator(
    settings: Optional[PipelineSettings] = None,
) -> AnswerGenerator:
    """
    Build the answer generator configured in shared runtime settings.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared project settings. Default settings are loaded when omitted.

    Returns
    -------
    AnswerGenerator
        Configured generator instance ready for grounded answer generation.
    """

    resolved_settings = settings or PipelineSettings()

    if not resolved_settings.response_generation_enabled:
        raise ValueError("Response generation is disabled in runtime settings.")

    provider_name = _resolve_provider_name(
        resolved_settings.response_generation_provider
    )
    generator_builder = _get_generator_builders().get(provider_name)

    if generator_builder is None:
        supported_providers = ", ".join(sorted(_get_generator_builders()))
        raise ValueError(
            "Unsupported response-generation provider configured in settings: "
            f"'{resolved_settings.response_generation_provider}'. "
            f"Supported providers: {supported_providers}."
        )

    return generator_builder(resolved_settings)


def _get_generator_builders() -> Dict[str, GeneratorBuilder]:
    """
    Return the internal registry of supported answer-generator builders.

    Returns
    -------
    Dict[str, GeneratorBuilder]
        Mapping between normalized provider names and builder callables.
    """

    return {
        "openai": _build_openai_answer_generator,
    }


def _resolve_provider_name(provider_name: str) -> str:
    """
    Normalize the configured answer-generation provider name.

    Parameters
    ----------
    provider_name : str
        Raw provider name from settings.

    Returns
    -------
    str
        Lowercase provider name without leading or trailing whitespace.
    """

    normalized_provider_name = provider_name.strip().lower()
    if not normalized_provider_name:
        raise ValueError("Response-generation provider name cannot be empty.")

    return normalized_provider_name


@dataclass(slots=True)
class OpenAIAnswerGenerator:
    """
    Generate grounded answers through the OpenAI chat-completions API.

    Parameters
    ----------
    model : str
        Chat model used for answer generation.

    grounded_fallback_enabled : bool
        Whether a deterministic fallback answer should be returned when no
        grounded context is available.
    """

    model: str
    grounded_fallback_enabled: bool = True

    def __post_init__(self) -> None:
        """
        Validate generator configuration after dataclass initialization.
        """

        normalized_model = self.model.strip()
        if not normalized_model:
            raise ValueError("Response-generation model cannot be empty.")

        self.model = normalized_model

    def generate_answer(
        self,
        generation_input: AnswerGenerationInput,
    ) -> GeneratedAnswer:
        """
        Generate one grounded answer for the provided question and context.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        GeneratedAnswer
            Answer text plus minimal metadata for retrieval orchestration.
        """

        self._validate_generation_input(generation_input)

        if (
            self.grounded_fallback_enabled
            and not generation_input.context.context_text
        ):
            return self._build_grounded_fallback_answer(generation_input)

        client = self._build_client()
        response = self._request_completion(
            client=client,
            generation_input=generation_input,
        )
        answer_text = self._extract_answer_text(response)

        if not answer_text:
            raise AnswerGenerationError(
                "OpenAI answer-generation response did not include answer text."
            )

        return GeneratedAnswer(
            answer_text=answer_text,
            grounded=bool(generation_input.context.context_text),
            metadata={
                "provider": "openai",
                "model": self.model,
                "used_grounded_fallback": False,
                "context_chunk_count": generation_input.context.chunk_count,
                "context_chunk_ids": [
                    chunk.chunk_id for chunk in generation_input.context.chunks
                ],
                "usage": self._extract_usage(response),
            },
        )

    def _validate_generation_input(
        self,
        generation_input: AnswerGenerationInput,
    ) -> None:
        """
        Validate the required answer-generation input fields.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Candidate generation payload supplied by the caller.
        """

        if not isinstance(generation_input, AnswerGenerationInput):
            raise ValueError(
                "Generation input must be an AnswerGenerationInput instance."
            )

        if not generation_input.question.question_text:
            raise ValueError("Question text cannot be empty.")

    def _build_grounded_fallback_answer(
        self,
        generation_input: AnswerGenerationInput,
    ) -> GeneratedAnswer:
        """
        Build a deterministic grounded fallback answer without calling the LLM.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized generation payload with missing context.

        Returns
        -------
        GeneratedAnswer
            Deterministic fallback answer preserving grounding discipline.
        """

        return GeneratedAnswer(
            answer_text=(
                "No reliable grounded context was retrieved for this question. "
                "A supported answer cannot be generated safely."
            ),
            grounded=False,
            metadata={
                "provider": "openai",
                "model": self.model,
                "used_grounded_fallback": True,
                "context_chunk_count": generation_input.context.chunk_count,
                "context_chunk_ids": [],
                "usage": {},
            },
        )

    def _build_client(self) -> Any:
        """
        Build the OpenAI client using the configured environment variable.

        Returns
        -------
        Any
            OpenAI SDK client instance.
        """

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise AnswerGenerationError(
                "Environment variable 'OPENAI_API_KEY' is required for "
                "OpenAI answer generation."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AnswerGenerationError(
                "Package 'openai' is required to use the OpenAI answer generator."
            ) from exc

        return OpenAI(api_key=api_key)

    def _request_completion(
        self,
        client: Any,
        generation_input: AnswerGenerationInput,
    ) -> Any:
        """
        Request one grounded completion from the OpenAI chat API.

        Parameters
        ----------
        client : Any
            OpenAI SDK client instance.

        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        Any
            Raw provider response returned by the SDK.
        """

        messages = self._build_messages(generation_input)

        try:
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except Exception as exc:
            raise AnswerGenerationError(
                "OpenAI answer-generation request failed with model "
                f"'{self.model}': {exc}"
            ) from exc

    def _build_messages(
        self,
        generation_input: AnswerGenerationInput,
    ) -> List[Dict[str, str]]:
        """
        Build the grounded chat prompt sent to the generation provider.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        List[Dict[str, str]]
            Ordered chat messages compatible with the OpenAI SDK.
        """

        system_instruction = self._build_system_instruction(generation_input)
        user_prompt = self._build_user_prompt(generation_input)

        return [
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def _build_system_instruction(
        self,
        generation_input: AnswerGenerationInput,
    ) -> str:
        """
        Build the effective system instruction for grounded generation.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        str
            Final system instruction sent to the provider.
        """

        instruction_parts = [
            "Answer the user using only the grounded context provided.",
            "Do not invent facts, citations, or legal conclusions not supported by the context.",
            "If the context is insufficient, state that clearly and keep the answer concise.",
        ]

        if generation_input.system_instruction:
            instruction_parts.append(generation_input.system_instruction)

        if generation_input.grounding_instruction:
            instruction_parts.append(generation_input.grounding_instruction)

        return "\n".join(instruction_parts)

    def _build_user_prompt(
        self,
        generation_input: AnswerGenerationInput,
    ) -> str:
        """
        Build the user prompt carrying context and question to the provider.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        str
            User prompt string sent to the provider.
        """

        context_text = generation_input.context.context_text.strip()
        question_text = generation_input.question.question_text.strip()

        return (
            "Grounded context:\n"
            f"{context_text}\n\n"
            "User question:\n"
            f"{question_text}"
        )

    def _extract_answer_text(self, response: Any) -> str:
        """
        Extract the first answer text from one OpenAI chat response.

        Parameters
        ----------
        response : Any
            Raw provider response returned by the SDK.

        Returns
        -------
        str
            Stripped answer text or an empty string when unavailable.
        """

        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            return ""

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            extracted_parts: List[str] = []

            for part in content:
                if isinstance(part, dict):
                    text_value = part.get("text", "")
                    if isinstance(text_value, str) and text_value.strip():
                        extracted_parts.append(text_value.strip())

            return "\n".join(extracted_parts).strip()

        return ""

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        """
        Extract normalized token-usage metadata from one provider response.

        Parameters
        ----------
        response : Any
            Raw provider response returned by the SDK.

        Returns
        -------
        Dict[str, int]
            Optional token-usage counters when available.
        """

        usage = getattr(response, "usage", None)
        if usage is None:
            return {}

        usage_mapping: Dict[str, int] = {}

        for attribute_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            raw_value = getattr(usage, attribute_name, None)
            if isinstance(raw_value, int):
                usage_mapping[attribute_name] = raw_value

        return usage_mapping


def _build_openai_answer_generator(
    settings: PipelineSettings,
) -> OpenAIAnswerGenerator:
    """
    Build the OpenAI answer generator from shared runtime settings.

    Parameters
    ----------
    settings : PipelineSettings
        Shared runtime settings.

    Returns
    -------
    OpenAIAnswerGenerator
        OpenAI generator configured with the selected chat model.
    """

    return OpenAIAnswerGenerator(
        model=settings.response_generation_model,
        grounded_fallback_enabled=(
            settings.response_generation_grounded_fallback_enabled
        ),
    )
