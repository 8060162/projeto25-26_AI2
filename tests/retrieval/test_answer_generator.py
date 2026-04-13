"""Regression tests for grounded answer-generation integration."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Chunking.config.settings import PipelineSettings
from retrieval.answer_generator import (
    OpenAIAnswerGenerator,
    create_answer_generator,
)
from retrieval.models import AnswerGenerationInput, RetrievalContext, UserQuestionInput


class AnswerGeneratorFactoryTests(unittest.TestCase):
    """Protect answer-generator selection through runtime settings."""

    def test_create_answer_generator_returns_openai_generator(self) -> None:
        """Ensure the configured OpenAI generator resolves correctly."""

        settings = PipelineSettings(
            response_generation_provider=" OPENAI ",
            response_generation_model="gpt-4o-mini",
        )

        generator = create_answer_generator(settings)

        self.assertIsInstance(generator, OpenAIAnswerGenerator)
        self.assertEqual(generator.model, "gpt-4o-mini")

    def test_create_answer_generator_raises_for_unsupported_provider(self) -> None:
        """Ensure unsupported answer-generation providers fail clearly."""

        settings = PipelineSettings(
            response_generation_provider="unsupported_provider",
            response_generation_model="gpt-4o",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported response-generation provider configured in settings",
        ):
            create_answer_generator(settings)


class OpenAIAnswerGeneratorTests(unittest.TestCase):
    """Protect the grounded OpenAI answer-generator contract."""

    def test_generate_answer_returns_grounded_fallback_without_context(self) -> None:
        """Ensure missing grounded context returns a deterministic fallback answer."""

        generator = OpenAIAnswerGenerator(
            model="gpt-4o",
            grounded_fallback_enabled=True,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(),
        )

        with patch.object(OpenAIAnswerGenerator, "_build_client") as mocked_build_client:
            result = generator.generate_answer(generation_input)

        mocked_build_client.assert_not_called()
        self.assertFalse(result.grounded)
        self.assertTrue(result.metadata["used_grounded_fallback"])
        self.assertIn("No reliable grounded context", result.answer_text)

    def test_generate_answer_calls_openai_with_grounded_prompt(self) -> None:
        """Ensure grounded generation forwards the expected prompt and metadata."""

        generator = OpenAIAnswerGenerator(
            model="gpt-4o",
            grounded_fallback_enabled=True,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What is the filing deadline?"),
            context=RetrievalContext(
                context_text="Article 5 states the filing deadline is 10 working days.",
                chunk_count=1,
            ),
            system_instruction="Use a formal tone.",
            grounding_instruction="Cite the article number when present.",
        )
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="According to Article 5, the filing deadline is 10 working days."
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=9,
                total_tokens=21,
            ),
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: fake_response)
            )
        )

        with patch.object(
            OpenAIAnswerGenerator,
            "_build_client",
            return_value=fake_client,
        ):
            with patch.object(
                OpenAIAnswerGenerator,
                "_request_completion",
                wraps=generator._request_completion,
            ) as wrapped_request:
                result = generator.generate_answer(generation_input)

        request_kwargs = wrapped_request.call_args.kwargs
        self.assertEqual(request_kwargs["client"], fake_client)
        self.assertEqual(
            request_kwargs["generation_input"].question.question_text,
            "What is the filing deadline?",
        )
        self.assertTrue(result.grounded)
        self.assertFalse(result.metadata["used_grounded_fallback"])
        self.assertEqual(result.metadata["provider"], "openai")
        self.assertEqual(result.metadata["model"], "gpt-4o")
        self.assertEqual(
            result.metadata["usage"],
            {
                "prompt_tokens": 12,
                "completion_tokens": 9,
                "total_tokens": 21,
            },
        )
        self.assertIn("Article 5", result.answer_text)


if __name__ == "__main__":
    unittest.main()
