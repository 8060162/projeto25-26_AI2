"""Regression tests for grounded answer-generation integration."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Chunking.config.settings import PipelineSettings
from retrieval.answer_generator import (
    AnswerGenerationError,
    OpenAIAnswerGenerator,
    create_answer_generator,
)
from retrieval.external_gpt4o_answer_generator import ExternalGPT4oAnswerGenerator
from retrieval.models import (
    AnswerGenerationInput,
    RetrievalContext,
    RetrievedChunkResult,
    UserQuestionInput,
)


class AnswerGeneratorFactoryTests(unittest.TestCase):
    """Protect answer-generator selection through runtime settings."""

    def test_create_answer_generator_returns_openai_generator(self) -> None:
        """Ensure the configured OpenAI generator resolves correctly."""

        settings = PipelineSettings(
            response_generation_provider=" OPENAI ",
            response_generation_model="gpt-4o-mini",
            response_generation_openai_api_key_env_var="DIRECT_OPENAI_API_KEY",
        )

        generator = create_answer_generator(settings)

        self.assertIsInstance(generator, OpenAIAnswerGenerator)
        self.assertEqual(generator.model, "gpt-4o-mini")
        self.assertEqual(generator.api_key_env_var, "DIRECT_OPENAI_API_KEY")

    def test_create_answer_generator_accepts_openai_direct_alias(self) -> None:
        """Ensure the direct OpenAI alias resolves to the OpenAI generator."""

        settings = PipelineSettings(
            response_generation_provider=" openai_direct ",
            response_generation_model="gpt-4o",
        )

        generator = create_answer_generator(settings)

        self.assertIsInstance(generator, OpenAIAnswerGenerator)
        self.assertEqual(generator.model, "gpt-4o")

    def test_create_answer_generator_returns_external_gpt4o_generator(self) -> None:
        """Ensure the configured external GPT-4o generator resolves correctly."""

        settings = PipelineSettings(
            response_generation_provider=" external_gpt4o ",
            response_generation_external_gpt4o_endpoint_url=(
                "https://example.test/gpt4o"
            ),
            response_generation_external_gpt4o_auth_env_var="GPT4O_PROXY_TOKEN",
            response_generation_external_gpt4o_question_field_name="message",
            response_generation_external_gpt4o_context_field_name="evidence",
            response_generation_external_gpt4o_instructions_field_name="guidance",
            response_generation_external_gpt4o_metadata_field_name="payload_metadata",
            response_generation_external_gpt4o_auth_header_name="x-api-key",
            response_generation_external_gpt4o_auth_header_prefix="",
            response_generation_external_gpt4o_timeout_seconds=12,
            response_generation_external_gpt4o_max_retries=1,
            response_generation_grounded_fallback_enabled=False,
        )

        generator = create_answer_generator(settings)

        self.assertIsInstance(generator, ExternalGPT4oAnswerGenerator)
        self.assertEqual(generator.endpoint_url, "https://example.test/gpt4o")
        self.assertEqual(generator.auth_env_var, "GPT4O_PROXY_TOKEN")
        self.assertEqual(generator.question_field_name, "message")
        self.assertEqual(generator.context_field_name, "evidence")
        self.assertEqual(generator.instructions_field_name, "guidance")
        self.assertEqual(generator.metadata_field_name, "payload_metadata")
        self.assertEqual(generator.auth_header_name, "x-api-key")
        self.assertEqual(generator.auth_header_prefix, "")
        self.assertEqual(generator.timeout_seconds, 12.0)
        self.assertEqual(generator.max_retries, 1)
        self.assertFalse(generator.grounded_fallback_enabled)

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

    def test_request_completion_logs_openai_rate_limit_errors(self) -> None:
        """Ensure OpenAI throttling is logged with provider-safe metadata."""

        class RateLimitLikeError(RuntimeError):
            """Test exception carrying an HTTP-style status code."""

            status_code = 429

        generator = OpenAIAnswerGenerator(
            model="gpt-4o",
            api_key_env_var="DIRECT_OPENAI_API_KEY",
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What is the filing deadline?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: (_ for _ in ()).throw(
                        RateLimitLikeError("Rate limit reached")
                    )
                )
            )
        )

        with self.assertLogs("retrieval.answer_generator", level="WARNING") as logs:
            with self.assertRaisesRegex(
                AnswerGenerationError,
                "Rate limit reached",
            ):
                generator._request_completion(
                    client=fake_client,
                    generation_input=generation_input,
                )

        self.assertIn("rate limiting", "\n".join(logs.output))
        self.assertIn("DIRECT_OPENAI_API_KEY", "\n".join(logs.output))

    def test_build_messages_explicitly_exposes_legal_anchors_and_query_separation(
        self,
    ) -> None:
        """Ensure the prompt separates question roles and exposes structural cues."""

        generator = OpenAIAnswerGenerator(
            model="gpt-4o",
            grounded_fallback_enabled=True,
        )
        retrieved_chunk = RetrievedChunkResult(
            chunk_id="chunk_1",
            doc_id="regulation_a",
            text="The filing deadline is 10 working days.",
            record_id="record_1",
            rank=1,
            similarity_score=0.95,
            chunk_metadata={
                "article_number": "5",
                "article_title": "Deadlines",
                "section_title": "Article 5",
                "parent_structure": ["Chapter II", "Applications"],
                "page_start": 3,
            },
            document_metadata={"document_title": "Regulation A"},
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(
                question_text=(
                    "Responde em PT-PT e indica o artigo aplicavel: "
                    "qual e o prazo de candidatura?"
                ),
                normalized_query_text="qual e o prazo de candidatura?",
                formatting_instructions=[
                    "Responde em PT-PT",
                    "indica o artigo aplicavel",
                ],
            ),
            context=RetrievalContext(
                chunks=[retrieved_chunk],
                context_text=(
                    "Source 1 | doc_id=regulation_a | chunk_id=chunk_1 | "
                    "document_title=Regulation A | article_number=5 | "
                    "article_title=Deadlines\n"
                    "The filing deadline is 10 working days."
                ),
            ),
        )

        messages = generator._build_messages(generation_input)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(
            "Treat article numbers, article titles, section titles, and document titles as explicit legal anchors",
            messages[0]["content"],
        )
        self.assertIn(
            "Apply formatting instructions only to the final presentation of the answer",
            messages[0]["content"],
        )
        self.assertIn("Grounded context:", messages[1]["content"])
        self.assertIn("Structural legal anchors:", messages[1]["content"])
        self.assertIn(
            "Source 1 | document=Regulation A | article=5 | article_title=Deadlines",
            messages[1]["content"],
        )
        self.assertIn("Original user question:", messages[1]["content"])
        self.assertIn("Normalized retrieval query:", messages[1]["content"])
        self.assertIn("Formatting expectations:", messages[1]["content"])
        self.assertIn(
            "Responde em PT-PT e indica o artigo aplicavel: qual e o prazo de candidatura?",
            messages[1]["content"],
        )
        self.assertIn("qual e o prazo de candidatura?", messages[1]["content"])
        self.assertIn("- Responde em PT-PT", messages[1]["content"])
        self.assertIn("- indica o artigo aplicavel", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
