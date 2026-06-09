"""Regression tests for the external GPT-4o answer-generation adapter."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from retrieval.answer_generator import AnswerGenerationError
from retrieval.external_gpt4o_answer_generator import ExternalGPT4oAnswerGenerator
from retrieval.models import (
    AnswerGenerationInput,
    RetrievalContext,
    RetrievedChunkResult,
    UserQuestionInput,
)


class ExternalGPT4oAnswerGeneratorTests(unittest.TestCase):
    """Protect the external GPT-4o multipart adapter contract."""

    def test_generate_answer_posts_multipart_payload_with_auth(self) -> None:
        """Ensure the adapter builds the configured multipart endpoint request."""

        captured_requests = []

        def fake_request_sender(request, timeout):
            """Capture the outgoing request and return a successful JSON body."""

            captured_requests.append((request, timeout))
            return SimpleNamespace(
                status=200,
                read=lambda: json.dumps(
                    {
                        "answer_text": "According to Article 5, the deadline is 10 days.",
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 8,
                            "total_tokens": 20,
                        },
                    }
                ).encode("utf-8"),
            )

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="GPT4O_PROXY_TOKEN",
            question_field_name="message",
            context_field_name="evidence",
            instructions_field_name="instructions",
            metadata_field_name="metadata",
            timeout_seconds=12,
            max_retries=0,
            request_sender=fake_request_sender,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(
                question_text="What is the filing deadline?",
                request_id="request-1",
                conversation_id="conversation-1",
                metadata={
                    "thread_id": "thread-1",
                    "channel_id": "channel-1",
                    "user_id": "user-1",
                },
                normalized_query_text="filing deadline",
                formatting_instructions=["Answer in PT-PT."],
            ),
            context=RetrievalContext(
                chunks=[
                    RetrievedChunkResult(
                        chunk_id="chunk-1",
                        doc_id="doc-1",
                        text="Article 5 states the filing deadline is 10 days.",
                    )
                ],
                context_text="Article 5 states the filing deadline is 10 days.",
                chunk_count=1,
            ),
            system_instruction="Use a formal tone.",
            grounding_instruction="Cite the article number when present.",
        )

        with patch.dict("os.environ", {"GPT4O_PROXY_TOKEN": "secret-token"}):
            result = generator.generate_answer(generation_input)

        self.assertEqual(result.answer_text, "According to Article 5, the deadline is 10 days.")
        self.assertTrue(result.grounded)
        self.assertEqual(result.metadata["provider"], "external_gpt4o")
        self.assertEqual(result.metadata["status_code"], 200)
        self.assertEqual(
            result.metadata["usage"],
            {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        )

        request, timeout = captured_requests[0]
        request_body = request.data.decode("utf-8")

        self.assertEqual(request.full_url, "https://example.test/gpt4o")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(timeout, 12.0)
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertIn("multipart/form-data; boundary=", request.get_header("Content-type"))
        self.assertIn('name="message"', request_body)
        self.assertIn("What is the filing deadline?", request_body)
        self.assertIn('name="evidence"', request_body)
        self.assertIn("Article 5 states the filing deadline is 10 days.", request_body)
        self.assertIn('name="instructions"', request_body)
        self.assertIn("Use a formal tone.", request_body)
        self.assertIn("Structural legal anchors:", request_body)
        self.assertIn('name="metadata"', request_body)
        self.assertIn('"request_id": "request-1"', request_body)
        self.assertIn('"context_chunk_ids": ["chunk-1"]', request_body)
        self.assertIn('name="thread_id"', request_body)
        self.assertIn("thread-1", request_body)
        self.assertIn('name="channel_id"', request_body)
        self.assertIn("channel-1", request_body)
        self.assertIn('name="user_info"', request_body)
        self.assertIn("{}", request_body)
        self.assertIn('name="user_id"', request_body)
        self.assertIn("user-1", request_body)

    def test_build_request_supports_iaedu_auth_and_message_contract(self) -> None:
        """Ensure the adapter supports the iaedu multipart endpoint contract."""

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://api.iaedu.pt/agent-chat/api/v1/agent/id/stream",
            auth_env_var="EXTERNAL_GPT4O_API_KEY",
            question_field_name="message",
            channel_id="configured-channel",
            thread_id="configured-thread",
            user_info='{"role": "student"}',
            auth_header_name="x-api-key",
            auth_header_prefix="",
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What is the value of X?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict(
            "os.environ",
            {
                "EXTERNAL_GPT4O_API_KEY": "secret-token",
            },
        ):
            request = generator._build_request(generation_input)

        request_body = request.data.decode("utf-8")

        self.assertEqual(request.get_header("X-api-key"), "secret-token")
        self.assertNotIn("Bearer secret-token", request.headers.values())
        self.assertIn('name="message"', request_body)
        self.assertIn("What is the value of X?", request_body)
        self.assertIn('name="channel_id"', request_body)
        self.assertIn("configured-channel", request_body)
        self.assertIn('name="thread_id"', request_body)
        self.assertIn("configured-thread", request_body)
        self.assertIn('name="user_info"', request_body)
        self.assertIn('{"role": "student"}', request_body)

    def test_request_metadata_overrides_configured_operational_fields(self) -> None:
        """Ensure per-request operational fields can override configured defaults."""

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://api.iaedu.pt/agent-chat/api/v1/agent/id/stream",
            auth_env_var="EXTERNAL_GPT4O_API_KEY",
            question_field_name="message",
            channel_id="configured-channel",
            thread_id="configured-thread",
            user_info='{"role": "student"}',
            auth_header_name="x-api-key",
            auth_header_prefix="",
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(
                question_text="What is the value of X?",
                metadata={
                    "channel_id": "metadata-channel",
                    "thread_id": "metadata-thread",
                    "user_info": '{"role": "staff"}',
                },
            ),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict(
            "os.environ",
            {"EXTERNAL_GPT4O_API_KEY": "secret-token"},
        ):
            request = generator._build_request(generation_input)

        request_body = request.data.decode("utf-8")

        self.assertIn("metadata-channel", request_body)
        self.assertIn("metadata-thread", request_body)
        self.assertIn('{"role": "staff"}', request_body)
        self.assertNotIn("configured-channel", request_body)
        self.assertNotIn("configured-thread", request_body)

    def test_send_request_uses_urlopen_timeout_keyword(self) -> None:
        """Ensure urllib receives timeout as timeout instead of request data."""

        sent_calls = []

        def fake_urlopen(request, data=None, timeout=None):
            """Capture urllib arguments and return a successful response."""

            sent_calls.append((request, data, timeout))
            return SimpleNamespace(status=200, read=lambda: b'{"answer": "Done."}')

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="GPT4O_PROXY_TOKEN",
            timeout_seconds=9,
            max_retries=0,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict("os.environ", {"GPT4O_PROXY_TOKEN": "secret-token"}):
            with patch("retrieval.external_gpt4o_answer_generator.urlopen", fake_urlopen):
                result = generator.generate_answer(generation_input)

        self.assertEqual(result.answer_text, "Done.")
        self.assertEqual(sent_calls[0][1], None)
        self.assertEqual(sent_calls[0][2], 9.0)

    def test_generate_answer_returns_fallback_without_context(self) -> None:
        """Ensure missing context does not call the external endpoint."""

        def fake_request_sender(request, timeout):
            """Fail if the adapter tries to call the endpoint."""

            raise AssertionError("request_sender should not be called")

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            request_sender=fake_request_sender,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(),
        )

        result = generator.generate_answer(generation_input)

        self.assertFalse(result.grounded)
        self.assertTrue(result.metadata["used_grounded_fallback"])
        self.assertEqual(result.metadata["provider"], "external_gpt4o")
        self.assertIn("No reliable grounded context", result.answer_text)

    def test_generate_answer_requires_configured_auth_env_var(self) -> None:
        """Ensure endpoint authentication is resolved only from the environment."""

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="MISSING_GPT4O_TOKEN",
            grounded_fallback_enabled=False,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                AnswerGenerationError,
                "MISSING_GPT4O_TOKEN",
            ):
                generator.generate_answer(generation_input)

    def test_generate_answer_retries_transient_status(self) -> None:
        """Ensure transient endpoint statuses are retried predictably."""

        responses = [
            SimpleNamespace(status=500, read=lambda: b'{"error": "temporary"}'),
            SimpleNamespace(status=200, read=lambda: b'{"answer": "Final answer."}'),
        ]

        def fake_request_sender(request, timeout):
            """Return one transient response followed by a successful response."""

            return responses.pop(0)

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="GPT4O_PROXY_TOKEN",
            max_retries=1,
            request_sender=fake_request_sender,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict("os.environ", {"GPT4O_PROXY_TOKEN": "secret-token"}):
            result = generator.generate_answer(generation_input)

        self.assertEqual(result.answer_text, "Final answer.")
        self.assertEqual(len(responses), 0)

    def test_generate_answer_raises_for_missing_answer_text(self) -> None:
        """Ensure malformed endpoint responses fail with a clear adapter error."""

        def fake_request_sender(request, timeout):
            """Return a JSON response that does not contain answer text."""

            return SimpleNamespace(status=200, read=lambda: b'{"usage": {}}')

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="GPT4O_PROXY_TOKEN",
            request_sender=fake_request_sender,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict("os.environ", {"GPT4O_PROXY_TOKEN": "secret-token"}):
            with self.assertRaisesRegex(
                AnswerGenerationError,
                "did not include answer text",
            ):
                generator.generate_answer(generation_input)

    def test_generate_answer_raises_for_stream_error_event(self) -> None:
        """Ensure endpoint stream error events are not returned as answers."""

        def fake_request_sender(request, timeout):
            """Return a stream-like endpoint error body."""

            return SimpleNamespace(
                status=200,
                read=lambda: (
                    "{'run_id': 'run-1', 'type': 'start', 'content': 'Processing'}\n\n"
                    "{'run_id': 'run-1', 'type': 'error', "
                    "'content': 'Rate limit reached (429)', 'messageId': 'None'}\n\n"
                    "{'run_id': 'run-1', 'type': 'done', 'content': 'run-1'}"
                ).encode("utf-8"),
            )

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="GPT4O_PROXY_TOKEN",
            request_sender=fake_request_sender,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict("os.environ", {"GPT4O_PROXY_TOKEN": "secret-token"}):
            with self.assertLogs(
                "retrieval.external_gpt4o_answer_generator",
                level="WARNING",
            ) as logs:
                with self.assertRaisesRegex(
                    AnswerGenerationError,
                    r"Rate limit reached \(429\)",
                ):
                    generator.generate_answer(generation_input)

        self.assertIn("rate limiting", "\n".join(logs.output))
        self.assertIn("GPT4O_PROXY_TOKEN", "\n".join(logs.output))

    def test_generate_answer_logs_http_rate_limit_status(self) -> None:
        """Ensure HTTP 429 endpoint throttling is logged before failing."""

        def fake_request_sender(request, timeout):
            """Return an HTTP 429 response from the configured endpoint."""

            return SimpleNamespace(status=429, read=lambda: b'{"error": "limit"}')

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="GPT4O_PROXY_TOKEN",
            max_retries=0,
            request_sender=fake_request_sender,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict("os.environ", {"GPT4O_PROXY_TOKEN": "secret-token"}):
            with self.assertLogs(
                "retrieval.external_gpt4o_answer_generator",
                level="WARNING",
            ) as logs:
                with self.assertRaisesRegex(
                    AnswerGenerationError,
                    "HTTP status 429",
                ):
                    generator.generate_answer(generation_input)

        self.assertIn("status_code=429", "\n".join(logs.output))

    def test_generate_answer_raises_for_key_value_stream_error_event(self) -> None:
        """Ensure non-JSON stream error lines are treated as endpoint failures."""

        def fake_request_sender(request, timeout):
            """Return a key-value endpoint error stream."""

            return SimpleNamespace(
                status=200,
                read=lambda: (
                    "type=start, content=Processing\n"
                    "type=error, content=Rate limit reached (429)\n"
                    "type=done"
                ).encode("utf-8"),
            )

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="GPT4O_PROXY_TOKEN",
            request_sender=fake_request_sender,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict("os.environ", {"GPT4O_PROXY_TOKEN": "secret-token"}):
            with self.assertLogs(
                "retrieval.external_gpt4o_answer_generator",
                level="WARNING",
            ):
                with self.assertRaisesRegex(
                    AnswerGenerationError,
                    r"Rate limit reached \(429\)",
                ):
                    generator.generate_answer(generation_input)

    def test_generate_answer_extracts_text_from_stream_events(self) -> None:
        """Ensure successful stream events are converted into answer text."""

        def fake_request_sender(request, timeout):
            """Return a stream-like endpoint success body."""

            return SimpleNamespace(
                status=200,
                read=lambda: (
                    "{'run_id': 'run-1', 'type': 'start', 'content': 'Processing'}\n\n"
                    "{'run_id': 'run-1', 'type': 'message', "
                    "'content': 'According to Article 5, payment is due.'}\n\n"
                    "{'run_id': 'run-1', 'type': 'done', 'content': 'run-1'}"
                ).encode("utf-8"),
            )

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="GPT4O_PROXY_TOKEN",
            request_sender=fake_request_sender,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict("os.environ", {"GPT4O_PROXY_TOKEN": "secret-token"}):
            result = generator.generate_answer(generation_input)

        self.assertEqual(
            result.answer_text,
            "According to Article 5, payment is due.",
        )
        self.assertTrue(result.grounded)

    def test_generate_answer_extracts_text_from_key_value_stream_events(self) -> None:
        """Ensure key-value stream answer events are converted into answer text."""

        def fake_request_sender(request, timeout):
            """Return a key-value endpoint success stream."""

            return SimpleNamespace(
                status=200,
                read=lambda: (
                    "type=start, content=Processing\n"
                    "type=message, content=According to Article 6, payment is due.\n"
                    "type=done, content=run-1"
                ).encode("utf-8"),
            )

        generator = ExternalGPT4oAnswerGenerator(
            endpoint_url="https://example.test/gpt4o",
            auth_env_var="GPT4O_PROXY_TOKEN",
            request_sender=fake_request_sender,
        )
        generation_input = AnswerGenerationInput(
            question=UserQuestionInput(question_text="What deadline applies?"),
            context=RetrievalContext(context_text="Grounded context.", chunk_count=1),
        )

        with patch.dict("os.environ", {"GPT4O_PROXY_TOKEN": "secret-token"}):
            result = generator.generate_answer(generation_input)

        self.assertEqual(
            result.answer_text,
            "According to Article 6, payment is due.",
        )
        self.assertTrue(result.grounded)


if __name__ == "__main__":
    unittest.main()
