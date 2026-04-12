"""Regression tests for the Sentence Transformers embedding provider."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from embedding.providers.sentence_transformers_provider import (
    EmbeddingProviderError,
    SentenceTransformersEmbeddingProvider,
)


class FakeArray:
    """Provide one deterministic `tolist` payload for provider assertions."""

    def __init__(self, payload: list[list[float]]) -> None:
        """Store the raw vector payload returned by the fake model."""
        self.payload = payload

    def tolist(self) -> list[list[float]]:
        """Return the stored vector payload."""
        return [list(vector) for vector in self.payload]


class FakeSentenceTransformerModel:
    """Record encode calls and return deterministic vectors by input text."""

    def __init__(self) -> None:
        """Initialize the captured call registry."""
        self.encode_calls: list[dict[str, object]] = []

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> FakeArray:
        """Return one fake vector per text while recording call arguments."""
        self.encode_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )

        payload = [
            [float(index + 1), float(len(text))]
            for index, text in enumerate(texts)
        ]
        return FakeArray(payload)


class SentenceTransformersEmbeddingProviderTests(unittest.TestCase):
    """Protect deterministic local embedding generation behavior."""

    def test_embed_texts_returns_vectors_in_input_order_across_batches(self) -> None:
        """Ensure valid text inputs are normalized, batched, and returned in order."""
        provider = SentenceTransformersEmbeddingProvider(
            model=" all-MiniLM-L6-v2 ",
            batch_size=2,
        )
        fake_model = FakeSentenceTransformerModel()

        with patch.object(
            SentenceTransformersEmbeddingProvider,
            "_build_model",
            return_value=fake_model,
        ):
            vectors = provider.embed_texts(["  first text  ", "second", " third "])

        self.assertEqual(
            vectors,
            [
                [1.0, 10.0],
                [2.0, 6.0],
                [1.0, 5.0],
            ],
        )
        self.assertEqual(
            fake_model.encode_calls,
            [
                {
                    "texts": ["first text", "second"],
                    "batch_size": 2,
                    "convert_to_numpy": True,
                    "show_progress_bar": False,
                },
                {
                    "texts": ["third"],
                    "batch_size": 2,
                    "convert_to_numpy": True,
                    "show_progress_bar": False,
                },
            ],
        )

    def test_embed_texts_returns_empty_list_without_loading_model_for_empty_input(self) -> None:
        """Ensure empty input short-circuits before any model interaction."""
        provider = SentenceTransformersEmbeddingProvider(
            model="all-MiniLM-L6-v2",
            batch_size=4,
        )

        with patch.object(
            SentenceTransformersEmbeddingProvider,
            "_build_model",
            side_effect=AssertionError("Model should not be built for empty inputs."),
        ) as build_model_mock:
            vectors = provider.embed_texts([])

        self.assertEqual(vectors, [])
        build_model_mock.assert_not_called()

    def test_embed_texts_raises_for_invalid_text_inputs(self) -> None:
        """Ensure invalid text items fail through predictable validation rules."""
        provider = SentenceTransformersEmbeddingProvider(
            model="all-MiniLM-L6-v2",
            batch_size=4,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Embedding text at index 0 cannot be empty.",
        ):
            provider.embed_texts(["   "])

        with self.assertRaisesRegex(
            ValueError,
            "Embedding text at index 1 must be a string",
        ):
            provider.embed_texts(["valid", 123])  # type: ignore[list-item]

    def test_embed_texts_raises_when_model_response_size_does_not_match_batch(self) -> None:
        """Ensure malformed model responses fail with a deterministic provider error."""
        provider = SentenceTransformersEmbeddingProvider(
            model="all-MiniLM-L6-v2",
            batch_size=2,
        )

        class WrongSizeModel:
            """Return fewer vectors than requested to emulate a broken provider payload."""

            def encode(
                self,
                texts: list[str],
                *,
                batch_size: int,
                convert_to_numpy: bool,
                show_progress_bar: bool,
            ) -> FakeArray:
                """Return one malformed vector payload."""
                return FakeArray([[0.1, 0.2]])

        with patch.object(
            SentenceTransformersEmbeddingProvider,
            "_build_model",
            return_value=WrongSizeModel(),
        ):
            with self.assertRaisesRegex(
                EmbeddingProviderError,
                "Sentence Transformers embedding response size does not match",
            ):
                provider.embed_texts(["alpha", "beta"])

    def test_build_model_raises_concise_error_for_torchcodec_runtime_failures(self) -> None:
        """Ensure runtime import failures become actionable provider errors."""
        provider = SentenceTransformersEmbeddingProvider(
            model="all-MiniLM-L6-v2",
            batch_size=2,
        )

        with patch.object(
            SentenceTransformersEmbeddingProvider,
            "_load_sentence_transformer_class",
            side_effect=EmbeddingProviderError(
                "Sentence Transformers could not start because the local "
                "PyTorch runtime is incompatible with the installed "
                "TorchCodec/CUDA libraries."
            ),
        ), patch.object(
            SentenceTransformersEmbeddingProvider,
            "_build_runtime_versions_summary",
            return_value="torch=2.11.0+cu130, torchcodec=0.11.0",
        ):
            with self.assertRaisesRegex(
                EmbeddingProviderError,
                "TorchCodec/CUDA libraries",
            ):
                provider._build_model()

    def test_runtime_dependency_error_message_reports_versions_and_fix_hint(self) -> None:
        """Ensure runtime dependency failures expose actionable environment details."""
        provider = SentenceTransformersEmbeddingProvider(
            model="all-MiniLM-L6-v2",
            batch_size=2,
        )

        with patch.object(
            SentenceTransformersEmbeddingProvider,
            "_build_runtime_versions_summary",
            return_value=(
                "torch=2.11.0+cu130, torchcodec=0.11.0, "
                "sentence-transformers=5.4.0, transformers=5.5.0"
            ),
        ), patch.object(
            SentenceTransformersEmbeddingProvider,
            "_build_runtime_fix_hint",
            return_value=(
                "Install a compatible CPU-only PyTorch stack in the active "
                "virtual environment, or remove 'torchcodec' if this project "
                "does not need media decoding."
            ),
        ):
            error_message = provider._build_runtime_dependency_error_message(
                RuntimeError("Could not load libtorchcodec. Likely causes:")
            )

        self.assertIn("torch=2.11.0+cu130", error_message)
        self.assertIn("torchcodec=0.11.0", error_message)
        self.assertIn("remove 'torchcodec' if this project does not need media decoding", error_message)

    def test_summarize_exception_returns_first_meaningful_line(self) -> None:
        """Ensure multiline dependency errors are reduced to one readable line."""
        provider = SentenceTransformersEmbeddingProvider(
            model="all-MiniLM-L6-v2",
            batch_size=2,
        )

        summarized_message = provider._summarize_exception(
            RuntimeError("\nfirst line\nsecond line\n")
        )

        self.assertEqual(summarized_message, "first line")


if __name__ == "__main__":
    unittest.main()
