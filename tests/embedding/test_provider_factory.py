"""Regression tests for embedding provider selection behavior."""

from __future__ import annotations

import unittest

from Chunking.config.settings import PipelineSettings
from embedding.provider_factory import create_embedding_provider
from embedding.providers.openai_provider import OpenAIEmbeddingProvider
from embedding.providers.sentence_transformers_provider import (
    SentenceTransformersEmbeddingProvider,
)


class EmbeddingProviderFactoryTests(unittest.TestCase):
    """Protect provider selection through centralized runtime settings."""

    def test_create_embedding_provider_returns_sentence_transformers_provider(self) -> None:
        """Ensure the configured Sentence Transformers provider resolves correctly."""

        settings = PipelineSettings(
            embedding_provider=" sentence_transformers ",
            embedding_model="all-MiniLM-L6-v2",
            embedding_batch_size=16,
        )

        provider = create_embedding_provider(settings)

        self.assertIsInstance(provider, SentenceTransformersEmbeddingProvider)
        self.assertEqual(provider.model, "all-MiniLM-L6-v2")
        self.assertEqual(provider.batch_size, 16)

    def test_create_embedding_provider_returns_openai_provider_when_requested(self) -> None:
        """Ensure the factory still resolves OpenAI while it remains supported."""

        settings = PipelineSettings(
            embedding_provider="OPENAI",
            embedding_model="text-embedding-3-small",
            embedding_batch_size=8,
        )

        provider = create_embedding_provider(settings)

        self.assertIsInstance(provider, OpenAIEmbeddingProvider)
        self.assertEqual(provider.model, "text-embedding-3-small")
        self.assertEqual(provider.batch_size, 8)

    def test_create_embedding_provider_raises_for_unsupported_provider(self) -> None:
        """Ensure unsupported provider names fail with a clear validation error."""

        settings = PipelineSettings(
            embedding_provider="unsupported_provider",
            embedding_model="all-MiniLM-L6-v2",
            embedding_batch_size=4,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported embedding provider configured in settings",
        ):
            create_embedding_provider(settings)


if __name__ == "__main__":
    unittest.main()
