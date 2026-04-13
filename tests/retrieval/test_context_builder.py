"""Regression tests for retrieval context selection and packing."""

from __future__ import annotations

import unittest

from Chunking.config.settings import PipelineSettings
from retrieval.context_builder import RetrievalContextBuilder
from retrieval.models import RetrievedChunkResult


class RetrievalContextBuilderTests(unittest.TestCase):
    """Protect deterministic context selection and packing behavior."""

    def test_build_context_orders_deduplicates_and_limits_chunks(self) -> None:
        """Ensure the builder keeps the best ordered unique chunks only."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=4,
                retrieval_context_max_chunks=2,
                retrieval_context_max_characters=500,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_2",
                    doc_id="doc_a",
                    text="Second ranked chunk.",
                    record_id="record_2",
                    rank=2,
                    similarity_score=0.90,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_1",
                    doc_id="doc_a",
                    text="First ranked chunk.",
                    record_id="record_1",
                    rank=1,
                    similarity_score=0.95,
                    source_file="data/chunks/doc_a.json",
                    chunk_metadata={"section_title": "Article 1", "page_start": 3},
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_1",
                    doc_id="doc_a",
                    text="First ranked chunk.",
                    record_id="record_1",
                    rank=3,
                    similarity_score=0.70,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_3",
                    doc_id="doc_b",
                    text="Third ranked chunk omitted by max_chunks.",
                    record_id="record_3",
                    rank=4,
                    similarity_score=0.80,
                ),
            ]
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_1", "chunk_2"])
        self.assertEqual(context.chunk_count, 2)
        self.assertFalse(context.truncated)
        self.assertIn("Source 1", context.context_text)
        self.assertIn("section_title=Article 1", context.context_text)
        self.assertEqual(context.metadata["duplicate_count"], 1)
        self.assertEqual(context.metadata["omitted_by_rank_limit_count"], 1)

    def test_build_context_applies_similarity_filter_and_budget_truncation(self) -> None:
        """Ensure score filtering and character-budget truncation remain deterministic."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=3,
                retrieval_context_max_chunks=3,
                retrieval_context_max_characters=85,
                retrieval_score_filtering_enabled=True,
                retrieval_min_similarity_score=0.80,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_high",
                    doc_id="doc_a",
                    text="This grounded excerpt is long enough to exceed the compact context budget.",
                    record_id="record_high",
                    rank=1,
                    similarity_score=0.93,
                ),
                RetrievedChunkResult(
                    chunk_id="chunk_low",
                    doc_id="doc_b",
                    text="This excerpt should be filtered by similarity.",
                    record_id="record_low",
                    rank=2,
                    similarity_score=0.40,
                ),
            ]
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_high"])
        self.assertTrue(context.truncated)
        self.assertLessEqual(context.character_count, 85)
        self.assertEqual(context.metadata["score_filtered_count"], 1)
        self.assertEqual(context.metadata["omitted_by_budget_count"], 0)

    def test_build_context_keeps_missing_similarity_scores_when_filtering_enabled(self) -> None:
        """Ensure missing similarity scores do not discard otherwise valid chunks."""
        builder = RetrievalContextBuilder(
            PipelineSettings(
                retrieval_top_k=2,
                retrieval_context_max_chunks=2,
                retrieval_context_max_characters=300,
                retrieval_score_filtering_enabled=True,
                retrieval_min_similarity_score=0.80,
            )
        )

        context = builder.build_context(
            [
                RetrievedChunkResult(
                    chunk_id="chunk_missing_score",
                    doc_id="doc_a",
                    text="Grounded excerpt without explicit similarity score.",
                    record_id="record_missing_score",
                    rank=1,
                )
            ]
        )

        self.assertEqual(context.chunk_count, 1)
        self.assertFalse(context.truncated)
        self.assertEqual(context.metadata["missing_similarity_score_count"], 1)
        self.assertIn("chunk_missing_score", context.metadata["selected_chunk_ids"])


if __name__ == "__main__":
    unittest.main()
