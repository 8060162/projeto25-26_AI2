"""Regression tests for embedding storage replacement behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Chunking.config.settings import PipelineSettings
from embedding.models import EmbeddingRunManifest, EmbeddingVectorRecord
from embedding.storage import LocalEmbeddingStorage


def build_vector_record() -> EmbeddingVectorRecord:
    """Build one stable embedding record used by storage regression tests."""
    return EmbeddingVectorRecord(
        chunk_id="chunk_1",
        doc_id="doc_1",
        vector=[0.1, 0.2, 0.3],
        metadata={
            "strategy": "article_smart",
            "page_start": 1,
            "page_end": 1,
        },
        model="text-embedding-3-large",
        provider="openai",
        source_file="data/chunks/doc_1/article_smart/05_chunks.json",
        text="Texto de teste para embeddings.",
    )


def build_run_manifest(run_id: str) -> EmbeddingRunManifest:
    """Build one stable run manifest used by storage regression tests."""
    return EmbeddingRunManifest(
        run_id=run_id,
        provider="openai",
        model="text-embedding-3-large",
        input_root="data/chunks",
        output_root="data/embeddings",
        input_text_field="text",
        batch_size=100,
        record_count=1,
        source_files=["data/chunks/doc_1/article_smart/05_chunks.json"],
        metadata={
            "strategy": "article_smart",
            "input_record_count": 1,
            "embedded_record_count": 1,
            "vector_dimension": 3,
        },
    )


class LocalEmbeddingStorageReplacementTests(unittest.TestCase):
    """Ensure new embedding runs replace previous outputs for the same strategy."""

    def test_save_run_removes_previous_strategy_run_directory(self) -> None:
        """Ensure one new save deletes the older run directory for that strategy."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            settings = PipelineSettings(embedding_output_root=output_root)
            storage = LocalEmbeddingStorage(settings)
            embedding_records = [build_vector_record()]

            first_result = storage.save_run(
                embedding_records=embedding_records,
                manifest=build_run_manifest("article_smart_20260408T220739Z"),
            )

            stale_marker_path = first_result.run_directory / "stale_marker.txt"
            stale_marker_path.write_text("obsolete", encoding="utf-8")

            second_result = storage.save_run(
                embedding_records=embedding_records,
                manifest=build_run_manifest("article_smart_20260408T221000Z"),
            )

            self.assertFalse(first_result.run_directory.exists())
            self.assertFalse(stale_marker_path.exists())
            self.assertTrue(second_result.run_directory.exists())
            self.assertTrue(second_result.records_path.exists())
            self.assertTrue(second_result.manifest_path.exists())

            strategy_root = output_root / "article_smart"
            self.assertEqual(
                sorted(path.name for path in strategy_root.iterdir()),
                ["article_smart_20260408T221000Z"],
            )

            manifest_payload = json.loads(
                second_result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest_payload["run_id"],
                "article_smart_20260408T221000Z",
            )
