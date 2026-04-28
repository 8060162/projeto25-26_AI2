"""Regression tests for the benchmark overlay generation CLI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

from Chunking.config.settings import PipelineSettings
from embedding.visualization import benchmark_overlay_main
from embedding.visualization.benchmark_overlay_main import (
    load_chunk_embedding_records,
    run_benchmark_overlay_main,
)


class BenchmarkOverlayMainTests(unittest.TestCase):
    """Protect the operational benchmark overlay generation entrypoint."""

    def test_run_generates_overlay_from_spotlight_vectors_and_benchmark_questions(
        self,
    ) -> None:
        """Ensure the runner loads fixtures, embeds questions, and writes JSONL."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            questions_path = temporary_root / "questions.jsonl"
            chunk_vectors_path = temporary_root / "spotlight_dataset.jsonl"
            output_path = temporary_root / "benchmark_overlay.jsonl"

            _write_question_fixture(questions_path)
            _write_spotlight_fixture(chunk_vectors_path)

            result = run_benchmark_overlay_main(
                settings=_build_settings(temporary_root),
                questions_path=questions_path,
                chunk_vectors_path=chunk_vectors_path,
                output_path=output_path,
                provider=_FakeEmbeddingProvider(vectors=[[0.3, 0.4]]),
            )

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(result.chunk_vectors_path, chunk_vectors_path.resolve())
            self.assertEqual(result.export_result.output_path, output_path)
            self.assertEqual(result.export_result.chunk_point_count, 1)
            self.assertEqual(result.export_result.benchmark_question_point_count, 1)
            self.assertEqual(result.export_result.embedding_dimension, 2)
            self.assertEqual(rows[0]["point_type"], "chunk")
            self.assertEqual(rows[0]["document_title"], "Fixture Regulation")
            self.assertEqual(rows[0]["article_number"], "5")
            self.assertEqual(rows[1]["point_type"], "benchmark_question")
            self.assertEqual(rows[1]["benchmark_case_id"], "case_one")

    def test_load_chunk_embedding_records_rejects_invalid_vector_rows(self) -> None:
        """Ensure invalid exported chunk vectors fail before overlay generation."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            chunk_vectors_path = Path(temporary_directory) / "spotlight_dataset.jsonl"
            chunk_vectors_path.write_text(
                json.dumps(
                    {
                        "chunk_id": "chunk_one",
                        "doc_id": "doc_one",
                        "text": "Chunk text",
                        "embedding": ["not numeric"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must contain only numbers"):
                load_chunk_embedding_records(chunk_vectors_path)

    def test_main_uses_configured_provider_and_writes_output(self) -> None:
        """Ensure the command-line path is callable with deterministic fixtures."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            questions_path = temporary_root / "questions.jsonl"
            chunk_vectors_path = temporary_root / "spotlight_dataset.jsonl"
            output_path = temporary_root / "benchmark_overlay.jsonl"

            _write_question_fixture(questions_path)
            _write_spotlight_fixture(chunk_vectors_path)

            with patch.object(
                benchmark_overlay_main,
                "PipelineSettings",
                return_value=_build_settings(temporary_root),
            ), patch.object(
                benchmark_overlay_main,
                "create_embedding_provider",
                return_value=_FakeEmbeddingProvider(vectors=[[0.3, 0.4]]),
            ):
                exit_code = benchmark_overlay_main.main(
                    [
                        "--questions-path",
                        str(questions_path),
                        "--chunk-vectors-path",
                        str(chunk_vectors_path),
                        "--output-path",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())


class _FakeEmbeddingProvider:
    """Return deterministic embeddings for ordered text batches."""

    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        """Store vectors returned by the fake provider."""

        self.vectors = [list(vector) for vector in vectors]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return configured vectors after checking the requested batch size."""

        if len(texts) != len(self.vectors):
            raise ValueError("Unexpected text count.")
        return [list(vector) for vector in self.vectors]


def _build_settings(temporary_root: Path) -> PipelineSettings:
    """Build settings isolated from repository output paths."""

    return PipelineSettings(
        embedding_output_root=temporary_root / "embeddings",
        embedding_visualization_benchmark_overlay_output_path=(
            temporary_root / "default_overlay.jsonl"
        ),
        benchmark_questions_path=temporary_root / "questions.jsonl",
        benchmark_guardrails_path=temporary_root / "guardrails.jsonl",
    )


def _write_spotlight_fixture(output_path: Path) -> None:
    """Write one minimal Spotlight chunk-vector fixture."""

    output_path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk_one",
                "doc_id": "doc_one",
                "text": "Chunk text",
                "document_title": "Fixture Regulation",
                "article_number": "5",
                "article_title": "Deadlines",
                "hierarchy_path": ["DOCUMENT:DOCUMENT", "ARTICLE:ART_5"],
                "page_start": 2,
                "page_end": 3,
                "embedding": [0.1, 0.2],
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
                "source_file": "chunks.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_question_fixture(output_path: Path) -> None:
    """Write one minimal benchmark question fixture."""

    output_path.write_text(
        json.dumps(
            {
                "case_id": "case_one",
                "question": "What deadline applies?",
                "case_type": "deadline",
                "expected_route": {
                    "route_name": "semantic",
                    "retrieval_scope": "broad",
                    "retrieval_profile": "default",
                    "target_document_titles": ["Fixture Regulation"],
                    "target_doc_ids": ["doc_one"],
                    "target_article_numbers": ["5"],
                },
                "expected_doc_id": "doc_one",
                "expected_article_numbers": ["5"],
                "expected_chunk_ids": ["chunk_one"],
                "required_facts": ["deadline"],
                "forbidden_facts": ["wrong deadline"],
                "expected_answer_behavior": "answer",
                "grounding_labels": {
                    "expected_citation_doc_ids": ["doc_one"],
                    "expected_citation_article_numbers": ["5"],
                    "ambiguity": "low",
                    "article_misattribution_risk": "medium",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
