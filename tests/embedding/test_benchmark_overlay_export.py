"""Regression tests for benchmark overlay embedding visualization exports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from embedding.models import EmbeddingVectorRecord
from embedding.visualization.benchmark_overlay_export import (
    export_benchmark_overlay_dataset,
)
from retrieval.evaluation.models import (
    BenchmarkQuestionCase,
    BenchmarkRouteExpectation,
)


class BenchmarkOverlayExportTests(unittest.TestCase):
    """Protect the chunk-vs-question overlay dataset contract."""

    def test_export_combines_chunk_and_benchmark_question_points(self) -> None:
        """Ensure chunk and question points share stable metadata fields."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "benchmark_overlay.jsonl"

            result = export_benchmark_overlay_dataset(
                chunk_embedding_records=[
                    _build_chunk_record(
                        chunk_id="chunk_1",
                        doc_id="doc_a",
                        vector=[0.1, 0.2],
                    ),
                    _build_chunk_record(
                        chunk_id="chunk_2",
                        doc_id="doc_b",
                        vector=[0.3, 0.4],
                    ),
                ],
                benchmark_question_cases=[_build_benchmark_case()],
                benchmark_question_embeddings={"case_one": [0.5, 0.6]},
                output_path=output_path,
            )

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(result.output_path, output_path)
            self.assertEqual(result.chunk_point_count, 2)
            self.assertEqual(result.benchmark_question_point_count, 1)
            self.assertEqual(result.record_count, 3)
            self.assertEqual(result.embedding_dimension, 2)
            self.assertEqual([row["point_type"] for row in rows], [
                "chunk",
                "chunk",
                "benchmark_question",
            ])
            self.assertEqual(rows[0]["doc_id"], "doc_a")
            self.assertEqual(rows[0]["document_title"], "Regulation doc_a")
            self.assertEqual(rows[0]["article_number"], "5")
            self.assertEqual(rows[0]["color_group"], "chunk:doc_a")
            self.assertEqual(rows[2]["benchmark_case_id"], "case_one")
            self.assertEqual(rows[2]["expected_doc_id"], "doc_a")
            self.assertEqual(rows[2]["expected_article_numbers"], ["5"])
            self.assertEqual(rows[2]["expected_chunk_ids"], ["chunk_1"])
            self.assertEqual(
                rows[2]["document_title"],
                "Expected Regulation",
            )
            self.assertEqual(rows[2]["color_group"], "benchmark_question:doc_a")
            self.assertTrue(
                all(
                    required_field in row
                    for row in rows
                    for required_field in (
                        "point_type",
                        "benchmark_case_id",
                        "expected_doc_id",
                        "expected_article_numbers",
                        "doc_id",
                        "document_title",
                        "color_group",
                        "embedding",
                    )
                )
            )

    def test_export_rejects_missing_question_embedding(self) -> None:
        """Ensure every benchmark question case has a corresponding vector."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError,
                "Missing benchmark-question embedding",
            ):
                export_benchmark_overlay_dataset(
                    chunk_embedding_records=[],
                    benchmark_question_cases=[_build_benchmark_case()],
                    benchmark_question_embeddings={},
                    output_path=Path(temporary_directory) / "overlay.jsonl",
                )

    def test_export_rejects_inconsistent_embedding_dimensions(self) -> None:
        """Ensure one visual dataset cannot mix different vector dimensions."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "Inconsistent embedding dimension"):
                export_benchmark_overlay_dataset(
                    chunk_embedding_records=[
                        _build_chunk_record(
                            chunk_id="chunk_1",
                            doc_id="doc_a",
                            vector=[0.1, 0.2],
                        )
                    ],
                    benchmark_question_cases=[_build_benchmark_case()],
                    benchmark_question_embeddings={"case_one": [0.3, 0.4, 0.5]},
                    output_path=Path(temporary_directory) / "overlay.jsonl",
                )


def _build_chunk_record(
    chunk_id: str,
    doc_id: str,
    vector: list[float],
) -> EmbeddingVectorRecord:
    """Build one deterministic chunk embedding record."""

    return EmbeddingVectorRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        vector=vector,
        metadata={
            "document_title": f"Regulation {doc_id}",
            "article_number": "5",
            "article_title": "Deadlines",
            "hierarchy_path": ["Chapter I", "Article 5"],
            "page_start": 2,
            "page_end": 3,
        },
        model="all-MiniLM-L6-v2",
        provider="sentence_transformers",
        source_file=f"{doc_id}.json",
        text=f"Text for {chunk_id}",
    )


def _build_benchmark_case() -> BenchmarkQuestionCase:
    """Build one deterministic benchmark question case."""

    return BenchmarkQuestionCase(
        case_id="case_one",
        question="What deadline applies?",
        case_type="deadline",
        expected_route=BenchmarkRouteExpectation(
            route_name="semantic",
            retrieval_scope="broad",
            retrieval_profile="default",
            target_document_titles=["Expected Regulation"],
            target_doc_ids=["doc_a"],
            target_article_numbers=["5"],
        ),
        expected_doc_id="doc_a",
        expected_article_numbers=["5"],
        expected_chunk_ids=["chunk_1"],
        required_facts=["deadline"],
        forbidden_facts=["wrong deadline"],
        expected_answer_behavior="answer",
    )


if __name__ == "__main__":
    unittest.main()
