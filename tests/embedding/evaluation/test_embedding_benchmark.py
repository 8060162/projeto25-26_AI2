"""Regression tests for embedding benchmark comparison."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Sequence

from Chunking.config.settings import PipelineSettings
from embedding.evaluation.embedding_benchmark import (
    QWEN_EMBEDDING_MODEL,
    run_embedding_benchmark_comparison,
)
from embedding.models import EmbeddingInputRecord


class EmbeddingBenchmarkComparisonTests(unittest.TestCase):
    """Protect deterministic embedding model benchmark comparison."""

    def test_runner_compares_model_retrieval_metrics_and_writes_summary(self) -> None:
        """Ensure model comparison reports metrics and baseline deltas."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            questions_path = temporary_root / "questions.jsonl"
            output_root = temporary_root / "embedding_comparison"
            questions_path.write_text(
                json.dumps(_build_question_record(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            result = run_embedding_benchmark_comparison(
                settings=PipelineSettings(
                    embedding_model="current-model",
                    embedding_comparison_candidate_models=[QWEN_EMBEDDING_MODEL],
                    retrieval_top_k=1,
                ),
                questions_path=questions_path,
                output_root=output_root,
                run_id="fixture_run",
                providers_by_model={
                    "current-model": _KeywordEmbeddingProvider("current"),
                    QWEN_EMBEDDING_MODEL: _KeywordEmbeddingProvider("qwen"),
                },
                input_records=_build_input_records(),
                top_k=1,
            )

            summary_path = (
                output_root / "fixture_run" / "embedding_comparison_summary.json"
            )

            self.assertEqual(result.baseline_model, "current-model")
            self.assertEqual(
                result.compared_models,
                ["current-model", QWEN_EMBEDDING_MODEL],
            )
            self.assertEqual(result.model_results[0].metrics["recall_at_k"], 1.0)
            self.assertEqual(result.model_results[1].metrics["recall_at_k"], 0.0)
            self.assertEqual(
                result.metric_deltas[QWEN_EMBEDDING_MODEL]["recall_at_k"],
                -1.0,
            )
            self.assertTrue(summary_path.exists())


class _KeywordEmbeddingProvider:
    """Return deterministic vectors keyed by text content and model profile."""

    def __init__(self, profile: str) -> None:
        """Create one deterministic provider profile."""

        self.profile = profile

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts with fixed two-dimensional vectors."""

        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> List[float]:
        """Resolve one vector from the deterministic keyword profile."""

        normalized_text = text.lower()

        if "deadline question" in normalized_text:
            return [1.0, 0.0]
        if "expected evidence" in normalized_text:
            return [1.0, 0.0] if self.profile == "current" else [0.0, 1.0]
        if "wrong evidence" in normalized_text:
            return [0.0, 1.0] if self.profile == "current" else [1.0, 0.0]
        return [0.0, 0.0]


def _build_question_record() -> dict[str, object]:
    """Build one valid benchmark question record."""

    return {
        "case_id": "case_one",
        "question": "deadline question",
        "case_type": "deadline",
        "expected_route": {
            "route_name": "semantic",
            "retrieval_scope": "broad",
            "retrieval_profile": "default",
        },
        "expected_doc_id": "doc_expected",
        "expected_article_numbers": ["5"],
        "expected_chunk_ids": ["expected_chunk"],
        "required_facts": ["expected evidence"],
        "forbidden_facts": ["wrong evidence"],
        "expected_answer_behavior": "answer",
        "grounding_labels": {
            "expected_citation_doc_ids": ["doc_expected"],
            "expected_citation_article_numbers": ["5"],
            "ambiguity": "low",
        },
    }


def _build_input_records() -> List[EmbeddingInputRecord]:
    """Build deterministic chunk inputs for embedding comparison."""

    return [
        EmbeddingInputRecord(
            chunk_id="expected_chunk",
            doc_id="doc_expected",
            text="expected evidence",
            chunk_metadata={"article_number": "5"},
            document_metadata={"document_title": "Expected document"},
        ),
        EmbeddingInputRecord(
            chunk_id="wrong_chunk",
            doc_id="doc_wrong",
            text="wrong evidence",
            chunk_metadata={"article_number": "9"},
            document_metadata={"document_title": "Wrong document"},
        ),
    ]


if __name__ == "__main__":
    unittest.main()
