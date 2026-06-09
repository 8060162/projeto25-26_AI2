"""Regression tests for embedding input loading behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Chunking.config.settings import PipelineSettings
from embedding.chunk_input_loader import load_embedding_input_records


def _write_chunk_file(
    input_root: Path,
    *,
    doc_id: str,
    strategy_name: str,
    payload: list[dict[str, object]],
) -> Path:
    """Write one chunk payload using the loader's expected strategy layout."""

    chunk_file_path = input_root / doc_id / strategy_name / "05_chunks.json"
    chunk_file_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chunk_file_path


class EmbeddingChunkInputLoaderTests(unittest.TestCase):
    """Protect sequencing and metadata preservation in the embedding loader."""

    def test_loader_builds_metadata_before_calling_embedding_text_builder(self) -> None:
        """Ensure structural metadata is available when the text builder runs."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory) / "chunks"
            _write_chunk_file(
                input_root,
                doc_id="doc_1",
                strategy_name="article_smart",
                payload=[
                    {
                        "chunk_id": "chunk_1",
                        "doc_id": "doc_1",
                        "text": "Article body text.",
                        "strategy": "article_smart",
                        "hierarchy_path": ["Title I", "Article 7"],
                        "page_start": 3,
                        "page_end": 3,
                        "metadata": {
                            "document_title": "Regulation A",
                            "article_number": "7",
                            "article_title": "Deadlines",
                            "section_title": "Article 7",
                        },
                    }
                ],
            )
            settings = PipelineSettings(
                chunking_strategy="article_smart",
                embedding_input_root=input_root,
                embedding_input_text_field="text",
            )

            captured_metadata: list[dict[str, object]] = []

            def fake_build_embedding_text(record: object) -> str:
                """Capture the builder input metadata and return stable text."""

                metadata = dict(getattr(record, "metadata"))
                captured_metadata.append(metadata)
                return str(getattr(record, "text")).strip()

            with patch(
                "embedding.chunk_input_loader.build_embedding_text",
                side_effect=fake_build_embedding_text,
            ):
                records = load_embedding_input_records(settings)

            self.assertEqual(len(records), 1)
            self.assertEqual(len(captured_metadata), 1)
            self.assertEqual(captured_metadata[0]["strategy"], "article_smart")
            self.assertEqual(captured_metadata[0]["article_number"], "7")
            self.assertEqual(captured_metadata[0]["article_title"], "Deadlines")
            self.assertEqual(captured_metadata[0]["section_title"], "Article 7")
            self.assertEqual(captured_metadata[0]["document_title"], "Regulation A")
            self.assertEqual(
                captured_metadata[0]["embedding_text_field"],
                "text",
            )

            record = records[0]
            self.assertEqual(record.text, "Article body text.")
            self.assertEqual(record.metadata["article_number"], "7")
            self.assertEqual(record.metadata["article_title"], "Deadlines")
            self.assertEqual(record.metadata["section_title"], "Article 7")
            self.assertEqual(record.metadata["document_title"], "Regulation A")

    def test_loader_keeps_text_free_of_metadata_anchors_when_configured_for_text(
        self,
    ) -> None:
        """Ensure configured body text remains separate from structural metadata."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory) / "chunks"
            _write_chunk_file(
                input_root,
                doc_id="doc_1",
                strategy_name="article_smart",
                payload=[
                    {
                        "chunk_id": "chunk_1",
                        "doc_id": "doc_1",
                        "text": "Body text only.",
                        "meta_text": "Article 7 - Deadlines\n\nBody text only.",
                        "strategy": "article_smart",
                        "metadata": {
                            "article_number": "7",
                            "article_title": "Deadlines",
                        },
                    }
                ],
            )
            settings = PipelineSettings(
                chunking_strategy="article_smart",
                embedding_input_root=input_root,
                embedding_input_text_field="text",
            )

            records = load_embedding_input_records(settings)

            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0].text,
                "Body text only.",
            )
            self.assertEqual(records[0].metadata["embedding_text_field"], "text")
            self.assertEqual(records[0].metadata["article_number"], "7")
            self.assertEqual(records[0].metadata["article_title"], "Deadlines")


if __name__ == "__main__":
    unittest.main()
