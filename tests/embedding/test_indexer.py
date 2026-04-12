"""End-to-end regression tests for the embedding indexer flow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Chunking.config.settings import PipelineSettings
from embedding.indexer import run_embedding_indexer


class FakeCollection:
    """Record ChromaDB-style collection operations for deterministic assertions."""

    def __init__(self) -> None:
        """Initialize empty call registries for one fake collection."""
        self.get_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []

    def get(self, where: dict[str, object], include: list[object]) -> dict[str, object]:
        """Record one strategy lookup and return an empty collection state."""
        self.get_calls.append({"where": dict(where), "include": list(include)})
        return {"ids": []}

    def delete(self, ids: list[str]) -> None:
        """Record one delete request emitted by the storage layer."""
        self.delete_calls.append({"ids": list(ids)})

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
    ) -> None:
        """Record one ChromaDB upsert payload."""
        self.upsert_calls.append(
            {
                "ids": list(ids),
                "documents": list(documents),
                "embeddings": [list(vector) for vector in embeddings],
                "metadatas": [dict(metadata) for metadata in metadatas],
            }
        )


class FakeEmbeddingProvider:
    """Provide deterministic vectors while recording embedded texts."""

    def __init__(self) -> None:
        """Initialize the provider call registry."""
        self.embed_calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one stable vector per text while preserving call order."""
        normalized_texts = [str(text) for text in texts]
        self.embed_calls.append(normalized_texts)
        return [
            [float(index + 1), float(len(text))]
            for index, text in enumerate(normalized_texts)
        ]


def _write_chunk_file(
    input_root: Path,
    *,
    doc_id: str,
    strategy_name: str,
    payload: list[dict[str, object]],
) -> Path:
    """Write one chunk payload under the strategy layout used by embedding."""
    chunk_file_path = input_root / doc_id / strategy_name / "05_chunks.json"
    chunk_file_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chunk_file_path


class EmbeddingIndexerEndToEndTests(unittest.TestCase):
    """Protect the full embedding orchestration contract without live services."""

    def _build_settings(self, temporary_directory: str) -> PipelineSettings:
        """Create runtime settings bound to one isolated temporary workspace."""
        temporary_root = Path(temporary_directory)
        return PipelineSettings(
            chunking_strategy="article_smart",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_input_root=temporary_root / "chunks",
            embedding_output_root=temporary_root / "embeddings",
            embedding_input_text_field="text",
            embedding_batch_size=1,
            embedding_visualization_enabled=True,
            embedding_visualization_spotlight_enabled=True,
            chromadb_mode="persistent",
            chromadb_persist_directory=temporary_root / "chromadb",
            chromadb_collection_name="rag_embeddings",
        )

    def test_run_embedding_indexer_persists_manifest_storage_and_spotlight_artifacts(
        self,
    ) -> None:
        """Ensure the indexer connects chunk loading, embedding, storage, and export."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = self._build_settings(temporary_directory)
            input_root = Path(temporary_directory) / "chunks"
            _write_chunk_file(
                input_root,
                doc_id="doc_a",
                strategy_name="article_smart",
                payload=[
                    {
                        "chunk_id": "chunk_1",
                        "doc_id": "doc_a",
                        "text": "  First regulation chunk.  ",
                        "strategy": "article_smart",
                        "hierarchy_path": ["Title I", "Article 1"],
                        "page_start": 1,
                        "page_end": 1,
                        "metadata": {
                            "document_title": "Regulation A",
                            "section_title": "Scope",
                        },
                    },
                    {
                        "chunk_id": "chunk_2",
                        "doc_id": "doc_a",
                        "text": "Second line\nwrapped chunk.",
                        "strategy": "article_smart",
                        "hierarchy_path": ["Title I", "Article 2"],
                        "page_start": 2,
                        "page_end": 3,
                        "metadata": {
                            "document_title": "Regulation A",
                            "section_title": "Definitions",
                        },
                    },
                ],
            )

            fake_collection = FakeCollection()
            fake_provider = FakeEmbeddingProvider()

            with patch(
                "embedding.indexer.create_embedding_provider",
                return_value=fake_provider,
            ), patch(
                "embedding.storage.ChromaEmbeddingStorage._get_collection",
                return_value=fake_collection,
            ), patch(
                "embedding.indexer._build_run_id",
                return_value="article_smart_20260410T120000Z",
            ):
                result = run_embedding_indexer(settings)

            self.assertEqual(result.run_id, "article_smart_20260410T120000Z")
            self.assertEqual(result.input_record_count, 2)
            self.assertEqual(result.embedded_record_count, 2)
            self.assertEqual(
                fake_provider.embed_calls,
                [["First regulation chunk."], ["Second line wrapped chunk."]],
            )
            self.assertEqual(
                fake_collection.get_calls,
                [{"where": {"strategy": "article_smart"}, "include": []}],
            )
            self.assertEqual(fake_collection.delete_calls, [])
            self.assertEqual(len(fake_collection.upsert_calls), 1)
            self.assertEqual(
                fake_collection.upsert_calls[0]["ids"],
                [
                    "emb_2e5cb9ccbc04ecfcce2d7876563909fe",
                    "emb_dc08ae99a82e6ed9a497af4bb2366724",
                ],
            )
            self.assertEqual(
                fake_collection.upsert_calls[0]["documents"],
                ["First regulation chunk.", "Second line wrapped chunk."],
            )

            manifest_path = Path(result.manifest_path)
            records_path = Path(result.records_path)
            spotlight_path = Path(result.spotlight_export_path or "")

            self.assertTrue(manifest_path.exists())
            self.assertTrue(records_path.exists())
            self.assertTrue(spotlight_path.exists())

            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["run_id"], result.run_id)
            self.assertEqual(manifest_payload["provider"], "sentence_transformers")
            self.assertEqual(manifest_payload["model"], "all-MiniLM-L6-v2")
            self.assertEqual(manifest_payload["record_count"], 2)
            self.assertEqual(manifest_payload["storage_backend"], "chromadb")
            self.assertEqual(manifest_payload["storage_collection"], "rag_embeddings")
            self.assertEqual(manifest_payload["metadata"]["strategy"], "article_smart")
            self.assertEqual(manifest_payload["metadata"]["vector_dimension"], 2)

            storage_payload = json.loads(records_path.read_text(encoding="utf-8"))
            self.assertEqual(storage_payload["run_id"], result.run_id)
            self.assertEqual(storage_payload["strategy"], "article_smart")
            self.assertEqual(storage_payload["storage_backend"], "chromadb")
            self.assertEqual(storage_payload["deleted_record_count"], 0)
            self.assertEqual(storage_payload["upserted_record_count"], 2)
            self.assertEqual(
                storage_payload["record_ids"],
                [
                    "emb_2e5cb9ccbc04ecfcce2d7876563909fe",
                    "emb_dc08ae99a82e6ed9a497af4bb2366724",
                ],
            )

            spotlight_rows = [
                json.loads(line)
                for line in spotlight_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(spotlight_rows), 2)
            self.assertEqual(spotlight_rows[0]["run_id"], result.run_id)
            self.assertEqual(spotlight_rows[0]["text"], "First regulation chunk.")
            self.assertEqual(spotlight_rows[0]["strategy"], "article_smart")
            self.assertEqual(spotlight_rows[0]["document_title"], "Regulation A")
            self.assertEqual(
                spotlight_rows[1]["text"],
                "Second line wrapped chunk.",
            )

    def test_run_embedding_indexer_skips_spotlight_export_when_visualization_is_disabled(
        self,
    ) -> None:
        """Ensure the indexer still completes when Spotlight export is disabled."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = self._build_settings(temporary_directory)
            settings.embedding_visualization_enabled = False
            input_root = Path(temporary_directory) / "chunks"
            _write_chunk_file(
                input_root,
                doc_id="doc_b",
                strategy_name="article_smart",
                payload=[
                    {
                        "chunk_id": "chunk_10",
                        "doc_id": "doc_b",
                        "text": "Standalone chunk for storage only.",
                        "strategy": "article_smart",
                        "hierarchy_path": ["Title II"],
                        "page_start": 4,
                        "page_end": 4,
                    }
                ],
            )

            fake_collection = FakeCollection()
            fake_provider = FakeEmbeddingProvider()

            with patch(
                "embedding.indexer.create_embedding_provider",
                return_value=fake_provider,
            ), patch(
                "embedding.storage.ChromaEmbeddingStorage._get_collection",
                return_value=fake_collection,
            ), patch(
                "embedding.indexer._build_run_id",
                return_value="article_smart_20260410T121500Z",
            ):
                result = run_embedding_indexer(settings)

            self.assertEqual(result.input_record_count, 1)
            self.assertEqual(result.embedded_record_count, 1)
            self.assertIsNone(result.spotlight_export_path)
            self.assertEqual(
                fake_collection.upsert_calls[0]["ids"],
                ["emb_f8e106c88b634889f3f17ad98ff5a0bf"],
            )
            self.assertTrue(Path(result.manifest_path).exists())
            self.assertTrue(Path(result.records_path).exists())


if __name__ == "__main__":
    unittest.main()
