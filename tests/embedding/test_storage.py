"""Regression tests for ChromaDB-backed embedding storage behavior."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Chunking.config.settings import PipelineSettings
from embedding.models import EmbeddingRunManifest, EmbeddingVectorRecord
from embedding.storage import LocalEmbeddingStorage
from retrieval.models import RetrievedChunkResult


class FakeCollection:
    """Record ChromaDB-style storage interactions for deterministic tests."""

    def __init__(self, existing_ids: list[str] | None = None) -> None:
        """Store the initial record set returned by `get`."""
        self.existing_ids = list(existing_ids or [])
        self.query_response: dict[str, object] = {}
        self.get_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []

    def get(self, where: dict[str, object], include: list[object]) -> dict[str, object]:
        """Return the configured existing identifiers for one strategy lookup."""
        self.get_calls.append({"where": where, "include": include})
        return {"ids": list(self.existing_ids)}

    def delete(self, ids: list[str]) -> None:
        """Record the identifiers deleted by the storage layer."""
        self.delete_calls.append({"ids": list(ids)})
        self.existing_ids = []

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, object] | None,
        include: list[object],
    ) -> dict[str, object]:
        """Record one ChromaDB query payload and return the configured response."""
        self.query_calls.append(
            {
                "query_embeddings": [list(vector) for vector in query_embeddings],
                "n_results": n_results,
                "where": None if where is None else dict(where),
                "include": list(include),
            }
        )
        return dict(self.query_response)

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


def build_vector_record(
    *,
    storage_record_id: str = "",
    record_id: str = "",
) -> EmbeddingVectorRecord:
    """Build one stable vector record for ChromaDB persistence assertions."""
    return EmbeddingVectorRecord(
        chunk_id="chunk_1",
        doc_id="doc_1",
        vector=[0.1, 0.2, 0.3],
        metadata={
            "strategy": "article_smart",
            "page_start": 1,
            "page_end": 2,
            "section title": "General Rules",
            "tags": ["alpha", 7, True, {"ignored": "value"}],
            "structured": {"scope": "public"},
            "skip_none": None,
        },
        model="all-MiniLM-L6-v2",
        provider="sentence_transformers",
        source_file="data/chunks/doc_1/article_smart/05_chunks.json",
        text="Test chunk text for embeddings.",
        record_id=record_id,
        storage_record_id=storage_record_id,
        chunk_metadata={
            "page_start": 1,
            "page_end": 2,
            "section_title": "General Rules",
        },
        document_metadata={
            "document_title": "Test Regulation",
            "jurisdiction": "PT",
        },
    )


def build_run_manifest(run_id: str) -> EmbeddingRunManifest:
    """Build one stable run manifest for storage regression tests."""
    return EmbeddingRunManifest(
        run_id=run_id,
        provider="sentence_transformers",
        model="all-MiniLM-L6-v2",
        input_root="data/chunks",
        output_root="data/embeddings",
        input_text_field="text",
        batch_size=32,
        record_count=1,
        source_files=["data/chunks/doc_1/article_smart/05_chunks.json"],
        metadata={
            "strategy": "article_smart",
            "input_record_count": 1,
            "embedded_record_count": 1,
            "vector_dimension": 3,
        },
    )


class ChromaEmbeddingStorageTests(unittest.TestCase):
    """Protect the ChromaDB storage contract and local audit artifacts."""

    def _build_storage(
        self,
        temporary_directory: str,
        collection: FakeCollection,
    ) -> LocalEmbeddingStorage:
        """Create one storage instance bound to a fake collection."""
        settings = PipelineSettings(
            embedding_output_root=Path(temporary_directory) / "embeddings",
            chromadb_mode="persistent",
            chromadb_persist_directory=Path(temporary_directory) / "chromadb",
            chromadb_collection_name="rag_embeddings",
        )
        storage = LocalEmbeddingStorage(settings)
        storage._get_collection = lambda: collection  # type: ignore[method-assign]
        return storage

    def test_save_run_replaces_previous_strategy_artifacts_and_chromadb_records(self) -> None:
        """Ensure one new run clears previous strategy data before persisting again."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            collection = FakeCollection(existing_ids=["stale_1", "stale_2"])
            storage = self._build_storage(temporary_directory, collection)
            embedding_records = [build_vector_record()]

            first_result = storage.save_run(
                embedding_records=embedding_records,
                manifest=build_run_manifest("article_smart_20260410T100000Z"),
            )

            stale_marker_path = first_result.run_directory / "stale_marker.txt"
            stale_marker_path.write_text("obsolete", encoding="utf-8")

            collection.existing_ids = ["chunk_1"]
            second_result = storage.save_run(
                embedding_records=embedding_records,
                manifest=build_run_manifest("article_smart_20260410T101500Z"),
            )

            self.assertFalse(first_result.run_directory.exists())
            self.assertFalse(stale_marker_path.exists())
            self.assertTrue(second_result.records_path.exists())
            self.assertTrue(second_result.manifest_path.exists())
            self.assertEqual(
                collection.delete_calls,
                [{"ids": ["stale_1", "stale_2"]}, {"ids": ["chunk_1"]}],
            )

            strategy_root = Path(temporary_directory) / "embeddings" / "article_smart"
            self.assertEqual(
                sorted(path.name for path in strategy_root.iterdir()),
                ["article_smart_20260410T101500Z"],
            )

            storage_payload = json.loads(
                second_result.records_path.read_text(encoding="utf-8")
            )
            self.assertEqual(storage_payload["storage_backend"], "chromadb")
            self.assertEqual(storage_payload["deleted_record_count"], 1)
            self.assertEqual(storage_payload["upserted_record_count"], 1)
            self.assertEqual(storage_payload["record_ids"], ["chunk_1"])

    def test_save_run_uses_stable_storage_ids_for_chromadb_upsert(self) -> None:
        """Ensure the storage layer prefers the explicit stable storage identifier."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            collection = FakeCollection()
            storage = self._build_storage(temporary_directory, collection)
            embedding_records = [
                build_vector_record(
                    storage_record_id="doc_1:article_smart:chunk_1",
                    record_id="legacy_record_id",
                )
            ]

            storage.save_run(
                embedding_records=embedding_records,
                manifest=build_run_manifest("article_smart_20260410T103000Z"),
            )

            self.assertEqual(len(collection.upsert_calls), 1)
            self.assertEqual(
                collection.upsert_calls[0]["ids"],
                ["doc_1:article_smart:chunk_1"],
            )

    def test_save_run_persists_bounded_metadata_in_chromadb_payload(self) -> None:
        """Ensure stored metadata remains traceable within ChromaDB key quotas."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            collection = FakeCollection()
            storage = self._build_storage(temporary_directory, collection)

            storage.save_run(
                embedding_records=[build_vector_record()],
                manifest=build_run_manifest("article_smart_20260410T104500Z"),
            )

            self.assertEqual(len(collection.upsert_calls), 1)
            metadata = collection.upsert_calls[0]["metadatas"][0]

            self.assertEqual(metadata["record_id"], "chunk_1")
            self.assertEqual(metadata["storage_record_id"], "chunk_1")
            self.assertEqual(metadata["chunk_id"], "chunk_1")
            self.assertEqual(metadata["doc_id"], "doc_1")
            self.assertEqual(metadata["strategy"], "article_smart")
            self.assertEqual(metadata["provider"], "sentence_transformers")
            self.assertEqual(metadata["model"], "all-MiniLM-L6-v2")
            self.assertEqual(
                metadata["source_file"],
                "data/chunks/doc_1/article_smart/05_chunks.json",
            )
            self.assertEqual(metadata["chunk_page_start"], 1)
            self.assertEqual(metadata["chunk_page_end"], 2)
            self.assertEqual(metadata["chunk_section_title"], "General Rules")
            self.assertEqual(metadata["document_document_title"], "Test Regulation")
            self.assertEqual(metadata["document_jurisdiction"], "PT")
            self.assertLessEqual(len(metadata), 32)
            self.assertIn("metadata_extras", metadata)
            self.assertNotIn("chunk_extras", metadata)

            metadata_extras = json.loads(str(metadata["metadata_extras"]))

            self.assertEqual(metadata_extras["page_start"], 1)
            self.assertEqual(metadata_extras["page_end"], 2)
            self.assertEqual(metadata_extras["section_title"], "General Rules")
            self.assertEqual(
                metadata_extras["tags"],
                ["alpha", 7, True, {"ignored": "value"}],
            )
            self.assertEqual(metadata_extras["structured"], {"scope": "public"})
            self.assertNotIn("skip_none", metadata_extras)

    def test_build_chromadb_client_passes_cloud_host_and_port(self) -> None:
        """Ensure cloud storage uses the configured Chroma region endpoint."""
        cloud_client_calls: list[dict[str, object]] = []

        def fake_cloud_client(**kwargs: object) -> object:
            """Record one CloudClient construction call."""
            cloud_client_calls.append(dict(kwargs))
            return object()

        settings = PipelineSettings(
            chromadb_mode="cloud",
            chromadb_collection_name="rag_embeddings",
            chromadb_cloud_tenant="90cd7e0f-0662-4e42-bfd0-785efb6dcb14",
            chromadb_cloud_database="RAG-AI2",
            chromadb_cloud_host="europe-west1.gcp.trychroma.com",
            chromadb_cloud_port=443,
            chromadb_cloud_api_key_env_var="CHROMA_API_KEY",
        )
        storage = LocalEmbeddingStorage(settings)

        with patch.dict(os.environ, {"CHROMA_API_KEY": "test_api_key"}, clear=False):
            with patch.dict(
                sys.modules,
                {"chromadb": SimpleNamespace(CloudClient=fake_cloud_client)},
            ):
                storage._build_chromadb_client()

        self.assertEqual(len(cloud_client_calls), 1)
        self.assertEqual(
            cloud_client_calls[0]["cloud_host"],
            "europe-west1.gcp.trychroma.com",
        )
        self.assertEqual(cloud_client_calls[0]["cloud_port"], 443)
        self.assertEqual(
            cloud_client_calls[0]["tenant"],
            "90cd7e0f-0662-4e42-bfd0-785efb6dcb14",
        )
        self.assertEqual(cloud_client_calls[0]["database"], "RAG-AI2")

    def test_get_collection_uses_configured_collection_name_with_persistent_client(self) -> None:
        """Ensure persistent mode resolves the shared ChromaDB collection name."""
        requested_collection_names: list[str] = []

        class FakePersistentClient:
            """Expose one deterministic collection lookup for storage tests."""

            def get_or_create_collection(self, name: str) -> FakeCollection:
                """Record the requested collection name and return one collection."""
                requested_collection_names.append(name)
                return FakeCollection()

        settings = PipelineSettings(
            chromadb_mode="persistent",
            chromadb_collection_name="rag_embeddings",
        )
        storage = LocalEmbeddingStorage(settings)
        storage._build_chromadb_client = lambda: FakePersistentClient()  # type: ignore[method-assign]

        collection = storage._get_collection()

        self.assertIsInstance(collection, FakeCollection)
        self.assertEqual(requested_collection_names, ["rag_embeddings"])

    def test_get_collection_uses_configured_collection_name_with_cloud_client(self) -> None:
        """Ensure cloud mode resolves the shared ChromaDB collection name."""
        requested_collection_names: list[str] = []

        class FakeCloudClient:
            """Expose one deterministic collection lookup for storage tests."""

            def get_or_create_collection(self, name: str) -> FakeCollection:
                """Record the requested collection name and return one collection."""
                requested_collection_names.append(name)
                return FakeCollection()

        settings = PipelineSettings(
            chromadb_mode="cloud",
            chromadb_collection_name="rag_embeddings",
        )
        storage = LocalEmbeddingStorage(settings)
        storage._build_chromadb_client = lambda: FakeCloudClient()  # type: ignore[method-assign]

        collection = storage._get_collection()

        self.assertIsInstance(collection, FakeCollection)
        self.assertEqual(requested_collection_names, ["rag_embeddings"])

    def test_query_similar_chunks_uses_shared_collection_and_returns_normalized_results(self) -> None:
        """Ensure the storage layer normalizes ChromaDB query results for retrieval."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            collection = FakeCollection()
            collection.query_response = {
                "ids": [["doc_1:article_smart:chunk_1", "doc_1:article_smart:chunk_2"]],
                "documents": [["Chunk text 1", "Chunk text 2"]],
                "metadatas": [
                    [
                        {
                            "record_id": "chunk_1",
                            "chunk_id": "chunk_1",
                            "doc_id": "doc_1",
                            "source_file": "data/chunks/doc_1/article_smart/05_chunks.json",
                            "strategy": "article_smart",
                            "metadata_chunk_file_path": "data/chunks/doc_1/article_smart/05_chunks.json",
                            "metadata_extras": "{\"tags\": [\"alpha\"]}",
                            "chunk_page_start": 1,
                            "chunk_page_end": 2,
                            "chunk_section_title": "General Rules",
                            "document_document_title": "Test Regulation",
                            "document_jurisdiction": "PT",
                        },
                        {
                            "record_id": "chunk_2",
                            "chunk_id": "chunk_2",
                            "doc_id": "doc_1",
                            "source_file": "data/chunks/doc_1/article_smart/05_chunks.json",
                            "chunk_page_start": 3,
                            "chunk_page_end": 3,
                            "document_document_title": "Test Regulation",
                        },
                    ]
                ],
                "distances": [[0.11, 0.27]],
            }
            storage = self._build_storage(temporary_directory, collection)

            results = storage.query_similar_chunks(
                query_vector=[0.5, 0.4, 0.3],
                where={"strategy": "article_smart"},
            )

            self.assertEqual(len(collection.query_calls), 1)
            self.assertEqual(collection.query_calls[0]["n_results"], 8)
            self.assertEqual(
                collection.query_calls[0]["query_embeddings"],
                [[0.5, 0.4, 0.3]],
            )
            self.assertEqual(
                collection.query_calls[0]["where"],
                {"strategy": "article_smart"},
            )
            self.assertEqual(
                collection.query_calls[0]["include"],
                ["documents", "metadatas", "distances"],
            )

            self.assertEqual(len(results), 2)
            self.assertTrue(
                all(isinstance(result, RetrievedChunkResult) for result in results)
            )

            first_result = results[0]
            self.assertEqual(first_result.chunk_id, "chunk_1")
            self.assertEqual(first_result.record_id, "doc_1:article_smart:chunk_1")
            self.assertEqual(first_result.doc_id, "doc_1")
            self.assertEqual(first_result.text, "Chunk text 1")
            self.assertEqual(first_result.rank, 1)
            self.assertEqual(first_result.distance, 0.11)
            self.assertEqual(
                first_result.source_file,
                "data/chunks/doc_1/article_smart/05_chunks.json",
            )
            self.assertEqual(
                first_result.metadata,
                {
                    "chunk_file_path": "data/chunks/doc_1/article_smart/05_chunks.json",
                    "tags": ["alpha"],
                },
            )
            self.assertEqual(
                first_result.chunk_metadata,
                {
                    "page_start": 1,
                    "page_end": 2,
                    "section_title": "General Rules",
                },
            )
            self.assertEqual(
                first_result.document_metadata,
                {
                    "document_title": "Test Regulation",
                    "jurisdiction": "PT",
                },
            )

    def test_query_similar_chunks_honors_explicit_top_k_and_handles_sparse_payloads(self) -> None:
        """Ensure sparse ChromaDB query responses still normalize deterministically."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            collection = FakeCollection()
            collection.query_response = {
                "ids": [["storage_id_only"]],
                "metadatas": [[{"doc_id": "doc_sparse"}]],
            }
            storage = self._build_storage(temporary_directory, collection)

            results = storage.query_similar_chunks(
                query_vector=[1.0, 2.0],
                top_k=1,
            )

            self.assertEqual(collection.query_calls[0]["n_results"], 1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].chunk_id, "storage_id_only")
            self.assertEqual(results[0].record_id, "storage_id_only")
            self.assertEqual(results[0].doc_id, "doc_sparse")
            self.assertEqual(results[0].text, "")
            self.assertEqual(results[0].distance, None)


if __name__ == "__main__":
    unittest.main()
