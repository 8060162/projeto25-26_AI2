"""
Testes do backend de storage local.

Usa tmp_path do pytest — sem estado global, sem disco real partilhado.
"""
from __future__ import annotations

import pytest

from rag_api.pipeline.storage.local_disk import LocalDiskBackend


@pytest.fixture
def backend(tmp_path):
    return LocalDiskBackend(storage_path=str(tmp_path))


class TestLocalDiskBackend:

    @pytest.mark.asyncio
    async def test_save_and_read_roundtrip(self, backend):
        data = b"%PDF-1.4 fake content"
        file_ref = await backend.save("doc-1", "regulamento.pdf", data)
        assert await backend.read(file_ref) == data

    @pytest.mark.asyncio
    async def test_save_creates_file(self, backend, tmp_path):
        await backend.save("doc-2", "test.pdf", b"content")
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "doc-2_test.pdf"

    @pytest.mark.asyncio
    async def test_exists_true_after_save(self, backend):
        file_ref = await backend.save("doc-3", "a.pdf", b"x")
        assert await backend.exists(file_ref) is True

    @pytest.mark.asyncio
    async def test_exists_false_for_unknown(self, backend):
        assert await backend.exists("/tmp/nonexistent_file.pdf") is False

    @pytest.mark.asyncio
    async def test_delete_removes_file(self, backend):
        file_ref = await backend.save("doc-4", "b.pdf", b"y")
        await backend.delete(file_ref)
        assert await backend.exists(file_ref) is False

    @pytest.mark.asyncio
    async def test_delete_is_idempotent(self, backend):
        """Segundo delete não deve levantar excepção."""
        file_ref = await backend.save("doc-5", "c.pdf", b"z")
        await backend.delete(file_ref)
        await backend.delete(file_ref)  # não deve falhar

    @pytest.mark.asyncio
    async def test_doc_id_prefix_prevents_collisions(self, backend):
        ref1 = await backend.save("doc-A", "same.pdf", b"version1")
        ref2 = await backend.save("doc-B", "same.pdf", b"version2")
        assert ref1 != ref2
        assert await backend.read(ref1) == b"version1"
        assert await backend.read(ref2) == b"version2"