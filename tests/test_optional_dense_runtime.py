from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.embeddings import SentenceTransformerEmbeddingProvider
from ragfaq.schemas import Chunk
from ragfaq.vector_store import ChromaVectorStore

pytestmark = pytest.mark.optional_dense


def _import_or_skip(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"optional dependency {module_name} unavailable: {type(exc).__name__}: {exc}")


def test_optional_chromadb_runtime_roundtrip(tmp_path: Path) -> None:
    _import_or_skip("chromadb")
    store = ChromaVectorStore(tmp_path / "chroma_db", collection_name="runtime_test")
    chunks = [
        Chunk(
            chunk_id="doc_a::chunk000",
            source_id="doc_a",
            title="Doc A",
            text="Self-attention compares tokens in the same sequence.",
            token_count=7,
            metadata={"source": "knowledge_base/docs/doc_a.md", "topic": "attention", "chunk_index": "0"},
        ),
        Chunk(
            chunk_id="doc_b::chunk000",
            source_id="doc_b",
            title="Doc B",
            text="Transformers can process tokens in parallel within a layer.",
            token_count=9,
            metadata={
                "source": "knowledge_base/docs/doc_b.md",
                "topic": "transformers",
                "chunk_index": "0",
            },
        ),
    ]
    count = store.index(chunks, [[1.0, 0.0], [0.0, 1.0]], rebuild=True)
    assert count == 2
    results = store.query([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].source_id == "doc_a"
    assert results[0].metadata["topic"] == "attention"


def test_optional_sentence_transformer_runtime_uses_local_cache_only(tmp_path: Path) -> None:
    _import_or_skip("sentence_transformers")
    provider = SentenceTransformerEmbeddingProvider(tmp_path)
    cached, detail = provider.local_model_status()
    if not cached:
        pytest.skip(f"MiniLM model not cached locally: {detail}")
    vectors = provider.embed_texts(["self attention test"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0
