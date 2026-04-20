from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.chunking import chunk_documents
from ragfaq.config import get_paths
from ragfaq.ingest import inspect_documents, load_documents
from ragfaq.retrievers import build_lexical_index, query_lexical_index


def _write_retrieval_faq_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "source_id,question,answer,topic,difficulty",
                (
                    "faq_attention_001,What is self-attention?,"
                    "Self-attention lets each token compare with other tokens in the same sequence.,"
                    "attention,intro"
                ),
                (
                    "faq_transformers_001,Why are transformers parallelizable?,"
                    "Transformers can process tokens in parallel within a layer instead of one step at a time.,"
                    "transformers,intro"
                ),
                (
                    "faq_rag_001,Why is retrieval useful in RAG?,"
                    "Retrieval provides external evidence before answer generation.,"
                    "rag,intermediate"
                ),
                (
                    "faq_chromadb_001,Why is metadata useful in vector search?,"
                    "Metadata helps trace retrieved chunks back to their source and topic.,"
                    "chromadb_vector_search,intermediate"
                ),
            ]
        ),
        encoding="utf-8",
    )


def test_repo_knowledge_base_loads_expected_file_mix() -> None:
    inspection = inspect_documents(get_paths())
    assert inspection["source_file_count"] >= 2
    assert inspection["faq_row_count"] >= 75
    assert "attention" in inspection["topics"]
    assert "transformers" in inspection["topics"]


def test_tfidf_retrieval_returns_top_three_ranked_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base_dir = tmp_path / "knowledge_base"
    knowledge_base_dir.mkdir()
    _write_retrieval_faq_csv(knowledge_base_dir / "faqs.csv")

    monkeypatch.setenv("RAGFAQ_ROOT", str(tmp_path))
    paths = get_paths()
    documents = load_documents(paths)
    chunks = chunk_documents(documents)
    build_lexical_index(chunks, paths)

    results = query_lexical_index(
        "How does self-attention help a token use context?",
        top_k=3,
        paths=paths,
    )
    assert len(results) == 3
    assert [chunk.rank for chunk in results] == [1, 2, 3]
    assert results[0].source_id == "faq_attention_001"
    assert results[0].score >= results[1].score >= results[2].score
    assert all(chunk.backend == "tfidf" for chunk in results)
    assert all(chunk.metadata["source"] == "knowledge_base/faqs.csv" for chunk in results)
    assert all("topic" in chunk.metadata for chunk in results)
