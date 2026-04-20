from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import rag_system
from ragfaq import embeddings as embedding_module
from ragfaq import vector_store as vector_store_module
from ragfaq.schemas import AnswerResult, BackendMode, Chunk, LlmMode, RetrievedChunk
from ragfaq.utils import RagFaqError, stable_text_hash


class _FakeVector(list):
    def tolist(self) -> list[float]:
        return list(self)


def test_embedding_provider_raises_friendly_error_when_package_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = embedding_module.SentenceTransformerEmbeddingProvider(tmp_path)

    def _raise_import_error():
        raise ImportError("missing sentence_transformers")

    monkeypatch.setattr(embedding_module, "_load_sentence_transformer_class", _raise_import_error)

    with pytest.raises(RagFaqError, match="sentence-transformers"):
        provider.embed_texts(["hello"])


def test_embedding_provider_raises_friendly_error_when_model_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = embedding_module.SentenceTransformerEmbeddingProvider(tmp_path)
    monkeypatch.setattr(provider, "local_model_status", lambda: (False, "cache not found"))
    monkeypatch.setattr(embedding_module, "_load_sentence_transformer_class", lambda: object)

    with pytest.raises(RagFaqError, match="not available locally"):
        provider.embed_texts(["hello"])


def test_embedding_provider_uses_cpu_batch_size_and_normalized_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls: list[dict[str, object]] = []
    encode_calls: list[dict[str, object]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs) -> None:
            init_calls.append({"model_name": model_name, **kwargs})

        def encode(self, texts, **kwargs):
            encode_calls.append({"texts": list(texts), **kwargs})
            return [_FakeVector([1.0, 0.0]), _FakeVector([0.0, 1.0])]

    provider = embedding_module.SentenceTransformerEmbeddingProvider(tmp_path)
    monkeypatch.setattr(provider, "local_model_status", lambda: (True, str(tmp_path)))
    monkeypatch.setattr(
        embedding_module,
        "_load_sentence_transformer_class",
        lambda: FakeSentenceTransformer,
    )

    vectors = provider.embed_texts(["first", "second"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert init_calls[0]["device"] == "cpu"
    assert init_calls[0]["local_files_only"] is True
    assert encode_calls[0]["batch_size"] == 16
    assert encode_calls[0]["normalize_embeddings"] is True
    assert encode_calls[0]["show_progress_bar"] is False


def test_chroma_vector_store_uses_add_and_query_with_expected_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.add_kwargs = None
            self.query_kwargs = None
            self.deleted_ids = None

        def add(self, **kwargs) -> None:
            self.add_kwargs = kwargs

        def delete(self, ids=None) -> None:
            self.deleted_ids = ids

        def count(self) -> int:
            return 3

        def query(self, **kwargs):
            self.query_kwargs = kwargs
            return {
                "documents": [["chunk one", "chunk two", "chunk three"]],
                "metadatas": [[
                    {
                        "chunk_id": "doc_a::chunk000",
                        "title": "Doc A",
                        "source": "knowledge_base/docs/doc_a.md",
                        "source_id": "doc_a",
                        "topic": "attention",
                        "chunk_index": "0",
                        "text_hash": "aaaa1111",
                    },
                    {
                        "chunk_id": "doc_b::chunk000",
                        "title": "Doc B",
                        "source": "knowledge_base/docs/doc_b.md",
                        "source_id": "doc_b",
                        "topic": "transformers",
                        "chunk_index": "0",
                        "text_hash": "bbbb2222",
                    },
                    {
                        "chunk_id": "doc_c::chunk000",
                        "title": "Doc C",
                        "source": "knowledge_base/docs/doc_c.md",
                        "source_id": "doc_c",
                        "topic": "rag",
                        "chunk_index": "0",
                        "text_hash": "cccc3333",
                    },
                ]],
                "distances": [[0.1, 0.2, 0.3]],
            }

    class FakeClient:
        def __init__(self) -> None:
            self.collection = FakeCollection()
            self.deleted_collection = None

        def get_or_create_collection(self, name: str):
            self.collection.name = name
            return self.collection

        def get_collection(self, name: str):
            self.collection.name = name
            return self.collection

        def delete_collection(self, name: str) -> None:
            self.deleted_collection = name

    fake_client = FakeClient()

    class FakeChromaModule:
        @staticmethod
        def PersistentClient(path: str):
            fake_client.path = path
            return fake_client

    monkeypatch.setattr(vector_store_module, "_load_chromadb_module", lambda: FakeChromaModule)

    store = vector_store_module.ChromaVectorStore(tmp_path, collection_name="demo_collection")
    chunks = [
        Chunk(
            chunk_id="doc_a::chunk000",
            source_id="doc_a",
            title="Doc A",
            text="First chunk text",
            token_count=3,
            metadata={
                "source": "knowledge_base/docs/doc_a.md",
                "topic": "attention",
                "chunk_index": "0",
            },
        ),
        Chunk(
            chunk_id="doc_b::chunk000",
            source_id="doc_b",
            title="Doc B",
            text="Second chunk text",
            token_count=3,
            metadata={
                "source": "knowledge_base/docs/doc_b.md",
                "topic": "transformers",
                "chunk_index": "0",
            },
        ),
        Chunk(
            chunk_id="doc_c::chunk000",
            source_id="doc_c",
            title="Doc C",
            text="Third chunk text",
            token_count=3,
            metadata={
                "source": "knowledge_base/docs/doc_c.md",
                "topic": "rag",
                "chunk_index": "0",
            },
        ),
    ]
    store.index(chunks, [[0.1], [0.2], [0.3]], rebuild=True)

    assert fake_client.deleted_collection == "demo_collection"
    assert fake_client.collection.add_kwargs is not None
    assert list(fake_client.collection.add_kwargs.keys()) == [
        "documents",
        "embeddings",
        "metadatas",
        "ids",
    ]
    metadata = fake_client.collection.add_kwargs["metadatas"][0]
    assert metadata["source"] == "knowledge_base/docs/doc_a.md"
    assert metadata["source_id"] == "doc_a"
    assert metadata["topic"] == "attention"
    assert metadata["chunk_index"] == "0"
    assert metadata["text_hash"] == stable_text_hash("First chunk text")

    results = store.query([0.5], top_k=3)
    assert fake_client.collection.query_kwargs["n_results"] == 3
    assert len(results) == 3
    assert results[0].rank == 1
    assert results[0].source_id == "doc_a"
    assert results[0].distance == 0.1
    assert results[0].metadata["source"] == "knowledge_base/docs/doc_a.md"


def test_ask_chroma_prints_three_ranked_retrieval_rows(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    retrieved_chunks = [
        RetrievedChunk(
            rank=1,
            chunk_id="faq_attention_003::chunk000",
            source_id="faq_attention_003",
            title="What is self attention?",
            text="Self attention lets tokens in the same sequence compare with one another.",
            score=0.91,
            backend="chroma",
            distance=0.10,
            metadata={"source": "knowledge_base/faqs.csv", "topic": "attention"},
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="faq_attention_006::chunk000",
            source_id="faq_attention_006",
            title="Why does self attention help long range dependencies?",
            text="Self attention can connect distant tokens directly.",
            score=0.85,
            backend="chroma",
            distance=0.18,
            metadata={"source": "knowledge_base/faqs.csv", "topic": "attention"},
        ),
        RetrievedChunk(
            rank=3,
            chunk_id="self_attention_and_transformer_architecture::chunk000",
            source_id="self_attention_and_transformer_architecture",
            title="Self-Attention and Transformer Architecture",
            text="Queries keys and values are learned projections used inside self attention.",
            score=0.80,
            backend="chroma",
            distance=0.25,
            metadata={
                "source": "knowledge_base/docs/self_attention_and_transformer_architecture.md",
                "topic": "attention",
            },
        ),
    ]

    monkeypatch.setattr(
        rag_system,
        "retrieve",
        lambda *args, **kwargs: (retrieved_chunks, BackendMode.CHROMA),
    )
    monkeypatch.setattr(
        rag_system,
        "answer_question",
        lambda **kwargs: AnswerResult(
            question=kwargs["question"],
            answer="Self-attention compares tokens within the same sequence.",
            sources=[chunk.source_id for chunk in retrieved_chunks],
            resolved_backend=BackendMode.CHROMA,
            resolved_llm=LlmMode.OFFLINE,
            retrieved_chunks=retrieved_chunks,
        ),
    )

    result = rag_system.main(
        [
            "ask",
            "--backend",
            "chroma",
            "--llm",
            "offline",
            "--question",
            "What is self-attention?",
        ]
    )
    output = capsys.readouterr().out
    assert result == 0
    assert "Resolved backend: chroma" in output
    assert "Retrieval results:" in output
    assert "1. source=knowledge_base/faqs.csv source_id=faq_attention_003" in output
    assert "2. source=knowledge_base/faqs.csv source_id=faq_attention_006" in output
    assert (
        "3. source=knowledge_base/docs/self_attention_and_transformer_architecture.md "
        "source_id=self_attention_and_transformer_architecture" in output
    )
