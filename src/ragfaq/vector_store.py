from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .config import COLLECTION_NAME, DENSE_TOP_K
from .schemas import Chunk, RetrievedChunk
from .utils import RagFaqError, stable_text_hash


def _load_chromadb_module():
    import chromadb

    return chromadb


class VectorStore(ABC):
    @abstractmethod
    def has_index(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def index(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        rebuild: bool = False,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int = DENSE_TOP_K) -> list[RetrievedChunk]:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: Path, collection_name: str = COLLECTION_NAME) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def _get_client(self):
        try:
            chromadb = _load_chromadb_module()
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            raise RagFaqError(
                f"ChromaDB is unavailable in this environment: {type(exc).__name__}: {exc}"
            ) from exc
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.persist_dir))

    def _get_collection(self, create: bool = True):
        client = self._get_client()
        if create:
            return client.get_or_create_collection(name=self.collection_name)
        return client.get_collection(name=self.collection_name)

    def _delete_existing_ids(self, collection, chunk_ids: list[str]) -> None:
        try:
            collection.delete(ids=chunk_ids)
        except Exception:
            return

    def _metadata_for_chunk(self, chunk: Chunk) -> dict[str, str]:
        return {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "source": chunk.metadata.get("source", ""),
            "source_id": chunk.source_id,
            "topic": chunk.metadata.get("topic", "general"),
            "chunk_index": chunk.metadata.get("chunk_index", "0"),
            "text_hash": stable_text_hash(chunk.text),
        }

    def index(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        rebuild: bool = False,
    ) -> int:
        client = self._get_client()
        if rebuild:
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
        collection = client.get_or_create_collection(name=self.collection_name)
        if not rebuild:
            self._delete_existing_ids(collection, [chunk.chunk_id for chunk in chunks])

        collection.add(
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[self._metadata_for_chunk(chunk) for chunk in chunks],
            ids=[chunk.chunk_id for chunk in chunks],
        )
        return collection.count()

    def has_index(self) -> bool:
        try:
            collection = self._get_collection(create=False)
        except Exception:
            return False
        return collection.count() > 0

    def query(self, query_embedding: list[float], top_k: int = DENSE_TOP_K) -> list[RetrievedChunk]:
        try:
            collection = self._get_collection(create=False)
        except Exception as exc:
            raise RagFaqError(
                "Dense retrieval index is missing. Run `python rag_system.py build "
                "--backend chroma --rebuild` after resolving dense dependency issues."
            ) from exc

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for index, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1,
        ):
            metadata = {k: str(v) for k, v in (metadata or {}).items()}
            distance_value = float(distance)
            retrieved.append(
                RetrievedChunk(
                    rank=index,
                    chunk_id=metadata.get("chunk_id", ""),
                    source_id=metadata.get("source_id", ""),
                    title=metadata.get("title", metadata.get("source_id", "")),
                    text=document,
                    score=1.0 / (1.0 + distance_value),
                    backend="chroma",
                    distance=distance_value,
                    dense_rank=index,
                    dense_score=1.0 / (1.0 + distance_value),
                    metadata=metadata,
                )
            )
        return retrieved
