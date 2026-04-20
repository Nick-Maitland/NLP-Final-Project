from __future__ import annotations

from dataclasses import asdict

from .config import COLLECTION_NAME
from .schemas import Chunk, RetrievedChunk
from .utils import RagFaqError


class ChromaStore:
    def __init__(self, persist_dir, collection_name: str = COLLECTION_NAME) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def _get_client(self):
        try:
            import chromadb
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

    def rebuild(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=self.collection_name)
        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "title": chunk.title,
                "token_count": str(chunk.token_count),
                **chunk.metadata,
            }
            for chunk in chunks
        ]
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return collection.count()

    def has_index(self) -> bool:
        try:
            collection = self._get_collection(create=False)
        except Exception:
            return False
        return collection.count() > 0

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[RetrievedChunk]:
        try:
            collection = self._get_collection(create=False)
        except Exception as exc:
            raise RagFaqError(
                "Dense retrieval index is missing. Run `python rag_system.py build "
                "--backend chroma` after resolving dense dependency issues."
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
        for document, metadata, distance in zip(documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            metadata = metadata or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=metadata.get("chunk_id", ""),
                    source_id=metadata.get("source_id", ""),
                    title=metadata.get("title", ""),
                    text=document,
                    score=score,
                    backend="chroma",
                    metadata={k: str(v) for k, v in metadata.items()},
                )
            )
        return retrieved

