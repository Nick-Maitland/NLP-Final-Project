from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone

from .config import (
    COLLECTION_NAME,
    DENSE_TOP_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
    HYBRID_MMR_LAMBDA,
    HYBRID_RRF_K,
    PathConfig,
    get_paths,
    get_runtime_availability,
    resolve_query_backend,
)
from .embeddings import SentenceTransformerEmbeddingProvider, _load_sentence_transformer_class
from .schemas import BackendMode, Chunk, RetrievalRunResult, RetrievedChunk
from .utils import (
    RagFaqError,
    content_tokens,
    dump_json,
    jaccard_similarity,
    load_json,
    token_signature,
    tokenize,
)
from .vector_store import ChromaVectorStore


def _chunk_to_payload(chunk: Chunk, term_frequencies: dict[str, int]) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "title": chunk.title,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata,
        "term_frequencies": term_frequencies,
        "doc_length": sum(term_frequencies.values()),
    }


def build_lexical_index(chunks: list[Chunk], paths: PathConfig | None = None) -> dict[str, object]:
    paths = paths or get_paths()
    document_frequency: Counter[str] = Counter()
    documents: list[dict[str, object]] = []
    total_doc_length = 0

    for chunk in chunks:
        tokens = content_tokens(chunk.text)
        term_frequencies = Counter(tokens)
        documents.append(_chunk_to_payload(chunk, dict(term_frequencies)))
        document_frequency.update(term_frequencies.keys())
        total_doc_length += len(tokens)

    avg_doc_length = total_doc_length / max(len(documents), 1)
    payload = {
        "version": 1,
        "document_frequency": dict(document_frequency),
        "documents": documents,
        "avg_doc_length": avg_doc_length,
        "document_count": len(documents),
    }
    dump_json(paths.lexical_index_path, payload)
    dump_json(
        paths.chunk_cache_path,
        {
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]
        },
    )
    return payload


def lexical_index_exists(paths: PathConfig | None = None) -> bool:
    paths = paths or get_paths()
    return paths.lexical_index_path.exists()


def _dense_backend_available(paths: PathConfig | None = None) -> tuple[bool, str]:
    paths = paths or get_paths()
    availability = get_runtime_availability()
    if not availability.chromadb.available:
        return False, f"chromadb unavailable: {availability.chromadb.reason}"
    try:
        _load_sentence_transformer_class()
    except Exception as exc:
        return (
            False,
            "sentence-transformers unavailable: "
            f"{type(exc).__name__}: {exc}",
        )
    provider = SentenceTransformerEmbeddingProvider(paths.cache_dir)
    cached, detail = provider.local_model_status()
    if not cached:
        return False, detail
    return True, ""


def dense_index_exists(
    paths: PathConfig | None = None,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    paths = paths or get_paths()
    return ChromaVectorStore(paths.chroma_dir, collection_name=collection_name).has_index()


def build_dense_index(
    chunks: list[Chunk],
    paths: PathConfig | None = None,
    collection_name: str = COLLECTION_NAME,
    rebuild: bool = False,
) -> dict[str, object]:
    paths = paths or get_paths()
    embedder = SentenceTransformerEmbeddingProvider(paths.cache_dir)
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    store = ChromaVectorStore(paths.chroma_dir, collection_name=collection_name)
    count = store.index(chunks, embeddings, rebuild=rebuild)
    return {"document_count": count}


def maybe_build_indexes(
    chunks: list[Chunk],
    requested_backend: BackendMode,
    paths: PathConfig | None = None,
    collection_name: str = COLLECTION_NAME,
    rebuild: bool = False,
) -> dict[str, object]:
    paths = paths or get_paths()
    summary: dict[str, object] = {}
    lexical_payload = build_lexical_index(chunks, paths)
    summary["lexical_index"] = {
        "chunk_count": lexical_payload["document_count"],
        "path": str(paths.lexical_index_path),
    }

    dense_requested = requested_backend in {
        BackendMode.CHROMA,
        BackendMode.HYBRID,
        BackendMode.AUTO,
    }
    if not dense_requested:
        summary["dense_index"] = {"built": False, "reason": "backend tfidf selected"}
        return summary

    dense_ok, reason = _dense_backend_available(paths)
    if not dense_ok:
        if requested_backend in {BackendMode.CHROMA, BackendMode.HYBRID}:
            raise RagFaqError(
                "Dense build was requested but unavailable. "
                f"Reason: {reason}"
            )
        summary["dense_index"] = {"built": False, "reason": reason}
        return summary

    try:
        dense_summary = build_dense_index(
            chunks,
            paths,
            collection_name=collection_name,
            rebuild=rebuild,
        )
    except RagFaqError as exc:
        if requested_backend in {BackendMode.CHROMA, BackendMode.HYBRID}:
            raise
        summary["dense_index"] = {"built": False, "reason": str(exc)}
        return summary
    summary["dense_index"] = {"built": True, **dense_summary}
    return summary


def _idf(term: str, document_frequency: dict[str, int], document_count: int) -> float:
    frequency = document_frequency.get(term, 0)
    return math.log(1.0 + ((document_count - frequency + 0.5) / (frequency + 0.5)))


def _bm25_score(
    query_tokens: list[str],
    document: dict[str, object],
    document_frequency: dict[str, int],
    document_count: int,
    avg_doc_length: float,
) -> float:
    k1 = 1.5
    b = 0.75
    term_frequencies = document.get("term_frequencies", {})
    doc_length = max(1, int(document.get("doc_length", 0)))
    score = 0.0
    for term in query_tokens:
        tf = int(term_frequencies.get(term, 0))
        if tf == 0:
            continue
        idf = _idf(term, document_frequency, document_count)
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * doc_length / max(avg_doc_length, 1.0))
        score += idf * (numerator / denominator)
    return score


def query_lexical_index(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    paths: PathConfig | None = None,
) -> list[RetrievedChunk]:
    paths = paths or get_paths()
    payload = load_json(paths.lexical_index_path, default=None)
    if not payload:
        raise RagFaqError(
            "Lexical index is missing. Run `python rag_system.py build --backend tfidf` first."
        )

    question_tokens = content_tokens(question) or tokenize(question)
    documents = payload["documents"]
    document_frequency = payload["document_frequency"]
    document_count = int(payload["document_count"])
    avg_doc_length = float(payload["avg_doc_length"])

    ranked = []
    for document in documents:
        score = _bm25_score(
            question_tokens,
            document,
            document_frequency,
            document_count,
            avg_doc_length,
        )
        ranked.append((score, document))

    ranked.sort(key=lambda item: item[0], reverse=True)
    top_ranked = ranked[:top_k]
    return [
        RetrievedChunk(
            rank=index,
            chunk_id=document["chunk_id"],
            source_id=document["source_id"],
            title=document["title"],
            text=document["text"],
            score=float(score),
            backend="tfidf",
            distance=None,
            lexical_rank=index,
            lexical_score=float(score),
            metadata={k: str(v) for k, v in document.get("metadata", {}).items()},
        )
        for index, (score, document) in enumerate(top_ranked, start=1)
    ]


def query_dense_index(
    question: str,
    top_k: int = DENSE_TOP_K,
    paths: PathConfig | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[RetrievedChunk]:
    paths = paths or get_paths()
    embedder = SentenceTransformerEmbeddingProvider(paths.cache_dir)
    query_embedding = embedder.embed_texts([question])[0]
    store = ChromaVectorStore(paths.chroma_dir, collection_name=collection_name)
    return store.query(query_embedding, top_k=top_k)


def _rrf_score(rank: int, rrf_k: int = HYBRID_RRF_K) -> float:
    return 1.0 / (rrf_k + rank)


def _trace_metadata(chunk: RetrievedChunk) -> dict[str, str]:
    return {
        "source": chunk.metadata.get("source", ""),
        "source_id": chunk.source_id,
        "topic": chunk.metadata.get("topic", "general"),
        "chunk_index": chunk.metadata.get("chunk_index", "0"),
    }


def _selection_reason(chunk: RetrievedChunk, novelty_penalty: float) -> str:
    if chunk.lexical_rank and chunk.dense_rank:
        reason = "selected because it ranked well in both dense and lexical retrieval"
    elif chunk.dense_rank:
        reason = "selected because dense retrieval found a strong semantic match"
    elif chunk.lexical_rank:
        reason = "selected because lexical retrieval matched key question terms"
    else:
        reason = "selected as a retrieved candidate"

    if novelty_penalty > 0.0:
        reason += " and MMR kept it because it added different context"
    return reason


def _candidate_trace_entry(
    chunk: RetrievedChunk,
    *,
    final_rank: int | None = None,
) -> dict[str, object]:
    metadata = _trace_metadata(chunk)
    return {
        "chunk_id": chunk.chunk_id,
        "source": metadata["source"],
        "source_id": metadata["source_id"],
        "topic": metadata["topic"],
        "chunk_index": metadata["chunk_index"],
        "dense_rank": chunk.dense_rank,
        "dense_score": chunk.dense_score,
        "lexical_rank": chunk.lexical_rank,
        "lexical_score": chunk.lexical_score,
        "fusion_score": chunk.fusion_score,
        "final_rank": final_rank,
        "mmr_score": chunk.mmr_score,
        "selection_reason": chunk.selection_reason,
    }


def _build_trace(
    *,
    question: str,
    backend: BackendMode,
    top_k: int,
    candidate_k: int,
    candidates: list[RetrievedChunk],
    finals: list[RetrievedChunk],
) -> dict[str, object]:
    return {
        "question": question,
        "backend": backend.value,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_chunks": [_candidate_trace_entry(chunk) for chunk in candidates],
        "final_chunks": [
            _candidate_trace_entry(chunk, final_rank=chunk.rank) for chunk in finals
        ],
    }


def _hybrid_candidates(
    question: str,
    *,
    candidate_k: int,
    paths: PathConfig | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[RetrievedChunk]:
    lexical = query_lexical_index(question, top_k=candidate_k, paths=paths)
    dense = query_dense_index(
        question,
        top_k=candidate_k,
        paths=paths,
        collection_name=collection_name,
    )
    lexical_map = {chunk.chunk_id: chunk for chunk in lexical}
    dense_map = {chunk.chunk_id: chunk for chunk in dense}

    merged: list[RetrievedChunk] = []
    for chunk_id in sorted(set(lexical_map) | set(dense_map)):
        lexical_chunk = lexical_map.get(chunk_id)
        dense_chunk = dense_map.get(chunk_id)
        base_chunk = dense_chunk or lexical_chunk
        if base_chunk is None:
            continue
        fusion_score = 0.0
        if lexical_chunk is not None:
            fusion_score += _rrf_score(lexical_chunk.rank)
        if dense_chunk is not None:
            fusion_score += _rrf_score(dense_chunk.rank)
        merged.append(
            RetrievedChunk(
                rank=0,
                chunk_id=base_chunk.chunk_id,
                source_id=base_chunk.source_id,
                title=base_chunk.title,
                text=base_chunk.text,
                score=fusion_score,
                backend="hybrid",
                distance=dense_chunk.distance if dense_chunk else None,
                lexical_rank=lexical_chunk.rank if lexical_chunk else None,
                lexical_score=lexical_chunk.score if lexical_chunk else None,
                dense_rank=dense_chunk.rank if dense_chunk else None,
                dense_score=dense_chunk.score if dense_chunk else None,
                fusion_score=fusion_score,
                metadata=base_chunk.metadata,
            )
        )

    return sorted(
        merged,
        key=lambda chunk: (
            chunk.fusion_score or 0.0,
            chunk.dense_score or 0.0,
            chunk.lexical_score or 0.0,
            chunk.chunk_id,
        ),
        reverse=True,
    )


def _mmr_rerank(candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        best_index = 0
        best_chunk = remaining[0]
        best_score = float("-inf")
        best_penalty = 0.0
        for index, candidate in enumerate(remaining):
            candidate_signature = token_signature(candidate.text)
            novelty_penalty = 0.0
            if selected:
                novelty_penalty = max(
                    jaccard_similarity(candidate_signature, token_signature(existing.text))
                    for existing in selected
                )
            relevance = candidate.fusion_score or candidate.score
            mmr_score = HYBRID_MMR_LAMBDA * relevance - (1.0 - HYBRID_MMR_LAMBDA) * novelty_penalty
            if mmr_score > best_score:
                best_index = index
                best_chunk = candidate
                best_score = mmr_score
                best_penalty = novelty_penalty

        remaining.pop(best_index)
        selected.append(
            RetrievedChunk(
                rank=len(selected) + 1,
                chunk_id=best_chunk.chunk_id,
                source_id=best_chunk.source_id,
                title=best_chunk.title,
                text=best_chunk.text,
                score=best_chunk.fusion_score or best_chunk.score,
                backend=best_chunk.backend,
                distance=best_chunk.distance,
                lexical_rank=best_chunk.lexical_rank,
                lexical_score=best_chunk.lexical_score,
                dense_rank=best_chunk.dense_rank,
                dense_score=best_chunk.dense_score,
                fusion_score=best_chunk.fusion_score,
                mmr_score=best_score,
                selection_reason=_selection_reason(best_chunk, best_penalty),
                metadata=best_chunk.metadata,
            )
        )
    return selected


def query_hybrid_index(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    paths: PathConfig | None = None,
    collection_name: str = COLLECTION_NAME,
) -> tuple[list[RetrievedChunk], dict[str, object]]:
    candidate_k = max(candidate_k, top_k)
    candidates = _hybrid_candidates(
        question,
        candidate_k=candidate_k,
        paths=paths,
        collection_name=collection_name,
    )
    finals = _mmr_rerank(candidates, top_k=top_k)
    return finals, _build_trace(
        question=question,
        backend=BackendMode.HYBRID,
        top_k=top_k,
        candidate_k=candidate_k,
        candidates=candidates,
        finals=finals,
    )


def retrieve(
    question: str,
    requested_backend: BackendMode,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    paths: PathConfig | None = None,
    collection_name: str = COLLECTION_NAME,
) -> RetrievalRunResult:
    paths = paths or get_paths()
    lexical_ready = lexical_index_exists(paths)
    if requested_backend is BackendMode.TFIDF:
        chunks = query_lexical_index(question, top_k=top_k, paths=paths)
        return RetrievalRunResult(
            chunks=chunks,
            resolved_backend=BackendMode.TFIDF,
            trace=_build_trace(
                question=question,
                backend=BackendMode.TFIDF,
                top_k=top_k,
                candidate_k=top_k,
                candidates=chunks,
                finals=chunks,
            ),
        )

    dense_backend_ok, dense_reason = _dense_backend_available(paths)
    dense_ready = dense_index_exists(paths, collection_name=collection_name) and dense_backend_ok
    resolved_backend = resolve_query_backend(requested_backend, lexical_ready, dense_ready)

    if requested_backend is BackendMode.CHROMA and not dense_ready:
        suffix = dense_reason if not dense_backend_ok else "Dense index not built yet."
        raise RagFaqError(
            "Dense retrieval requested but unavailable. "
            "Run `python rag_system.py build --backend chroma --rebuild` after fixing dependencies. "
            f"{suffix}"
        )
    if requested_backend is BackendMode.HYBRID and not (lexical_ready and dense_ready):
        suffix = dense_reason if not dense_backend_ok else "Both lexical and dense indexes must exist."
        raise RagFaqError(
            "Hybrid retrieval requested but unavailable. "
            "Run `python rag_system.py build --backend hybrid` after fixing dependencies. "
            f"{suffix}"
        )

    if resolved_backend is BackendMode.TFIDF:
        chunks = query_lexical_index(question, top_k=top_k, paths=paths)
        return RetrievalRunResult(
            chunks=chunks,
            resolved_backend=resolved_backend,
            trace=_build_trace(
                question=question,
                backend=resolved_backend,
                top_k=top_k,
                candidate_k=top_k,
                candidates=chunks,
                finals=chunks,
            ),
        )
    if resolved_backend is BackendMode.CHROMA:
        chunks = query_dense_index(
            question,
            top_k=top_k,
            paths=paths,
            collection_name=collection_name,
        )
        return RetrievalRunResult(
            chunks=chunks,
            resolved_backend=resolved_backend,
            trace=_build_trace(
                question=question,
                backend=resolved_backend,
                top_k=top_k,
                candidate_k=top_k,
                candidates=chunks,
                finals=chunks,
            ),
        )
    chunks, trace = query_hybrid_index(
        question,
        top_k=top_k,
        candidate_k=candidate_k,
        paths=paths,
        collection_name=collection_name,
    )
    return RetrievalRunResult(
        chunks=chunks,
        resolved_backend=resolved_backend,
        trace=trace,
    )


def inspect_index_state(
    paths: PathConfig | None = None,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, object]:
    paths = paths or get_paths()
    availability = get_runtime_availability()
    dense_ok, dense_reason = _dense_backend_available(paths)
    return {
        "lexical_index_ready": lexical_index_exists(paths),
        "dense_index_ready": dense_index_exists(paths, collection_name=collection_name),
        "dense_runtime_available": dense_ok,
        "dense_runtime_reason": dense_reason,
        "chroma_sdk_available": availability.chromadb.available,
        "sentence_transformers_available": availability.sentence_transformers.available,
        "openai_key_available": availability.openai_key_available,
        "openai_sdk_available": availability.openai_sdk.available,
    }
