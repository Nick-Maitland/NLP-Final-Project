from __future__ import annotations

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .schemas import Chunk, Document
from .utils import content_tokens, jaccard_similarity, normalize_text, sentence_split, tokenize

MIN_USEFUL_CONTENT_TOKENS = 12
NEAR_DUPLICATE_SIMILARITY = 0.92


def _build_chunk(
    document: Document,
    chunk_index: int,
    sentences: list[str],
    token_count: int,
) -> Chunk:
    return Chunk(
        chunk_id=f"{document.source_id}::chunk{chunk_index:03d}",
        source_id=document.source_id,
        title=document.title,
        text=" ".join(sentences).strip(),
        token_count=token_count,
        metadata={
            **dict(document.metadata),
            "source_id": document.source_id,
            "chunk_index": str(chunk_index),
        },
    )


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    sentences = sentence_split(document.text)
    if not sentences:
        return []
    effective_overlap = 0 if document.metadata.get("kind") == "faq" else overlap

    sentence_tokens = [tokenize(sentence) for sentence in sentences]
    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(sentences):
        token_total = 0
        end = start
        collected: list[str] = []
        while end < len(sentences):
            sentence_token_count = max(1, len(sentence_tokens[end]))
            if collected and token_total + sentence_token_count > chunk_size:
                break
            collected.append(sentences[end])
            token_total += sentence_token_count
            end += 1

        chunks.append(_build_chunk(document, chunk_index, collected, token_total))
        chunk_index += 1

        if end >= len(sentences):
            break

        overlap_tokens = 0
        next_start = end - 1
        while next_start > start and overlap_tokens < effective_overlap:
            overlap_tokens += max(1, len(sentence_tokens[next_start]))
            next_start -= 1
        start = max(start + 1, next_start + 1)

    return chunks


def _are_near_duplicates(first: Chunk, second: Chunk) -> bool:
    first_text = normalize_text(first.text).lower()
    second_text = normalize_text(second.text).lower()
    if first_text == second_text:
        return True

    first_tokens = content_tokens(first.text)
    second_tokens = content_tokens(second.text)
    similarity = jaccard_similarity(first_tokens, second_tokens)
    if similarity < NEAR_DUPLICATE_SIMILARITY:
        return False

    shorter = max(1, min(len(first_tokens), len(second_tokens)))
    longer = max(len(first_tokens), len(second_tokens))
    return (longer / shorter) <= 1.25


def _deduplicate_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], list[str]]:
    unique_chunks: list[Chunk] = []
    duplicate_chunk_ids: list[str] = []
    for chunk in chunks:
        if any(_are_near_duplicates(chunk, existing_chunk) for existing_chunk in unique_chunks):
            duplicate_chunk_ids.append(chunk.chunk_id)
            continue
        unique_chunks.append(chunk)
    return unique_chunks, duplicate_chunk_ids


def inspect_chunk_collection(chunks: list[Chunk]) -> dict[str, object]:
    sorted_chunks = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    too_short_chunk_ids = [
        chunk.chunk_id
        for chunk in sorted_chunks
        if len(content_tokens(chunk.text)) < MIN_USEFUL_CONTENT_TOKENS
    ]
    topics = sorted({chunk.metadata.get("topic", "general") for chunk in sorted_chunks})
    return {
        "chunk_count": len(sorted_chunks),
        "topics": topics,
        "top_chunk_ids": [chunk.chunk_id for chunk in sorted_chunks[:10]],
        "too_short_chunk_ids": too_short_chunk_ids,
    }


def chunk_documents_with_report(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[Chunk], dict[str, object]]:
    raw_chunks: list[Chunk] = []
    for document in documents:
        raw_chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))

    unique_chunks, duplicate_chunk_ids = _deduplicate_chunks(raw_chunks)
    report = inspect_chunk_collection(unique_chunks)
    report["raw_chunk_count"] = len(raw_chunks)
    report["duplicate_chunk_ids"] = duplicate_chunk_ids
    report["duplicate_chunk_count"] = len(duplicate_chunk_ids)
    return unique_chunks, report


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks, _ = chunk_documents_with_report(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return chunks
