from __future__ import annotations

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .schemas import Chunk, Document
from .utils import sentence_split, tokenize


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
        metadata=dict(document.metadata),
    )


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    sentences = sentence_split(document.text)
    if not sentences:
        return []

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
        while next_start > start and overlap_tokens < overlap:
            overlap_tokens += max(1, len(sentence_tokens[next_start]))
            next_start -= 1
        start = max(start + 1, next_start + 1)

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks

