from __future__ import annotations

import os

from .config import resolve_llm_mode
from .schemas import AnswerResult, BackendMode, LlmMode, RetrievedChunk
from .utils import RagFaqError, content_tokens, normalize_text, sentence_split, tokenize

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer using ONLY the retrieved context. "
    "If the context is insufficient, say that the retrieved context is insufficient. "
    "Do not use outside knowledge."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[source={chunk.source_id} chunk={chunk.chunk_id}]\n{chunk.text.strip()}"
        )
    return "\n\n".join(blocks)


def build_openai_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    context = format_context(chunks)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer the question using ONLY the retrieved context. "
        "If the retrieved context is insufficient, say so plainly."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", "")
    if output_text:
        return normalize_text(output_text)
    try:
        return normalize_text(response.choices[0].message.content)
    except Exception as exc:  # pragma: no cover - SDK-dependent edge case
        raise RagFaqError(
            "OpenAI returned a response in an unexpected format."
        ) from exc


def generate_with_openai(question: str, chunks: list[RetrievedChunk]) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RagFaqError(
            "OpenAI LLM mode was requested but OPENAI_API_KEY is not set."
        )

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RagFaqError(
            f"OpenAI LLM mode is unavailable: {type(exc).__name__}: {exc}"
        ) from exc

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = build_openai_messages(question, chunks)
    try:
        response = client.responses.create(model="gpt-4o-mini", input=messages)
        return _extract_response_text(response)
    except Exception:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return _extract_response_text(response)


def _sentence_relevance(question: str, sentence: str, chunk_rank: int) -> float:
    question_tokens = set(content_tokens(question))
    sentence_tokens = content_tokens(sentence)
    if not sentence_tokens:
        return 0.0
    overlap = sum(1 for token in sentence_tokens if token in question_tokens)
    novelty = len(set(sentence_tokens)) / max(len(sentence_tokens), 1)
    rank_bonus = 1.0 / (chunk_rank + 1)
    return overlap * 2.5 + novelty + rank_bonus


def generate_offline_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    candidates: list[tuple[float, int, str]] = []
    for chunk_rank, chunk in enumerate(chunks):
        for sentence in sentence_split(chunk.text):
            score = _sentence_relevance(question, sentence, chunk_rank)
            candidates.append((score, chunk_rank, sentence))

    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected: list[str] = []
    seen = set()
    for score, _, sentence in candidates:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        if len(selected) >= 3:
            break

    if not selected:
        return "The retrieved context is insufficient to answer the question."

    if candidates and candidates[0][0] <= 1.0:
        return (
            "The retrieved context does not contain a direct answer. "
            f"The closest relevant statement is: {selected[0]}"
        )

    return normalize_text(" ".join(selected))


def answer_question(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    requested_llm: LlmMode,
    resolved_backend: BackendMode,
) -> AnswerResult:
    resolved_llm = resolve_llm_mode(requested_llm)

    if resolved_llm is LlmMode.OPENAI:
        answer = generate_with_openai(question, retrieved_chunks)
    else:
        answer = generate_offline_answer(question, retrieved_chunks)

    source_ids: list[str] = []
    for chunk in retrieved_chunks:
        if chunk.source_id not in source_ids:
            source_ids.append(chunk.source_id)

    return AnswerResult(
        question=question,
        answer=answer,
        sources=source_ids,
        resolved_backend=resolved_backend,
        resolved_llm=resolved_llm,
        retrieved_chunks=retrieved_chunks,
    )
