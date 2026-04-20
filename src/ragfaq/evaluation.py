from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import COLLECTION_NAME, DEFAULT_CANDIDATE_K, DEFAULT_TOP_K, PathConfig, get_paths
from .generation import ABSTENTION_TEXT, answer_question, strip_citation_markers
from .reporting import generate_evaluation_report, generate_failure_report, summarize_results
from .retrievers import retrieve
from .schemas import BackendMode, EvaluationRow, LlmMode, RetrievedChunk
from .utils import (
    content_tokens,
    dump_json,
    normalize_text,
    read_csv_rows,
    sentence_split,
    tokenize,
    write_csv_rows,
)

ROOT_QUESTION_FIELDNAMES = [
    "question_id",
    "question",
    "expected_source_id",
    "expected_topic",
    "answerable",
    "retrieved_source_ids",
    "retrieval_recall_at_3",
    "reciprocal_rank",
    "faithfulness_score",
    "citation_valid",
    "abstention_correct",
    "answer",
    "notes",
]

SCORED_FIELDNAMES = [
    *ROOT_QUESTION_FIELDNAMES,
    "latency_ms",
    "resolved_backend",
    "resolved_llm",
    "citation_warnings",
    "retrieved_chunk_ids",
]


def _split_values(value: str) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def _parse_bool(value: str) -> bool:
    if value is None:
        return False
    return value.strip().lower() == "true"


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _format_bool(value: bool) -> str:
    return str(value).lower()


def _score_recall_at_3(expected_source_ids: list[str], retrieved_source_ids: list[str]) -> float | None:
    if not expected_source_ids:
        return None
    hits = sum(1 for source_id in expected_source_ids if source_id in retrieved_source_ids[:3])
    return round(hits / max(len(expected_source_ids), 1), 2)


def _score_reciprocal_rank(expected_source_ids: list[str], retrieved_source_ids: list[str]) -> float | None:
    if not expected_source_ids:
        return None
    for index, source_id in enumerate(retrieved_source_ids[:3], start=1):
        if source_id in expected_source_ids:
            return round(1.0 / index, 2)
    return 0.0


def _sentence_supported(sentence: str, context: str, context_tokens: set[str]) -> bool:
    normalized_sentence = sentence.strip().lower()
    if not normalized_sentence:
        return False
    if normalized_sentence in context.lower():
        return True
    sentence_tokens = [token for token in content_tokens(sentence) if len(token) > 2]
    if not sentence_tokens:
        return True
    overlap = sum(1 for token in sentence_tokens if token in context_tokens)
    return (overlap / len(sentence_tokens)) >= 0.6


def score_faithfulness(
    question: str,
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    citation_valid: bool,
    abstained: bool,
) -> float:
    question_tokens = content_tokens(question)
    if abstained:
        if normalize_text(answer) != ABSTENTION_TEXT:
            return 0.0
        context = " ".join(chunk.text for chunk in retrieved_chunks)
        context_tokens = set(tokenize(context))
        if not question_tokens:
            return 1.0
        relevance_overlap = sum(1 for token in question_tokens if token in context_tokens) / len(
            question_tokens
        )
        if relevance_overlap >= 0.65:
            return 0.35
        if relevance_overlap >= 0.4:
            return 0.6
        return 1.0

    raw_answer = strip_citation_markers(answer)
    context = " ".join(chunk.text for chunk in retrieved_chunks)
    context_tokens = set(tokenize(context))
    answer_tokens = content_tokens(raw_answer)
    answer_token_set = set(answer_tokens)
    token_overlap = 0.0
    if answer_tokens:
        token_overlap = sum(1 for token in answer_tokens if token in context_tokens) / len(answer_tokens)

    question_context_overlap = 0.0
    question_answer_overlap = 0.0
    if question_tokens:
        question_context_overlap = sum(
            1 for token in question_tokens if token in context_tokens
        ) / len(question_tokens)
        question_answer_overlap = sum(
            1 for token in question_tokens if token in answer_token_set
        ) / len(question_tokens)
    relevance_overlap = max(question_context_overlap, question_answer_overlap)

    sentences = sentence_split(raw_answer)
    sentence_support = 0.0
    if sentences:
        supported = sum(
            1 for sentence in sentences if _sentence_supported(sentence, context, context_tokens)
        )
        sentence_support = supported / len(sentences)

    cited_context = " ".join(
        chunk.text for chunk in retrieved_chunks if f"[{chunk.rank}]" in answer
    )
    cited_tokens = set(tokenize(cited_context))
    cited_overlap = 0.0
    if answer_tokens and cited_tokens:
        cited_overlap = sum(1 for token in answer_tokens if token in cited_tokens) / len(answer_tokens)
    elif citation_valid:
        cited_overlap = token_overlap

    score = (
        token_overlap
        + sentence_support
        + cited_overlap
        + relevance_overlap
        + relevance_overlap
    ) / 5.0
    if relevance_overlap < 0.65:
        score *= max(relevance_overlap / 0.65, 0.25)
    if not citation_valid:
        score *= 0.8
    return round(score, 2)


def _abstention_correct(answerable: bool, abstained: bool) -> bool:
    return abstained if not answerable else not abstained


def _retrieved_chunk_summaries(chunks: list[RetrievedChunk]) -> str:
    summaries = []
    for chunk in chunks:
        snippet = normalize_text(chunk.text).replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117].rstrip() + "..."
        summaries.append(f"[{chunk.rank}] {chunk.source_id}: {snippet}")
    return " || ".join(summaries)


def _row_from_result(
    row: dict[str, str],
    *,
    retrieved_chunks: list[RetrievedChunk],
    answer,
    latency_ms: float,
) -> EvaluationRow:
    expected_source_ids = _split_values(row["expected_source_id"])
    expected_topics = _split_values(row["expected_topic"])
    answerable = _parse_bool(row["answerable"])
    retrieved_source_ids: list[str] = []
    for chunk in retrieved_chunks:
        if chunk.source_id not in retrieved_source_ids:
            retrieved_source_ids.append(chunk.source_id)

    recall = _score_recall_at_3(expected_source_ids, retrieved_source_ids) if answerable else None
    reciprocal_rank = (
        _score_reciprocal_rank(expected_source_ids, retrieved_source_ids) if answerable else None
    )
    citation_valid = len(answer.citation_warnings) == 0
    abstention_correct = _abstention_correct(answerable, answer.abstained)
    faithfulness_score = score_faithfulness(
        row["question"],
        answer.answer_text,
        retrieved_chunks,
        citation_valid=citation_valid,
        abstained=answer.abstained,
    )

    return EvaluationRow(
        question_id=row["question_id"],
        question=row["question"],
        expected_source_id=";".join(expected_source_ids),
        expected_topic=";".join(expected_topics) if expected_topics else row["expected_topic"],
        answerable=answerable,
        retrieved_source_ids=";".join(retrieved_source_ids),
        retrieval_recall_at_3=recall,
        reciprocal_rank=reciprocal_rank,
        faithfulness_score=faithfulness_score,
        citation_valid=citation_valid,
        abstention_correct=abstention_correct,
        answer=answer.answer_text,
        notes=row.get("notes") or "",
        abstained=answer.abstained,
        latency_ms=round(latency_ms, 2),
        resolved_backend=answer.resolved_backend.value,
        resolved_llm=answer.resolved_llm.value,
        citation_warnings="; ".join(answer.citation_warnings),
        retrieved_chunk_ids=";".join(chunk.chunk_id for chunk in retrieved_chunks),
        retrieved_chunk_summaries=_retrieved_chunk_summaries(retrieved_chunks),
    )


def _to_root_row(result: EvaluationRow) -> dict[str, str]:
    return {
        "question_id": result.question_id,
        "question": result.question,
        "expected_source_id": result.expected_source_id,
        "expected_topic": result.expected_topic,
        "answerable": _format_bool(result.answerable),
        "retrieved_source_ids": result.retrieved_source_ids,
        "retrieval_recall_at_3": _format_float(result.retrieval_recall_at_3),
        "reciprocal_rank": _format_float(result.reciprocal_rank),
        "faithfulness_score": _format_float(result.faithfulness_score),
        "citation_valid": _format_bool(result.citation_valid),
        "abstention_correct": _format_bool(result.abstention_correct),
        "answer": result.answer,
        "notes": result.notes,
    }


def _to_scored_row(result: EvaluationRow) -> dict[str, str]:
    return {
        **_to_root_row(result),
        "latency_ms": f"{result.latency_ms:.2f}",
        "resolved_backend": result.resolved_backend,
        "resolved_llm": result.resolved_llm,
        "citation_warnings": result.citation_warnings,
        "retrieved_chunk_ids": result.retrieved_chunk_ids,
    }


def run_evaluation(
    requested_backend: BackendMode,
    requested_llm: LlmMode,
    paths: PathConfig | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    collection_name: str = COLLECTION_NAME,
    show_context: bool = False,
    trace_output_path: Path | None = None,
) -> tuple[list[EvaluationRow], dict[str, object]]:
    paths = paths or get_paths()
    rows = read_csv_rows(paths.test_questions_path)
    results: list[EvaluationRow] = []
    traces: list[dict[str, object]] = []

    for row in rows:
        question = row["question"]
        started = time.perf_counter()
        retrieval = retrieve(
            question,
            requested_backend=requested_backend,
            top_k=top_k,
            candidate_k=candidate_k,
            paths=paths,
            collection_name=collection_name,
        )
        answer = answer_question(
            question=question,
            retrieved_chunks=retrieval.chunks,
            requested_llm=requested_llm,
            resolved_backend=retrieval.resolved_backend,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if show_context:
            print(f"[{row['question_id']}] {question}")
            for chunk in retrieval.chunks:
                print(f"{chunk.rank}. {chunk.chunk_id}")
                print(chunk.text)
                print("")
        if retrieval.trace is not None:
            traces.append(retrieval.trace)

        results.append(
            _row_from_result(
                row,
                retrieved_chunks=retrieval.chunks,
                answer=answer,
                latency_ms=latency_ms,
            )
        )

    summary = summarize_results(results)

    write_csv_rows(
        paths.test_questions_path,
        ROOT_QUESTION_FIELDNAMES,
        [_to_root_row(result) for result in results],
    )
    write_csv_rows(
        paths.scored_questions_path,
        SCORED_FIELDNAMES,
        [_to_scored_row(result) for result in results],
    )
    dump_json(paths.evaluation_summary_path, summary)
    generate_evaluation_report(results, summary, paths.evaluation_report_path)
    generate_failure_report(results, paths.failure_report_path)

    if trace_output_path is not None:
        dump_json(
            trace_output_path,
            {
                "mode": "evaluation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_count": len(traces),
                "traces": traces,
            },
        )
    return results, summary
