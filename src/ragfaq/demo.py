from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import PathConfig
from .generation import answer_question
from .retrievers import retrieve
from .schemas import AnswerResult, BackendMode, LlmMode, RetrievalRunResult, RetrievedChunk
from .utils import ensure_parent_dir, normalize_text


@dataclass(frozen=True)
class DemoQuestion:
    question: str
    rationale: str
    category: str


@dataclass(frozen=True)
class DemoEntry:
    question: str
    rationale: str
    category: str
    retrieval: RetrievalRunResult
    answer: AnswerResult
    latency_ms: float
    offline_fallback_used: bool


DEFAULT_DEMO_QUESTIONS = [
    DemoQuestion(
        question="What is self-attention?",
        rationale="Shows the core transformer mechanism with a direct in-scope definition.",
        category="answerable",
    ),
    DemoQuestion(
        question="Why can transformers train more in parallel than RNNs?",
        rationale="Highlights a classic architectural comparison and retrieval of a conceptual explanation.",
        category="answerable",
    ),
    DemoQuestion(
        question="Why is retrieval helpful before answer generation in a RAG system?",
        rationale="Demonstrates the project's retrieval-augmented workflow on a portfolio-relevant question.",
        category="answerable",
    ),
    DemoQuestion(
        question="Why does metadata matter after a vector store retrieves a chunk?",
        rationale="Shows source tracing and citation-oriented evidence handling.",
        category="answerable",
    ),
    DemoQuestion(
        question="What is the capital city of France?",
        rationale="Shows the offline system abstaining on an out-of-scope question.",
        category="out_of_scope",
    ),
]


def demo_questions_for_run(ad_hoc_question: str | None) -> list[DemoQuestion]:
    if ad_hoc_question:
        return [
            DemoQuestion(
                question=ad_hoc_question,
                rationale="Ad hoc demo question supplied by the user.",
                category="custom",
            )
        ]
    return DEFAULT_DEMO_QUESTIONS


def _offline_fallback_used(
    requested_backend: BackendMode,
    requested_llm: LlmMode,
    answer: AnswerResult,
) -> bool:
    return (
        requested_backend is BackendMode.AUTO and answer.resolved_backend is BackendMode.TFIDF
    ) or (requested_llm is LlmMode.AUTO and answer.resolved_llm is LlmMode.OFFLINE)


def run_demo(
    *,
    requested_backend: BackendMode,
    requested_llm: LlmMode,
    questions: list[DemoQuestion],
    paths: PathConfig,
    top_k: int,
    candidate_k: int,
    collection_name: str,
) -> list[DemoEntry]:
    entries: list[DemoEntry] = []
    for prompt in questions:
        started = time.perf_counter()
        retrieval = retrieve(
            question=prompt.question,
            requested_backend=requested_backend,
            top_k=top_k,
            candidate_k=candidate_k,
            paths=paths,
            collection_name=collection_name,
        )
        answer = answer_question(
            question=prompt.question,
            retrieved_chunks=retrieval.chunks,
            requested_llm=requested_llm,
            resolved_backend=retrieval.resolved_backend,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        entries.append(
            DemoEntry(
                question=prompt.question,
                rationale=prompt.rationale,
                category=prompt.category,
                retrieval=retrieval,
                answer=answer,
                latency_ms=round(latency_ms, 2),
                offline_fallback_used=_offline_fallback_used(
                    requested_backend,
                    requested_llm,
                    answer,
                ),
            )
        )
    return entries


def build_demo_trace_payload(entries: list[DemoEntry]) -> dict[str, object]:
    return {
        "mode": "demo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_count": len(entries),
        "traces": [entry.retrieval.trace for entry in entries if entry.retrieval.trace is not None],
    }


def _format_source_line(chunk: RetrievedChunk) -> str:
    topic = chunk.metadata.get("topic", "general")
    source = chunk.metadata.get("source", "unknown")
    chunk_index = chunk.metadata.get("chunk_index", "0")
    return f"[{chunk.rank}] {chunk.source_id} | {topic} | {source} | {chunk_index}"


def _context_snippet(chunk: RetrievedChunk, limit: int = 220) -> str:
    snippet = normalize_text(chunk.text).replace("\n", " ")
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3].rstrip() + "..."


def _trace_summary(entry: DemoEntry) -> list[str]:
    if entry.retrieval.trace is None:
        return []
    finals = entry.retrieval.trace.get("final_chunks", [])
    lines = []
    for final_chunk in finals[:3]:
        rank = final_chunk.get("final_rank", "?")
        source_id = final_chunk.get("source_id", "unknown")
        reason = final_chunk.get("selection_reason") or "selected as a retrieved candidate"
        lines.append(f"- [{rank}] {source_id}: {reason}")
    return lines


def render_demo_markdown(entries: list[DemoEntry], *, requested_backend: BackendMode, requested_llm: LlmMode, show_context: bool) -> str:
    resolved_backends = sorted({entry.answer.resolved_backend.value for entry in entries})
    resolved_llms = sorted({entry.answer.resolved_llm.value for entry in entries})
    any_fallback = any(entry.offline_fallback_used for entry in entries)
    average_latency = sum(entry.latency_ms for entry in entries) / max(len(entries), 1)

    lines = [
        "# Demo Run",
        "",
        f"- Timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"- Requested backend: `{requested_backend.value}`",
        f"- Requested llm: `{requested_llm.value}`",
        f"- Question count: {len(entries)}",
        f"- Resolved backends used: {', '.join(resolved_backends)}",
        f"- Resolved llm modes used: {', '.join(resolved_llms)}",
        f"- Any offline fallback used: {str(any_fallback).lower()}",
        f"- Average latency (ms): {average_latency:.2f}",
        "",
    ]

    for index, entry in enumerate(entries, start=1):
        lines.extend(
            [
                f"## {index}. {entry.question}",
                "",
                f"- Rationale: {entry.rationale}",
                f"- Expected behavior category: `{entry.category}`",
                f"- Resolved backend: `{entry.answer.resolved_backend.value}`",
                f"- Resolved llm: `{entry.answer.resolved_llm.value}`",
                f"- Latency (ms): {entry.latency_ms:.2f}",
                f"- Offline fallback used: {str(entry.offline_fallback_used).lower()}",
                "",
                "### Answer",
                "",
                entry.answer.answer_text or entry.answer.answer,
                "",
                "### Sources",
                "",
            ]
        )
        for chunk in entry.answer.retrieved_chunks:
            lines.append(f"- {_format_source_line(chunk)}")
        if show_context:
            lines.extend(["", "### Context", ""])
            for chunk in entry.answer.retrieved_chunks:
                lines.append(f"- [{chunk.rank}] {_context_snippet(chunk)}")
        trace_lines = _trace_summary(entry)
        if trace_lines:
            lines.extend(["", "### Retrieval Trace Summary", ""])
            lines.extend(trace_lines)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_demo_markdown(
    entries: list[DemoEntry],
    *,
    requested_backend: BackendMode,
    requested_llm: LlmMode,
    paths: PathConfig,
    show_context: bool,
) -> None:
    ensure_parent_dir(paths.demo_run_path)
    paths.demo_run_path.write_text(
        render_demo_markdown(
            entries,
            requested_backend=requested_backend,
            requested_llm=requested_llm,
            show_context=show_context,
        ),
        encoding="utf-8",
    )
