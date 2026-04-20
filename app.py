from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.chunking import chunk_documents
from ragfaq.config import (
    COLLECTION_NAME,
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
    ensure_runtime_directories,
    get_paths,
    get_runtime_availability,
)
from ragfaq.embeddings import find_local_model_cache
from ragfaq.generation import answer_question
from ragfaq.ingest import load_documents
from ragfaq.retrievers import dense_index_exists, lexical_index_exists, maybe_build_indexes, retrieve
from ragfaq.schemas import AnswerResult, BackendMode, LlmMode, RetrievalRunResult
from ragfaq.utils import RagFaqError

DEFAULT_BACKEND_SELECTION = BackendMode.TFIDF
DEFAULT_LLM_SELECTION = LlmMode.OFFLINE
BACKEND_OPTIONS = [
    BackendMode.AUTO,
    BackendMode.TFIDF,
    BackendMode.CHROMA,
    BackendMode.HYBRID,
]
LLM_OPTIONS = [
    LlmMode.AUTO,
    LlmMode.OFFLINE,
    LlmMode.OPENAI,
]


@dataclass(frozen=True)
class AppMessage:
    level: str
    text: str


@dataclass(frozen=True)
class RuntimeSnapshot:
    lexical_index_ready: bool
    dense_index_ready: bool
    chromadb_available: bool
    chromadb_reason: str
    sentence_transformers_available: bool
    sentence_transformers_reason: str
    dense_model_cached: bool
    dense_model_detail: str
    openai_sdk_available: bool
    openai_sdk_reason: str
    openai_key_available: bool

    @property
    def dense_runtime_ready(self) -> bool:
        return (
            self.chromadb_available
            and self.sentence_transformers_available
            and self.dense_model_cached
        )

    @property
    def dense_runtime_reason(self) -> str:
        if not self.chromadb_available:
            return f"chromadb unavailable: {self.chromadb_reason}"
        if not self.sentence_transformers_available:
            return (
                "sentence-transformers unavailable: "
                f"{self.sentence_transformers_reason}"
            )
        if not self.dense_model_cached:
            return (
                "MiniLM model not cached locally: "
                f"{self.dense_model_detail}"
            )
        return "dense runtime ready"


@dataclass(frozen=True)
class PreflightDecision:
    can_run: bool
    effective_llm: LlmMode
    messages: list[AppMessage]
    adjusted_llm_note: str | None = None


def _load_streamlit():
    try:
        return importlib.import_module("streamlit")
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise SystemExit(
            "Streamlit is optional and not installed. Install it manually with "
            "`pip install streamlit` and then run `streamlit run app.py`."
        ) from exc


def _ensure_lexical_index(paths) -> None:
    if paths.lexical_index_path.exists():
        return
    documents = load_documents(paths)
    chunks = chunk_documents(documents)
    maybe_build_indexes(
        chunks,
        requested_backend=BackendMode.TFIDF,
        paths=paths,
        collection_name=COLLECTION_NAME,
    )


def build_runtime_snapshot(paths, collection_name: str) -> RuntimeSnapshot:
    availability = get_runtime_availability()
    dense_model_cached, dense_model_detail = find_local_model_cache(paths.cache_dir)
    dense_ready = False
    if availability.chromadb.available:
        dense_ready = dense_index_exists(paths, collection_name=collection_name)
    return RuntimeSnapshot(
        lexical_index_ready=lexical_index_exists(paths),
        dense_index_ready=dense_ready,
        chromadb_available=availability.chromadb.available,
        chromadb_reason=availability.chromadb.reason,
        sentence_transformers_available=availability.sentence_transformers.available,
        sentence_transformers_reason=availability.sentence_transformers.reason,
        dense_model_cached=dense_model_cached,
        dense_model_detail=dense_model_detail,
        openai_sdk_available=availability.openai_sdk.available,
        openai_sdk_reason=availability.openai_sdk.reason,
        openai_key_available=availability.openai_key_available,
    )


def _dense_build_command(backend: BackendMode) -> str:
    if backend is BackendMode.HYBRID:
        return "python rag_system.py build --backend hybrid"
    return "python rag_system.py build --backend chroma --rebuild"


def build_selection_messages(
    backend: BackendMode,
    llm: LlmMode,
    snapshot: RuntimeSnapshot,
) -> list[AppMessage]:
    messages: list[AppMessage] = []

    if backend in {BackendMode.CHROMA, BackendMode.HYBRID}:
        if not snapshot.dense_runtime_ready:
            messages.append(
                AppMessage(
                    "error",
                    "Dense retrieval is unavailable for the selected backend. "
                    f"{snapshot.dense_runtime_reason}",
                )
            )
        elif not snapshot.dense_index_ready:
            messages.append(
                AppMessage(
                    "error",
                    "Dense retrieval is configured but the dense index is not built yet. "
                    f"Run `{_dense_build_command(backend)}` first.",
                )
            )
    elif backend is BackendMode.AUTO:
        if not snapshot.dense_runtime_ready:
            messages.append(
                AppMessage(
                    "warning",
                    "Auto backend will stay on the offline-safe TF-IDF path because "
                    f"dense retrieval is unavailable. {snapshot.dense_runtime_reason}",
                )
            )
        elif not snapshot.dense_index_ready:
            messages.append(
                AppMessage(
                    "warning",
                    "Auto backend can use dense retrieval only after the dense index is built. "
                    f"Until then it will resolve to TF-IDF. Run `{_dense_build_command(BackendMode.CHROMA)}`.",
                )
            )

    if llm is LlmMode.OPENAI:
        if not snapshot.openai_key_available:
            messages.append(
                AppMessage(
                    "error",
                    "OpenAI mode was selected, but OPENAI_API_KEY is not set.",
                )
            )
        elif not snapshot.openai_sdk_available:
            messages.append(
                AppMessage(
                    "error",
                    "OpenAI mode was selected, but the OpenAI SDK is unavailable. "
                    f"Reason: {snapshot.openai_sdk_reason}",
                )
            )
    elif llm is LlmMode.AUTO:
        if not snapshot.openai_key_available:
            messages.append(
                AppMessage(
                    "info",
                    "Auto LLM will resolve to offline mode because OPENAI_API_KEY is not set.",
                )
            )
        elif not snapshot.openai_sdk_available:
            messages.append(
                AppMessage(
                    "warning",
                    "Auto LLM will resolve to offline mode in the app because the OpenAI SDK "
                    f"is unavailable. Reason: {snapshot.openai_sdk_reason}",
                )
            )

    return messages


def build_preflight_decision(
    backend: BackendMode,
    llm: LlmMode,
    snapshot: RuntimeSnapshot,
) -> PreflightDecision:
    messages = build_selection_messages(backend, llm, snapshot)
    effective_llm = llm
    adjusted_llm_note: str | None = None
    can_run = True

    if backend in {BackendMode.CHROMA, BackendMode.HYBRID} and any(
        message.level == "error" for message in messages
    ):
        can_run = False

    if llm is LlmMode.OPENAI and any(
        message.level == "error" and "OpenAI mode" in message.text for message in messages
    ):
        can_run = False

    if llm is LlmMode.AUTO and (
        not snapshot.openai_key_available or not snapshot.openai_sdk_available
    ):
        effective_llm = LlmMode.OFFLINE
        adjusted_llm_note = (
            "The app used offline generation for this run because OpenAI prerequisites "
            "were not fully available."
        )

    return PreflightDecision(
        can_run=can_run,
        effective_llm=effective_llm,
        messages=messages,
        adjusted_llm_note=adjusted_llm_note,
    )


def build_result_summary(
    retrieval: RetrievalRunResult,
    answer: AnswerResult,
    latency_ms: float,
) -> dict[str, object]:
    return {
        "backend_used": answer.resolved_backend.value,
        "llm_used": answer.resolved_llm.value,
        "latency_ms": round(latency_ms, 2),
        "abstained": answer.abstained,
        "confidence_score": answer.confidence_score,
        "confidence_gate_triggered": answer.confidence_gate_triggered,
        "question_type": answer.question_type or "n/a",
        "fallback_reason": retrieval.fallback_reason or "",
        "citation_warning_count": len(answer.citation_warnings),
        "source_count": len(answer.sources),
    }


def _environment_status_lines(snapshot: RuntimeSnapshot) -> list[str]:
    lines = [
        f"- Lexical index ready: `{str(snapshot.lexical_index_ready).lower()}`",
        f"- Dense index ready: `{str(snapshot.dense_index_ready).lower()}`",
        f"- ChromaDB available: `{str(snapshot.chromadb_available).lower()}`",
        f"- sentence-transformers available: `{str(snapshot.sentence_transformers_available).lower()}`",
        f"- MiniLM cached locally: `{str(snapshot.dense_model_cached).lower()}`",
        f"- OpenAI SDK available: `{str(snapshot.openai_sdk_available).lower()}`",
        f"- OPENAI_API_KEY set: `{str(snapshot.openai_key_available).lower()}`",
    ]
    if not snapshot.dense_runtime_ready:
        lines.append(f"- Dense runtime note: `{snapshot.dense_runtime_reason}`")
    if not snapshot.openai_sdk_available:
        lines.append(f"- OpenAI SDK note: `{snapshot.openai_sdk_reason}`")
    return lines


def _sources_rows(answer: AnswerResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for chunk in answer.retrieved_chunks:
        metrics = [f"score={chunk.score:.4f}", f"backend={chunk.backend}"]
        if chunk.distance is not None:
            metrics.append(f"distance={chunk.distance:.4f}")
        if chunk.lexical_rank is not None:
            metrics.append(f"lexical_rank={chunk.lexical_rank}")
        if chunk.dense_rank is not None:
            metrics.append(f"dense_rank={chunk.dense_rank}")
        if chunk.fusion_score is not None:
            metrics.append(f"fusion={chunk.fusion_score:.4f}")
        if chunk.mmr_score is not None:
            metrics.append(f"mmr={chunk.mmr_score:.4f}")
        rows.append(
            {
                "rank": chunk.rank,
                "source_id": chunk.source_id,
                "topic": chunk.metadata.get("topic", "general"),
                "source": chunk.metadata.get("source", "unknown"),
                "metrics": " | ".join(metrics),
            }
        )
    return rows


def _trace_summary_lines(retrieval: RetrievalRunResult) -> list[str]:
    if not retrieval.trace:
        return []
    lines: list[str] = []
    if retrieval.trace.get("auto_fallback_reason"):
        lines.append(
            "Auto fallback reason: "
            f"{retrieval.trace['auto_fallback_reason']}"
        )
    for chunk in retrieval.trace.get("final_chunks", [])[:3]:
        rank = chunk.get("final_rank", "?")
        source_id = chunk.get("source_id", "unknown")
        reason = chunk.get("selection_reason") or "selected as a retrieved candidate"
        lines.append(f"[{rank}] {source_id}: {reason}")
    return lines


def _render_message(st, message: AppMessage) -> None:
    if message.level == "error":
        st.error(message.text)
        return
    if message.level == "warning":
        st.warning(message.text)
        return
    st.info(message.text)


def _render_summary_metrics(st, summary: dict[str, object]) -> None:
    columns = st.columns(4)
    columns[0].metric("Backend used", str(summary["backend_used"]))
    columns[1].metric("LLM mode used", str(summary["llm_used"]))
    columns[2].metric("Latency", f"{summary['latency_ms']:.2f} ms")
    confidence_value = f"{float(summary['confidence_score']):.2f}"
    if bool(summary["abstained"]):
        confidence_value = f"{confidence_value} abstained"
    columns[3].metric("Confidence", confidence_value)


def _render_answer_panel(
    st,
    *,
    answer: AnswerResult,
    retrieval: RetrievalRunResult,
    adjusted_llm_note: str | None,
) -> None:
    if answer.abstained:
        st.warning("The system abstained because the retrieved context was insufficient.")
    st.write(answer.answer_text or answer.answer)

    diagnostics = [
        f"- Question type: `{answer.question_type or 'n/a'}`",
        f"- Confidence score: `{answer.confidence_score:.2f}`",
        f"- Confidence gate triggered: `{str(answer.confidence_gate_triggered).lower()}`",
        f"- Citation warnings: `{len(answer.citation_warnings)}`",
    ]
    st.markdown("### Diagnostics")
    st.markdown("\n".join(diagnostics))

    if adjusted_llm_note:
        st.info(adjusted_llm_note)
    if retrieval.fallback_reason:
        st.info(f"Retrieval fallback note: {retrieval.fallback_reason}")
    if answer.citation_warnings:
        for warning in answer.citation_warnings:
            st.warning(warning)
    if answer.confidence_reasons:
        st.markdown("### Confidence reasons")
        for reason in answer.confidence_reasons:
            st.markdown(f"- {reason}")


def _render_sources_panel(st, answer: AnswerResult) -> None:
    st.table(_sources_rows(answer))


def _render_context_panel(st, answer: AnswerResult) -> None:
    for chunk in answer.retrieved_chunks:
        header = f"[{chunk.rank}] {chunk.source_id}"
        with st.expander(header):
            st.caption(
                f"Topic: {chunk.metadata.get('topic', 'general')} | "
                f"Source: {chunk.metadata.get('source', 'unknown')}"
            )
            st.write(chunk.text)


def _render_trace_panel(st, retrieval: RetrievalRunResult) -> None:
    summary_lines = _trace_summary_lines(retrieval)
    if summary_lines:
        st.markdown("### Trace summary")
        for line in summary_lines:
            st.markdown(f"- {line}")
    st.markdown("### Raw trace")
    st.json(retrieval.trace or {"trace": None})


def _sidebar_controls(st) -> tuple[BackendMode, LlmMode, int, int, str]:
    with st.sidebar:
        st.header("Controls")
        backend = st.selectbox(
            "Backend",
            [mode.value for mode in BACKEND_OPTIONS],
            index=BACKEND_OPTIONS.index(DEFAULT_BACKEND_SELECTION),
        )
        llm = st.selectbox(
            "LLM mode",
            [mode.value for mode in LLM_OPTIONS],
            index=LLM_OPTIONS.index(DEFAULT_LLM_SELECTION),
        )
        top_k = st.number_input("Top-k", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        candidate_k = st.number_input(
            "Candidate-k",
            min_value=1,
            max_value=30,
            value=DEFAULT_CANDIDATE_K,
        )
        collection_name = st.text_input("Collection name", value=COLLECTION_NAME)
    return BackendMode(backend), LlmMode(llm), int(top_k), int(candidate_k), collection_name


def _run_query(
    *,
    question: str,
    backend: BackendMode,
    llm: LlmMode,
    top_k: int,
    candidate_k: int,
    paths,
    collection_name: str,
) -> tuple[RetrievalRunResult, AnswerResult, float]:
    started = time.perf_counter()
    retrieval = retrieve(
        question=question,
        requested_backend=backend,
        top_k=top_k,
        candidate_k=candidate_k,
        paths=paths,
        collection_name=collection_name,
    )
    answer = answer_question(
        question=question,
        retrieved_chunks=retrieval.chunks,
        requested_llm=llm,
        resolved_backend=retrieval.resolved_backend,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return retrieval, answer, latency_ms


def main() -> None:
    st = _load_streamlit()
    st.set_page_config(page_title="RAG FAQ Demo", page_icon="📚", layout="wide")

    paths = ensure_runtime_directories(get_paths())

    st.title("RAG FAQ Resume Demo")
    st.caption(
        "Optional Streamlit UI for demonstrating retrieval, fallback behavior, and "
        "grounded answering. The CLI remains the primary course submission path."
    )

    question = st.text_area(
        "Question input",
        value="What is self-attention?",
        height=120,
        help="Use an in-scope NLP question for the strongest demo. Offline-safe defaults are preselected.",
    )
    backend, llm, top_k, candidate_k, collection_name = _sidebar_controls(st)
    snapshot = build_runtime_snapshot(paths, collection_name)
    with st.sidebar:
        st.markdown("### Environment status")
        st.markdown("\n".join(_environment_status_lines(snapshot)))
    decision = build_preflight_decision(backend, llm, snapshot)

    st.markdown("### Runtime warnings")
    if decision.messages:
        for message in decision.messages:
            _render_message(st, message)
    else:
        st.info("Selected configuration is ready to run.")

    if not st.button("Run query", type="primary", use_container_width=True):
        return

    if not question.strip():
        st.error("Enter a question before running the demo.")
        return

    if not decision.can_run:
        st.error("The selected configuration is blocked. Resolve the warnings above and try again.")
        return

    try:
        _ensure_lexical_index(paths)
        retrieval, answer, latency_ms = _run_query(
            question=question.strip(),
            backend=backend,
            llm=decision.effective_llm,
            top_k=top_k,
            candidate_k=candidate_k,
            paths=paths,
            collection_name=collection_name,
        )
    except RagFaqError as exc:
        st.error(str(exc))
        return

    summary = build_result_summary(retrieval, answer, latency_ms)
    st.markdown("## Run summary")
    _render_summary_metrics(st, summary)

    tabs = st.tabs(["Answer", "Sources", "Retrieved Context", "Retrieval Trace"])
    with tabs[0]:
        _render_answer_panel(
            st,
            answer=answer,
            retrieval=retrieval,
            adjusted_llm_note=decision.adjusted_llm_note,
        )
    with tabs[1]:
        _render_sources_panel(st, answer)
    with tabs[2]:
        _render_context_panel(st, answer)
    with tabs[3]:
        _render_trace_panel(st, retrieval)


if __name__ == "__main__":
    main()
