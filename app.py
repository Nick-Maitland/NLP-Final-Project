from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Streamlit is optional and not installed. Install it manually with "
        "`pip install streamlit` and then run `streamlit run app.py`."
    ) from exc

from ragfaq.chunking import chunk_documents
from ragfaq.config import COLLECTION_NAME, DEFAULT_CANDIDATE_K, DEFAULT_TOP_K, ensure_runtime_directories, get_paths
from ragfaq.generation import answer_question
from ragfaq.ingest import load_documents
from ragfaq.retrievers import maybe_build_indexes, retrieve
from ragfaq.schemas import BackendMode, LlmMode
from ragfaq.utils import RagFaqError


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


def main() -> None:
    st.set_page_config(page_title="RAG FAQ Demo", page_icon="📚", layout="wide")
    st.title("RAG FAQ Demo")
    st.caption("Optional local UI. Defaults stay offline-safe for an M1 Mac.")

    with st.sidebar:
        backend = st.selectbox("Backend", [mode.value for mode in BackendMode], index=1)
        llm = st.selectbox("LLM mode", [mode.value for mode in LlmMode], index=1)
        top_k = st.number_input("Top-k", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        candidate_k = st.number_input(
            "Candidate-k",
            min_value=1,
            max_value=30,
            value=DEFAULT_CANDIDATE_K,
        )
        collection_name = st.text_input("Collection name", value=COLLECTION_NAME)

    question = st.text_area(
        "Question",
        value="What is self-attention?",
        height=120,
    )
    run = st.button("Run query", type="primary")

    if not run:
        return

    paths = ensure_runtime_directories(get_paths())
    try:
        _ensure_lexical_index(paths)
        retrieval = retrieve(
            question=question,
            requested_backend=BackendMode(backend),
            top_k=int(top_k),
            candidate_k=int(candidate_k),
            paths=paths,
            collection_name=collection_name,
        )
        answer = answer_question(
            question=question,
            retrieved_chunks=retrieval.chunks,
            requested_llm=LlmMode(llm),
            resolved_backend=retrieval.resolved_backend,
        )
    except RagFaqError as exc:
        st.error(str(exc))
        return

    st.subheader("Answer")
    st.write(answer.answer_text or answer.answer)
    st.caption(
        f"Resolved backend: {answer.resolved_backend.value} | "
        f"Resolved llm: {answer.resolved_llm.value}"
    )

    st.subheader("Sources")
    for chunk in answer.retrieved_chunks:
        topic = chunk.metadata.get("topic", "general")
        source = chunk.metadata.get("source", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", "0")
        st.markdown(
            f"- **[{chunk.rank}] {chunk.source_id}** | {topic} | {source} | {chunk_index}"
        )

    st.subheader("Retrieved Context")
    for chunk in answer.retrieved_chunks:
        with st.expander(f"[{chunk.rank}] {chunk.source_id}"):
            st.write(chunk.text)

    st.subheader("Retrieval Trace")
    st.json(retrieval.trace or {"trace": None})


if __name__ == "__main__":
    main()
