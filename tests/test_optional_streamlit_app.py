from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.schemas import AnswerResult, BackendMode, LlmMode, RetrievalRunResult, RetrievedChunk

APP_PATH = ROOT_DIR / "app.py"


def _load_app_module():
    existing = sys.modules.get("app")
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location("app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load app.py for tests.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["app"] = module
    spec.loader.exec_module(module)
    return module


app = _load_app_module()


def _snapshot(**overrides) -> app.RuntimeSnapshot:
    defaults = {
        "lexical_index_ready": True,
        "dense_index_ready": True,
        "chromadb_available": True,
        "chromadb_reason": "",
        "sentence_transformers_available": True,
        "sentence_transformers_reason": "",
        "dense_model_cached": True,
        "dense_model_detail": "/tmp/model-cache",
        "openai_sdk_available": True,
        "openai_sdk_reason": "",
        "openai_key_available": True,
    }
    defaults.update(overrides)
    return app.RuntimeSnapshot(**defaults)


def _retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        chunk_id="faq_attention_003::chunk000",
        source_id="faq_attention_003",
        title="What is self-attention?",
        text="Self-attention lets tokens compare with one another in the same sequence.",
        score=0.91,
        backend="tfidf",
        lexical_rank=1,
        lexical_score=0.91,
        metadata={
            "source": "knowledge_base/faqs.csv",
            "topic": "attention",
            "chunk_index": "0",
        },
    )


def test_app_import_does_not_require_streamlit() -> None:
    module = importlib.import_module("app")
    assert module.DEFAULT_BACKEND_SELECTION is BackendMode.TFIDF
    assert module.DEFAULT_LLM_SELECTION is LlmMode.OFFLINE


def test_load_streamlit_is_lazy_and_errors_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = app.importlib.import_module

    def _fake_import_module(name: str):
        if name == "streamlit":
            raise ImportError("streamlit missing for test")
        return real_import_module(name)

    monkeypatch.setattr(app.importlib, "import_module", _fake_import_module)

    with pytest.raises(SystemExit) as exc_info:
        app._load_streamlit()

    assert "pip install streamlit" in str(exc_info.value)


def test_default_app_controls_are_offline_safe_and_ordered() -> None:
    assert app.BACKEND_OPTIONS == [
        BackendMode.AUTO,
        BackendMode.TFIDF,
        BackendMode.CHROMA,
        BackendMode.HYBRID,
    ]
    assert app.LLM_OPTIONS == [
        LlmMode.AUTO,
        LlmMode.OFFLINE,
        LlmMode.OPENAI,
    ]
    assert app.DEFAULT_BACKEND_SELECTION is BackendMode.TFIDF
    assert app.DEFAULT_LLM_SELECTION is LlmMode.OFFLINE


def test_auto_selection_messages_explain_offline_safe_fallbacks() -> None:
    snapshot = _snapshot(
        chromadb_available=False,
        chromadb_reason="ModuleNotFoundError: No module named 'chromadb'",
        dense_index_ready=False,
        openai_key_available=False,
    )

    messages = app.build_selection_messages(BackendMode.AUTO, LlmMode.AUTO, snapshot)
    texts = [message.text for message in messages]

    assert any("Auto backend will stay on the offline-safe TF-IDF path" in text for text in texts)
    assert any("Auto LLM will resolve to offline mode because OPENAI_API_KEY is not set" in text for text in texts)


def test_explicit_dense_backend_reports_missing_runtime() -> None:
    snapshot = _snapshot(
        chromadb_available=False,
        chromadb_reason="ModuleNotFoundError: No module named 'chromadb'",
        dense_index_ready=False,
    )

    messages = app.build_selection_messages(BackendMode.CHROMA, LlmMode.OFFLINE, snapshot)

    assert messages[0].level == "error"
    assert "Dense retrieval is unavailable" in messages[0].text
    assert "chromadb unavailable" in messages[0].text


def test_explicit_dense_backend_reports_missing_index() -> None:
    snapshot = _snapshot(dense_index_ready=False)

    messages = app.build_selection_messages(BackendMode.HYBRID, LlmMode.OFFLINE, snapshot)

    assert messages[0].level == "error"
    assert "dense index is not built yet" in messages[0].text
    assert "python rag_system.py build --backend hybrid" in messages[0].text


def test_explicit_openai_reports_missing_key() -> None:
    snapshot = _snapshot(openai_key_available=False)

    messages = app.build_selection_messages(BackendMode.TFIDF, LlmMode.OPENAI, snapshot)

    assert messages[0].level == "error"
    assert "OPENAI_API_KEY is not set" in messages[0].text


def test_explicit_openai_reports_missing_sdk() -> None:
    snapshot = _snapshot(
        openai_key_available=True,
        openai_sdk_available=False,
        openai_sdk_reason="ModuleNotFoundError: No module named 'openai'",
    )

    messages = app.build_selection_messages(BackendMode.TFIDF, LlmMode.OPENAI, snapshot)

    assert messages[0].level == "error"
    assert "OpenAI SDK is unavailable" in messages[0].text


def test_preflight_auto_llm_falls_back_to_offline_when_sdk_missing() -> None:
    snapshot = _snapshot(
        openai_key_available=True,
        openai_sdk_available=False,
        openai_sdk_reason="ModuleNotFoundError: No module named 'openai'",
    )

    decision = app.build_preflight_decision(BackendMode.AUTO, LlmMode.AUTO, snapshot)

    assert decision.can_run is True
    assert decision.effective_llm is LlmMode.OFFLINE
    assert decision.adjusted_llm_note is not None


def test_preflight_blocks_explicit_openai_without_key() -> None:
    snapshot = _snapshot(openai_key_available=False)

    decision = app.build_preflight_decision(BackendMode.TFIDF, LlmMode.OPENAI, snapshot)

    assert decision.can_run is False
    assert decision.effective_llm is LlmMode.OPENAI


def test_build_result_summary_surfaces_runtime_metrics_and_confidence() -> None:
    chunk = _retrieved_chunk()
    retrieval = RetrievalRunResult(
        chunks=[chunk],
        resolved_backend=BackendMode.TFIDF,
        trace={"backend": "tfidf"},
        fallback_reason="dense retrieval unavailable",
    )
    answer = AnswerResult(
        question="What is self-attention?",
        answer="Self-attention compares tokens. [1]",
        sources=[chunk.source_id],
        resolved_backend=BackendMode.TFIDF,
        resolved_llm=LlmMode.OFFLINE,
        retrieved_chunks=[chunk],
        answer_text="Self-attention compares tokens. [1]",
        raw_answer_text="Self-attention compares tokens.",
        citation_warnings=[],
        abstained=False,
        confidence_score=0.73,
        confidence_reasons=[],
        confidence_gate_triggered=False,
        question_type="definition",
    )

    summary = app.build_result_summary(retrieval, answer, latency_ms=12.345)

    assert summary["backend_used"] == "tfidf"
    assert summary["llm_used"] == "offline"
    assert summary["latency_ms"] == 12.35
    assert summary["confidence_score"] == 0.73
    assert summary["question_type"] == "definition"
    assert summary["fallback_reason"] == "dense retrieval unavailable"
