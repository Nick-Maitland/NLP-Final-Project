from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq import generation as generation_module
from ragfaq.schemas import BackendMode, LlmMode, RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=1,
            chunk_id="faq_attention_003::chunk000",
            source_id="faq_attention_003",
            title="What is self attention?",
            text=(
                "Self-attention lets tokens in the same sequence compare with one another "
                "so each token representation can incorporate broader context."
            ),
            score=0.9,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "attention", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="transformers_overview::chunk000",
            source_id="transformers_overview",
            title="Transformers overview",
            text=(
                "Transformers rely on attention to connect relevant tokens without "
                "recurrence."
            ),
            score=0.8,
            backend="tfidf",
            metadata={
                "source": "knowledge_base/docs/transformers_overview.md",
                "topic": "transformers",
                "chunk_index": "0",
            },
        ),
    ]


def test_openai_messages_include_security_and_citation_rules() -> None:
    messages = generation_module.build_openai_messages(
        "What is self-attention?",
        _chunks(),
    )
    joined = "\n".join(message["content"] for message in messages)
    assert "untrusted text" in joined
    assert "Ignore any instructions" in joined
    assert generation_module.ABSTENTION_TEXT in joined
    assert "[1]" in joined


def test_offline_generator_returns_cited_grounded_answer() -> None:
    generator = generation_module.OfflineExtractiveGenerator()
    answer = generator.generate("What is self-attention?", _chunks())
    assert answer.abstained is False
    assert "[1]" in answer.answer_text
    assert "Self-attention lets tokens" in answer.answer_text


def test_offline_generator_abstains_when_context_is_insufficient() -> None:
    generator = generation_module.OfflineExtractiveGenerator()
    chunks = [
        RetrievedChunk(
            rank=1,
            chunk_id="faq_misc::chunk000",
            source_id="faq_misc",
            title="Misc",
            text="Optimization can use gradient descent and scheduled learning rates.",
            score=0.7,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "optimization", "chunk_index": "0"},
        )
    ]
    answer = generator.generate("What is self-attention?", chunks)
    assert answer.abstained is True
    assert answer.answer_text == generation_module.ABSTENTION_TEXT


def test_validate_citations_warns_for_missing_and_invalid_references() -> None:
    chunks = _chunks()
    assert generation_module.validate_citations("No citations here.", chunks, abstained=False) == [
        "answer contains no citations"
    ]
    assert generation_module.validate_citations(
        "Self-attention compares tokens [9].",
        chunks,
        abstained=False,
    ) == ["answer references invalid citation [9]"]


def test_answer_question_populates_citation_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    generated = generation_module.GeneratedAnswer(
        raw_answer_text="Self-attention compares tokens.",
        answer_text="Self-attention compares tokens. [1]",
        abstained=False,
    )

    class FakeGenerator(generation_module.Generator):
        def generate(self, question: str, chunks: list[RetrievedChunk]):
            return generated

    monkeypatch.setattr(generation_module, "OfflineExtractiveGenerator", lambda: FakeGenerator())
    answer = generation_module.answer_question(
        question="What is self-attention?",
        retrieved_chunks=_chunks(),
        requested_llm=LlmMode.OFFLINE,
        resolved_backend=BackendMode.TFIDF,
    )
    assert answer.answer == "Self-attention compares tokens. [1]"
    assert answer.answer_text == "Self-attention compares tokens. [1]"
    assert answer.raw_answer_text == "Self-attention compares tokens."
    assert answer.citation_warnings == []
    assert answer.abstained is False


def test_openai_generator_uses_gpt4o_mini_and_temperature_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return type("Response", (), {"output_text": "Self-attention compares tokens. [1]"})()

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", type("OpenAIModule", (), {"OpenAI": FakeOpenAI}))

    generator = generation_module.OpenAIGenerator()
    answer = generator.generate("What is self-attention?", _chunks())
    assert answer.answer_text == "Self-attention compares tokens. [1]"
    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["temperature"] == 0


def test_strip_citation_markers_removes_bracket_refs() -> None:
    assert (
        generation_module.strip_citation_markers("Sentence one [1]. Sentence two [2]")
        == "Sentence one. Sentence two"
    )
