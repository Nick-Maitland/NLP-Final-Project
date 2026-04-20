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


def _embedding_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=1,
            chunk_id="faq_embeddings_002::chunk000",
            source_id="faq_embeddings_002",
            title="Why are embeddings preferred over one hot vectors in modern NLP?",
            text=(
                "Question: Why are embeddings preferred over one hot vectors in modern NLP? "
                "Answer: Embeddings are compact and can place related items near each other "
                "while one hot vectors are sparse and do not encode similarity by themselves."
            ),
            score=16.37,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "embeddings", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="faq_rnns_006::chunk000",
            source_id="faq_rnns_006",
            title="Why are RNNs less parallelizable than transformers?",
            text=(
                "Question: Why are RNNs less parallelizable than transformers? Answer: "
                "RNNs usually process one step after another because each hidden state "
                "depends on the previous one."
            ),
            score=6.33,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "rnns", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=3,
            chunk_id="faq_embeddings_005::chunk000",
            source_id="faq_embeddings_005",
            title="What does proximity in embedding space suggest?",
            text=(
                "Question: What does proximity in embedding space suggest? Answer: "
                "It usually means the model has learned that the two items appear in related "
                "contexts or have similar functional roles."
            ),
            score=5.88,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "embeddings", "chunk_index": "0"},
        ),
    ]


def _count_answer_sentences(answer_text: str) -> int:
    normalized = generation_module.strip_citation_markers(answer_text)
    return len([sentence for sentence in generation_module.sentence_split(normalized) if sentence])


def _assert_openai_prompt_contract(messages: list[dict[str, str]]) -> None:
    joined = "\n".join(message["content"] for message in messages)
    assert generation_module.CONTEXT_ONLY_RULE in joined
    assert generation_module.PROMPT_INJECTION_RULE in joined
    assert generation_module.ABSTENTION_RULE in joined
    assert generation_module.CITATION_RULE in joined
    assert generation_module.SOURCE_ID_RULE in joined
    assert "source_id=faq_attention_003" in joined
    assert "[1]" in joined


def test_openai_messages_include_security_and_citation_rules() -> None:
    messages = generation_module.build_openai_messages(
        "What is self-attention?",
        _chunks(),
    )
    _assert_openai_prompt_contract(messages)


def test_offline_generator_returns_cited_grounded_answer() -> None:
    generator = generation_module.OfflineExtractiveGenerator()
    answer = generator.generate("What is self-attention?", _chunks())
    assert answer.abstained is False
    assert "[1]" in answer.answer_text
    assert "Self-attention lets tokens" in answer.answer_text
    assert _count_answer_sentences(answer.answer_text) <= 2


def test_offline_generator_returns_concise_embedding_answer_without_topic_drift() -> None:
    generator = generation_module.OfflineExtractiveGenerator()
    answer = generator.generate(
        "Why are embeddings more useful than one-hot vectors for meaning?",
        _embedding_chunks(),
    )
    assert answer.abstained is False
    assert "[1]" in answer.answer_text
    assert "Embeddings are compact" in answer.answer_text
    assert "RNNs usually process one step after another" not in answer.answer_text
    assert "transformers" not in answer.answer_text.lower()
    assert _count_answer_sentences(answer.answer_text) <= 2


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


def test_offline_generator_abstains_for_topical_but_unsupported_factoid() -> None:
    generator = generation_module.OfflineExtractiveGenerator()
    chunks = [
        RetrievedChunk(
            rank=1,
            chunk_id="pytorch::chunk000",
            source_id="tensors_and_pytorch_basics",
            title="PyTorch basics",
            text=(
                "PyTorch is popular because it supports automatic differentiation. "
                "PyTorch tensors can move between CPU and GPU."
            ),
            score=4.02,
            backend="tfidf",
            metadata={
                "source": "knowledge_base/docs/tensors_and_pytorch_basics.md",
                "topic": "tensors_and_pytorch",
                "chunk_index": "0",
            },
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="pytorch::chunk001",
            source_id="tensors_and_pytorch_basics",
            title="PyTorch basics",
            text="PyTorch uses tensors to store parameters, activations, gradients, and outputs.",
            score=3.42,
            backend="tfidf",
            metadata={
                "source": "knowledge_base/docs/tensors_and_pytorch_basics.md",
                "topic": "tensors_and_pytorch",
                "chunk_index": "1",
            },
        ),
        RetrievedChunk(
            rank=3,
            chunk_id="nn::chunk000",
            source_id="neural_networks_basics",
            title="Neural networks",
            text="A perceptron adds weighted inputs and a bias before applying an activation.",
            score=0.0,
            backend="tfidf",
            metadata={
                "source": "knowledge_base/docs/neural_networks_basics.md",
                "topic": "neural_networks",
                "chunk_index": "0",
            },
        ),
    ]
    answer = generator.generate("Who founded PyTorch?", chunks)
    assert answer.abstained is True
    assert answer.answer_text == generation_module.ABSTENTION_TEXT
    assert answer.confidence_gate_triggered is True
    assert any("evidence type needed" in reason for reason in answer.confidence_reasons or [])


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
    assert answer.confidence_gate_triggered is False


def test_openai_generator_uses_responses_api_with_gpt4o_mini_and_prompt_contract(
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
    _assert_openai_prompt_contract(calls[0]["input"])


def test_openai_generator_falls_back_to_chat_completions_with_same_prompt_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_calls: list[dict[str, object]] = []

    class FakeResponses:
        def create(self, **kwargs):
            raise RuntimeError("responses API unavailable in this runtime")

    class FakeChatCompletions:
        def create(self, **kwargs):
            chat_calls.append(kwargs)
            message = type("Message", (), {"content": "Self-attention compares tokens. [1]"})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeChatCompletions()

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.responses = FakeResponses()
            self.chat = FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", type("OpenAIModule", (), {"OpenAI": FakeOpenAI}))

    generator = generation_module.OpenAIGenerator()
    answer = generator.generate("What is self-attention?", _chunks())
    assert answer.answer_text == "Self-attention compares tokens. [1]"
    assert chat_calls[0]["model"] == "gpt-4o-mini"
    assert chat_calls[0]["temperature"] == 0
    _assert_openai_prompt_contract(chat_calls[0]["messages"])


def test_strip_citation_markers_removes_bracket_refs() -> None:
    assert (
        generation_module.strip_citation_markers("Sentence one [1]. Sentence two [2]")
        == "Sentence one. Sentence two"
    )


def test_non_abstention_answer_keeps_valid_citations_for_embedding_answer() -> None:
    generator = generation_module.OfflineExtractiveGenerator()
    answer = generator.generate(
        "Why are embeddings more useful than one-hot vectors for meaning?",
        _embedding_chunks(),
    )
    warnings = generation_module.validate_citations(
        answer.answer_text,
        _embedding_chunks(),
        abstained=answer.abstained,
    )
    assert warnings == []


def test_multi_hop_answer_stays_within_three_sentences() -> None:
    generator = generation_module.OfflineExtractiveGenerator()
    chunks = [
        RetrievedChunk(
            rank=1,
            chunk_id="faq_rag_002::chunk000",
            source_id="faq_rag_002",
            title="Why is retrieval helpful before answer generation in a RAG system?",
            text=(
                "Question: Why is retrieval helpful before answer generation in a RAG system? "
                "Answer: Retrieval provides external evidence before generation so the answer can "
                "be grounded in relevant documents."
            ),
            score=8.0,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "rag", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="faq_chromadb_vector_search_003::chunk000",
            source_id="faq_chromadb_vector_search_003",
            title="Why is metadata important in vector search?",
            text=(
                "Question: Why is metadata important in vector search? Answer: Metadata lets the "
                "system trace a retrieved chunk back to its source topic and other attributes needed "
                "for inspection or citation."
            ),
            score=7.0,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "chromadb_vector_search", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=3,
            chunk_id="faq_evaluation_metrics_002::chunk000",
            source_id="faq_evaluation_metrics_002",
            title="Why are citations helpful in a grounded QA system?",
            text=(
                "Question: Why are citations helpful in a grounded QA system? Answer: Citations let "
                "a reader inspect which retrieved evidence supported the answer."
            ),
            score=6.2,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "evaluation_metrics", "chunk_index": "0"},
        ),
    ]
    answer = generator.generate(
        "How do retrieval and metadata together support citation inspection in a RAG workflow?",
        chunks,
    )
    assert answer.abstained is False
    assert 1 <= _count_answer_sentences(answer.answer_text) <= 3
