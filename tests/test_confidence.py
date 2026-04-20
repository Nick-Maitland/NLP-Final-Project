from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.confidence import QuestionType, RetrievalConfidenceGate
from ragfaq.schemas import RetrievedChunk


def _answerable_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=1,
            chunk_id="faq_attention_003::chunk000",
            source_id="faq_attention_003",
            title="What is self attention?",
            text=(
                "Question: What is self attention? Answer: Self attention lets tokens in the "
                "same sequence compare with one another so each token representation can "
                "incorporate broader context."
            ),
            score=6.95,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "attention", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="faq_attention_006::chunk000",
            source_id="faq_attention_006",
            title="Attention advantage",
            text=(
                "Self attention can connect distant tokens directly without passing information "
                "step by step through intermediate positions."
            ),
            score=6.25,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "attention", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=3,
            chunk_id="self_attention::chunk000",
            source_id="self_attention_and_transformer_architecture",
            title="Self-attention and transformer architecture",
            text=(
                "Self-attention is a mechanism that lets each token compare itself with other "
                "tokens in the same sequence and compute a weighted combination of their "
                "representations."
            ),
            score=5.76,
            backend="tfidf",
            metadata={
                "source": "knowledge_base/docs/self_attention_and_transformer_architecture.md",
                "topic": "attention",
                "chunk_index": "0",
            },
        ),
    ]


def _unsupported_factoid_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=1,
            chunk_id="pytorch::chunk001",
            source_id="tensors_and_pytorch_basics",
            title="PyTorch basics",
            text="PyTorch is popular because it supports automatic differentiation.",
            score=4.02,
            backend="tfidf",
            metadata={"source": "knowledge_base/docs/tensors_and_pytorch_basics.md", "topic": "pytorch", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="pytorch::chunk002",
            source_id="tensors_and_pytorch_basics",
            title="PyTorch basics",
            text="PyTorch tensors can move between CPU and GPU.",
            score=3.42,
            backend="tfidf",
            metadata={"source": "knowledge_base/docs/tensors_and_pytorch_basics.md", "topic": "pytorch", "chunk_index": "1"},
        ),
        RetrievedChunk(
            rank=3,
            chunk_id="nn::chunk000",
            source_id="neural_networks_basics",
            title="Neural networks",
            text="A perceptron adds weighted inputs and a bias before an activation.",
            score=0.0,
            backend="tfidf",
            metadata={"source": "knowledge_base/docs/neural_networks_basics.md", "topic": "neural_networks", "chunk_index": "0"},
        ),
    ]


def _irrelevant_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=1,
            chunk_id="misc::chunk000",
            source_id="misc",
            title="Optimization",
            text="Gradient descent updates parameters using gradients.",
            score=0.0,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "optimization", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="misc::chunk001",
            source_id="misc",
            title="Optimization",
            text="Learning rates control the step size during optimization.",
            score=0.0,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "optimization", "chunk_index": "1"},
        ),
        RetrievedChunk(
            rank=3,
            chunk_id="misc::chunk002",
            source_id="misc",
            title="Optimization",
            text="Momentum smooths parameter updates across steps.",
            score=0.0,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "optimization", "chunk_index": "2"},
        ),
    ]


def test_confidence_gate_allows_supported_definition_question() -> None:
    gate = RetrievalConfidenceGate()
    retrieval = gate.evaluate_retrieval("What is self-attention?", _answerable_chunks())
    assert retrieval.question_type is QuestionType.DEFINITION
    assert retrieval.should_abstain is False

    answer = gate.validate_answer(
        "What is self-attention?",
        _answerable_chunks(),
        "Self attention lets tokens in the same sequence compare with one another. [1]",
    )
    assert answer.should_abstain is False
    assert answer.confidence_score >= 0.58


def test_confidence_gate_rejects_topical_but_unsupported_factoid() -> None:
    gate = RetrievalConfidenceGate()
    retrieval = gate.evaluate_retrieval("Who founded PyTorch?", _unsupported_factoid_chunks())
    assert retrieval.question_type is QuestionType.PERSON
    assert retrieval.should_abstain is True
    assert any("evidence type needed" in reason for reason in retrieval.reasons)


def test_confidence_gate_rejects_low_confidence_retrieval() -> None:
    gate = RetrievalConfidenceGate()
    decision = gate.evaluate_retrieval("What is the capital city of France?", _irrelevant_chunks())
    assert decision.should_abstain is True
    assert any("near zero" in reason or "too little overlap" in reason for reason in decision.reasons)


def test_confidence_gate_rejects_unsupported_final_answer() -> None:
    gate = RetrievalConfidenceGate()
    decision = gate.validate_answer(
        "Why is retrieval helpful in RAG?",
        [
            RetrievedChunk(
                rank=1,
                chunk_id="faq_rag::chunk000",
                source_id="faq_rag_001",
                title="RAG",
                text="Retrieval provides external evidence before answer generation.",
                score=7.0,
                backend="tfidf",
                metadata={"source": "knowledge_base/faqs.csv", "topic": "rag", "chunk_index": "0"},
            )
        ],
        "Retrieval helps because it uses hidden states. [1]",
    )
    assert decision.should_abstain is True
    assert any("fully supported" in reason or "below the abstention threshold" in reason for reason in decision.reasons)
