from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq import evaluation as evaluation_module
from ragfaq.generation import ABSTENTION_TEXT
from ragfaq.schemas import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=1,
            chunk_id="faq_attention_001::chunk000",
            source_id="faq_attention_001",
            title="Attention",
            text="Self-attention lets each token compare with other tokens in the same sequence.",
            score=0.9,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "attention", "chunk_index": "0"},
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="faq_transformers_001::chunk000",
            source_id="faq_transformers_001",
            title="Transformers",
            text="Transformers process tokens in parallel within a layer instead of one step at a time.",
            score=0.8,
            backend="tfidf",
            metadata={
                "source": "knowledge_base/faqs.csv",
                "topic": "transformers",
                "chunk_index": "0",
            },
        ),
        RetrievedChunk(
            rank=3,
            chunk_id="faq_rag_001::chunk000",
            source_id="faq_rag_001",
            title="RAG",
            text="Retrieval gives answer generation external evidence before composing a response.",
            score=0.7,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "rag", "chunk_index": "0"},
        ),
    ]


def test_recall_and_reciprocal_rank_cover_single_and_multi_hop_cases() -> None:
    retrieved = ["faq_transformers_001", "faq_attention_001", "faq_rag_001"]
    assert evaluation_module._score_recall_at_3(["faq_attention_001"], retrieved) == 1.0
    assert evaluation_module._score_reciprocal_rank(["faq_attention_001"], retrieved) == 0.5
    assert (
        evaluation_module._score_recall_at_3(
            ["faq_attention_001", "faq_chromadb_001"],
            retrieved,
        )
        == 0.5
    )
    assert evaluation_module._score_reciprocal_rank([], retrieved) is None


def test_faithfulness_rewards_grounded_answers_and_penalizes_irrelevant_answers() -> None:
    grounded = evaluation_module.score_faithfulness(
        "How does self-attention help?",
        "Self-attention lets each token compare with other tokens in the same sequence. [1]",
        _chunks(),
        citation_valid=True,
        abstained=False,
    )
    irrelevant = evaluation_module.score_faithfulness(
        "Who founded PyTorch?",
        "PyTorch is popular because it supports automatic differentiation. [1]",
        _chunks(),
        citation_valid=True,
        abstained=False,
    )
    assert grounded > irrelevant
    assert grounded >= 0.8
    assert irrelevant <= 0.5


def test_faithfulness_handles_abstention_consistency() -> None:
    supportive_chunks = _chunks()
    irrelevant_chunks = [
        RetrievedChunk(
            rank=1,
            chunk_id="faq_misc::chunk000",
            source_id="faq_misc",
            title="Misc",
            text="Gradient descent updates parameters using gradients.",
            score=0.5,
            backend="tfidf",
            metadata={"source": "knowledge_base/faqs.csv", "topic": "optimization", "chunk_index": "0"},
        )
    ]
    assert (
        evaluation_module.score_faithfulness(
            "What is the capital city of France?",
            ABSTENTION_TEXT,
            irrelevant_chunks,
            citation_valid=True,
            abstained=True,
        )
        == 1.0
    )
    assert (
        evaluation_module.score_faithfulness(
            "What is self-attention?",
            ABSTENTION_TEXT,
            supportive_chunks,
            citation_valid=True,
            abstained=True,
        )
        < 1.0
    )
