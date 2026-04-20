from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BackendMode(str, Enum):
    CHROMA = "chroma"
    TFIDF = "tfidf"
    HYBRID = "hybrid"
    AUTO = "auto"


class LlmMode(str, Enum):
    OPENAI = "openai"
    OFFLINE = "offline"
    AUTO = "auto"


@dataclass(frozen=True)
class Document:
    source_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    text: str
    token_count: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    chunk_id: str
    source_id: str
    title: str
    text: str
    score: float
    backend: str
    distance: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    fusion_score: float | None = None
    mmr_score: float | None = None
    selection_reason: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalRunResult:
    chunks: list[RetrievedChunk]
    resolved_backend: BackendMode
    trace: dict[str, object] | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    sources: list[str]
    resolved_backend: BackendMode
    resolved_llm: LlmMode
    retrieved_chunks: list[RetrievedChunk]
    answer_text: str = ""
    raw_answer_text: str = ""
    citation_warnings: list[str] = field(default_factory=list)
    abstained: bool = False
    confidence_score: float = 0.0
    confidence_reasons: list[str] = field(default_factory=list)
    confidence_gate_triggered: bool = False
    question_type: str = ""


@dataclass(frozen=True)
class EvaluationRow:
    question_id: str
    question: str
    expected_source_id: str
    expected_topic: str
    answerable: bool
    retrieved_source_ids: str
    retrieval_recall_at_3: float | None
    reciprocal_rank: float | None
    faithfulness_score: float
    citation_valid: bool
    abstention_correct: bool
    answer: str
    notes: str
    abstained: bool = False
    latency_ms: float = 0.0
    resolved_backend: str = ""
    resolved_llm: str = ""
    citation_warnings: str = ""
    retrieved_chunk_ids: str = ""
    retrieved_chunk_summaries: str = ""
    failure_type: str = ""
