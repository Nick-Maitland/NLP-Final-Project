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
    chunk_id: str
    source_id: str
    title: str
    text: str
    score: float
    backend: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    sources: list[str]
    resolved_backend: BackendMode
    resolved_llm: LlmMode
    retrieved_chunks: list[RetrievedChunk]


@dataclass(frozen=True)
class EvaluationRow:
    question_id: str
    question: str
    expected_answer: str
    expected_source_id: str
    expected_keywords: str
    retrieved_source_ids: str
    recall_at_3: float
    generated_answer: str
    faithfulness_score: float
    resolved_backend: str
    resolved_llm: str
    notes: str

