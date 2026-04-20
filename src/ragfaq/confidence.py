from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .config import (
    ABSTENTION_CHROMA_DISTANCE_NORMALIZER,
    ABSTENTION_CHROMA_GAP_NORMALIZER,
    ABSTENTION_CONFIDENCE_THRESHOLD,
    ABSTENTION_MIN_CONTEXT_OVERLAP,
    ABSTENTION_MIN_SENTENCE_QUESTION_OVERLAP,
    ABSTENTION_MIN_SUPPORT_RATIO,
    ABSTENTION_TFIDF_GAP_NORMALIZER,
    ABSTENTION_TFIDF_SCORE_NORMALIZER,
    ABSTENTION_TOP_SIGNAL_FLOOR,
)
from .schemas import RetrievedChunk
from .utils import content_tokens, normalize_text, sentence_split, tokenize

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
PERSON_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
NUMERIC_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\b|%|\bpercent\b|\bpercentage\b|\bexchange rate\b|"
    r"\bcelsius\b|\bfahrenheit\b|\bdegrees?\b|\btemperature\b|\busd\b|\beur\b|\$|€"
)
MONTH_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\b",
    re.IGNORECASE,
)
PERSON_NAME_STOPWORDS = {
    "Answer",
    "Architecture",
    "Attention",
    "Basics",
    "Bert",
    "Chunk",
    "Encoder",
    "Faq",
    "Gradient",
    "Knowledge",
    "Metadata",
    "Model",
    "Networks",
    "Neural",
    "Overview",
    "PyTorch",
    "Question",
    "Rag",
    "Retrieval",
    "Self",
    "Tensor",
    "Tensors",
    "Transformer",
    "Transformers",
    "Vector",
}


class QuestionType(str, Enum):
    PERSON = "person"
    DATE_OR_YEAR = "date_or_year"
    NUMERIC = "numeric"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    REASON = "reason"
    GENERIC = "generic"


@dataclass(frozen=True)
class ConfidenceDecision:
    should_abstain: bool
    confidence_score: float
    question_type: QuestionType
    reasons: list[str]
    top_signal: float
    gap_signal: float
    question_context_overlap: float
    answer_question_overlap: float
    support_ratio: float
    answer_type_evidence_found: bool


def strip_citations(text: str) -> str:
    text = CITATION_PATTERN.sub("", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return normalize_text(text)


def classify_question_type(question: str) -> QuestionType:
    normalized = normalize_text(question).lower()
    if normalized.startswith("who ") or any(
        phrase in normalized for phrase in ("who founded", "who created", "who developed", "who invented")
    ):
        return QuestionType.PERSON
    if re.search(r"\b(in what year|what year|when was|what date|release date|released)\b", normalized):
        return QuestionType.DATE_OR_YEAR
    if re.search(
        r"\b(how many|how much|number of|percentage|percent|exchange rate|temperature|"
        r"boiling point|price|cost|amount)\b",
        normalized,
    ):
        return QuestionType.NUMERIC
    if any(token in normalized for token in ("difference", "different", "compare", "compared", "versus", " than ")):
        return QuestionType.COMPARISON
    if normalized.startswith(("what is ", "what are ", "what does ", "what do ")):
        return QuestionType.DEFINITION
    if normalized.startswith(("why ", "how does ", "how do ", "how can ", "how is ", "how are ")):
        return QuestionType.REASON
    return QuestionType.GENERIC


def _content_overlap_ratio(left_text: str, right_text: str) -> float:
    left_tokens = set(content_tokens(left_text))
    if not left_tokens:
        return 0.0
    right_tokens = set(content_tokens(right_text))
    return round(sum(1 for token in left_tokens if token in right_tokens) / len(left_tokens), 2)


def _sentence_question_overlap_count(sentence: str, question_tokens: set[str]) -> int:
    sentence_tokens = set(content_tokens(sentence))
    return sum(1 for token in sentence_tokens if token in question_tokens)


def _sentence_supported(sentence: str, context: str, context_tokens: set[str]) -> bool:
    normalized_sentence = sentence.strip().lower()
    if not normalized_sentence:
        return False
    if normalized_sentence in context.lower():
        return True
    sentence_tokens = [token for token in content_tokens(sentence) if len(token) > 2]
    if not sentence_tokens:
        return True
    overlap = sum(1 for token in sentence_tokens if token in context_tokens)
    return (overlap / len(sentence_tokens)) >= 0.6


def _detect_answer_type_evidence(question_type: QuestionType, text: str) -> bool:
    if not text.strip():
        return False
    lowered = text.lower()
    if question_type is QuestionType.PERSON:
        candidate_names = []
        for match in PERSON_NAME_PATTERN.findall(text):
            parts = match.split()
            if all(part not in PERSON_NAME_STOPWORDS for part in parts):
                candidate_names.append(match)
        return (
            any(
                phrase in lowered
                for phrase in ("founded by", "created by", "developed by", "invented by")
            )
            or bool(candidate_names)
        )
    if question_type is QuestionType.DATE_OR_YEAR:
        return bool(YEAR_PATTERN.search(text)) or bool(MONTH_PATTERN.search(text)) or "released" in lowered
    if question_type is QuestionType.NUMERIC:
        return bool(NUMERIC_PATTERN.search(text))
    if question_type is QuestionType.DEFINITION:
        return any(
            phrase in lowered for phrase in (" is ", " are ", "means", "refers to", "mechanism", "lets")
        )
    if question_type is QuestionType.COMPARISON:
        return any(
            phrase in lowered for phrase in (" than ", " while ", " instead ", " compared ", "difference")
        )
    if question_type is QuestionType.REASON:
        return any(
            phrase in lowered for phrase in ("because", "helps", "allows", "enables", "so that", "due")
        )
    return bool(content_tokens(text))


def _top_signal(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    chunk = chunks[0]
    if chunk.backend == "chroma":
        if chunk.distance is None:
            return 0.0
        return round(max(0.0, 1.0 - (chunk.distance / ABSTENTION_CHROMA_DISTANCE_NORMALIZER)), 2)
    base_score = chunk.score
    if chunk.backend == "hybrid" and chunk.lexical_score is not None:
        base_score = chunk.lexical_score
    return round(
        min(max(base_score, 0.0) / ABSTENTION_TFIDF_SCORE_NORMALIZER, 1.0),
        2,
    )


def _gap_signal(chunks: list[RetrievedChunk]) -> float:
    if len(chunks) < 3:
        return 0.0
    first = chunks[0]
    third = chunks[2]
    if first.backend == "chroma":
        if first.distance is None or third.distance is None:
            return 0.0
        gap = max(third.distance - first.distance, 0.0)
        return round(min(gap / ABSTENTION_CHROMA_GAP_NORMALIZER, 1.0), 2)
    first_score = first.score
    third_score = third.score
    if first.backend == "hybrid" and first.lexical_score is not None and third.lexical_score is not None:
        first_score = first.lexical_score
        third_score = third.lexical_score
    gap = max(first_score - third_score, 0.0)
    return round(min(gap / ABSTENTION_TFIDF_GAP_NORMALIZER, 1.0), 2)


def _cited_context_for_sentence(sentence: str, chunks: list[RetrievedChunk]) -> str:
    citations = {int(match) for match in CITATION_PATTERN.findall(sentence)}
    if not citations:
        return ""
    cited_chunks = [chunk.text for chunk in chunks if chunk.rank in citations]
    return " ".join(cited_chunks)


def _answer_segments(answer_text: str) -> list[str]:
    normalized = normalize_text(answer_text)
    if not normalized:
        return []
    if CITATION_PATTERN.search(normalized):
        return [segment.strip() for segment in re.split(r"(?<=\])\s+", normalized) if segment.strip()]
    return sentence_split(normalized)


def _support_ratio(answer_text: str, chunks: list[RetrievedChunk]) -> float:
    segments = _answer_segments(answer_text)
    if not segments:
        return 0.0
    supported = 0
    for segment in segments:
        sentence_text = strip_citations(segment)
        context = _cited_context_for_sentence(segment, chunks)
        if not context:
            continue
        context_tokens = set(tokenize(context))
        if _sentence_supported(sentence_text, context, context_tokens):
            supported += 1
    return round(supported / len(segments), 2)


def _answer_question_overlap(question: str, answer_text: str) -> float:
    return _content_overlap_ratio(question, answer_text)


def _confidence_score(
    *,
    question_context_overlap: float,
    top_signal: float,
    gap_signal: float,
    answer_question_overlap: float,
    support_ratio: float,
) -> float:
    score = (
        0.35 * question_context_overlap
        + 0.20 * top_signal
        + 0.10 * gap_signal
        + 0.20 * answer_question_overlap
        + 0.15 * support_ratio
    )
    return round(score, 2)


class RetrievalConfidenceGate:
    def evaluate_retrieval(self, question: str, chunks: list[RetrievedChunk]) -> ConfidenceDecision:
        question_type = classify_question_type(question)
        context = " ".join(chunk.text for chunk in chunks[:3])
        question_context_overlap = _content_overlap_ratio(question, context)
        top_signal = _top_signal(chunks[:3])
        gap_signal = _gap_signal(chunks[:3])
        answer_type_evidence_found = _detect_answer_type_evidence(question_type, context)
        reasons: list[str] = []

        if not chunks:
            reasons.append("no chunks were retrieved")
        if top_signal <= ABSTENTION_TOP_SIGNAL_FLOOR:
            reasons.append("top retrieval signal was near zero")
        if question_context_overlap < ABSTENTION_MIN_CONTEXT_OVERLAP:
            reasons.append("retrieved context had too little overlap with the question")
        if question_type in {
            QuestionType.PERSON,
            QuestionType.DATE_OR_YEAR,
            QuestionType.NUMERIC,
        } and not answer_type_evidence_found:
            reasons.append("retrieved context lacked the evidence type needed for this question")

        confidence_score = _confidence_score(
            question_context_overlap=question_context_overlap,
            top_signal=top_signal,
            gap_signal=gap_signal,
            answer_question_overlap=0.0,
            support_ratio=0.0,
        )
        return ConfidenceDecision(
            should_abstain=bool(reasons),
            confidence_score=confidence_score,
            question_type=question_type,
            reasons=reasons,
            top_signal=top_signal,
            gap_signal=gap_signal,
            question_context_overlap=question_context_overlap,
            answer_question_overlap=0.0,
            support_ratio=0.0,
            answer_type_evidence_found=answer_type_evidence_found,
        )

    def validate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        proposed_answer: str,
    ) -> ConfidenceDecision:
        retrieval_decision = self.evaluate_retrieval(question, chunks)
        question_type = retrieval_decision.question_type
        reasons: list[str] = list(retrieval_decision.reasons)
        raw_answer = strip_citations(proposed_answer)
        answer_sentences = sentence_split(raw_answer)
        question_tokens = set(content_tokens(question))
        answer_question_overlap = _answer_question_overlap(question, raw_answer)
        support_ratio = _support_ratio(proposed_answer, chunks)
        answer_type_evidence_found = _detect_answer_type_evidence(question_type, raw_answer)

        if not answer_sentences:
            reasons.append("no answer sentences passed extractive selection")
        max_sentence_overlap = 0
        if answer_sentences:
            max_sentence_overlap = max(
                _sentence_question_overlap_count(sentence, question_tokens)
                for sentence in answer_sentences
            )
        if max_sentence_overlap < ABSTENTION_MIN_SENTENCE_QUESTION_OVERLAP:
            reasons.append("answer sentences did not match enough question terms")
        if support_ratio < ABSTENTION_MIN_SUPPORT_RATIO:
            reasons.append("final answer was not fully supported by its cited chunks")
        if question_type in {
            QuestionType.PERSON,
            QuestionType.DATE_OR_YEAR,
            QuestionType.NUMERIC,
        } and not answer_type_evidence_found:
            reasons.append("final answer still lacked the evidence type needed for this question")

        confidence_score = _confidence_score(
            question_context_overlap=retrieval_decision.question_context_overlap,
            top_signal=retrieval_decision.top_signal,
            gap_signal=retrieval_decision.gap_signal,
            answer_question_overlap=answer_question_overlap,
            support_ratio=support_ratio,
        )
        if confidence_score < ABSTENTION_CONFIDENCE_THRESHOLD:
            reasons.append("composite confidence score was below the abstention threshold")

        deduped_reasons = list(dict.fromkeys(reasons))
        return ConfidenceDecision(
            should_abstain=bool(deduped_reasons),
            confidence_score=confidence_score,
            question_type=question_type,
            reasons=deduped_reasons,
            top_signal=retrieval_decision.top_signal,
            gap_signal=retrieval_decision.gap_signal,
            question_context_overlap=retrieval_decision.question_context_overlap,
            answer_question_overlap=answer_question_overlap,
            support_ratio=support_ratio,
            answer_type_evidence_found=answer_type_evidence_found,
        )
