from __future__ import annotations

from dataclasses import asdict

from .config import DEFAULT_TOP_K, PathConfig, get_paths
from .generation import answer_question
from .reporting import generate_failure_report
from .retrievers import retrieve
from .schemas import BackendMode, EvaluationRow, LlmMode
from .utils import read_csv_rows, sentence_split, tokenize, write_csv_rows

QUESTION_FIELDNAMES = [
    "question_id",
    "question",
    "expected_answer",
    "expected_source_id",
    "expected_keywords",
    "retrieved_source_ids",
    "recall_at_3",
    "generated_answer",
    "faithfulness_score",
    "resolved_backend",
    "resolved_llm",
    "notes",
]


def _score_recall_at_3(expected_source_id: str, retrieved_source_ids: list[str]) -> float:
    return 1.0 if expected_source_id in retrieved_source_ids[:3] else 0.0


def _sentence_supported(sentence: str, context: str, context_tokens: set[str]) -> bool:
    normalized_sentence = sentence.strip().lower()
    if not normalized_sentence:
        return False
    if normalized_sentence in context.lower():
        return True
    sentence_tokens = [token for token in tokenize(sentence) if len(token) > 2]
    if not sentence_tokens:
        return True
    overlap = sum(1 for token in sentence_tokens if token in context_tokens)
    return (overlap / len(sentence_tokens)) >= 0.6


def score_faithfulness(answer: str, context_chunks: list[str]) -> float:
    context = " ".join(context_chunks)
    context_tokens = set(tokenize(context))
    sentences = sentence_split(answer)
    if not sentences:
        return 0.0
    supported = sum(
        1 for sentence in sentences if _sentence_supported(sentence, context, context_tokens)
    )
    return round(supported / max(len(sentences), 1), 2)


def _row_notes(recall_at_3: float, faithfulness_score: float) -> str:
    notes = []
    if recall_at_3 < 1.0:
        notes.append("expected source not retrieved in top 3")
    if faithfulness_score < 1.0:
        notes.append("answer only partially supported by retrieved context")
    return "; ".join(notes)


def run_evaluation(
    requested_backend: BackendMode,
    requested_llm: LlmMode,
    paths: PathConfig | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[EvaluationRow]:
    paths = paths or get_paths()
    rows = read_csv_rows(paths.test_questions_path)
    results: list[EvaluationRow] = []

    for row in rows:
        question = row["question"]
        retrieved_chunks, resolved_backend = retrieve(
            question,
            requested_backend=requested_backend,
            top_k=top_k,
            paths=paths,
        )
        answer = answer_question(
            question=question,
            retrieved_chunks=retrieved_chunks,
            requested_llm=requested_llm,
            resolved_backend=resolved_backend,
        )
        retrieved_source_ids = []
        for chunk in retrieved_chunks:
            if chunk.source_id not in retrieved_source_ids:
                retrieved_source_ids.append(chunk.source_id)

        recall_at_3 = _score_recall_at_3(row["expected_source_id"], retrieved_source_ids)
        faithfulness_score = score_faithfulness(
            answer.answer, [chunk.text for chunk in retrieved_chunks]
        )
        result = EvaluationRow(
            question_id=row["question_id"],
            question=row["question"],
            expected_answer=row["expected_answer"],
            expected_source_id=row["expected_source_id"],
            expected_keywords=row["expected_keywords"],
            retrieved_source_ids=";".join(retrieved_source_ids),
            recall_at_3=recall_at_3,
            generated_answer=answer.answer,
            faithfulness_score=faithfulness_score,
            resolved_backend=answer.resolved_backend.value,
            resolved_llm=answer.resolved_llm.value,
            notes=_row_notes(recall_at_3, faithfulness_score),
        )
        results.append(result)

    write_csv_rows(
        paths.test_questions_path,
        QUESTION_FIELDNAMES,
        [
            {
                **asdict(result),
                "recall_at_3": f"{result.recall_at_3:.2f}",
                "faithfulness_score": f"{result.faithfulness_score:.2f}",
            }
            for result in results
        ],
    )
    generate_failure_report(results, paths.failure_report_path)
    return results

