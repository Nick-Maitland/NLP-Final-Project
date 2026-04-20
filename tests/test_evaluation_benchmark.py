from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "evaluation_questions.csv"
EXPECTED_COLUMNS = [
    "question_id",
    "question",
    "expected_source_id",
    "expected_topic",
    "answerable",
    "question_type",
    "difficulty",
    "notes",
]
DISALLOWED_COLUMNS = {
    "retrieved_source_ids",
    "retrieval_recall_at_3",
    "faithfulness_score",
    "answer",
    "latency_ms",
}


def _read_benchmark() -> tuple[list[dict[str, str]], list[str]]:
    with BENCHMARK_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def test_evaluation_questions_csv_has_expected_structure_and_minimum_counts() -> None:
    rows, fieldnames = _read_benchmark()
    assert fieldnames == EXPECTED_COLUMNS
    assert len(rows) >= 30
    assert len({row["question_id"] for row in rows}) == len(rows)
    assert sum(row["answerable"] == "true" for row in rows) >= 24
    assert sum(row["answerable"] == "false" for row in rows) >= 6


def test_evaluation_questions_csv_stays_clean_of_generated_columns() -> None:
    _, fieldnames = _read_benchmark()
    assert DISALLOWED_COLUMNS.isdisjoint(fieldnames)
